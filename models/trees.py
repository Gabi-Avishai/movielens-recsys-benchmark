"""
Gradient-boosted trees over the same protocol.

Two representations, deliberately:

`LightGBMSparse` consumes the *identical* 13,882-column design matrix the FM is trained
on. This is the controlled comparison -- same rows, same labels, same splits -- and it is
expected to do badly. Trees split one feature at a time, so a 6,040-way one-hot id gives
almost no information gain per split and offers no mechanism to generalise across ids the
way a factorised embedding does. Reporting only this number would measure nothing except
that trees dislike one-hot ids.

`LightGBMNative` consumes the features a practitioner would actually build: native
categoricals for the ids, plus dense aggregates. Every dense feature here is a
hand-engineered stand-in for something the FM gets implicitly from <v_u, v_i>, which is
the point of the comparison rather than an embarrassment about it.

The gap between the two is the real result: how much of "trees versus FM" is actually
"representation suits the model".

User-side features are derived from `Dataset.history_matrix`, the same causal window the
FM's history block uses, so the two models see the same information about a user at the
same moment. Item-side aggregates come from the training split only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

import lightgbm as lgb

from data import ALL_BLOCKS, GENRES
from .base import Recommender

CATEGORICALS = ["u", "i", "gender", "age", "occupation"]
GENRE_FLAGS = ["Film-Noir", "Documentary", "War", "Drama", "Horror",
               "Children's", "Sci-Fi", "Comedy"]


class _LGBBase(Recommender):
    def __init__(self, objective: str = "binary", n_estimators: int = 400,
                 learning_rate: float = 0.05, num_leaves: int = 63,
                 min_child_samples: int = 50, subsample: float = 0.8,
                 colsample: float = 0.8, neg_policy: str = "unobserved_only",
                 neg_ratio: int = 1, seed: int = 42, early_stopping: int = 30,
                 verbose: bool = True, name: str | None = None):
        self.objective, self.n_estimators = objective, n_estimators
        self.learning_rate, self.num_leaves = learning_rate, num_leaves
        self.min_child_samples, self.subsample, self.colsample = min_child_samples, subsample, colsample
        self.neg_policy, self.neg_ratio, self.seed = neg_policy, neg_ratio, seed
        self.early_stopping, self.verbose = early_stopping, verbose
        self.negatives = f"{neg_policy} 1:{neg_ratio}"
        if name: self.name = name
        self.history_ = []

    def _params(self):
        p = dict(objective=self.objective, learning_rate=self.learning_rate,
                 num_leaves=self.num_leaves, min_child_samples=self.min_child_samples,
                 bagging_fraction=self.subsample, bagging_freq=1,
                 feature_fraction=self.colsample, seed=self.seed,
                 verbosity=-1, num_threads=0)
        if self.objective == "lambdarank":
            p.update(metric="ndcg", ndcg_eval_at=[10], lambdarank_truncation_level=30)
        else:
            p.update(metric="binary_logloss")
        return p

    def _fit_booster(self, Xtr, ytr, gtr, Xva, yva, gva, feature_name=None, categorical=None):
        dtr = lgb.Dataset(Xtr, label=ytr, group=gtr, feature_name=feature_name or "auto",
                          categorical_feature=categorical or "auto", free_raw_data=False)
        valid = []
        if Xva is not None:
            valid = [lgb.Dataset(Xva, label=yva, group=gva, reference=dtr, free_raw_data=False)]
        cbs = [lgb.log_evaluation(period=50 if self.verbose else 0)]
        if valid:
            cbs.append(lgb.early_stopping(self.early_stopping, verbose=self.verbose))
        evals = {}
        cbs.append(lgb.record_evaluation(evals))
        self.booster = lgb.train(self._params(), dtr, num_boost_round=self.n_estimators,
                                 valid_sets=valid, callbacks=cbs)
        self.best_iteration_ = self.booster.best_iteration or self.n_estimators
        for metric, vals in (evals.get("valid_0", {}) or {}).items():
            self.history_ = [{"iteration": i + 1, f"val_{metric}": v}
                             for i, v in enumerate(vals)]
        return self


class LightGBMSparse(_LGBBase):
    """The controlled arm: the FM's exact design matrix, fed to a tree."""
    name = "lgbm_sparse"

    def __init__(self, blocks=ALL_BLOCKS, **kw):
        super().__init__(**kw)
        self.blocks = tuple(blocks)

    def _design(self, ds, rows, seed):
        r = ds.sample_negatives(rows, self.neg_policy, self.neg_ratio, seed=seed)
        if self.objective == "lambdarank":
            r = r.sort_values("u", kind="mergesort").reset_index(drop=True)
        X = ds.feature_matrix(r, self.blocks).astype(np.float32)
        g = r.groupby("u", sort=False).size().to_numpy() if self.objective == "lambdarank" else None
        return X, r["label"].to_numpy(np.float32), g

    def fit(self, ds, train_rows, val_rows=None):
        Xtr, ytr, gtr = self._design(ds, train_rows, self.seed)
        Xva = yva = gva = None
        if val_rows is not None:
            Xva, yva, gva = self._design(ds, val_rows, self.seed + 1)
        self._ds_blocks = self.blocks
        return self._fit_booster(Xtr, ytr, gtr, Xva, yva, gva)

    def score(self, ds, u, items):
        frame = ds.scoring_frame(u, items)
        X = ds.feature_matrix(frame, self.blocks).astype(np.float32)
        return self.booster.predict(X, num_iteration=self.best_iteration_)


class LightGBMNative(_LGBBase):
    """The practitioner arm: native categoricals plus dense engineered aggregates.

    Each dense feature stands in for something the FM learns rather than is told:

      item_pop / item_mean / item_std     what an item factor encodes about quality+exposure
      user_n / user_mean / user_std       what a user factor encodes about activity+generosity
      genre_affinity                      the user x item genre cross, computed not learned
      genre_overlap_frac                  how much of the causal window shares a genre
      year_gap                            distance from the user's typical release era
      hist_len                            how much evidence the user-side features rest on

    Everything user-side is read from the same causal history window the FM's history
    block uses, so neither model sees more about a user than the other at the same row.
    """
    name = "lgbm_native"

    def __init__(self, use_content: bool = True, cf_score=None, **kw):
        super().__init__(**kw)
        self.use_content = use_content
        self.cf_score = cf_score          # optional (n_users x n_items) matrix, see Phase 7

    # ------------------------------------------------------------ features

    def _prepare(self, ds):
        tr = ds.split_rows("train")
        n = ds.n_items
        cnt = np.bincount(tr["i"], minlength=n).astype(np.float32)
        tot = np.bincount(tr["i"], weights=tr["rating"], minlength=n).astype(np.float32)
        sq = np.bincount(tr["i"], weights=tr["rating"] ** 2, minlength=n).astype(np.float32)
        mean = np.divide(tot, cnt, out=np.full(n, tr["rating"].mean(), np.float32), where=cnt > 0)
        var = np.divide(sq, cnt, out=np.zeros(n, np.float32), where=cnt > 0) - mean ** 2
        self.item_pop_, self.item_mean_, self.item_std_ = cnt, mean, np.sqrt(np.maximum(var, 0))

        un = np.bincount(tr["u"], minlength=ds.n_users).astype(np.float32)
        utot = np.bincount(tr["u"], weights=tr["rating"], minlength=ds.n_users).astype(np.float32)
        usq = np.bincount(tr["u"], weights=tr["rating"] ** 2, minlength=ds.n_users).astype(np.float32)
        umean = np.divide(utot, un, out=np.full(ds.n_users, tr["rating"].mean(), np.float32), where=un > 0)
        uvar = np.divide(usq, un, out=np.zeros(ds.n_users, np.float32), where=un > 0) - umean ** 2
        self.user_n_, self.user_mean_, self.user_std_ = un, umean, np.sqrt(np.maximum(uvar, 0))

        self.G_ = ds.item_genre.toarray().astype(np.float32)
        gn = np.linalg.norm(self.G_, axis=1, keepdims=True)
        self.Gn_ = self.G_ / np.maximum(gn, 1e-6)
        self.year_ = ds.item_year_raw.astype(np.float32)
        demo = ds.user_demo.toarray()
        self.gender_ = demo[:, [c.startswith("gender_") for c in ds.user_demo_names]].argmax(1)
        self.age_ = demo[:, [c.startswith("age_") for c in ds.user_demo_names]].argmax(1)
        self.occ_ = demo[:, [c.startswith("occ_") for c in ds.user_demo_names]].argmax(1)
        self._prepared = True

    def _features(self, ds, rows):
        """Dense feature frame for arbitrary (u, i) rows, using the causal history window."""
        if not getattr(self, "_prepared", False):
            self._prepare(ds)
        u = rows["u"].to_numpy(np.int32)
        i = rows["i"].to_numpy(np.int32)
        H = ds.history_matrix(rows)                       # causal, row-aligned
        hist_len = np.asarray((H > 0).sum(axis=1)).ravel().astype(np.float32)
        UG = np.asarray(H @ self.G_, dtype=np.float32)    # genre counts in the window
        ug_sum = np.maximum(UG.sum(axis=1, keepdims=True), 1e-6)
        UGn = UG / ug_sum
        gi = self.Gn_[i]
        genre_affinity = (UGn * gi).sum(axis=1)
        genre_overlap = (UG * (self.G_[i] > 0)).sum(axis=1) / ug_sum.ravel()
        hist_year = np.divide(np.asarray(H @ self.year_).ravel(),
                              np.maximum(hist_len, 1), out=np.zeros(len(rows), np.float32),
                              where=hist_len > 0)

        d = {
            "u": u.astype(np.int32), "i": i.astype(np.int32),
            "gender": self.gender_[u].astype(np.int32),
            "age": self.age_[u].astype(np.int32),
            "occupation": self.occ_[u].astype(np.int32),
            "item_pop": np.log1p(self.item_pop_[i]),
            "item_mean": self.item_mean_[i],
            "item_std": self.item_std_[i],
            "user_n": np.log1p(self.user_n_[u]),
            "user_mean": self.user_mean_[u],
            "user_std": self.user_std_[u],
            "hist_len": hist_len,
            "genre_affinity": genre_affinity.astype(np.float32),
            "genre_overlap_frac": genre_overlap.astype(np.float32),
        }
        if self.use_content:
            d["year"] = self.year_[i]
            d["year_gap"] = (self.year_[i] - hist_year).astype(np.float32)
            # A handful of the highest-signal genres as explicit flags (Phase 1
            # measured Film-Noir +18.0 pts and Horror -12.9 pts on positive rate).
            for g in GENRE_FLAGS:
                d[f"is_{g}"] = self.G_[i, GENRES.index(g)]
        if self.cf_score is not None:
            d["cf_score"] = self.cf_score[u, i].astype(np.float32)
        return pd.DataFrame(d)

    # ---------------------------------------------------------------- fit

    def _design(self, ds, rows, seed):
        r = ds.sample_negatives(rows, self.neg_policy, self.neg_ratio, seed=seed)
        if self.objective == "lambdarank":
            r = r.sort_values("u", kind="mergesort").reset_index(drop=True)
        X = self._features(ds, r)
        g = r.groupby("u", sort=False).size().to_numpy() if self.objective == "lambdarank" else None
        return X, r["label"].to_numpy(np.float32), g

    def fit(self, ds, train_rows, val_rows=None):
        Xtr, ytr, gtr = self._design(ds, train_rows, self.seed)
        Xva = yva = gva = None
        if val_rows is not None:
            Xva, yva, gva = self._design(ds, val_rows, self.seed + 1)
        self.feature_names_ = list(Xtr.columns)
        cats = [c for c in CATEGORICALS if c in Xtr.columns]
        return self._fit_booster(Xtr, ytr, gtr, Xva, yva, gva,
                                 feature_name=self.feature_names_, categorical=cats)

    def score(self, ds, u, items):
        X = self._features(ds, ds.scoring_frame(u, items))
        return self.booster.predict(X, num_iteration=self.best_iteration_)

    def importances(self, kind="gain"):
        return (pd.Series(self.booster.feature_importance(kind), index=self.feature_names_)
                .sort_values(ascending=False))
