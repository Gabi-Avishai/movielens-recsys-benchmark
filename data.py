"""
Canonical MovieLens-1M loading, splitting and feature construction.

This module defines the *frozen protocol* for the benchmark. Every model consumes
splits and feature matrices from here; no model may re-split, re-threshold, or
build its own history features. If a comparison between two models is to mean
anything, this file is the only thing they are allowed to share.

Key invariants, all asserted rather than assumed:
  - History features for a row use only interactions strictly earlier than that row.
  - Test-time candidate sets exclude train+val history only, never test items.
  - The split is deterministic given `Config`; timestamp ties break by item id.
"""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd
import scipy.sparse as sp

GENRES = [
    "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
    "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
]

AGE_MAP = {1: "Under 18", 18: "18-24", 25: "25-34", 35: "35-44",
           45: "45-49", 50: "50-55", 56: "56+"}

YEAR_BINS = [0, 1939, 1949, 1959, 1969, 1979, 1989, 1994, 1997, 2000, 9999]
YEAR_LABELS = ["<1940", "1940s", "1950s", "1960s", "1970s", "1980s",
               "1990-94", "1995-97", "1998-2000", ">2000"]

# Feature blocks available to feature_matrix(). The ablation in Phase 4 is
# expressed purely as a choice of subset here.
ALL_BLOCKS = ("user_id", "item_id", "user_demo", "item_genre", "item_year",
              "history", "genre_history")
MF_BLOCKS = ("user_id", "item_id")

_SEARCH_PATHS = [
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"),
]


@dataclass(frozen=True)
class Config:
    """The frozen protocol. Changing any field changes the cache key."""
    pos_threshold: int = 4          # rating >= this is a positive
    train_frac: float = 0.70
    val_frac: float = 0.15          # test_frac is the remainder
    max_history: int = 50           # recency window for history features
    snap_ties: bool = True          # move split boundaries off tied timestamps
    user_frac: float = 1.0          # deterministic user subsample, for scaling studies
    min_item_ratings: int = 50      # for the "filtered" candidate setting
    seed: int = 42

    def key(self) -> str:
        raw = repr(sorted(asdict(self).items())).encode()
        return hashlib.md5(raw).hexdigest()[:10]


def find_data_dir() -> str:
    for p in _SEARCH_PATHS:
        if os.path.exists(os.path.join(p, "ratings.dat")):
            return p
    raise FileNotFoundError(
        "ML-1M .dat files not found. Searched:\n  " + "\n  ".join(_SEARCH_PATHS))


def load_raw(data_dir: str | None = None):
    """Read the three ML-1M .dat files with their original column names."""
    d = data_dir or find_data_dir()
    ratings = pd.read_csv(os.path.join(d, "ratings.dat"), sep="::", header=None,
                          engine="python", names=["UserID", "MovieID", "Rating", "Timestamp"])
    movies = pd.read_csv(os.path.join(d, "movies.dat"), sep="::", header=None,
                         engine="python", encoding="latin1", names=["MovieID", "Title", "Genre"])
    users = pd.read_csv(os.path.join(d, "users.dat"), sep="::", header=None,
                        engine="python", names=["UserID", "Gender", "Age", "Occupation", "Zip"])
    return ratings, movies, users


def _extract_year(title: str) -> int:
    m = re.findall(r"\((\d{4})\)", str(title))
    return int(m[-1]) if m else 0


class Dataset:
    """ML-1M under the frozen protocol.

    Attributes
    ----------
    inter : DataFrame
        One row per observed rating, sorted by (u, ts, i). Columns:
        u, i (contiguous 0-based indices), rating, ts, label, seq, split.
        `seq` is the row's 0-based position in that user's chronological
        sequence and is the sole basis for causal history construction.
    """

    def __init__(self, cfg: Config = Config(), data_dir: str | None = None, verbose: bool = True):
        self.cfg = cfg
        self._log = print if verbose else (lambda *a, **k: None)
        self._build(data_dir)

    # ---------------------------------------------------------------- build

    def _build(self, data_dir):
        ratings, movies, users = load_raw(data_dir)
        cfg = self.cfg

        # Contiguous indices. Item space is the FULL catalogue (3883), not just
        # rated items (3706) -- otherwise 177 items are silently unrecommendable
        # and coverage is measured against the wrong denominator.
        self.item_ids = np.sort(movies["MovieID"].unique())
        self.user_ids = np.sort(users["UserID"].unique())
        self.i2idx = {m: k for k, m in enumerate(self.item_ids)}
        self.u2idx = {u: k for k, u in enumerate(self.user_ids)}
        self.n_users, self.n_items = len(self.user_ids), len(self.item_ids)

        # A user subsample keeps the protocol identical and shrinks only the
        # population, which is what a scaling curve needs: every other choice --
        # split, threshold, candidate rule -- stays fixed across scales.
        if cfg.user_frac < 1.0:
            rng = np.random.default_rng(cfg.seed)
            keep = rng.choice(self.user_ids, size=int(len(self.user_ids) * cfg.user_frac),
                              replace=False)
            self.user_ids = np.sort(keep)
            self.u2idx = {u: k for k, u in enumerate(self.user_ids)}
            self.n_users = len(self.user_ids)
            users = users[users["UserID"].isin(self.user_ids)]
            ratings = ratings[ratings["UserID"].isin(self.user_ids)]

        df = ratings.copy()
        df["u"] = df["UserID"].map(self.u2idx).astype(np.int32)
        df["i"] = df["MovieID"].map(self.i2idx).astype(np.int32)
        assert df["i"].notna().all(), "rating references an unknown MovieID"

        # Deterministic chronological order. 53% of rows share a timestamp with
        # another rating by the same user (bulk rating sessions), so ties are
        # broken by item id to make the split reproducible rather than arbitrary.
        df = df.sort_values(["u", "Timestamp", "i"], kind="mergesort").reset_index(drop=True)
        df["seq"] = df.groupby("u", sort=False).cumcount().astype(np.int32)
        df["n_u"] = df.groupby("u", sort=False)["u"].transform("size").astype(np.int32)
        df["label"] = (df["Rating"] >= cfg.pos_threshold).astype(np.int8)

        df["split"] = pd.Categorical(self._assign_splits(df, cfg),
                                     categories=["train", "val", "test"])

        self.inter = df[["u", "i", "Rating", "Timestamp", "label", "seq", "n_u", "split"]] \
            .rename(columns={"Rating": "rating", "Timestamp": "ts"})

        # Per-user chronological item sequence, the substrate for causal history.
        self._user_items = {
            u: g["i"].to_numpy(np.int32)
            for u, g in self.inter.groupby("u", sort=True)
        }
        self._user_offset = self.inter.groupby("u", sort=True).indices  # row ids per user

        # Popularity is computed on TRAIN ONLY. Using full-data popularity would
        # leak the test period into every popularity baseline.
        tr = self.inter[self.inter["split"] == "train"]
        self.item_train_count = np.bincount(tr["i"], minlength=self.n_items)
        self.popular_items = np.argsort(-self.item_train_count)
        self.established = np.where(self.item_train_count >= cfg.min_item_ratings)[0]

        # Candidate exclusion and ground truth, per evaluation target.
        # Tuning on `val` must not see val items in the candidate-exclusion set, and
        # must never touch test -- otherwise hyperparameter selection leaks the very
        # split the benchmark reports on.
        self._seen_by = {}
        self._truth_by = {}
        for target, prior in (("val", ("train",)), ("test", ("train", "val"))):
            seen = self.inter[self.inter["split"].isin(prior)]
            self._seen_by[target] = seen.groupby("u")["i"].apply(
                lambda s: set(s.to_numpy())).to_dict()
            tg = self.inter[(self.inter["split"] == target) & (self.inter["label"] == 1)]
            self._truth_by[target] = tg.groupby("u")["i"].apply(
                lambda s: set(s.to_numpy())).to_dict()
        # Back-compat aliases: the unqualified names always mean the test target.
        self._seen_by_user = self._seen_by["test"]
        self._truth = self._truth_by["test"]

        # Per-user interaction counts, exposed canonically so every caller slices
        # on the same definition rather than reinventing one.
        self.user_train_count = np.bincount(tr["u"], minlength=self.n_users)
        self.user_seen_count = np.array(
            [len(self._seen_by_user.get(u, set())) for u in range(self.n_users)])
        self.user_total_count = np.bincount(self.inter["u"], minlength=self.n_users)

        # One-hot blocks are rebuilt on every feature_matrix() call otherwise.
        self._eye_u = sp.identity(self.n_users, format="csr", dtype=np.float32)
        self._eye_i = sp.identity(self.n_items, format="csr", dtype=np.float32)

        self._encode_side_features(movies, users)
        self._check_invariants()
        self._log(f"Dataset ready: {len(self.inter):,} interactions | "
                  f"{self.n_users:,} users | {self.n_items:,} items | cfg {cfg.key()}")
        for s in ("train", "val", "test"):
            n = int((self.inter["split"] == s).sum())
            p = float(self.inter.loc[self.inter["split"] == s, "label"].mean())
            self._log(f"  {s:<5} {n:>9,}  ({100*n/len(self.inter):4.1f}%)  positive rate {p:.3f}")

    @staticmethod
    def _assign_splits(df: pd.DataFrame, cfg: Config) -> np.ndarray:
        """Per-user chronological split.

        With `snap_ties`, each boundary moves to the nearest timestamp edge so that a
        burst of ratings sharing one second is never divided. 52.9% of rows share a
        timestamp with a sibling, and a purely positional cut therefore splits a burst
        for roughly half of all users on each boundary -- letting same-session
        interactions land on opposite sides of the split.

        A user whose bursts are large enough that snapping would empty one of their
        splits falls back to the positional cut. That is 72 users (1.2%); the remaining
        98.8% of boundary crossings are removed, split shares land closer to target
        (69.9/15.0/15.0 vs 70.3/15.0/14.7), and per-split positive rates move by at
        most 0.19 points.
        """
        out = np.empty(len(df), dtype=object)
        for _, g in df.groupby("u", sort=False):
            idx = g.index.to_numpy()
            t = g["Timestamp"].to_numpy()
            n = len(t)
            b1, b2 = int(cfg.train_frac * n), int((cfg.train_frac + cfg.val_frac) * n)
            if cfg.snap_ties:
                edges = np.flatnonzero(np.r_[True, t[1:] != t[:-1]])
                s1 = int(edges[np.argmin(np.abs(edges - cfg.train_frac * n))])
                s2 = int(edges[np.argmin(np.abs(edges - (cfg.train_frac + cfg.val_frac) * n))])
                if 0 < s1 < s2 < n:
                    b1, b2 = s1, s2
            b1 = min(max(b1, 1), n - 2)
            b2 = min(max(b2, b1 + 1), n - 1)
            out[idx[:b1]] = "train"
            out[idx[b1:b2]] = "val"
            out[idx[b2:]] = "test"
        return out

    def _encode_side_features(self, movies, users):
        m = movies.copy()
        m["idx"] = m["MovieID"].map(self.i2idx)
        m = m.sort_values("idx")
        g = m["Genre"].str.get_dummies(sep="|")
        for gn in GENRES:
            if gn not in g:
                g[gn] = 0
        self.item_genre = sp.csr_matrix(g[GENRES].to_numpy(np.float32))
        self.item_year_raw = m["Title"].map(_extract_year).to_numpy()
        yb = pd.cut(self.item_year_raw, bins=YEAR_BINS, labels=YEAR_LABELS)
        self.item_year = sp.csr_matrix(pd.get_dummies(yb).reindex(
            columns=YEAR_LABELS, fill_value=0).to_numpy(np.float32))
        self.item_title = m["Title"].to_numpy()
        self.item_genre_tokens = m["Genre"].fillna("").str.replace("|", " ", regex=False).to_numpy()

        u = users.copy()
        u["idx"] = u["UserID"].map(self.u2idx)
        u = u.sort_values("idx")
        demo = pd.get_dummies(pd.DataFrame({
            "gender": u["Gender"].astype(str),
            "age": u["Age"].map(AGE_MAP).astype(str),
            "occ": u["Occupation"].astype(str),
        }))
        self.user_demo = sp.csr_matrix(demo.to_numpy(np.float32))
        self.user_demo_names = list(demo.columns)

    # ----------------------------------------------------------- invariants

    def _check_invariants(self):
        """Cheap structural checks that would otherwise fail silently."""
        d = self.inter
        assert d.groupby("u")["seq"].apply(lambda s: (s.to_numpy() == np.arange(len(s))).all()).all(), \
            "seq is not a dense 0..n-1 chronological rank within each user"
        assert d.groupby("u")["ts"].apply(lambda s: s.is_monotonic_increasing).all(), \
            "rows are not chronologically ordered within user"
        # Every user must appear in all three splits (min 20 ratings guarantees it).
        per = d.groupby("u")["split"].nunique()
        assert (per == 3).all(), f"{(per != 3).sum()} users missing from a split"
        # Candidate exclusion must never remove a test positive.
        for u, truth in list(self._truth.items())[:200]:
            assert not (truth & self._seen_by_user.get(u, set())), \
                f"user {u}: a test positive also appears in train/val history"

    # ------------------------------------------------------------- history

    def history_matrix(self, rows: pd.DataFrame, max_history: int | None = None) -> sp.csr_matrix:
        """Causal implicit-feedback matrix: row r gets the user's items strictly
        before position `seq[r]`, capped to the last `max_history`, scaled by
        1/sqrt(n) in the SVD++ style.

        Built from `seq` alone, so it is causal by construction rather than by
        careful sequencing of calls -- the failure mode that makes leakage in
        recommender pipelines so easy to ship.
        """
        K = self.cfg.max_history if max_history is None else max_history
        if len(rows) == 0:
            return sp.csr_matrix((0, self.n_items), dtype=np.float32)
        u_all = rows["u"].to_numpy(np.int32)
        s_all = rows["seq"].to_numpy(np.int64)
        r_idx, c_idx, v = [], [], []

        # Grouped per user so the gather stays a single vectorised expression.
        order = np.argsort(u_all, kind="stable")
        bounds = np.flatnonzero(np.r_[True, np.diff(u_all[order]) != 0, True])
        for a, b in zip(bounds[:-1], bounds[1:]):
            sel = order[a:b]
            uu = int(u_all[sel[0]])
            seq = s_all[sel]
            starts = np.maximum(0, seq - K)
            lens = seq - starts
            tot = int(lens.sum())
            if tot == 0:
                continue
            ends = np.cumsum(lens)
            begins = ends - lens
            local = (np.arange(tot) - np.repeat(begins, lens)) + np.repeat(starts, lens)
            r_idx.append(np.repeat(sel, lens))
            c_idx.append(self._user_items[uu][local])
            v.append(np.repeat(1.0 / np.sqrt(np.maximum(lens, 1)), lens))

        if not r_idx:
            return sp.csr_matrix((len(rows), self.n_items), dtype=np.float32)
        M = sp.csr_matrix(
            (np.concatenate(v).astype(np.float32),
             (np.concatenate(r_idx), np.concatenate(c_idx))),
            shape=(len(rows), self.n_items))
        M.sum_duplicates()
        return M

    def genre_history_matrix(self, rows: pd.DataFrame, max_history: int | None = None) -> sp.csr_matrix:
        """Genre distribution of the same causal window (L1-normalised)."""
        H = self.history_matrix(rows, max_history)
        G = (H @ self.item_genre).astype(np.float32)
        norms = np.asarray(np.abs(G).sum(axis=1)).ravel()
        norms[norms == 0] = 1.0
        return sp.csr_matrix(G / norms[:, None])

    # ------------------------------------------------------------ features

    def feature_matrix(self, rows: pd.DataFrame, blocks=ALL_BLOCKS) -> sp.csr_matrix:
        """Horizontally stack the requested feature blocks.

        The model ablation is a choice of `blocks`: MF_BLOCKS reproduces biased
        matrix factorization, ALL_BLOCKS the full feature-rich FM. Same code path.
        """
        unknown = set(blocks) - set(ALL_BLOCKS)
        assert not unknown, f"unknown feature blocks: {unknown}"
        u = rows["u"].to_numpy(np.int32)
        i = rows["i"].to_numpy(np.int32)
        parts, names = [], []
        for b in blocks:
            if b == "user_id":
                parts.append(self._eye_u[u])
            elif b == "item_id":
                parts.append(self._eye_i[i])
            elif b == "user_demo":
                parts.append(self.user_demo[u])
            elif b == "item_genre":
                parts.append(self.item_genre[i])
            elif b == "item_year":
                parts.append(self.item_year[i])
            elif b == "history":
                parts.append(self.history_matrix(rows))
            elif b == "genre_history":
                parts.append(self.genre_history_matrix(rows))
            names.append((b, parts[-1].shape[1]))
        self.last_block_layout = names
        return sp.hstack(parts, format="csr", dtype=np.float32)

    # ------------------------------------------------------------ negatives

    def sample_negatives(self, rows: pd.DataFrame, policy: str = "unobserved_only",
                         ratio: int = 1, seed: int | None = None) -> pd.DataFrame:
        """Return `rows` augmented per the negative-class policy.

        policy
          'observed_only'   : no sampling; observed rows only, label = rating>=threshold.
                              The negative class is 'watched but rated low'.
          'unobserved_only' : observed positives + `ratio` sampled never-interacted
                              items per positive. The standard implicit-feedback setup.
          'both'            : all observed rows plus the sampled unobserved items.

        Sampled rows inherit the anchor's `seq`, so their causal history window is
        the user's history at that moment -- which is what a real scoring request
        would have.
        """
        assert policy in ("observed_only", "unobserved_only", "both"), policy
        if policy == "observed_only":
            return rows.reset_index(drop=True)

        rng = np.random.default_rng(self.cfg.seed if seed is None else seed)
        base = rows[rows["label"] == 1] if policy == "unobserved_only" else rows
        anchors = rows[rows["label"] == 1]

        neg_u, neg_i, neg_seq = [], [], []
        for u, g in anchors.groupby("u", sort=False):
            forbidden = set(self._user_items[u].tolist())   # everything ever observed
            need = len(g) * ratio
            avail = self.n_items - len(forbidden)
            if avail <= 0:
                continue
            need = min(need, avail)
            picked, guard = set(), 0
            while len(picked) < need and guard < 50:
                cand = rng.integers(0, self.n_items, size=(need - len(picked)) * 2 + 8)
                for c in cand:
                    c = int(c)
                    if c not in forbidden and c not in picked:
                        picked.add(c)
                        if len(picked) == need:
                            break
                guard += 1
            picked = list(picked)
            seqs = rng.choice(g["seq"].to_numpy(), size=len(picked), replace=True)
            neg_u.extend([u] * len(picked)); neg_i.extend(picked); neg_seq.extend(seqs.tolist())

        negs = pd.DataFrame({
            "u": np.asarray(neg_u, np.int32), "i": np.asarray(neg_i, np.int32),
            "rating": 0, "ts": 0, "label": np.int8(0),
            "seq": np.asarray(neg_seq, np.int32), "n_u": 0,
            "split": rows["split"].iloc[0] if len(rows) else "train",
        })
        out = pd.concat([base, negs], ignore_index=True)
        return out.sample(frac=1.0,
                          random_state=self.cfg.seed if seed is None else seed
                          ).reset_index(drop=True)

    # ----------------------------------------------------------- evaluation

    def interaction_matrix(self, split: str = "train", positives_only: bool = True,
                           binary: bool = True) -> sp.csr_matrix:
        """User x item matrix from one split. Positives only by default: the task is
        ranking items a user will like, so a low-rated interaction is not a positive
        signal to propagate through a neighbourhood model."""
        rows = self.split_rows(split)
        if positives_only:
            rows = rows[rows["label"] == 1]
        v = np.ones(len(rows), np.float32) if binary else rows["rating"].to_numpy(np.float32)
        return sp.csr_matrix((v, (rows["u"].to_numpy(), rows["i"].to_numpy())),
                             shape=(self.n_users, self.n_items))

    def split_rows(self, name: str) -> pd.DataFrame:
        return self.inter[self.inter["split"] == name].reset_index(drop=True)

    def candidates(self, u: int, filtered: bool = True, target: str = "test") -> np.ndarray:
        """Scorable items for user `u`: catalogue minus train+val history.

        `filtered=True` additionally restricts to items with >= min_item_ratings
        training interactions. Both settings are reported in the benchmark; the
        filtered one flatters accuracy and suppresses coverage.
        """
        pool = self.established if filtered else np.arange(self.n_items)
        seen = self._seen_by[target].get(u, set())
        if not seen:
            return pool
        mask = ~np.isin(pool, np.fromiter(seen, np.int32, len(seen)))
        return pool[mask]

    def ground_truth(self, u: int, target: str = "test") -> set:
        """Items this user rated >= pos_threshold in the target split."""
        return self._truth_by[target].get(u, set())

    def reachable_truth(self, u: int, filtered: bool = True, target: str = "test") -> set:
        """Ground-truth items that the candidate pool actually allows a model to
        return. Metrics are scored against the *full* truth -- an unreachable
        target is a real product cost, not an excuse -- but the gap is reported
        as a ceiling so a low recall is not misread as a ranking failure."""
        t = self._truth_by[target].get(u, set())
        if not filtered:
            return t
        return t & set(self.established.tolist())

    def eval_users(self, filtered: bool = True, target: str = "test") -> np.ndarray:
        """Users with at least one reachable test positive -- the only users for
        whom Recall@K is defined. Filtering the candidate pool makes some users
        unevaluable, which is itself a cost of filtering and is reported."""
        pool = set(self.established.tolist()) if filtered else None
        out = []
        for u, t in self._truth_by[target].items():
            if pool is None or (t & pool):
                out.append(u)
        return np.sort(np.asarray(out, np.int32))

    def scoring_frame(self, u: int, items: np.ndarray, target: str = "test") -> pd.DataFrame:
        """Rows for scoring user `u` against `items` at test time, with `seq` set
        past the end of the user's observed history so the causal window is the
        user's full train+val past."""
        # Because the split is by chronological position, the first `n_seen`
        # entries of the user's sequence are exactly their train+val items --
        # so seq=n_seen makes the causal window the user's full pre-test past.
        n_seen = len(self._seen_by[target].get(u, set()))
        assert n_seen <= len(self._user_items[u])
        return pd.DataFrame({
            "u": np.full(len(items), u, np.int32), "i": items.astype(np.int32),
            "rating": 0, "ts": 0, "label": np.int8(0),
            "seq": np.full(len(items), n_seen, np.int32),
            "n_u": len(self._user_items[u]), "split": "test",
        })


def build(cfg: Config = Config(), **kw) -> Dataset:
    return Dataset(cfg, **kw)


if __name__ == "__main__":
    ds = build()
