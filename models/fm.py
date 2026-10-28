"""
The factorization machine, and the models it degenerates into.

This file exists to make Phase 3's ablation credible. Logistic regression, biased
matrix factorization and a full feature-rich FM are not three implementations here --
they are one training loop under three configurations:

    LR   = FactorizationMachine(order=1, blocks=ALL_BLOCKS)   # no interaction term
    MF   = FactorizationMachine(order=2, blocks=MF_BLOCKS)    # interactions, IDs only
    FM   = FactorizationMachine(order=2, blocks=ALL_BLOCKS)   # both

Any difference between their scores is therefore attributable to the configuration and
not to an implementation detail, which is the only way the comparison means anything.

Scoring uses an exact decomposition rather than materialising a feature row per
candidate. Feature blocks split into user-side (user id, demographics, interaction
history, genre history) and item-side (item id, genre, year), and those column sets are
disjoint, so with x = x_u + x_i:

    sum_j v_jf x_j = S_uf + S_if

    y(u,i) = [w0 + <w,x_u> + half*sum_f(S_uf^2 - Q_uf)]      <- constant per user
           + [      <w,x_i> + half*sum_f(S_if^2 - Q_if)]      <- constant per item
           + <S_u, S_i>                                       <- the only coupled term

So an FM scores a candidate in O(k), exactly as a dot-product retrieval model does --
which is also the reason an FM can be served at all, and worth knowing before Phase 6
contrasts it with a two-tower.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.special import expit

from data import ALL_BLOCKS, MF_BLOCKS
from .base import Recommender

USER_SIDE = ("user_id", "user_demo", "history", "genre_history")
ITEM_SIDE = ("item_id", "item_genre", "item_year")


def _logloss(y, p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


class _Adam:
    """Adam state for one parameter array."""

    def __init__(self, shape, lr, b1=0.9, b2=0.999, eps=1e-8):
        self.m = np.zeros(shape, np.float32)
        self.v = np.zeros(shape, np.float32)
        self.lr, self.b1, self.b2, self.eps, self.t = lr, b1, b2, eps, 0

    def step(self, param, grad):
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * grad
        self.v = self.b2 * self.v + (1 - self.b2) * (grad * grad)
        mh = self.m / (1 - self.b1 ** self.t)
        vh = self.v / (1 - self.b2 ** self.t)
        param -= self.lr * mh / (np.sqrt(vh) + self.eps)
        return param


class FactorizationMachine(Recommender):
    """Second-order FM with shared latent factors, trained by Adam on log loss.

    `order=1` drops the interaction term entirely and leaves plain logistic
    regression; `blocks=MF_BLOCKS` restricts the features to the user and item id
    one-hots, at which point the model *is* biased matrix factorization.
    """

    #: Objectives sharing one gradient path. Each reduces to an "err" term multiplying
    #: the same dy_hat/dtheta, so the model class is held genuinely constant:
    #:   logistic    err = sigma(y_hat) - y       pointwise, absolute target
    #:   regression  err = y_hat - rating         pointwise, absolute target
    #:   softmax     err = p - onehot             listwise over 1 positive + N negatives
    #:   bpr         err = p - onehot, N = 1      pairwise -- the softmax with one negative
    GROUPED = ("softmax", "bpr")

    def __init__(self, k: int = 16, blocks=ALL_BLOCKS, order: int = 2,
                 lr: float = 0.02, reg_w: float = 1e-5, reg_v: float = 1e-5,
                 epochs: int = 25, batch_size: int = 4096, patience: int = 3,
                 neg_policy: str = "unobserved_only", neg_ratio: int = 1,
                 objective: str = "logistic", n_neg: int = 8,
                 seed: int = 42, name: str | None = None, verbose: bool = True):
        assert objective in ("logistic", "regression", "softmax", "bpr"), objective
        self.objective = objective
        self.n_neg = 1 if objective == "bpr" else n_neg
        self.k, self.blocks, self.order = k, tuple(blocks), order
        self.lr, self.reg_w, self.reg_v = lr, reg_w, reg_v
        self.epochs, self.batch_size, self.patience = epochs, batch_size, patience
        self.neg_policy, self.neg_ratio, self.seed = neg_policy, neg_ratio, seed
        self.verbose = verbose
        self.name = name or ("logreg" if order == 1 else
                             "mf" if set(self.blocks) == set(MF_BLOCKS) else "fm")
        if objective != "logistic":
            self.name = f"{self.name}_{objective}"
        self.negatives = f"{neg_policy} 1:{neg_ratio}"
        self.history_ = []

    # ------------------------------------------------------------------ fit

    def _design(self, ds, rows, seed):
        """Rows and targets for the configured objective.

        For the grouped objectives the frame is laid out so group g occupies rows
        [g*(N+1), (g+1)*(N+1)) with the positive first, so the softmax target is always
        index 0 and batching happens over groups rather than rows.
        """
        if self.objective == "regression":
            # Only observed ratings carry a regression target at all -- the constraint
            # this study exists to expose. The model can be fitted only where a user
            # chose to watch, then must rank what they did not.
            r = rows.reset_index(drop=True)
            X = ds.feature_matrix(r, self.blocks).astype(np.float32)
            return X, r["rating"].to_numpy(np.float32)

        if self.objective not in self.GROUPED:
            r = ds.sample_negatives(rows, self.neg_policy, self.neg_ratio, seed=seed)
            X = ds.feature_matrix(r, self.blocks)
            return X.astype(np.float32), r["label"].to_numpy(np.float32)

        rng = np.random.default_rng(seed)
        pos = rows[rows["label"] == 1].reset_index(drop=True)
        N = self.n_neg
        gu = pos["u"].to_numpy(np.int32)
        u = np.repeat(gu, N + 1)
        seq = np.repeat(pos["seq"].to_numpy(np.int32), N + 1)
        items = np.empty(len(pos) * (N + 1), np.int32)
        items[0::N + 1] = pos["i"].to_numpy(np.int32)
        draw = self._draw_negatives(ds, gu, rng)
        for c in range(N):
            items[c + 1::N + 1] = draw[:, c]
        frame = pd.DataFrame({"u": u, "i": items, "rating": 0, "ts": 0,
                              "label": np.int8(0), "seq": seq, "n_u": 0, "split": "train"})
        X = ds.feature_matrix(frame, self.blocks).astype(np.float32)
        self._last_layout = (gu, items.copy())
        return X, np.zeros(len(pos), np.int64)      # target is always index 0

    def _raw(self, X):
        out = np.full(X.shape[0], self.w0, np.float32) + X @ self.w
        if self.order >= 2:
            XV = X @ self.V
            out += 0.5 * ((XV ** 2).sum(1) - X.power(2) @ (self.V ** 2).sum(1))
        return out

    def _draw_negatives(self, ds, users, rng):
        """Uniform over items the user never interacted with -- the same rule as
        `Dataset.sample_negatives`, so the grouped objectives and the logistic one differ
        only in the loss. An unfiltered draw would make ~9% of negatives items the user
        actually rated, and a softmax spends its gradient on exactly those."""
        if not hasattr(self, "_seen_mask"):
            M = np.zeros((ds.n_users, ds.n_items), dtype=bool)
            for uu, its in ds._user_items.items():
                M[uu, its] = True
            self._seen_mask = M
        cand = rng.integers(0, ds.n_items, size=(len(users), self.n_neg)).astype(np.int32)
        bad = self._seen_mask[users[:, None], cand]
        while bad.any():
            cand[bad] = rng.integers(0, ds.n_items, size=int(bad.sum())).astype(np.int32)
            bad = self._seen_mask[users[:, None], cand]
        return cand

    def _resample(self, ds, X, rng):
        """Redraw every negative in place. Only the item-id block depends on which
        negative was drawn -- user id and the causal history are the anchor's -- so this
        swaps one indicator per negative row instead of rebuilding the matrix."""
        N, G = self.n_neg, X.shape[0] // (self.n_neg + 1)
        off = self._block_slices(ds)["item_id"].start
        neg_rows = ((np.arange(G)[:, None] * (N + 1)) + np.arange(1, N + 1)).ravel()
        old = self._grp_items[neg_rows]
        new = self._draw_negatives(ds, self._grp_users, rng).ravel()
        shape = X.shape
        ones = np.ones(len(neg_rows), np.float32)
        delta = (sp.csr_matrix((ones, (neg_rows, off + new)), shape=shape)
                 - sp.csr_matrix((ones, (neg_rows, off + old)), shape=shape))
        X = (X + delta).tocsr(); X.eliminate_zeros()
        self._grp_items[neg_rows] = new
        return X

    def _err(self, raw, y):
        """dL/d(y_hat). Every objective funnels through this single term."""
        if self.objective == "regression":
            return raw - y
        if self.objective not in self.GROUPED:
            return expit(raw) - y
        z = raw.reshape(-1, self.n_neg + 1)
        z = z - z.max(axis=1, keepdims=True)
        p = np.exp(z); p /= p.sum(axis=1, keepdims=True)
        p[:, 0] -= 1.0                                  # subtract the one-hot target
        return p.ravel()

    def _obj_loss(self, raw, y):
        if self.objective == "regression":
            return float(np.mean((raw - y) ** 2))
        if self.objective not in self.GROUPED:
            return _logloss(y, expit(raw))
        z = raw.reshape(-1, self.n_neg + 1)
        z = z - z.max(axis=1, keepdims=True)
        return float(np.mean(-z[:, 0] + np.log(np.exp(z).sum(axis=1))))

    def fit(self, ds, train_rows, val_rows=None):
        rng = np.random.default_rng(self.seed)
        X, y = self._design(ds, train_rows, self.seed)
        if self.objective in self.GROUPED:
            self._grp_users, self._grp_items = self._last_layout
            can_swap = not any(b in self.blocks for b in ("item_genre", "item_year"))
            assert can_swap, "in-place resampling assumes the item side is the id block only"
        Xv = yv = None
        if val_rows is not None:
            Xv, yv = self._design(ds, val_rows, self.seed + 1)

        n = X.shape[1]
        self.w0 = np.float32(0.0)
        self.w = np.zeros(n, np.float32)
        self.V = (rng.normal(0, 0.01, (n, self.k)).astype(np.float32)
                  if self.order >= 2 else None)
        opt_w0, opt_w = _Adam((), self.lr), _Adam(n, self.lr)
        opt_v = _Adam((n, self.k), self.lr) if self.order >= 2 else None

        best, best_state, bad = np.inf, None, 0
        # Grouped objectives shuffle and batch over GROUPS; a group must stay intact
        # because the softmax is taken across it.
        G = (self.n_neg + 1) if self.objective in self.GROUPED else 1
        m = X.shape[0] // G
        bs = max(1, self.batch_size // G)
        for ep in range(1, self.epochs + 1):
            if ep > 1 and G > 1:
                X = self._resample(ds, X, rng)          # fresh negatives every epoch
            order = rng.permutation(m)
            for s in range(0, m, bs):
                g_idx = order[s: s + bs]
                idx = ((g_idx[:, None] * G) + np.arange(G)).ravel() if G > 1 else g_idx
                Xb = X[idx]
                yb = y[g_idx] if G > 1 else y[idx]
                b = Xb.shape[0]
                lin = self.w0 + Xb @ self.w
                if self.order >= 2:
                    XV = Xb @ self.V
                    Xb2 = Xb.power(2)
                    lin = lin + 0.5 * ((XV ** 2).sum(1) - Xb2 @ (self.V ** 2).sum(1))
                err = self._err(lin, yb).astype(np.float32)

                self.w0 = opt_w0.step(np.float32(self.w0), np.float32(err.mean()))
                gw = (Xb.T @ err) / b + self.reg_w * self.w
                self.w = opt_w.step(self.w, gw.astype(np.float32))
                if self.order >= 2:
                    gv = (Xb.T @ (err[:, None] * XV) - Xb2.T @ err[:, None] * self.V) / b
                    self.V = opt_v.step(self.V, (gv + self.reg_v * self.V).astype(np.float32))

            tr_loss = self._obj_loss(self._raw(X), y)
            rec = {"epoch": ep, "train_logloss": tr_loss}
            if Xv is not None:
                va_loss = self._obj_loss(self._raw(Xv), yv)
                rec["val_logloss"] = va_loss
                if va_loss < best - 1e-5:
                    best, bad = va_loss, 0
                    best_state = (self.w0, self.w.copy(),
                                  None if self.V is None else self.V.copy())
                else:
                    bad += 1
            self.history_.append(rec)
            if self.verbose:
                msg = f"    [{self.name}] epoch {ep:>2}  train {tr_loss:.4f}"
                if Xv is not None:
                    msg += f"  val {rec['val_logloss']:.4f}"
                print(msg, flush=True)
            if bad >= self.patience:
                if self.verbose:
                    print(f"    [{self.name}] early stop at epoch {ep}", flush=True)
                break
        if best_state is not None:
            self.w0, self.w, self.V = best_state
        self._precompute(ds)
        return self

    # -------------------------------------------------------------- scoring

    def _block_slices(self, ds):
        """Column span of each block, in the order feature_matrix stacks them."""
        widths = {"user_id": ds.n_users, "item_id": ds.n_items,
                  "user_demo": ds.user_demo.shape[1], "item_genre": ds.item_genre.shape[1],
                  "item_year": ds.item_year.shape[1], "history": ds.n_items,
                  "genre_history": ds.item_genre.shape[1]}
        out, o = {}, 0
        for b in self.blocks:
            out[b] = slice(o, o + widths[b])
            o += widths[b]
        return out

    def _side_matrix(self, ds, which):
        """Feature rows for one side only, zero-padded into the full column space."""
        sl = self._block_slices(ds)
        n = sum(s.stop - s.start for s in sl.values())
        parts = []
        rows_n = ds.n_users if which == "user" else ds.n_items
        for b in self.blocks:
            w = sl[b].stop - sl[b].start
            on_side = b in (USER_SIDE if which == "user" else ITEM_SIDE)
            if not on_side:
                parts.append(sp.csr_matrix((rows_n, w), dtype=np.float32))
            elif b == "user_id":
                parts.append(sp.identity(ds.n_users, format="csr", dtype=np.float32))
            elif b == "item_id":
                parts.append(sp.identity(ds.n_items, format="csr", dtype=np.float32))
            elif b == "user_demo":
                parts.append(ds.user_demo)
            elif b == "item_genre":
                parts.append(ds.item_genre)
            elif b == "item_year":
                parts.append(ds.item_year)
            elif b in ("history", "genre_history"):
                # built per user at score time; a placeholder keeps the layout aligned
                parts.append(sp.csr_matrix((rows_n, w), dtype=np.float32))
        M = sp.hstack(parts, format="csr", dtype=np.float32)
        assert M.shape[1] == n
        return M

    def _precompute(self, ds):
        Mi = self._side_matrix(ds, "item")
        self.item_lin_ = np.asarray(Mi @ self.w).ravel()
        if self.order >= 2:
            self.item_S_ = np.asarray(Mi @ self.V)
            self.item_Q_ = np.asarray(Mi.power(2) @ (self.V ** 2))
            self.item_const_ = 0.5 * (self.item_S_ ** 2 - self.item_Q_).sum(1)
        else:
            self.item_S_ = self.item_const_ = None
        self._uses_history = any(b in ("history", "genre_history") for b in self.blocks)
        self._ucache = {}

    def _user_side(self, ds, u):
        if u in self._ucache:
            return self._ucache[u]
        # One scoring row carries this user's full causal pre-test context; the
        # item columns in it are zero, so it is exactly the user-side vector.
        frame = ds.scoring_frame(u, np.array([0], np.int32))
        Xu = ds.feature_matrix(frame, self.blocks).astype(np.float32)
        sl = self._block_slices(ds)
        Xu = Xu.tolil()
        for b in self.blocks:
            if b in ITEM_SIDE:
                Xu[0, sl[b]] = 0.0
        Xu = Xu.tocsr()
        lin = float(np.asarray(Xu @ self.w).ravel()[0])
        if self.order >= 2:
            S = np.asarray(Xu @ self.V).ravel()
            Q = np.asarray(Xu.power(2) @ (self.V ** 2)).ravel()
            const = 0.5 * float((S ** 2 - Q).sum())
        else:
            S, const = None, 0.0
        out = (lin, S, const)
        self._ucache[u] = out
        return out

    def score(self, ds, u, items):
        lin_u, S_u, const_u = self._user_side(ds, int(u))
        s = float(self.w0) + lin_u + const_u + self.item_lin_[items]
        if self.order >= 2:
            s = s + self.item_const_[items] + self.item_S_[items] @ S_u
        return s

    def score_rows(self, ds, rows):
        """Pointwise output. Only `logistic` yields a calibrated probability; the
        grouped objectives learn an order with no absolute scale, and `regression`
        predicts a rating on the 1-5 range."""
        raw = self._raw(ds.feature_matrix(rows, self.blocks).astype(np.float32))
        return expit(raw) if self.objective == "logistic" else raw

    @property
    def calibrated(self) -> bool:
        """Whether score_rows means anything on an absolute scale."""
        return self.objective in ("logistic", "regression")


class Poly2(Recommender):
    """Second-order model with an INDEPENDENT weight per feature pair.

    Restricted to the user-id and item-id blocks, where each training row has exactly
    two non-zeros and therefore exactly one pair, so the model holds one free
    parameter per observed (user, item) cell. It is the direct control for parameter
    sharing: an FM with the same features spends its capacity on k factors per feature
    and can score a pair it never saw, while Poly2 must have seen the exact pair or it
    has nothing to say. On a matrix that is 1.8% dense, that difference is the whole
    argument for factorisation.
    """
    name = "poly2"

    def __init__(self, lr: float = 0.05, reg: float = 1e-5, epochs: int = 25,
                 batch_size: int = 4096, patience: int = 3,
                 neg_policy: str = "unobserved_only", neg_ratio: int = 1,
                 seed: int = 42, verbose: bool = True):
        self.lr, self.reg, self.epochs = lr, reg, epochs
        self.batch_size, self.patience = batch_size, patience
        self.neg_policy, self.neg_ratio, self.seed = neg_policy, neg_ratio, seed
        self.verbose = verbose
        self.negatives = f"{neg_policy} 1:{neg_ratio}"
        self.history_ = []

    def _design(self, ds, rows, seed):
        r = ds.sample_negatives(rows, self.neg_policy, self.neg_ratio, seed=seed)
        return (r["u"].to_numpy(np.int64), r["i"].to_numpy(np.int64),
                r["label"].to_numpy(np.float32))

    def _raw(self, u, i):
        return self.w0 + self.bu[u] + self.bi[i] + np.asarray(self.W[u, i]).ravel()

    def fit(self, ds, train_rows, val_rows=None):
        rng = np.random.default_rng(self.seed)
        u, i, y = self._design(ds, train_rows, self.seed)
        vu = vi = vy = None
        if val_rows is not None:
            vu, vi, vy = self._design(ds, val_rows, self.seed + 1)

        self.w0 = 0.0
        self.bu = np.zeros(ds.n_users, np.float32)
        self.bi = np.zeros(ds.n_items, np.float32)
        # Only pairs that actually occur can ever be updated; every other cell of the
        # 6,040 x 3,883 grid stays at zero, which is precisely the failure being measured.
        pairs = np.unique(u * ds.n_items + i)
        self.pair_index = {int(p): n for n, p in enumerate(pairs)}
        self.wp = np.zeros(len(pairs), np.float32)
        self.n_items_ = ds.n_items
        pid_tr = np.fromiter((self.pair_index[int(p)] for p in u * ds.n_items + i),
                             np.int64, len(u))
        pid_va = (np.fromiter((self.pair_index.get(int(p), -1) for p in vu * ds.n_items + vi),
                              np.int64, len(vu)) if vu is not None else None)

        ow0, obu, obi = _Adam((), self.lr), _Adam(ds.n_users, self.lr), _Adam(ds.n_items, self.lr)
        owp = _Adam(len(pairs), self.lr)
        best, best_state, bad = np.inf, None, 0
        m = len(u)

        def raw(uu, ii, pid):
            out = self.w0 + self.bu[uu] + self.bi[ii]
            seen = pid >= 0
            out[seen] += self.wp[pid[seen]]
            return out

        for ep in range(1, self.epochs + 1):
            perm = rng.permutation(m)
            for s in range(0, m, self.batch_size):
                b = perm[s: s + self.batch_size]
                ub, ib, pb, yb = u[b], i[b], pid_tr[b], y[b]
                err = (expit(raw(ub, ib, pb)) - yb).astype(np.float32)
                nb = len(b)
                self.w0 = float(ow0.step(np.float32(self.w0), np.float32(err.mean())))
                gbu = np.zeros_like(self.bu); np.add.at(gbu, ub, err)
                gbi = np.zeros_like(self.bi); np.add.at(gbi, ib, err)
                gwp = np.zeros_like(self.wp); np.add.at(gwp, pb, err)
                self.bu = obu.step(self.bu, gbu / nb + self.reg * self.bu)
                self.bi = obi.step(self.bi, gbi / nb + self.reg * self.bi)
                self.wp = owp.step(self.wp, gwp / nb + self.reg * self.wp)

            tr = _logloss(y, expit(raw(u, i, pid_tr)))
            rec = {"epoch": ep, "train_logloss": tr}
            if vu is not None:
                va = _logloss(vy, expit(raw(vu, vi, pid_va)))
                rec["val_logloss"] = va
                if va < best - 1e-5:
                    best, bad = va, 0
                    best_state = (self.w0, self.bu.copy(), self.bi.copy(), self.wp.copy())
                else:
                    bad += 1
            self.history_.append(rec)
            if self.verbose:
                msg = f"    [poly2] epoch {ep:>2}  train {tr:.4f}"
                if vu is not None: msg += f"  val {rec['val_logloss']:.4f}"
                print(msg, flush=True)
            if bad >= self.patience:
                if self.verbose: print(f"    [poly2] early stop at epoch {ep}", flush=True)
                break
        if best_state is not None:
            self.w0, self.bu, self.bi, self.wp = best_state
        return self

    def score(self, ds, u, items):
        keys = int(u) * self.n_items_ + items.astype(np.int64)
        out = self.w0 + self.bu[int(u)] + self.bi[items]
        pid = np.fromiter((self.pair_index.get(int(kk), -1) for kk in keys), np.int64, len(keys))
        seen = pid >= 0
        out = out.astype(np.float64)
        out[seen] += self.wp[pid[seen]]
        return out
