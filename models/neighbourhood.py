"""
Memory-based collaborative baselines: no learned parameters, only counting.

These matter because they separate two things that a single "popularity is hard to
beat" claim conflates. Raw co-occurrence counts are dominated by popularity -- a
popular item co-occurs with everything -- while cosine normalises that away. Running
both shows how much of a neighbourhood model's apparent skill is just the head of the
catalogue wearing a different hat.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from .base import Recommender


def _dense(M) -> np.ndarray:
    """Sparse-times-sparse stays sparse; every score grid here must be dense."""
    return (M.toarray() if sp.issparse(M) else np.asarray(M)).astype(np.float32, copy=False)


class _Precomputed(Recommender):
    """Scores the whole user x item grid once in fit().

    Per-user scoring would recompute the same sparse product 5,877 times; one
    matmul over the full grid is both faster and easier to reason about. The grid
    is n_users x n_items float32 -- 94 MB here, which is affordable and would not
    be at industrial scale, a limitation these baselines are allowed to have.
    """
    scores: np.ndarray

    def score(self, ds, u, items):
        return self.scores[u, items]

    def _finish(self, S, R):
        self.scores = _dense(R @ S)
        # never re-recommend something already interacted with in training
        self.scores[R.nonzero()] = -np.inf
        return self


class ItemItemCooccurrence(_Precomputed):
    """score(u, i) = number of times i was liked by users who liked what u liked.

    Deliberately unnormalised. This is the counting baseline, and its failure mode
    (collapsing onto popular items) is the thing worth measuring.
    """
    name = "item_item_cooc"

    def fit(self, ds, train_rows, val_rows=None):
        R = ds.interaction_matrix("train")
        C = (R.T @ R).astype(np.float32)
        C.setdiag(0); C.eliminate_zeros()
        return self._finish(C, R)


class ItemItemCosine(_Precomputed):
    """Classic item-item neighbourhood CF with cosine similarity.

    `topk` keeps only each item's strongest neighbours: an unrestricted similarity
    matrix lets thousands of weak, noisy similarities accumulate into a score that
    again tracks popularity.
    """
    name = "item_item_cosine"

    def __init__(self, topk: int | None = 200, shrink: float = 0.0):
        self.topk, self.shrink = topk, shrink

    def fit(self, ds, train_rows, val_rows=None):
        R = ds.interaction_matrix("train")
        norms = np.sqrt(np.asarray(R.multiply(R).sum(axis=0)).ravel())
        inv = np.divide(1.0, norms, out=np.zeros_like(norms), where=norms > 0)
        Rn = R @ sp.diags(inv.astype(np.float32))
        S = _dense(Rn.T @ Rn)
        if self.shrink:
            co = _dense(R.T @ R)
            S *= co / (co + self.shrink)
        np.fill_diagonal(S, 0.0)
        if self.topk and self.topk < S.shape[0]:
            cut = np.partition(S, -self.topk, axis=1)[:, -self.topk][:, None]
            S[S < cut] = 0.0
        self.S = S
        return self._finish(S, R)


class UserUserCosine(_Precomputed):
    """The transpose view: recommend what similar users liked."""
    name = "user_user_cosine"

    def __init__(self, topk: int | None = 100):
        self.topk = topk

    def fit(self, ds, train_rows, val_rows=None):
        R = ds.interaction_matrix("train")
        norms = np.sqrt(np.asarray(R.multiply(R).sum(axis=1)).ravel())
        inv = np.divide(1.0, norms, out=np.zeros_like(norms), where=norms > 0)
        Rn = sp.diags(inv.astype(np.float32)) @ R
        Su = _dense(Rn @ Rn.T)
        np.fill_diagonal(Su, 0.0)
        if self.topk and self.topk < Su.shape[0]:
            cut = np.partition(Su, -self.topk, axis=1)[:, -self.topk][:, None]
            Su[Su < cut] = 0.0
        self.scores = _dense(Su @ R)
        self.scores[R.nonzero()] = -np.inf
        return self
