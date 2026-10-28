"""
Zero-parameter baselines: the floor every learned model must clear.

The damped mean is the one that matters. A raw mean rating ranks obscure items
with a single 5-star rating above everything else; damping shrinks each item's
mean toward the global mean in proportion to how little evidence supports it,
which is a one-line Bayesian prior and is, in practice, hard to beat at top-10.
"""

from __future__ import annotations

import numpy as np

from .base import Recommender


class MostPopular(Recommender):
    """Rank by raw training interaction count. The crudest possible lookup table."""
    name = "pop_count"

    def fit(self, ds, train_rows, val_rows=None):
        self.s = ds.item_train_count.astype(np.float64)
        return self

    def score(self, ds, u, items):
        return self.s[items]


class DampedMean(Recommender):
    """Bayesian-damped mean rating: (C*m + sum_r) / (C + n).

    C is the prior strength in units of ratings -- an item needs roughly C ratings
    before its own mean outweighs the global mean. C is chosen on validation, not
    guessed, because the value silently controls a popularity/quality trade-off.
    """
    name = "damped_mean"

    def __init__(self, C: float = 25.0):
        self.C = C

    def fit(self, ds, train_rows, val_rows=None):
        n = ds.n_items
        cnt = np.bincount(train_rows["i"], minlength=n).astype(np.float64)
        tot = np.bincount(train_rows["i"], weights=train_rows["rating"], minlength=n)
        m = tot.sum() / max(cnt.sum(), 1)
        self.global_mean = m
        denom = self.C + cnt
        self.s = np.divide(self.C * m + tot, denom, out=np.zeros_like(tot),
                           where=denom > 0)
        self.s[cnt == 0] = 0.0   # never recommend an item with no training evidence
        return self

    def score(self, ds, u, items):
        return self.s[items]
