"""
Content-based baselines over the only item content MovieLens ships: 18 genre flags
and a release year.

These are the only rungs of the ladder that can score an item with no interaction
history at all -- 237 items, 6.1% of the catalogue, are invisible to every
collaborative model above. That is the whole reason they are here; their aggregate
accuracy is expected to be poor.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from .base import Recommender


def _dense(M) -> np.ndarray:
    """Sparse-times-sparse stays sparse; every score grid here must be dense."""
    return (M.toarray() if sp.issparse(M) else np.asarray(M)).astype(np.float32, copy=False)


class ContentKNN(Recommender):
    """Cosine between a user's taste profile and each item's content vector.

    The profile is the mean content vector of the items the user liked in training,
    so the model has no notion of popularity at all -- which makes it a useful
    contrast to every other baseline, all of which do.
    """
    name = "content_knn"

    def __init__(self, use_year: bool = True, idf: bool = True):
        self.use_year, self.idf = use_year, idf

    def fit(self, ds, train_rows, val_rows=None):
        blocks = [ds.item_genre] + ([ds.item_year] if self.use_year else [])
        F = sp.hstack(blocks, format="csr").astype(np.float32)
        if self.idf:
            # Rare genres (Film-Noir, Western) should count for more than Drama,
            # which half the catalogue carries and which therefore discriminates
            # between almost nothing.
            df = np.asarray((F > 0).sum(axis=0)).ravel().astype(np.float32)
            w = np.log(F.shape[0] / np.maximum(df, 1.0)) + 1.0
            F = F @ sp.diags(w)
        F = self._l2(F)
        R = ds.interaction_matrix("train")
        P = self._l2(sp.csr_matrix(R @ F))
        self.scores = _dense(P @ F.T)
        self.scores[R.nonzero()] = -np.inf
        return self

    @staticmethod
    def _l2(M):
        n = np.sqrt(np.asarray(M.multiply(M).sum(axis=1)).ravel())
        inv = np.divide(1.0, n, out=np.zeros_like(n), where=n > 0)
        return sp.diags(inv.astype(np.float32)) @ M

    def score(self, ds, u, items):
        return self.scores[u, items]


class GenrePopularity(Recommender):
    """Popularity, restricted to the genres a user actually watches.

    The point of this rung is that it is *personalised without being learned*: it
    has no parameters, only a per-user genre histogram multiplied by a popularity
    prior. If a factorization machine cannot beat it, the factorization is not
    earning its keep.
    """
    name = "genre_popularity"

    def __init__(self, pop_weight: float = 1.0):
        self.pop_weight = pop_weight

    def fit(self, ds, train_rows, val_rows=None):
        R = ds.interaction_matrix("train")
        G = ds.item_genre.astype(np.float32)                  # items x genres
        A = _dense(R @ G)                                     # users x genres
        A /= np.maximum(A.sum(axis=1, keepdims=True), 1.0)    # taste distribution
        Gn = ContentKNN._l2(G)
        affinity = A @ _dense(Gn).T                           # users x items
        pop = np.log1p(ds.item_train_count).astype(np.float32)
        self.scores = affinity * (pop ** self.pop_weight)[None, :]
        self.scores[R.nonzero()] = -np.inf
        return self

    def score(self, ds, u, items):
        return self.scores[u, items]
