"""
The interface every benchmark model implements, and the ranking loop they share.

The contract is deliberately narrow -- fit, then score a user against an array of
candidate items -- because anything wider lets a model reach into the protocol and
quietly evaluate itself on different terms than its neighbours in the table.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class Recommender(ABC):
    name: str = "unnamed"
    #: free-text note recording the negative-sampling policy used for training,
    #: carried into the results table so pointwise AUC is never compared blind.
    negatives: str = "n/a"

    @abstractmethod
    def fit(self, ds, train_rows: pd.DataFrame, val_rows: pd.DataFrame | None = None) -> "Recommender":
        ...

    @abstractmethod
    def score(self, ds, u: int, items: np.ndarray) -> np.ndarray:
        """Scores for one user against `items`. Higher is better. Any monotone
        scale is acceptable -- ranking metrics only read the order, which is why
        a pairwise model with no calibrated output is still rankable here."""
        ...

    def score_rows(self, ds, rows: pd.DataFrame) -> np.ndarray:
        """Pointwise scores for explicit (u, i) rows, for the classification
        metrics. Models with no meaningful absolute scale should leave this
        unimplemented rather than inventing one."""
        raise NotImplementedError(f"{self.name} has no calibrated pointwise score")

    # -- convenience ------------------------------------------------------
    def __repr__(self):
        return f"<{type(self).__name__} name={self.name!r}>"


def rank_users(model: Recommender, ds, users=None, filtered: bool = True,
               k: int = 10, progress_every: int = 1000,
               target: str = "test") -> tuple[dict, float]:
    """Score every candidate for every user and return {user: ranked item ids}.

    Ranks the *full* candidate set rather than a sampled subset: sampled-candidate
    ranking inflates every metric and does so unevenly across model families, which
    would make the comparison this benchmark exists to draw meaningless.

    `target` selects the evaluation split and MUST match the target whose ground truth
    the caller will score against. Mismatching them silently excludes every relevant
    item from the candidate pool and yields all-zero metrics.
    """
    users = ds.eval_users(filtered=filtered, target=target) if users is None else np.asarray(users)
    out, t0 = {}, time.time()
    for n, u in enumerate(users, 1):
        u = int(u)
        cand = ds.candidates(u, filtered=filtered, target=target)
        if len(cand) == 0:
            continue
        s = np.asarray(model.score(ds, u, cand), np.float64)
        assert s.shape == cand.shape, f"{model.name}: score/candidate shape mismatch"
        top = np.argpartition(-s, min(k, len(s) - 1))[:k] if len(s) > k else np.arange(len(s))
        out[u] = cand[top[np.argsort(-s[top], kind="stable")]]
        if progress_every and n % progress_every == 0:
            print(f"    {n}/{len(users)} users  ({n/(time.time()-t0):.0f}/s)", flush=True)
    return out, time.time() - t0
