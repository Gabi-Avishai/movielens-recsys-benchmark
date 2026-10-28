"""
Evaluation metrics for the benchmark.

Three families, reported together for every model:

  accuracy        -- Recall@K, NDCG@K, MAP@K, HitRate@K, Precision@K
  beyond-accuracy -- catalogue coverage, Gini of the served distribution,
                     mean popularity percentile of served items
  classification  -- AUC, logloss, PR-AUC, RMSE

The second family exists because the first can be won by serving the same fifty
popular items to everybody. A model is only interesting if it moves accuracy
without collapsing coverage, and that trade is invisible unless both are printed
side by side.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (average_precision_score, log_loss, roc_auc_score)

K_DEFAULT = 10


# ------------------------------------------------------------- per-user

def hits(ranked: np.ndarray, truth: set, k: int) -> np.ndarray:
    """Binary relevance of the top-k, in rank order."""
    return np.fromiter((1 if int(i) in truth else 0 for i in ranked[:k]), np.int8,
                       min(k, len(ranked)))


def precision_at_k(ranked, truth, k=K_DEFAULT) -> float:
    return float(hits(ranked, truth, k).sum()) / k


def recall_at_k(ranked, truth, k=K_DEFAULT) -> float:
    return float(hits(ranked, truth, k).sum()) / len(truth) if truth else 0.0


def hit_rate_at_k(ranked, truth, k=K_DEFAULT) -> float:
    return float(hits(ranked, truth, k).any())


def ndcg_at_k(ranked, truth, k=K_DEFAULT) -> float:
    """Binary-relevance NDCG. The ideal DCG is capped at min(k, |truth|), so a
    user with three relevant items can still score 1.0 at k=10 -- without the
    cap, users with few test positives are permanently penalised and the mean
    becomes a measure of test-set size rather than of ranking quality."""
    h = hits(ranked, truth, k)
    if not truth or h.size == 0:
        return 0.0
    disc = 1.0 / np.log2(np.arange(2, h.size + 2))
    dcg = float((h * disc).sum())
    n_ideal = min(k, len(truth))
    idcg = float((1.0 / np.log2(np.arange(2, n_ideal + 2))).sum())
    return dcg / idcg if idcg > 0 else 0.0


def average_precision_at_k(ranked, truth, k=K_DEFAULT) -> float:
    h = hits(ranked, truth, k)
    if not truth or h.sum() == 0:
        return 0.0
    csum = np.cumsum(h)
    prec = csum / np.arange(1, h.size + 1)
    return float((prec * h).sum() / min(len(truth), k))


# ------------------------------------------------------- beyond accuracy

def gini(counts: np.ndarray) -> float:
    """Gini of the served-item distribution. 0 = every item served equally,
    1 = one item takes everything. Computed over the full catalogue, so items
    that are never served count as zeros and correctly drag the score up."""
    x = np.sort(np.asarray(counts, np.float64))
    n = x.size
    if n == 0 or x.sum() == 0:
        return 0.0
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def coverage(served_counts: np.ndarray) -> float:
    """Fraction of the catalogue that appeared in at least one top-k list."""
    return float((served_counts > 0).mean())


def mean_popularity_percentile(served_counts: np.ndarray, train_counts: np.ndarray) -> float:
    """Average popularity percentile of served items (1.0 = always serves the
    single most popular item, 0.5 = serves the median item on average)."""
    order = np.argsort(np.argsort(train_counts))
    pct = order / max(len(order) - 1, 1)
    tot = served_counts.sum()
    return float((pct * served_counts).sum() / tot) if tot else 0.0


# ------------------------------------------------------------ aggregate

def evaluate_ranking(rankings: dict[int, np.ndarray], truths: dict[int, set],
                     n_items: int, train_counts: np.ndarray,
                     k: int = K_DEFAULT, reachable: dict[int, set] | None = None) -> dict:
    """Aggregate all ranking metrics over users.

    `rankings` maps user -> item ids ordered best-first (already restricted to
    that user's candidate set). Users with no reachable ground truth are skipped
    and counted, never scored as zero -- scoring them as zero would let a model
    look worse simply because the candidate filter removed its targets.
    """
    served = np.zeros(n_items, np.int64)
    ceil = []
    acc = {m: [] for m in ("precision", "recall", "ndcg", "map", "hit_rate")}
    skipped = 0
    for u, ranked in rankings.items():
        truth = truths.get(u, set())
        if not truth:
            skipped += 1
            continue
        served[np.asarray(ranked[:k], np.int64)] += 1
        acc["precision"].append(precision_at_k(ranked, truth, k))
        acc["recall"].append(recall_at_k(ranked, truth, k))
        acc["ndcg"].append(ndcg_at_k(ranked, truth, k))
        acc["map"].append(average_precision_at_k(ranked, truth, k))
        acc["hit_rate"].append(hit_rate_at_k(ranked, truth, k))
        reach = truth if reachable is None else reachable.get(u, truth)
        ceil.append(min(k, len(reach)) / len(truth))

    out = {f"{m}@{k}": float(np.mean(v)) if v else float("nan") for m, v in acc.items()}
    out.update({
        # Recall@k cannot reach 1.0 when users have more than k relevant items,
        # nor when the candidate filter hides some of them. Reporting the ceiling
        # keeps a "low" recall from being read as a ranking failure when it is
        # arithmetic.
        f"recall_ceiling@{k}": float(np.mean(ceil)) if ceil else float("nan"),
        "coverage": coverage(served),
        "gini": gini(served),
        "pop_percentile": mean_popularity_percentile(served, train_counts),
        "n_users_scored": len(acc["recall"]),
        "n_users_skipped": skipped,
    })
    return out


def evaluate_classification(y_true, y_score, note: str = "") -> dict:
    """Pointwise metrics. `note` records the negative-sampling policy: AUC is not
    comparable across models that sampled negatives differently, so the policy
    travels with the number rather than being lost in the results table."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, np.float64)
    out = {"rmse": float(np.sqrt(np.mean((y_score - y_true) ** 2)))}
    uniq = np.unique(y_true)
    # Ranking metrics apply to any target; AUC and PR-AUC are binary-only, and the
    # regression arm of the objective study passes raw 1-5 ratings through here.
    if uniq.size > 1 and set(uniq.tolist()) <= {0, 1}:
        out["auc"] = float(roc_auc_score(y_true, y_score))
        out["pr_auc"] = float(average_precision_score(y_true, y_score))
        if y_score.size and y_score.min() >= 0 and y_score.max() <= 1:
            out["logloss"] = float(log_loss(y_true, np.clip(y_score, 1e-7, 1 - 1e-7)))
    out["negatives"] = note or "unspecified"
    return out


# --------------------------------------------------------------- slices

def slice_by_user_activity(rankings, truths, n_items, train_counts, hist_len: dict,
                           k=K_DEFAULT, edges=(0, 35, 150, 10 ** 9),
                           names=("cold", "warm", "heavy"), reachable=None) -> dict:
    """Re-run the aggregate within user-history-length bands.

    Bands are on training-interaction count (`ds.user_train_count`), which ranges
    14-2000+ with quartiles 31/68/146; MovieLens guarantees >=20 ratings per user
    *in total*, so slicing on total count leaves the cold band empty and hides the
    segment the slice exists to expose.

    Aggregate metrics on MovieLens are dominated by heavy users, who are a small
    minority of accounts but a large majority of interactions. Cold-user
    performance is the segment a real deployment cares about and the one an
    overall mean is most effective at hiding.
    """
    out = {}
    for lo, hi, name in zip(edges[:-1], edges[1:], names):
        sub = {u: r for u, r in rankings.items() if lo <= hist_len.get(u, 0) < hi}
        if sub:
            out[name] = evaluate_ranking(sub, truths, n_items, train_counts, k, reachable)
            out[name]["n_users"] = len(sub)
    return out


def format_row(name: str, m: dict, k: int = K_DEFAULT) -> str:
    return (f"{name:<28} "
            f"R@{k} {m[f'recall@{k}']:.4f}  NDCG@{k} {m[f'ndcg@{k}']:.4f}  "
            f"(ceil {m[f'recall_ceiling@{k}']:.3f})  "
            f"MAP {m[f'map@{k}']:.4f}  HR {m[f'hit_rate@{k}']:.3f}  "
            f"cov {m['coverage']:.3f}  gini {m['gini']:.3f}  "
            f"pop {m['pop_percentile']:.3f}")
