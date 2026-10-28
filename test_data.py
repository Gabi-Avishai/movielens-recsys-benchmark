"""Structural tests for the frozen protocol.

These target the failure modes that are silent: leakage through history
features, candidate sets that quietly delete the ground truth, and negative
samples drawn from items the user actually watched. A green run here is the
precondition for trusting any number the benchmark produces.
"""
import numpy as np
import data as D

ds = D.build(verbose=False)
rng = np.random.default_rng(0)
ok = lambda m: print(f"  PASS  {m}")


def test_causal_history():
    """A row's history may contain only that user's strictly-earlier items."""
    for split in ("train", "val", "test"):
        rows = ds.split_rows(split)
        sub = rows.iloc[rng.choice(len(rows), 4000, replace=False)].reset_index(drop=True)
        H = ds.history_matrix(sub)
        K = ds.cfg.max_history
        for r in range(0, len(sub), 37):
            u, seq = int(sub.u[r]), int(sub.seq[r])
            got = set(H.indices[H.indptr[r]:H.indptr[r + 1]].tolist())
            want = set(ds._user_items[u][max(0, seq - K):seq].tolist())
            assert got == want, f"{split} row {r}: history != causal window"
            assert len(got) <= K
        ok(f"{split}: history == items in [seq-{K}, seq) for all sampled rows")


def test_history_spans_splits():
    """Test rows must see train/val history -- causality forbids the future, not the past."""
    te = ds.split_rows("test")
    H = ds.history_matrix(te.iloc[:5000])
    nnz = np.diff(H.indptr)
    assert nnz.min() > 0, "some test row has empty history"
    assert nnz.mean() > 40, f"test history suspiciously thin: mean {nnz.mean():.1f}"
    ok(f"test rows carry pre-test history (mean {nnz.mean():.1f} items)")


def test_no_future_in_scoring_frame():
    """scoring_frame must reconstruct the user's train+val past, never test items."""
    for u in rng.choice(ds.n_users, 60, replace=False):
        u = int(u)
        sf = ds.scoring_frame(u, np.array([0, 1, 2], np.int32))
        H = ds.history_matrix(sf)
        got = set(H.indices[H.indptr[0]:H.indptr[1]].tolist())
        future = {int(x) for x in ds.split_rows("test").query("u == @u")["i"]}
        assert not (got & future), f"user {u}: scoring history contains test items"
        assert got <= ds._seen_by_user.get(u, set())
    ok("scoring_frame history ⊆ train+val history, disjoint from test")


def test_candidates_and_truth():
    for filt in (True, False):
        for u in rng.choice(ds.n_users, 150, replace=False):
            u = int(u)
            cand = ds.candidates(u, filtered=filt)
            seen = ds._seen_by_user.get(u, set())
            assert not (set(cand.tolist()) & seen), f"user {u}: candidate already seen"
            if not filt:
                assert ds.ground_truth(u) <= set(cand.tolist()), \
                    f"user {u}: ground truth not reachable in unfiltered pool"
        ok(f"candidates(filtered={filt}): exclude train+val history"
           + ("" if filt else "; ground truth reachable"))


def test_popularity_is_train_only():
    tr = ds.split_rows("train")
    assert ds.item_train_count.sum() == len(tr), "popularity counted outside train"
    assert ds.item_train_count.sum() < len(ds.inter)
    ok(f"popularity from train only ({ds.item_train_count.sum():,} of {len(ds.inter):,})")


def test_feature_blocks():
    rows = ds.split_rows("train").iloc[:3000]
    X_mf = ds.feature_matrix(rows, D.MF_BLOCKS)
    assert X_mf.shape[1] == ds.n_users + ds.n_items
    assert (np.diff(X_mf.indptr) == 2).all(), "MF row is not exactly two one-hots"
    X_all = ds.feature_matrix(rows, D.ALL_BLOCKS)
    assert X_all.shape[1] == sum(w for _, w in ds.last_block_layout)
    ok(f"MF blocks -> {X_mf.shape[1]:,} cols (2 nnz/row); "
       f"all blocks -> {X_all.shape[1]:,} cols")
    print("        layout:", ", ".join(f"{b}:{w}" for b, w in ds.last_block_layout))


def test_negative_policies():
    rows = ds.split_rows("train")
    a = ds.sample_negatives(rows, "observed_only")
    assert len(a) == len(rows) and a["label"].mean() == rows["label"].mean()
    ok(f"observed_only: {len(a):,} rows, positive rate {a['label'].mean():.3f}")

    for policy in ("unobserved_only", "both"):
        b = ds.sample_negatives(rows, policy, ratio=1)
        synth = b[(b["label"] == 0) & (b["ts"] == 0)]
        bad = sum(int(i) in set(ds._user_items[int(u)].tolist())
                  for u, i in zip(synth["u"].head(4000), synth["i"].head(4000)))
        assert bad == 0, f"{policy}: {bad} sampled negatives are actually observed"
        ok(f"{policy}: {len(b):,} rows, positive rate {b['label'].mean():.3f}, "
           f"{len(synth):,} sampled negatives, none observed")


def test_determinism():
    d2 = D.build(verbose=False)
    assert d2.inter[["u", "i", "seq"]].equals(ds.inter[["u", "i", "seq"]])
    assert (d2.inter["split"].to_numpy() == ds.inter["split"].to_numpy()).all()
    n1 = ds.sample_negatives(ds.split_rows("val"), "unobserved_only")
    n2 = d2.sample_negatives(d2.split_rows("val"), "unobserved_only")
    assert n1[["u", "i"]].equals(n2[["u", "i"]]), "negative sampling not reproducible"
    ok("split and negative sampling reproduce exactly")


def test_regressions():
    """Fixed defects, pinned so they cannot silently return."""
    # default policy must be a policy the assert actually accepts
    r = ds.sample_negatives(ds.split_rows("val").head(500))
    assert len(r) > 0
    ok("sample_negatives() works with default arguments")

    # the seed argument must actually be honoured
    rows = ds.split_rows("val").head(4000)
    a = ds.sample_negatives(rows, "unobserved_only", seed=1)
    b = ds.sample_negatives(rows, "unobserved_only", seed=2)
    c = ds.sample_negatives(rows, "unobserved_only", seed=1)
    assert a[["u", "i"]].equals(c[["u", "i"]]), "same seed gave different output"
    assert not a[["u", "i"]].equals(b[["u", "i"]]), "seed argument is ignored"
    ok("sample_negatives(seed=...) controls sampling and row order")

    # empty frames must not raise
    empty = ds.split_rows("val").head(0)
    assert ds.history_matrix(empty).shape == (0, ds.n_items)
    assert ds.genre_history_matrix(empty).shape == (0, len(D.GENRES))
    assert ds.feature_matrix(empty).shape[0] == 0
    ok("history/genre/feature matrices handle empty input")

    # non-binary targets must not reach roc_auc_score
    import metrics as M
    out = M.evaluate_classification(np.array([1, 2, 3, 4, 5] * 20), rng.random(100))
    assert "rmse" in out and "auc" not in out, "binary metrics computed on 1-5 ratings"
    bin_out = M.evaluate_classification(np.array([0, 1] * 50), rng.random(100))
    assert "auc" in bin_out and "logloss" in bin_out
    ok("evaluate_classification: AUC/PR-AUC binary-only, RMSE always")


def test_recall_ceiling():
    """Recall@k is capped by |truth| vs k and by candidate filtering; the cap is reported."""
    import metrics as M
    users = ds.eval_users(True)[:400]
    truths = {int(u): ds.ground_truth(int(u)) for u in users}
    reach = {int(u): ds.reachable_truth(int(u), True) for u in users}
    rank = {int(u): ds.candidates(int(u), True)[:10] for u in users}
    res = M.evaluate_ranking(rank, truths, ds.n_items, ds.item_train_count, 10, reach)
    c = res["recall_ceiling@10"]
    assert 0 < c < 1.0, f"ceiling {c} should be strictly below 1 on this data"
    assert res["recall@10"] <= c + 1e-9, "measured recall exceeds its own ceiling"
    ok(f"recall ceiling reported ({c:.3f}); measured recall respects it")


def test_activity_bands_populated():
    """Every activity band must contain users, or the slice hides what it exists to show."""
    cnt = ds.user_train_count
    for lo, hi, nm in zip((0, 35, 150), (35, 150, 10 ** 9), ("cold", "warm", "heavy")):
        n = int(((cnt >= lo) & (cnt < hi)).sum())
        assert n > 0, f"activity band {nm} is empty"
        print(f"  PASS  band {nm:<6} n={n:>5} users")
    assert ds.user_train_count.min() < 20 <= ds.user_total_count.min(), \
        "train-count slicing is what makes a cold band possible here"
    ok("cold/warm/heavy bands all populated on user_train_count")


def test_split_boundaries_avoid_ties():
    """Split boundaries must not divide a burst of ratings sharing one timestamp."""
    bad = 0
    for _, g in ds.inter.groupby("u", sort=False):
        t = g["ts"].to_numpy(); s = g["split"].to_numpy()
        b1 = int(np.searchsorted(s != "train", True))
        b2 = int(np.searchsorted(s == "test", True))
        if 0 < b1 < len(t) and t[b1 - 1] == t[b1]: bad += 1
        if 0 < b2 < len(t) and t[b2 - 1] == t[b2]: bad += 1
    frac = bad / ds.n_users
    assert frac < 0.03, f"{bad} boundary crossings ({frac:.1%}) -- snapping is not working"
    ok(f"{bad} of {ds.n_users:,} users ({frac:.1%}) have a boundary inside a tie group")

    unsnapped = D.build(D.Config(snap_ties=False), verbose=False)
    bad_u = 0
    for _, g in unsnapped.inter.groupby("u", sort=False):
        t = g["ts"].to_numpy(); s = g["split"].to_numpy()
        b1 = int(np.searchsorted(s != "train", True)); b2 = int(np.searchsorted(s == "test", True))
        if 0 < b1 < len(t) and t[b1 - 1] == t[b1]: bad_u += 1
        if 0 < b2 < len(t) and t[b2 - 1] == t[b2]: bad_u += 1
    assert bad_u > 10 * bad, "snapping should be a large improvement over positional cuts"
    ok(f"snapping reduces crossings {bad_u:,} -> {bad:,} ({1 - bad/bad_u:.1%} fewer)")


def test_eval_targets_are_isolated():
    """Validation-target evaluation must not see val items, nor touch test at all."""
    for u in rng.choice(ds.n_users, 200, replace=False):
        u = int(u)
        cv = set(ds.candidates(u, True, target="val").tolist())
        tv = ds.ground_truth(u, target="val")
        # val targets must be reachable in the val candidate pool (when established)
        assert not (cv & ds._seen_by["val"].get(u, set())), "val candidates include train history"
        assert tv <= set(ds.candidates(u, False, target="val").tolist()) or not tv, \
            "val ground truth unreachable in the unfiltered val pool"
        # test excludes train+val, val excludes train only, so test candidates
        # must be a subset of val candidates.
        ct = set(ds.candidates(u, True, target="test").tolist())
        assert ct <= cv, "test candidate pool is not a subset of the val pool"
    tr_only = sum(len(v) for v in ds._seen_by["val"].values())
    tr_val = sum(len(v) for v in ds._seen_by["test"].values())
    assert tr_only < tr_val, "val exclusion set should be strictly smaller than test's"
    ok(f"val target excludes train only ({tr_only:,} items); test excludes train+val ({tr_val:,})")
    a = set(map(int, ds.eval_users(True, target="val")))
    b = set(map(int, ds.eval_users(True, target="test")))
    ok(f"evaluable users: val {len(a):,}, test {len(b):,}")


def test_interaction_matrix():
    R = ds.interaction_matrix("train")
    tr = ds.split_rows("train")
    assert R.nnz == int((tr["label"] == 1).sum()), "positives-only matrix has wrong nnz"
    assert R.shape == (ds.n_users, ds.n_items)
    assert set(np.unique(R.data).tolist()) == {1.0}, "binary matrix is not binary"
    Rr = ds.interaction_matrix("train", positives_only=False, binary=False)
    assert Rr.nnz == len(tr) and Rr.data.max() == 5
    ok(f"interaction_matrix: {R.nnz:,} positive train interactions, density {R.nnz/(R.shape[0]*R.shape[1]):.4f}")


def test_rank_users_respects_target():
    """rank_users must build its candidate pool from the same split it is scored on."""
    import metrics as MM
    from models.base import rank_users
    from models.popularity import MostPopular
    m = MostPopular().fit(ds, ds.split_rows("train"))
    users = ds.eval_users(True, target="val")[:300]
    truths = {int(u): ds.ground_truth(int(u), target="val") for u in users}
    good, _ = rank_users(m, ds, users=users, filtered=True, k=10,
                         progress_every=0, target="val")
    bad, _ = rank_users(m, ds, users=users, filtered=True, k=10,
                        progress_every=0, target="test")
    g = MM.evaluate_ranking(good, truths, ds.n_items, ds.item_train_count, 10)
    b = MM.evaluate_ranking(bad, truths, ds.n_items, ds.item_train_count, 10)
    assert g["recall@10"] > 0, "val-target ranking scores zero against val truth"
    assert g["recall@10"] > b["recall@10"], "target argument has no effect"
    ok(f"target=val recall {g['recall@10']:.4f} vs mismatched target=test {b['recall@10']:.4f}")


def test_fm_fast_scoring_matches_naive():
    """The decomposed O(k) score must equal scoring a materialised feature row.

    score() never builds a feature row per candidate; it exploits the fact that the
    user-side and item-side column blocks are disjoint. If that shortcut drifts from
    the model it claims to evaluate, every Phase 3 number is quietly wrong.
    """
    from models.fm import FactorizationMachine
    tr = ds.split_rows("train").sample(20000, random_state=0)
    va = ds.split_rows("val").sample(4000, random_state=0)
    for blocks, order in ((D.MF_BLOCKS, 2), (D.ALL_BLOCKS, 2), (D.ALL_BLOCKS, 1)):
        m = FactorizationMachine(k=8, blocks=blocks, order=order, epochs=1,
                                 verbose=False).fit(ds, tr, va)
        worst = 0.0
        for u in (7, 76, 2000):
            items = ds.candidates(u, filtered=True)[:250]
            fast = m.score(ds, u, items)
            naive = m._raw(ds.feature_matrix(ds.scoring_frame(u, items),
                                             m.blocks).astype(np.float32))
            worst = max(worst, float(np.abs(fast - naive).max()))
        assert worst < 1e-3, f"blocks={len(blocks)} order={order}: max diff {worst:.2e}"
        ok(f"order={order} blocks={len(blocks):>1}: fast == naive (max diff {worst:.1e})")


def test_mf_blocks_are_exactly_two_ids():
    """MF must be FM-on-ids, not FM-on-ids-plus-something."""
    X = ds.feature_matrix(ds.split_rows("train").head(500), D.MF_BLOCKS)
    assert X.shape[1] == ds.n_users + ds.n_items
    assert (np.diff(X.indptr) == 2).all()
    assert set(np.unique(X.data).tolist()) == {1.0}
    ok("MF design matrix is exactly two one-hot ids per row")


def test_grouped_negatives_are_unobserved():
    """Softmax/BPR negatives must follow the same rule as sample_negatives: never an item
    the user interacted with. An unfiltered draw made ~9% of them real ratings."""
    from models.fm import FactorizationMachine
    tr = ds.split_rows("train").sample(15000, random_state=2)
    m = FactorizationMachine(k=4, blocks=tuple(D.MF_BLOCKS) + ("history",),
                             objective="softmax", n_neg=6, epochs=1, verbose=False)
    X, _ = m._design(ds, tr, 11)
    m._grp_users, m._grp_items = m._last_layout
    N = 6; G = X.shape[0] // (N + 1)
    rows = ((np.arange(G)[:, None] * (N + 1)) + np.arange(1, N + 1)).ravel()
    assert m._seen_mask[np.repeat(m._grp_users, N), m._grp_items[rows]].sum() == 0
    X2 = m._resample(ds, X, np.random.default_rng(5))
    assert m._seen_mask[np.repeat(m._grp_users, N), m._grp_items[rows]].sum() == 0
    assert X2.nnz == X.nnz, "resampling changed the number of non-zeros"
    ok("grouped negatives exclude observed items, before and after resampling")


if __name__ == "__main__":
    import time
    for fn in [test_causal_history, test_history_spans_splits, test_no_future_in_scoring_frame,
               test_candidates_and_truth, test_popularity_is_train_only, test_feature_blocks,
               test_negative_policies, test_determinism, test_regressions,
               test_recall_ceiling, test_activity_bands_populated,
               test_split_boundaries_avoid_ties, test_eval_targets_are_isolated,
               test_interaction_matrix, test_rank_users_respects_target,
               test_fm_fast_scoring_matches_naive, test_mf_blocks_are_exactly_two_ids,
               test_grouped_negatives_are_unobserved]:
        print(f"\n{fn.__name__}  --  {fn.__doc__ or ''}".rstrip())
        t = time.time(); fn(); print(f"  ({time.time()-t:.1f}s)")
    print("\nAll protocol invariants hold.")
