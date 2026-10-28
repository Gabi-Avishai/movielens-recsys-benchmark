# Recommender Design-Space Benchmark (MovieLens-1M)

A controlled comparison of recommender architectures — from a zero-parameter lookup
table to a two-tower retrieval network — under one dataset, one split, and one
evaluation protocol.

The point is not "which model wins." The point is to isolate **what each additional
modelling assumption actually buys**, and to be able to say so with a number.

## Summaries

- **[What Each Assumption Buys](docs/technical_summary.md)** — technical
  summary: protocol, metrics, standings and every phase's result.
- **[Picking the Next Film](docs/plain_language_summary.md)** — non-technical
  summary of what was built and learned.

## Where it stands

All phases complete (0–9). Test split, candidate pool filtered to items with ≥50 training
ratings, k=10. Recall ceiling is 0.787 for every row.

| Model | NDCG@10 | Recall@10 | Coverage | Gini | Phase |
|---|---|---|---|---|---|
| **Two-tower, sampled negatives** | **0.1653 ± 0.0025** | **0.1282** | **0.453** | **0.856** | 6 |
| Two-tower, unobserved-or-disliked | 0.1552 | 0.1226 | 0.461 | 0.854 | 6 |
| FM, ids + history | 0.1159 ± 0.0015 | 0.0895 | 0.204 | 0.956 | 4 |
| Two-tower, in-batch negatives | 0.1128 | 0.0995 | 0.492 | 0.813 | 6 |
| FM, all 7 blocks | 0.0990 ± 0.0019 | 0.0779 | 0.193 | 0.958 | 4 |
| item_item_cosine | 0.0948 | 0.0675 | 0.118 | 0.983 | 2 |
| user_user_cosine | 0.0932 | 0.0678 | 0.146 | 0.969 | 2 |
| MF (ids only) | 0.0922 ± 0.0005 | 0.0603 | 0.159 | 0.980 | 4 |
| LightGBM, sparse | 0.0862 ± 0.0004 | 0.0651 | 0.120 | 0.979 | 5 |
| LightGBM, engineered | 0.0833 ± 0.0006 | 0.0617 | 0.145 | 0.980 | 5 |
| item_item_cooc | 0.0836 | 0.0555 | 0.040 | 0.993 | 2 |
| poly2 | 0.0761 | 0.0506 | 0.035 | 0.994 | 3 |
| pop_count | 0.0737 | 0.0475 | 0.034 | 0.994 | 2 |
| damped_mean (tuned) | 0.0723 | 0.0482 | 0.033 | 0.995 | 2 |
| logreg | 0.0720 | 0.0466 | 0.036 | 0.994 | 3 |
| genre_popularity | 0.0493 | 0.0391 | 0.210 | 0.962 | 2 |
| content_knn | 0.0220 | 0.0216 | 0.520 | 0.848 | 2 |

Rows with `±` are means over 3–5 seeds at the tuned `k=64`; the rest are single runs, and
Phase 4 showed single runs of the feature-rich FM carry a seed spread of 0.0073 — larger than
several gaps in this table. Treat unseeded rows as indicative.

**Nine results that were not the expected ones:**

- A tuned damped-mean lookup table **loses to raw popularity** (0.0723 vs 0.0737). Exposure
  predicts a test positive better than quality does.
- Logistic regression on all 13,882 features **also loses to raw popularity** (0.0720). What
  is learnable here is almost entirely in the interactions.
- Refusing to share parameters costs more than most of the feature engineering: `poly2` → `mf`
  is **+22.1%**, because 98.3% of `poly2`'s pair weights can never fire at inference.
- The best model **drops four of its seven feature blocks**. `item_genre` is actively harmful
  by two independent tests.
- **No gradient-boosted configuration reaches the memory-based floor**, and the FM and the
  tree want *opposite* feature sets — the signs reverse on both block additions.
- The **two-tower beats the FM by 43%** on NDCG@10 at 2.2× the coverage, inverting the
  expectation that decoupled scoring costs accuracy. Only **10%** of that gap is explained by
  its larger negative sample.
- An oracle over the tower and FM's pooled lists shows **+98.3% headroom** — and no realizable
  combination captured more than **1.5%** of it. Complementary outputs are not combinable
  outputs.
- **The best model on RMSE is the worst ranker.** Rating regression beats the mean baseline by
  18% on RMSE and ranks below raw popularity.
- **The best model is the most fragile.** Taste drift hurts the two-tower nearly twice as much as
  the FM, and 29.4% of users get no relevant item from any model.

## Framing: what problem is actually being solved

MovieLens-1M is treated here as a stand-in for the **candidate generation + ranking
stage of a catalogue recommender**: a user arrives, the system has ~3.9k items, some
user metadata, and that user's interaction history, and it must return an ordered
short list. That framing sets what "good" means:

- **The task is ranking, not rating prediction.** Nobody ships an RMSE. The user sees
  a top-10 list, so the metric is position-sensitive (NDCG@K) and the offline protocol
  must score a *full candidate set*, not a pre-sampled handful of easy negatives.
- **Popularity is a real competitor, not a strawman.** Item popularity has a Gini of 0.677
  on the training split, so a non-personalised list is already decent. *Measured:* raw
  popularity reaches NDCG@10 = 0.0737, and a damped mean rating tuned on validation does
  worse. The floor a learned model actually has to clear is normalised item-item
  co-occurrence at 0.0948, not popularity.
- **A list that is accurate and identical for everyone is a failed product.** Accuracy
  metrics reward popularity bias. Coverage and catalogue-Gini of the served lists are
  reported alongside accuracy as first-class metrics, not as a footnote.
- **Cold start is the expensive case — but this dataset cannot fully represent it.**
  Aggregate metrics are dominated by heavy users (the top 10% write 38.1% of ratings), so
  results are sliced by user-history length. GroupLens filtered MovieLens to users with ≥20
  ratings, so the coldest user here has 10 training interactions: "cold" in this project means
  *relatively light*, and is reported as such.

## Research questions

These are the axes the model ladder is built to answer. Each is a controlled comparison,
not a leaderboard entry. All seven are now settled.

1. ✅ **Does personalisation pay?** Damped-mean popularity vs. anything learned.
   → **Yes, but the trivial baselines are stronger than expected.** A tuned damped mean
   *loses* to raw popularity (0.0723 vs 0.0737), and the real floor is normalised item-item
   co-occurrence at 0.0948. The best learned model reaches 0.1159, **+22.2% over that floor**.
   [Phase 2](#phase-2-the-baseline-ladder), [Phase 4](#phase-4-which-features-actually-matter-and-how-much-is-noise)
2. ✅ **What does the 2nd-order interaction term buy?** Logistic regression and a
   factorization machine on the *identical* sparse feature matrix differ only by the
   `½Σ((XV)² − X²V²)` term. The gap is the answer.
   → **+32.9%**, the largest single factor measured. Logistic regression on all 13,882
   columns scores 0.0720 — *below* raw popularity.
   [Phase 3](#phase-3-interaction-order-parameter-sharing-and-side-features)
3. ✅ **What do shared latent factors buy over per-pair weights?** FM's shared `V` vs. an
   explicit Poly2 weight per feature pair, at matched capacity.
   → **+22.1%**, nearly as much as the interaction term itself. The mechanism is measured,
   not assumed: 98.3% of `poly2`'s pair weights never fire at inference, and of the 1.7% that
   do, 100% were created by negative sampling. [Phase 3](#phase-3-interaction-order-parameter-sharing-and-side-features)
4. ✅ **What do side features buy over pure collaborative signal?** Feature blocks
   (demographics, genre, year, interaction history) added back one at a time.
   → **The question was badly posed.** "Side features" is not one thing: `history` alone is
   worth +25.6%, while `item_genre` is *harmful* by two independent tests, and the best model
   uses three of the seven available blocks. All five blocks together are 17% worse than
   `history` alone. [Phase 4](#phase-4-which-features-actually-matter-and-how-much-is-noise)
5. ✅ **Does decoupling the towers cost accuracy?** A two-tower model scores by dot product
   over independent user/item embeddings — sublinear retrieval via ANN, but no feature
   crossing before the dot product.
   → **No — it gains 43%.** The tower reaches 0.1653 against the FM's 0.1159, at 2.2× the
   coverage. Negative count explains ~10% of that and the loss function none of it, so the lead
   belongs to the architecture: nonlinear towers comparing normalised embeddings.
   [Phase 6](#phase-6-two-tower-retrieval), [Phase 8](#phase-8-the-objective-function)
6. ✅ **Does a two-stage cascade recover it?** Two-tower retrieves, a second model re-ranks.
   → **No, and the reason generalises.** The tower and FM share only 2.7 of 10 picks and an
   oracle over their pooled lists nearly doubles NDCG, yet every realizable combination lost to
   the tower alone. An oracle gap is necessary for combination to help, never sufficient.
   [Phase 7](#phase-7-is-there-anything-to-combine)
7. ✅ **Does the objective function change the ranking — and can it even be compared?**
   Rating regression, thresholded classification, and pairwise BPR train on *different data*
   and admit *different metrics*.
   → **Among ranking objectives, barely; across framings, completely.** Logistic, softmax and BPR
   land within 7% of each other on the FM. Regression wins RMSE and ranks below popularity, and
   three of the four objectives cannot produce an RMSE at all.
   [Phase 8](#phase-8-the-objective-function)

## Dataset and protocol (frozen before any model is written)

**MovieLens-1M** — 1,000,209 ratings, 6,040 users, 3,706 rated items (3,883 catalogue),
4.26% density, collected 2000-04 → 2003-02. Ratings are integer 1–5.

| Choice | Value | Why |
|---|---|---|
| Positive label | `rating ≥ 4` | Yields 57.5% positive rate — near-balanced, so AUC/logloss stay interpretable |
| Split | Per-user chronological 70/15/15 | Every user appears in all three splits. Avoids the leakage of a random split while keeping every user evaluable |
| Candidate set at eval | Full catalogue minus the user's train+val history | No sampled-negative shortcut. Sampled-negative AUC is available in `metrics.py` but deliberately not reported: it depends on each model's sampling policy and is not comparable across models |
| Decision threshold | **None — scores are only ranked** | No model converts a score into a yes/no. Each user receives the top 10 by score, so the operating point is the list length K=10, set by the product rather than tuned. Ranking metrics read only the order of scores and are unchanged by any monotone rescaling, so there is no threshold to select and no precision-recall curve to draw. Recall@K across K (Phase 6) is the ranking analogue of sweeping one |
| Popularity floor on candidates | Reported both with and without the `≥50 ratings` filter | On train-only counts the filter keeps 2,167 of 3,883 items and 96.2% of interactions — it removes **44.2% of the catalogue to gain 3.8% of the interactions**: nearly free on accuracy, ruinous for coverage. Both settings get a row |
| Tie-break | `(user, timestamp, item_id)` | 529,046 rows — 52.9% of the data — share a timestamp with another rating by the same user (bulk rating sessions, up to 30 ratings in one second). Within-burst order is unrecoverable; the tie-break makes it at least *reproducible* |
| Boundary snapping | `snap_ties = True` | A purely positional cut divides a tied burst at 6,085 user-boundaries — about half of all users on each boundary — letting same-session items land on opposite sides of the split. Snapping each boundary to the nearest timestamp edge leaves 128, with a positional fallback for users whose bursts would empty a split |
| History features | Strictly causal, capped to the last 50 items | Causality is the one place leakage is easy and fatal. The cap is a memory necessity — uncapped causal history is 194.5M nonzeros (~2.3 GB) — and doubles as a defensible recency window |
| Item space | All 3,883 catalogue items, not the 3,706 rated ones | Restricting to rated items makes 177 items silently unrecommendable and measures coverage against the wrong denominator |
| User subsampling | `user_frac` (default 1.0) | Scaling studies shrink the user population while holding split, threshold and candidate rule fixed, so the only thing varying is how much data a model has |
| Popularity statistics | Computed on **train only** | Full-data counts would leak the test period into every popularity baseline, i.e. into the floor the whole benchmark is measured against |

Everything above lives in `data.py` and is generated once. No model may re-split.

## Model ladder

| # | Model | Learned params | Axis it isolates | NDCG@10 |
|---|---|---|---|---|
| 0a | Global top-N by count | 0 | Floor | 0.0737 |
| 0b | **Damped mean rating** `(C·m̄ + Σr)/(C + n)` | 0 | The real lookup-table baseline | 0.0723 |
| 0c | Top-N within user's dominant genre | 0 | Personalisation without learning | 0.0493 |
| 0d | Item-item co-occurrence counts | 0 | Collaborative signal without learning | 0.0836 |
| 1 | Item-item cosine CF on the rating matrix | 0 (memory-based) | Classic neighbourhood CF | **0.0948** |
| 2 | Content kNN over genre/year item vectors | 0 (memory-based) | Content-only, cold-start-capable | 0.0220 |
| 3 | Logistic regression on the sparse matrix | O(n) | 1st order only | 0.0720 |
| 4 | Biased MF (= FM on ID blocks only) | O(n·k) | 2nd order, IDs only | 0.0922 |
| 5 | Poly2 | O(n²) | 2nd order, no parameter sharing | 0.0761 |
| 6 | **Factorization Machine** | O(n·k) | 2nd order + sharing + side features | **0.1159** |
| 7 | LightGBM on the same features | trees | What a production team would actually try first | 0.0862 |
| 8 | Two-tower (sampled softmax) | O((n+m)·k) + MLP | Retrieval, decoupled scoring | **0.1653** |
| 9 | Two-tower → reranker cascade | — | Retrieval + ranking | 0.1545 |
| 10 | DeepFM / FM+MLP *(stretch)* | — | Implicit higher-order interaction | — |

Rung 6 is the tuned `ids + history` configuration, which is the best model found; the
all-blocks FM scores 0.0990. Rung 7 is the best of six LightGBM configurations, none of which
reaches rung 1. Rung 2's poor accuracy is the price of being the only rung that
can score an item with no interaction history — 237 items, 6.1% of the catalogue.

Models 3–6 share one feature matrix and one training loop; they differ by configuration
flags, which is what makes the ablation credible rather than three separate codebases.

## The objective study: regression vs. classification vs. ranking

This is broken out because it is the phase most likely to produce a result worth stating,
and because the three framings are routinely conflated.

**The model is not the issue.** A factorization machine is a scoring function
`ŷ = w₀ + ⟨w,x⟩ + ½Σ((XV)² − X²V²)`; the loss sits on top of it. Logistic and squared
loss share the same gradient expression w.r.t. the score (`ŷ − y`), so the same training
loop serves both — the classification variant simply passes the score through a sigmoid
first. Rendle's original formulation covers regression, binary classification and ranking
for this reason.

**The data available is the issue.**

| Framing | Trains on | Native metric | Core assumption |
|---|---|---|---|
| Rating regression | observed ratings only | RMSE / MAE | Ratings are cardinal; selection bias is ignorable |
| Binary classification | observed + sampled unobserved | logloss / AUC | Unobserved ≈ negative |
| Pairwise (BPR) | (positive, negative) pairs per user | NDCG / MAP | Only relative order is meaningful; no absolute scale exists |

A regression model can only be fit where a target exists, i.e. on pairs the user *chose*
to rate. It therefore learns `E[rating | the user selected this item]`. At serving time it
is asked to rank the full catalogue — overwhelmingly pairs the user never selected. This is
missing-not-at-random: users watch what they expect to like and then rate it, which is why
57.5% of ML-1M ratings are ≥ 4 and only 5.6% are 1. The observed matrix is a sample filtered
through the user's own prior. Negative sampling does not solve this so much as replace it
with a different, more tractable assumption.

The consequence is documented — RMSE-optimal models underperform at top-N ranking
(Cremonesi, Koren & Turrin, RecSys 2010). The mechanism is metric mismatch: RMSE weights
all errors equally, top-N weights only the head of the list. An 0.05 error that transposes
ranks 1 and 2 is invisible to RMSE and material to NDCG.

### The experiment

One FM backbone, four objectives, cross-evaluated on both metric families:

| Objective | RMSE | NDCG@10 |
|---|---|---|
| Regression on observed ratings | expected best | |
| Classification, unobserved-as-negative | n/a | |
| Classification, observed-negatives only | n/a | |
| BPR pairwise | n/a | expected best |

The `n/a` cells are the finding, not a gap: **a BPR-trained model cannot produce an RMSE**,
because it learns an order with no calibrated scale. Three of the four cannot be scored on
the metric the fourth optimises. That asymmetry is the argument that these are different
tasks wearing similar notation.

### Two sub-ablations that ride along

- **What is a negative?** The current codebase is inconsistent about this and the two
  choices have never been compared. Negatives may be drawn from unobserved pairs, from
  observed-but-low-rated pairs, or both — these carry different evidence (absence of
  information vs. evidence of distaste) and lumping them is a modelling decision that
  should be measured rather than defaulted into.
- **Where is the positive threshold?** `≥ 4` (57.5% positive) vs. `≥ 3` (83.6%) vs.
  per-user centring (`rating >` that user's own mean), which absorbs rater bias — some
  users never award a 5. Threshold choice moves class balance by 26 points and is rarely
  justified in published work.

### Noted and not pursued

Ratings are **ordinal**, not cardinal: regression asserts that the 1→2 gap equals the
4→5 gap, and 5-way classification discards the ordering entirely. Ordinal regression via a
cumulative-link model is the framing that actually matches the data type. It is out of
scope here, and flagged because the mismatch is near-universal in the literature and
almost never stated.

## Metrics

Reported for every model, on the same candidate set:

- **Accuracy:** Recall@10, NDCG@10, MAP@10, HitRate@10, each reported alongside its
  **recall ceiling** — the best Recall@10 attainable given that users average 13.0
  relevant test items against k=10, and that candidate filtering hides some of them.
  Raw recall is not comparable across user segments without it
- **Beyond-accuracy:** catalogue coverage, Gini of served-item distribution, mean
  popularity-rank of served items
- **Calibration / classification view:** AUC, logloss, PR-AUC on the held-out set
  (labelled as the sampled-negative view where applicable)
- **Cost:** training time, model size, per-user scoring latency
- **Slices:** by user-history length (cold / warm / heavy), by item popularity decile

## Roadmap

| Phase | Goal | What |
|---|---|---|
| 0 (done) | Harness | `data.py` (loader, frozen splits, causal history, negative-sampling policies), `metrics.py`, model interface `fit(train) / score(user, candidates)` — see [Phase 0 notes](#phase-0-the-harness) |
| 1 (done) | EDA | `01_eda.ipynb` — justifies every protocol parameter from the data, and tests for second-order signal before any model is fitted — see [Phase 1 notes](#phase-1-eda) |
| 2 (done) | Baseline ladder | Models 0a–0d, 1, 2, tuned on validation. Produces the floor every later phase is measured against — see [Phase 2 results](#phase-2-the-baseline-ladder) |
| 3 (done) | The linear→FM ablation | Models 3–6 on the identical matrix and training loop. The core scientific contribution — see [Phase 3 results](#phase-3-interaction-order-parameter-sharing-and-side-features) |
| 4 (done) | Feature-block ablation | Blocks added and removed one at a time, over seeds, at tuned `k` — see [Phase 4 results](#phase-4-which-features-actually-matter-and-how-much-is-noise) |
| 5 (done) | Trees | LightGBM on two representations and the same block sets as the FM — see [Phase 5 results](#phase-5-gradient-boosted-trees-and-what-the-same-features-means) |
| 6 (done) | Retrieval | Two-tower, headline metric Recall@K. Negative-policy comparison and a subsample scaling curve — see [Phase 6 results](#phase-6-two-tower-retrieval) |
| 7 (done) | Combination | Restructured from "build a cascade" to "is there anything to combine?" — oracle gate, score blending, LightGBM reranker — see [Phase 7 results](#phase-7-is-there-anything-to-combine) |
| 8 (done) | Objective study | Four losses on one FM backbone, cross-evaluated on RMSE and NDCG, plus negative-count and positive-threshold sweeps — see [Phase 8 results](#phase-8-the-objective-function) |
| 9 (done) | Error analysis | Who fails, which segments, taste drift, the long tail, shared failures — see [Phase 9 results](#phase-9-error-analysis) |
| — | Cold start *(not planned)* | Dropped: GroupLens pre-filtered to ≥20 ratings, so no user here is genuinely cold, and the activity slices in Phases 2–4 already cover what is learnable |
| — | Serving *(not planned)* | Dropped: the Airflow / FastAPI infrastructure already exists in the original three projects and would add no findings |

## Phase 0: the harness

```
data.py                    frozen protocol: loading, splits, causal history, feature
                           blocks, negative-sampling policies, candidates, ground truth
metrics.py                 ranking / beyond-accuracy / classification metrics, slices
models/base.py             the Recommender interface + shared full-candidate ranking loop
models/popularity.py       pop_count, damped_mean
models/neighbourhood.py    item-item co-occurrence and cosine, user-user cosine
models/content.py          content kNN, genre popularity
models/fm.py               logreg / MF / FM (one loop, three configs) and Poly2
models/trees.py            LightGBM over the sparse matrix and over engineered features
models/two_tower.py        PyTorch two-tower, softmax over in-batch or sampled negatives
test_data.py               18 structural tests for the protocol invariants
01_eda.ipynb ... 09_error_analysis.ipynb    one notebook per phase
results_phase{2..8}.csv    per-phase results, one row per model x setting x seed
results_phase9_users.csv   per-user outcomes and descriptors from the error analysis
cache/                     fitted models and eval results, keyed on configuration
```

**Feature blocks.** `feature_matrix(rows, blocks)` assembles any subset of
`user_id, item_id, user_demo, item_genre, item_year, history, genre_history`
(6,040 + 3,883 + 30 + 18 + 10 + 3,883 + 18 = 13,882 columns in full). The
interaction-order and feature-block ablations are therefore a choice of argument,
not a second implementation: `MF_BLOCKS` yields exactly two non-zeros per row and
reproduces biased matrix factorization, `ALL_BLOCKS` the feature-rich FM.

**Negative-class policies.** `sample_negatives(rows, policy)` implements the three
arms of the objective study — `observed_only` (702,811 rows, 59.5% positive),
`unobserved_only` (836,074 rows, 50.0% positive), `both` (1,120,848 rows, 37.3%
positive). Sampled rows inherit their anchor's sequence position, so their causal
history window is the user's history at that moment rather than a leak from the future.

**Split.** 699,630 / 150,354 / 150,225 interactions (69.9 / 15.0 / 15.0), with positive
rate drifting 0.595 → 0.541 → 0.519 across train → val → test. That drift is a property of
the protocol worth stating up front: later interactions in a user's history are rated
slightly lower, so a model that ignores time is being scored on a distribution mildly
harder than the one it trained on.

**Evaluable users.** 5,877 of 6,040 users have at least one test positive reachable
inside the filtered candidate pool, with a mean of 13.3 relevant test items each.
Users with no reachable ground truth are *skipped and counted*, never scored as
zero — scoring them as zero would penalise a model for the candidate filter's
behaviour rather than its own.

**The recall ceiling.** Metrics are scored against the *full* ground truth: an
unreachable target is a real product cost, not an excuse. But Recall@10 cannot reach
1.0 on this data, and it is essential to know by how much. Maximum attainable
Recall@10 is **0.787 filtered** and **0.804 unfiltered** — so candidate filtering
costs only 0.017, and the binding constraint is arithmetic: the average user has 13.3
relevant test items competing for 10 slots.

The ceiling is not a footnote, because it varies by segment and inverts conclusions
drawn without it. Ranking by raw popularity gives Recall@10 of 0.052 for cold users
against a ceiling of 0.978, and 0.045 for heavy users against a ceiling of 0.447 —
raw recall says cold users are served *better*, while recall-against-attainable says
heavy users are served roughly twice as well. Any cold-start claim made on
unnormalised recall would point the wrong way.

**Activity bands.** Slices use training-interaction count (`ds.user_train_count`,
range 10–1,619, quartiles 31/67/145), banded at <35 / 35–150 / ≥150 to give
1,741 / 2,845 / 1,454 users. MovieLens guarantees ≥20 ratings per user *in total*, so
slicing on total count leaves the cold band empty and hides the segment the slice
exists to expose.

**Invariants.** `test_data.py` (18 checks) asserts, rather than assumes: that a row's history is
exactly the items in `[seq-K, seq)` of that user's sequence; that test rows carry
pre-test history but no test items; that candidate sets exclude train+val history
while leaving ground truth reachable; that popularity is counted on train only;
that no sampled negative is an item the user actually watched; and that the split
and the negative sampling reproduce bit-for-bit across runs. A regression group pins
previously-fixed defects: default arguments that no policy accepted, a `seed`
parameter that was accepted and ignored, empty input frames that raised, binary-only
AUC reached by the regression arm's 1–5 targets, and activity bands that could be
silently empty, and that boundary snapping actually suppresses tie crossings.

**Harness validation.** Two zero-parameter models were run end-to-end to confirm the
loop closes. With untuned settings, ranking the full filtered candidate set for all
5,870 users takes well under a second per model and produces NDCG@10 in the 0.03–0.07
range at **3.3% catalogue coverage and 0.994 Gini** — the popularity pathology the
beyond-accuracy columns exist to expose, visible in the very first row of the table.
Tuned baselines are Phase 2's subject; these numbers establish only that the
measurement apparatus works.

## Phase 1: EDA

`01_eda.ipynb` derives each protocol parameter from the data rather than asserting it, and
closes with a table mapping every choice to the section that justifies it. Three findings
changed the protocol or the reading of later results.

**Boundary snapping was adopted here.** Measuring the tie problem properly showed that a
purely positional cut divides a same-second burst at 6,085 user-boundaries — roughly half
of all users on each boundary. Snapping boundaries to timestamp edges cuts that to 128
(97.9%), moves split shares *closer* to target (69.9/15.0/15.0 vs the positional
69.7/15.0/15.3), and shifts per-split positive rates by at most 0.13 points. Adopted as
`Config(snap_ties=True)`, reversible, and covered by a test.

**Second-order signal is present, and measured before any model was fitted.** Across 122
age-band × genre cells, the residual from a strictly additive (first-order) model has a
standard deviation 2.2× the mean binomial standard error, and **17.2% of cells fall beyond
three standard errors** where an additive model predicts about 0.3%. Gender × genre is
stronger at 4.6×. The effects are interpretable — under-18s rate Horror 1.29× above the
additive prediction, the 56+ band rates Sci-Fi 0.79×, women rate Musicals 1.19× and
Westerns 0.78×. This is the empirical warrant for the second-order rungs of the ladder,
and it is a *lower bound*: the high-cardinality user×item interaction an FM learns through
its latent factors cannot be tabulated this way at all.

**The drift is within-user, not calendar.** Positive rate falls cleanly from 64% to 50%
across a user's own sequence, while the month-by-month series drifts only mildly and
noisily. The 7.8-point train→test gap is therefore structural, and a train/test gap in
Phase 3 onwards must not be read as overfitting by default.

Two limits are worth carrying forward. GroupLens filtered MovieLens to users with ≥20
ratings, so the coldest user here has 10 training interactions — "cold start" in this
project means *relatively light*, and cannot be reported as the real thing. And 237 items
have zero training interactions: 6.1% of the catalogue is invisible to every collaborative
model, reachable only through content features, which caps what the content-based rungs
can contribute.

## Phase 2: the baseline ladder

`02_baselines.ipynb`. Hyperparameters selected on a validation target whose candidate pool
excludes training history only, so tuning never sees a test item. Test results, candidate
pool filtered to items with ≥50 training ratings:

| | Model | NDCG@10 | Recall@10 | Coverage | Gini |
|---|---|---|---|---|---|
| 1i | **item_item_cosine** | **0.0948** | 0.0675 | 0.118 | 0.983 |
| 1u | user_user_cosine | 0.0932 | 0.0678 | 0.146 | 0.969 |
| 0d | item_item_cooc | 0.0836 | 0.0555 | 0.040 | 0.993 |
| 0a | pop_count | 0.0737 | 0.0475 | 0.034 | 0.994 |
| 0b | damped_mean | 0.0723 | 0.0482 | 0.033 | 0.995 |
| 0c | genre_popularity | 0.0493 | 0.0391 | 0.210 | 0.962 |
| 2 | content_knn | 0.0220 | 0.0216 | 0.520 | 0.848 |

Recall ceiling is 0.787 for every row. **The floor for Phase 3 is 0.0948, not 0.0737.**

**A tuned damped mean loses to raw popularity** (0.0723 vs 0.0737). `C` was swept across five
orders of magnitude; NDCG rises monotonically throughout, so no amount of trusting a raw mean
rating helps. The limit is not popularity either — the ordering converges exactly
(Kendall τ = +1.000) onto `nᵢ(r̄ᵢ − r̄)`, popularity weighted by signed quality margin — but it
approaches popularity at the head without overtaking it. This is the selection bias from
Phase 1 showing up in a metric: a test positive requires the user to have *watched* the film,
and exposure predicts that better than merit does.

**Normalisation is what earns the neighbourhood gain, not neighbourhood structure.** Raw
co-occurrence already beats popularity (0.0836) but serves 4.0% of the catalogue — it is
largely popularity in disguise, since a popular item co-occurs with everything. Dividing by
the norms yields 0.0948 at 11.8% coverage: 13% more accuracy and three times the reach.

**The candidate filter discriminates between model families.** Removing it moves NDCG by
−0.4% for every collaborative model and by **−27.5% for `content_knn`**, whose coverage
simultaneously rises from 52.0% to 82.2%. Reporting only the filtered setting would
understate content-based methods by nearly a third and leave every other row untouched. The
"report both settings" rule was adopted in Phase 1 on theoretical grounds; this is the
measurement that justifies it.

**Accuracy and coverage are inversely ordered across the whole ladder.** `content_knn` serves
15× the catalogue `pop_count` does at under a third of its NDCG, and no model achieves both.
Whether a learned model can bend that frontier rather than slide along it is the open
question Phase 3 inherits.

**Two segment-level findings.** `genre_popularity` beats `pop_count` by 11.8% for cold users
and loses to it by 48% for heavy users — a zero-parameter content prior is worth most exactly
where collaborative signal is thinnest. And ceiling-normalisation halves the apparent
cold-start gap: `item_item_cosine` looks 2.5× better for heavy users on raw NDCG (0.1669 vs
0.0660) but is 1.3× better as a fraction of attainable recall (0.121 vs 0.092). Most of the
headline deficit is arithmetic, not model failure.

## Phase 3: interaction order, parameter sharing, and side features

`03_interaction_ablation.ipynb`. Four configurations of **one** training loop — `logreg` is
`order=1`, `mf` is `blocks=MF_BLOCKS`, `fm` is both switched on — so every difference is
attributable to the configuration rather than to an implementation. Test, filtered pool:

| Model | Val logloss | NDCG@10 | Recall@10 | MAP@10 | Coverage | Gini |
|---|---|---|---|---|---|---|
| `logreg` | 0.4520 | 0.0720 | 0.0466 | 0.0344 | 0.036 | 0.994 |
| `poly2` | 0.4591 | 0.0761 | 0.0506 | 0.0360 | 0.035 | 0.994 |
| `mf` | 0.4018 | 0.0929 | 0.0610 | 0.0457 | 0.155 | 0.981 |
| **`fm`** | **0.3687** | **0.0956** | **0.0739** | 0.0452 | **0.173** | **0.961** |
| *item_item_cosine (Phase 2 floor)* | — | *0.0948* | *0.0675* | *0.0474* | *0.118* | *0.983* |

**The three contrasts.** Interaction term (`logreg`→`fm`) **+32.9%**; parameter sharing
(`poly2`→`mf`) **+22.1%**; side features (`mf`→`fm`) **+2.9%**.

> Every figure in this section comes from a **single training run per configuration**. The
> first two contrasts are far larger than the seed-to-seed variation measured in Phase 4 and
> stand as reported. The third is not: at +0.0027 absolute it sits inside the noise of a
> single seed, and Phase 4 retracts it as a demonstrated effect. Read the side-feature claims
> below against that.

**First-order is not enough to beat counting.** Logistic regression on all 13,882 columns
reaches 0.0720 — below raw popularity's 0.0737. Whatever is learnable here lives almost
entirely in the interactions.

**Parameter sharing is worth nearly as much as the interaction itself, and the mechanism is
measured rather than asserted.** `poly2` and `mf` saw identical features, rows, labels and
loop; only the pair parameterisation differs. `poly2` holds 831,968 pair weights, but of the
1.63M (user, candidate) pairs scored at test time **98.3% have no weight at all** — a candidate
is by definition an item the user never interacted with, so no positive ever created one. Of
the 1.7% that do fire, **100% were created by negative sampling and 0% by a positive**, mean
value −0.052, so they can only push an item down. `poly2` collapses onto ranking by learned
item bias (Kendall τ = +0.999 against bias-only). An FM has not seen the pair either, but it
has seen the user and the item, and composes an answer from shared factors.

**Side features appear to be worth little on NDCG** — under 3% from adding demographics, genre,
year and history to a model that already has user and item ids, despite Phase 1 measuring real
demographic × genre interaction. Whether that 3% is an effect at all, and which block produces
it, is settled in Phase 4 rather than here.

**The honest headline: the FM beats the Phase 2 floor by 0.9%** (0.0956 vs 0.0948) — a margin
smaller than this configuration's seed-to-seed spread, and therefore not a result. Quoting the
29.7% gain over popularity while omitting that normalised co-occurrence already captures nearly
all of it would misrepresent it twice over.

**The accuracy number is the wrong place to look.** Recall is 9.5% higher and coverage 47%
higher (0.173 vs 0.118) at a *lower* Gini. Every Phase 2 accuracy gain was bought by
concentrating further on the head; this one is not. That answers the open question Phase 2
left — a learned model can bend the accuracy/coverage frontier rather than slide along it.
Against that, MAP@10 is lower (0.0452 vs 0.0474): the FM surfaces more relevant items in the
top ten but orders them slightly worse.

The FM shows the widest generalisation gap of the four (0.147 vs `mf`'s 0.111) and early-stops
at epoch 5. Neither `k` nor regularisation was tuned, so the +2.9% from side features is
plausibly a floor rather than a ceiling.

## Phase 4: which features actually matter, and how much is noise

`04_feature_blocks.ipynb`. Every configuration is run over 3–5 seeds, varying both weight
initialisation and negative sampling, and reported with a standard error. `k` and
regularisation are tuned on validation first, because an ablation run on an undertuned model
measures the tuning rather than the feature.

**Seed variance changes what Phase 3 could claim.** Over five seeds the FM's NDCG spread is
**0.0073, four times the MF's 0.0017**. At Phase 3's `k=16` the side-feature effect is
+0.0026 with SE 0.0013 — **p = 0.085, not significant** — and the FM's worst seed (0.0918)
falls below both the MF mean and the Phase 2 floor. Recall (p = 0.0001) and Gini (p < 0.0001)
separate cleanly where NDCG does not.

**Tuning: `k = 64`, `reg = 1e-5`,** with `k` confirmed as a genuine interior peak (96 → 0.181,
192 → 0.175 on validation). At the tuned `k` the side-feature effect becomes +7.3%
(p = 0.025) — undertuning had hidden a real effect.

**But "side features" is not one thing, and the blocks disagree in sign.**

| Block | Added to ids-only | Removed from all | Verdict |
|---|---|---|---|
| `history` | **+0.0236** (p<0.001) | **−0.0109** (p=0.007) | strongly helps |
| `genre_history` | −0.0023 (p=0.055) | −0.0034 (p=0.35) | no NDCG effect; helps recall |
| `user_demo` | −0.0030 (p=0.043) | −0.0030 (p=0.26) | none or slightly harmful |
| `item_year` | −0.0084 (p=0.028) | −0.0034 (p=0.48) | harmful alone, redundant in company |
| `item_genre` | **−0.0157** (p<0.001) | **+0.0071** (p=0.048) | **actively harmful, both tests** |

**The best model uses three blocks, not seven.**

| Configuration | Blocks | NDCG@10 | Recall@10 | Coverage | Gini |
|---|---|---|---|---|---|
| **ids + history** | **3** | **0.1159 ± 0.0015** | **0.0895** | **0.204** | **0.956** |
| ids + history + genre_history | 4 | 0.1136 ± 0.0014 | 0.0879 | 0.204 | 0.957 |
| all blocks | 7 | 0.0990 ± 0.0019 | 0.0779 | 0.193 | 0.958 |
| ids only (MF) | 2 | 0.0922 ± 0.0005 | 0.0603 | 0.159 | 0.980 |
| *item_item_cosine (Phase 2 floor)* | — | *0.0948* | *0.0675* | *0.118* | *0.983* |

Dropping four of seven blocks improves NDCG **17%** over using all of them (p = 0.002). The
best configuration beats the Phase 2 floor by **+22.2%** while serving 20.4% of the catalogue
against the floor's 11.8%, at a lower Gini — the frontier-bending result Phase 3 gestured at,
now at a margin that survives the seed test.

**Why genre hurts** — most likely redundancy plus capacity. An item's genre is already
recoverable from its id, and the id factor is estimated from that item's own interactions
rather than pooled across the 1,603 films sharing the Drama tag. A dense, low-cardinality
block invites genre-level interactions that generalise worse than the item-level ones. The
two most dense, lowest-cardinality blocks (`item_genre`, `item_year`) are the two most
harmful, which fits. This does not contradict Phase 1: genre carries signal, but supplying it
*alongside item ids* to a model that can already infer it is a different proposition.

**Two points that generalise.** Validation log loss and validation NDCG select different
models — Spearman −0.615 across the grid, so broad agreement, yet the loss-optimal `k` is 16
and the NDCG-optimal is 64, and choosing on loss gives up 9.4% NDCG. And every result from
here on is reported over ≥3 seeds: two conclusions in this project have already changed sign
or significance once seeds were added.

## Phase 5: gradient-boosted trees, and what "the same features" means

`05_trees.ipynb`. Two representations, three seeds each. `lgbm_sparse` consumes the FM's
identical 13,882-column matrix; `lgbm_native` consumes native categoricals plus dense
engineered aggregates — each a hand-built stand-in for something the FM gets from
`⟨v_u, v_i⟩`. User-side features are read from the same causal history window the FM uses.

| Model | NDCG@10 | Recall@10 | Coverage | Rank time |
|---|---|---|---|---|
| *FM, ids + history (Phase 4)* | *0.1159* | *0.0895* | *0.204* | *~14 s* |
| *item_item_cosine (Phase 2 floor)* | *0.0948* | *0.0675* | *0.118* | *<1 s* |
| `lgbm_sparse` | 0.0862 ± 0.0004 | 0.0651 | 0.120 | 158 s |
| `lgbm_native_lambdarank` | 0.0839 ± 0.0007 | 0.0618 | 0.082 | 103 s |
| `lgbm_native` | 0.0833 ± 0.0006 | 0.0617 | 0.145 | 176 s |
| `lgbm_native_nocontent` | 0.0817 ± 0.0002 | 0.0592 | 0.101 | 92 s |

**No tree configuration reaches the Phase 2 floor**, and the best FM is 34% ahead of the best
tree.

**The two families want opposite feature sets.** Running the identical block sets through both:

| Block set | FM | LightGBM |
|---|---|---|
| ids only | 0.0922 | 0.0745 |
| ids + history | 0.1159 (**+25.7%**) | 0.0751 (**+0.8%**) |
| all blocks | 0.0990 (**−14.6%**) | 0.0862 (**+14.8%**) |

Sparse high-cardinality blocks (`history`, 3,883 columns) are worth +25.7% to the FM and
nothing to the tree: the FM projects them into a dense latent space where a user's history
pattern is a `k`-vector, while a tree can only split on "did this user watch item 1,234" one
column at a time. Dense low-cardinality blocks (genre, year, demographics) are worth −14.6%
to the FM and **+14.8%** to the tree, for the mirror-image reason.

**Phase 4's explanation is confirmed; this phase's own hypothesis is refuted.** Phase 4
predicted trees would be less harmed by content — they are not merely unharmed, content is
the largest gain available to them. But the notebook opened by hypothesising that
`lgbm_sparse` beats `lgbm_native` because it gets `history` at full resolution; adding
`history` to the sparse tree is worth **+0.0006**. It wins on the raw content blocks instead,
which is the opposite mechanism. Recorded rather than quietly rewritten.

**The native tree is mostly a popularity model.** `item_pop` accounts for 48% of total split
gain; the engineered interaction features contribute about 11% between them. Hand-engineered
stand-ins for `⟨v_u, v_i⟩` are much weaker than the learned thing — the argument for
factorisation from the other direction.

**Serving cost runs the wrong way too.** The FM ranks all 5,877 users in ~14 s because
user-side and item-side blocks are disjoint, so a candidate costs O(k). A GBDT admits no such
decomposition and needs a materialised row per (user, candidate) pair — 12.7M of them, 91–176 s.
**6–12× more expensive to serve and less accurate**; there is no operating point on this
problem where it wins. LambdaRank does not rescue it either: 0.0839 vs 0.0833, within noise,
while halving coverage.

## Phase 6: two-tower retrieval

`06_retrieval.ipynb`. Headline metric is **Recall@K**, not NDCG@10 — a retrieval stage's job
is getting relevant items into a shortlist. The user tower reads the same causal history
window the FM's `history` block uses, so neither model knows more about a user. BPR is
deferred to Phase 8. Tuning was capped at a 2×2 grid, 55 s total.

| Model | Recall@100 | Recall@500 | NDCG@10 | Coverage | Gini | Fit |
|---|---|---|---|---|---|---|
| **tower, sampled negatives** | **0.4724 ± 0.0014** | **0.8086** | **0.1653** | 0.453 | 0.856 | 459 s |
| tower, unobserved-or-disliked | 0.4671 ± 0.0012 | 0.8059 | 0.1552 | 0.461 | 0.854 | 358 s |
| tower, in-batch negatives | 0.4236 ± 0.0007 | 0.7775 | 0.1128 | 0.492 | 0.813 | 13 s |
| FM, ids + history (Phase 4 best) | 0.4195 ± 0.0062 | 0.7987 | 0.1159 | 0.204 | 0.956 | 40 s |
| tower, no history | 0.3611 ± 0.0008 | 0.7690 | 0.0813 | 0.449 | 0.867 | 6 s |

**The phase's own prediction failed.** It opened by expecting the tower to lose on NDCG@10
because it cannot cross features before the dot product. It wins by **43%**, at 2.2× the
catalogue coverage. The framing was too coarse: each tower is an MLP, so it crosses features
freely *within* each side and gives up crossing only *across* the user/item boundary. The FM
crosses the boundary but is linear in its factors. The real contrast is where the nonlinearity
sits, and here a nonlinearity inside each tower beats a bilinear term across them.

**Negative count explains almost none of the gap.** Raising the FM to 2, 4 and 8 negatives
closes only **10.1%**, and the FM peaks at 1:2 then *degrades* — binary log loss with more
negatives shifts probabilities down rather than sharpening order. The remaining explanation is
the softmax objective plus the tower nonlinearity, which Phase 8 is positioned to separate.

**The inherited negative-sampling disagreement, settled.** `recommender_2` used every
unobserved item as a negative; `recommender_3` also drew from items rated below threshold. On
one backbone with everything else fixed, **uniform-unobserved wins** (0.1653 vs 0.1552). The
likely reason is instructive: an item rated 1–3 is one the user *chose to watch*, so it is
popular and plausible — training against it teaches the model to suppress exactly what it
should retrieve. Absence of evidence is a cleaner negative than evidence of dislike.

**History is decisive across three model families.** Removing it costs the tower 24% of
Recall@100 and 51% of NDCG@10; Phase 4 found it worth +25.6% to the FM; Phase 5 found it worth
nothing to a tree.

**Scale does not explain the tower's advantage.** Subsampling the user population 10× with the
protocol fixed, the Recall@100 ratio is flat at ~1.16 while the NDCG ratio *shrinks* from 1.77
to 1.41 — the FM closes ground as data grows. Note the limit: this subsamples *users*, not
items, and the catalogue stays at 3,883 throughout, so it rules out "the tower wins because it
has more data" without addressing whether a larger catalogue would change things.

**The latency argument is asymptotic and is stated as such.** With 3,883 items an ANN index is
pointless and both models score a candidate in O(k), so on this dataset they are equally cheap
to serve. The tower's structural advantage needs a catalogue north of 10⁵ items to pay for
itself. A fabricated speedup would be the easiest claim in this project to get caught on.

## Phase 7: is there anything to combine?

`07_combination.ipynb`. The planned cascade (tower retrieves, FM reranks) was undermined by
Phase 6 — the FM ranks *worse* than the tower, so reranking would degrade a good list, and at
3,883 items there is nothing to narrow. The phase was restructured to ask the question the
mechanism depends on, with a gate before any machinery was built.

**The gate passed convincingly.** The two models share only **2.70 of 10** top-10 items, 11.3%
of users get lists with nothing in common, the union holds **37.7% more relevant items** than
the tower alone, and for 31.1% of users the FM surfaces a hit the tower missed.

| | NDCG@10 | vs tower alone |
|---|---|---|
| tower alone | 0.1613 | — |
| blend, weight chosen on validation | 0.1595 | **−1.1%** |
| cascade, LightGBM reranker without ids | 0.1545 | −4.3% |
| cascade, LightGBM reranker with ids | 0.1444 | −10.5% |
| *blend at the test-optimal weight (illegitimate)* | *0.1637* | *+1.5%* |
| **oracle: best 10 of the pooled union** | **0.3199** | **+98.3%** |

**Every realizable scheme lost.** Even allowing the illegitimate move of selecting the blend
weight on test, the ceiling is +1.5% — roughly **1.5% of the oracle headroom**.

**Why the oracle bound was misleading, and this is the transferable lesson.** The bound answers
"does the union contain better items?" — it does. Exploiting it requires knowing *which* items
in the union are relevant, which is the original ranking problem restated. The oracle is handed
the test labels; a real combiner has only two scores, and **neither carries information about
when the other model is right**. An oracle gap is a necessary condition for a combination to
help, never a sufficient one.

**The validation weight did not transfer.** Blending gained +6.6% on validation (w=0.35) and
lost 1.1% on test, where the optimum is w=0.20. The cause is structural: the validation
candidate pool excludes only training history while test excludes train+val, and the tower
scores 0.2377 on validation against 0.1613 on test — materially different tasks. Note what this
does *not* undermine: every hyperparameter selected on validation in Phases 2–6 was an
*intra-model* choice and transferred fine. It is specifically the *inter-model* mixing weight
that failed, because it depends on the models' relative difficulty on the target.

**The reranker memorised ids rather than combining scores.** The first LightGBM reranker drew
**78.7% of its split gain from the user and item id categoricals** and 4.7% from the model
scores it was given — it learned which items were relevant *for validation* and carried none of
it to test. Dropping the ids forced it to use `tower_score` (46.0%) and `fm_score` (22.2%); that
version behaves sensibly and still loses 4.3%. Phase 5's finding recurring, sharpened into
outright memorisation by the train/eval target mismatch.

**What would have to change for a cascade to earn its place:** a reranker with features neither
retriever has (session context, dwell time, price); a cheap retriever paired with an expensive
ranker, rather than one model that is better at both; and a catalogue large enough that
exhaustive scoring is infeasible. None hold here.

## Phase 8: the objective function

`08_objectives.ipynb`. One FM backbone (`k=64`, ids + history), four losses sharing one gradient
path — each reduces to an error term multiplying the same `∂ŷ/∂θ`. BPR falls out of the algebra
as the softmax with a single negative. Three seeds each.

| FM objective | NDCG@10 | RMSE on ratings |
|---|---|---|
| logistic (pointwise) | **0.1159 ± 0.0015** | n/a |
| softmax, 8 negatives (listwise) | 0.1143 ± 0.0001 | n/a |
| BPR (pairwise) | 0.1078 ± 0.0008 | n/a |
| regression on ratings | 0.0524 ± 0.0015 | **0.944** |
| *two-tower (Phase 6)* | *0.1653* | |

**The loss does not explain the tower's lead.** Softmax and logistic are indistinguishable on the
FM (p = 0.33) and neither moves it toward the tower. Pushing the softmax from 1 to 32 negatives
peaks at 4 (0.1150) and then *falls*, while coverage collapses 0.204 → 0.117. With Phase 6 showing
negative count explains ~10% and this phase showing the loss explains none, what remains is
architectural: nonlinear MLPs inside each tower, L2-normalised embeddings compared through a
temperature with no per-item bias, and the content inputs the towers carry. Those are eliminated
as a group rather than separated; the clean single-factor test would be the tower without its
MLPs. A hypothesis for the FM's decline with more negatives: uniform negatives are mostly
unpopular items, and the FM's free per-item bias can learn "popular beats unpopular" directly —
consistent with the coverage collapse, and a shortcut the tower's bias-free cosine does not have.

**Listwise losses are fragile to false negatives.** Grouped objectives must exclude items the
user interacted with. An unfiltered uniform draw makes **9.4%** of negatives real ratings (59.6%
for the heaviest user) and alone costs the FM softmax **14%** (0.1143 → 0.0987) and BPR 9%. A
softmax concentrates gradient on the highest-scoring negatives, which is exactly where a
mislabelled positive sits. Now enforced by a protocol test.

**The best model on RMSE is the worst ranker.** Regression beats the mean-rating baseline by 18%
(0.944 vs 1.157) — it genuinely learned the rating scale — and ranks below raw popularity. It can
only be trained on items users chose to watch, then must rank what they did not choose: Phase 1's
selection bias, visible as a 55% ranking deficit. Three of the four objectives cannot be scored
on RMSE at all — BPR and softmax learn an order with no scale, raw scores above 40, and their
losses are unchanged by adding a constant to every score. The prediction going in was half
right: regression won RMSE, but the expected ranking winner, BPR, came last of the three.

**The positive threshold.** Trained at `≥3`, `≥4`, `≥5` and evaluated against the same `≥4`
truth — valid because splits, sequences and candidate pools are verified identical across
thresholds: `≥3` and `≥4` are indistinguishable (0.1145 vs 0.1159) with `≥3` reaching more of the
catalogue (coverage 0.231 vs 0.204), while `≥5` loses 10.6%. Phase 1 chose `≥4` for class balance,
a reason that stops applying once negatives are sampled. The choice survives but is not uniquely
right.

## Phase 9: error analysis

`09_error_analysis.ipynb`. The two-tower, the FM (ids + history) and item-item cosine, examined
per user and per item. Reuses Phase 7's cached score vectors — nothing is retrained.

**Averages hide a population that gets nothing.** Zero relevant items in the top 10: tower
43.7%, FM 52.7%, cosine 60.4%. **1,726 users (29.4%) are failed by all three.** The top fifth of
users hold 61.5% of the tower's total NDCG. A random list would fail 93.5%, so the models work —
but their quality is concentrated.

**What predicts failure, with confounds removed.** Every raw descriptor is tangled with how many
relevant test items a user has (activity correlates with it at ρ = +0.846), so effects come from
a multivariate model with standardised predictors and robust errors:

| Predictor (per 1 sd) | tower | FM | cosine |
|---|---|---|---|
| number of test targets | +0.078 | +0.071 | +0.075 |
| **excess taste drift** | **−0.032** | −0.017 | −0.020 |
| training activity | −0.022 | −0.022 | −0.024 |
| rating generosity | −0.019 | −0.018 | −0.023 |
| mainstream taste | n.s. | +0.006 | +0.007 |

**The best model is the most fragile to changing taste.** Drift is the tower's largest effect
after target count and nearly twice its effect on the others; its NDCG falls 38% from the most
stable fifth of users to the most drifted, and its lead over the FM shrinks from +65% to +35%.
Drift needed a correction first: raw genre divergence correlates with test-set size at
ρ = −0.533, because a few test items look "drifted" by sampling noise alone. Subtracting each
user's expected divergence under no change brings that to +0.058.

**Two raw effects were confounds.** Activity *reverses*: at a fixed number of targets, heavier
users are harder. The apparent penalty for mainstream taste vanishes for the tower. And the
descriptors together explain only ~11% of per-user variance — most of what separates well- and
poorly-served users is not captured by any of them.

**The long tail is where recommendations go missing, and where the tower wins.** Half of all
relevant items sit in the five least popular deciles; on reachable items the tower retrieves
3.6% of them, the FM 0.5%, cosine 0.0%. On the most popular decile the order **reverses** —
cosine 43.3%, FM 35.4%, tower 25.9%.

**This is the mechanism behind Phase 7.** Items only the tower finds have a median of 703 training
ratings; items only the FM finds, 1,305. The models are complementary along an axis a combiner
cannot see in advance — whether a user's next liked film is a blockbuster or a tail item — which
is why a 98% oracle headroom yielded no usable gain.

**Directions this points to, untested:** recency-weighted or session-aware history to address
drift; a *popularity-conditioned* combination (FM for head items, tower for the tail) rather than
Phase 7's global blend; and tail exposure as an explicit objective.

## Formulas

Exact definitions for every non-learned baseline, plus the metric definitions, are in
[FORMULAS.md](FORMULAS.md) — each verified against the shipped code by independent
reimplementation.

A plain-language companion, **[Seven Ways to Rank a Film](docs/seven_ways_to_rank_a_film.md)**,
walks all seven baselines through one worked example: user 76 (a comedy watcher with seven
liked films) scoring *The Empire Strikes Back*. The same film comes out ranked 1st by two
models and 1,262nd by another, which makes the difference between the models concrete in a
way the notation does not.

## Running it

**Data.** MovieLens-1M is not redistributed here: its licence does not permit
redistribution without separate permission from GroupLens. Download `ml-1m.zip` from
[GroupLens](https://grouplens.org/datasets/movielens/1m/) and put `ratings.dat`,
`movies.dat` and `users.dat` in a `data/` directory next to `data.py`.

**Environment.** Python 3.12 with the pinned top-level dependencies in `requirements.txt`
(`pip install -r requirements.txt`). It is not a complete lockfile, so transitive versions may
differ from the environment that produced the results. The two-tower model uses a CUDA GPU when
one is available and otherwise runs on the CPU, more slowly.

```
python test_data.py        # protocol invariants -- run before trusting any result
python -m models.base      # interface reference
```

**Notebooks.** Run `01_eda` to `09_error_analysis` in numerical order. Notebooks 02–09 write
their own `results_phase*.csv`; some later notebooks read earlier artifacts, and fitted models
are cached under `cache/`. None of these are tracked.

## Known risks to audit before trusting any number

- **History-feature leakage.** History state must advance strictly causally and never
  incorporate the row being scored. This is checked with an explicit assertion in the
  harness, not by inspection.
- **Evaluation pool contamination.** The candidate-exclusion set at test time must be
  train+val history only. Using full history silently removes the test items.
- **Sampled-negative AUC is not comparable across models** that sample negatives
  differently. It is reported per-model with its sampling policy stated, never as a
  headline number.
- **Unfiltered negatives in a listwise loss.** *Encountered.* Drawing softmax/BPR negatives
  uniformly from the whole catalogue made 9.4% of them items the user had rated and cost the
  softmax 14% NDCG. Pointwise losses are far less exposed to this than listwise ones.
- **An oracle gap read as a green light.** *Encountered.* A +98.3% oracle bound over two
  models' pooled lists produced no realizable gain above 1.5%. The bound measures whether
  better items are *present*, not whether anything can *identify* them.
- **Coverage collapse hidden by accuracy.** A model can win NDCG@10 by serving the same
  fifty items to everyone. This is why the Gini column exists.
- **Single-seed conclusions.** *Encountered.* The feature-rich FM has a seed-to-seed NDCG
  spread of 0.0073, larger than several differences this benchmark reports. Phase 3 drew a
  conclusion from one run per configuration that Phase 4 had to retract. Every result from
  Phase 4 on is reported over ≥3 seeds with a standard error.
- **Ablating an untuned model.** *Encountered.* At `k=16` the side-feature effect was not
  significant; at the tuned `k=64` it is (p = 0.025). An ablation run on an undertuned model
  measures the tuning, not the feature.
- **Selecting on the wrong criterion.** Validation log loss and validation NDCG correlate at
  Spearman −0.615 across the tuning grid but pick different models — the loss-optimal `k` is
  16, the NDCG-optimal is 64, and choosing on loss gives up 9.4% NDCG.

## Scope boundaries

Stated so they aren't implicitly promised:

- **No online evaluation.** Every number here is offline and therefore a proxy. Offline
  NDCG improvements are known to correlate imperfectly with online engagement.
- **No sequential/session models** (GRU4Rec, SASRec, transformers). ML-1M has timestamps
  but the design space studied here is feature-interaction and retrieval architecture,
  not sequence modelling. A defensible extension, not a gap.
- **No text or image content features.** Item content is limited to genre and year.
- **ML-1M is small and old.** Conclusions about parameter sharing under sparsity should
  hold directionally; absolute numbers should not be read as production estimates.
- **No calibration, and no yes/no decisions.** Predicted probabilities are not calibrated to
  any real-world rate. The FM trains on one sampled negative per positive, so it sees a world
  where half of all pairs are positive; at serving time a typical user likes a handful of films
  among ~2,000 candidates. A score of 0.5 therefore means "as likely as a positive in a balanced
  training set", not a 50% chance of liking the film. That is harmless for ranking, which reads
  only order, but it would break any product decision that needs a probability — whether to send
  a recommendation email at all, or to hide a low-confidence row. Supporting those would need
  recalibration against the true base rate and a threshold chosen from a precision-recall curve
  computed under the serving distribution, not the sampled training one. Neither was studied.
