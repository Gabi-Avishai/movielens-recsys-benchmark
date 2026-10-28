# What Each Assumption Buys

Thirteen recommender models on one dataset, one split, one evaluation protocol — built so that
each pair differs by exactly one modelling assumption, and the gap between them is the
measurement.

`MovieLens-1M · 1,000,209 ratings · 6,040 users · 3,883 items · 4.26% dense · phases 0–9 complete`

## The problem

*Not "which model wins" — that produces a leaderboard nobody learns from.*

The dataset stands in for the **candidate generation and ranking stage of a catalogue
recommender**: a user arrives, the system has ~3.9k items and that user's history, and it must
return an ordered short list. Every model is measured on that job under identical conditions,
so any difference is attributable to the assumption that changed and not to preprocessing,
splitting, or tuning effort.

> The design constraint throughout: **models that differ in one respect share everything
> else**, including the training loop. Logistic regression, matrix factorization and the
> factorization machine are one implementation under three configuration flags; four loss
> functions share one gradient path.

## What it adds up to

- **The two-tower is the best model, by 43%** over the best factorization machine, at 2.2× the
  catalogue coverage. Negative-sample count explains ~10% of that lead and the loss function
  none of it — the lead is architectural.
- **Interaction history is the single most valuable input** across three model families, while
  the item content that was expected to help (genre, year) made the FM measurably worse.
- **Two models that disagree on 73% of their picks could not be combined.** They are
  complementary by item popularity, an axis no combiner can see in advance.
- **The best model is the most fragile.** Taste drift hurts it nearly twice as much as the
  others, and 29.4% of users get nothing useful from any model.

## Standings

*Test split, candidates filtered to items with ≥50 training ratings, k=10. Rows with ± are
means over 3–5 seeds.*

| Model | NDCG@10 | Recall@10 | Coverage | Gini | Phase |
|---|---|---|---|---|---|
| **Two-tower, sampled negatives** | **0.1653 ±.0025** | **0.1282** | **0.453** | **0.856** | 6 |
| Two-tower, disliked-as-negative | 0.1552 ±.0016 | 0.1226 | 0.461 | 0.854 | 6 |
| FM — ids + history, logistic | 0.1159 ±.0015 | 0.0895 | 0.204 | 0.956 | 4 |
| FM — ids + history, softmax loss | 0.1143 ±.0001 | 0.0900 | 0.179 | 0.958 | 8 |
| Two-tower, in-batch negatives | 0.1128 ±.0030 | 0.0995 | 0.492 | 0.813 | 6 |
| FM — all 7 feature blocks | 0.0990 ±.0019 | 0.0779 | 0.193 | 0.958 | 4 |
| *Item-item cosine — the floor* | *0.0948* | *0.0675* | *0.118* | *0.983* | *2* |
| Matrix factorization (ids only) | 0.0922 ±.0005 | 0.0603 | 0.159 | 0.980 | 4 |
| LightGBM (best of 6 configs) | 0.0862 ±.0004 | 0.0651 | 0.120 | 0.979 | 5 |
| Item-item co-occurrence | 0.0836 | 0.0555 | 0.040 | 0.993 | 2 |
| Poly2 | 0.0761 | 0.0506 | 0.035 | 0.994 | 3 |
| Raw popularity | 0.0737 | 0.0475 | 0.034 | 0.994 | 2 |
| Damped mean rating (tuned) | 0.0723 | 0.0482 | 0.033 | 0.995 | 2 |
| Logistic regression | 0.0720 | 0.0466 | 0.036 | 0.994 | 3 |
| FM — rating regression | 0.0524 ±.0015 | 0.0340 | 0.206 | 0.969 | 8 |
| Content kNN (genre + year) | 0.0220 | 0.0216 | 0.520 | 0.848 | 2 |

## The metrics, and why each is there

*Accuracy alone can be won by serving the same fifty films to everyone.*

- **NDCG@10.** Position-weighted hit rate in the top ten. The product is a ranked list, so the
  metric has to care where in the list a hit lands — not just whether it appears.
- **Recall@K.** Share of a user's relevant items that reach the top K. The headline for
  retrieval models at K=100/500, because a retrieval stage's job is getting candidates into a
  shortlist, not ordering them.
- **Recall ceiling.** Users average 13.3 relevant test items against k=10, so Recall@10 is
  arithmetically capped near **0.787** — not 1.0. Reported beside recall because a "low"
  recall is often just arithmetic. It varies by segment, and ignoring that inverts conclusions.
- **Coverage.** Fraction of the 3,883-item catalogue that appears in anyone's top-10. A model
  serving 3% of the catalogue is a failed product however good its NDCG.
- **Gini.** Concentration of the served-item distribution. 0 = every item served equally,
  1 = one item takes everything. Detects accuracy bought purely by leaning harder on the head.
- **Popularity percentile.** Average popularity rank of served items. 1.0 means the model
  always serves the single most popular item.
- **Why not RMSE.** Rating regression can only train where a rating exists — on items the user
  *chose* to watch — then must rank a catalogue of items they didn't. Measured in Phase 8: the
  model with the best RMSE ranks **below raw popularity**.

## Protocol decisions that matter

- **Chronological split per user (70/15/15).** A random split lets a model train on a user's
  future. Positive rate falls 64%→50% across a user's own sequence, so time carries real signal.
- **Split boundaries snapped off tied timestamps.** 52.9% of rows share a timestamp with a
  sibling; a naive cut divides a rating burst for ~half of all users. Snapping cut 6,085 such
  crossings to 128.
- **Full candidate set, no sampled shortcuts.** Every model ranks the whole pool.
  Sampled-candidate evaluation inflates metrics unevenly across model families.
- **No decision threshold — scores are only ranked.** No model turns a score into a yes/no;
  each user gets the top 10, so the operating point is the list length K=10, set by the
  product. Ranking metrics read only the order of scores, so there is no threshold to tune and
  no precision-recall curve to draw. Recall@K across K is the ranking analogue of sweeping one.
- **Both candidate settings reported.** The ≥50-rating filter removes 44% of the catalogue to
  gain 4% of interactions — nearly free on accuracy, ruinous for coverage, and it penalises
  content models specifically (−27.5% vs −0.4%).
- **Negatives never include an item the user interacted with — for every loss.** An unfiltered
  draw contaminated 9.4% of listwise negatives and cost the softmax 14%.
- **Popularity counted on training data only**, and **≥3 seeds with standard errors from
  Phase 4 on**. Two conclusions changed significance once seeds were added.

## The phases

Each asks one question.

### 0–1. What does the data actually permit us to claim?

No genuinely cold users exist (min 20 ratings), 6.1% of the catalogue has zero training
interactions, and **17.2% of age×genre cells deviate beyond three standard errors from an
additive model** — second-order signal is real, measured before any model was fitted.

The harness freezes loading, splitting, causal history and candidate rules in one module no
model may bypass. Eighteen structural tests assert the invariants — most importantly that a
row's history contains exactly the items strictly before it.

**Why test for interaction signal first?** An FM's central claim is that the effect of one
feature depends on another. That's an empirical claim about this dataset, testable by comparing
each age×genre cell against an additive prediction and its own binomial standard error — no
model required.

### 2. How good is a model with no parameters at all?

Better than expected. Normalised item-item co-occurrence reaches **0.0948** — that, not
popularity's 0.0737, is the floor everything later must clear. A tuned damped mean rating
*loses* to raw popularity.

Six zero-parameter models, hyperparameters tuned on a validation split that never touches test.
The damped mean was swept across five orders of magnitude of shrinkage and still lost —
exposure predicts a test positive better than quality does, which follows from the selection
bias measured in Phase 1.

**Why this matters:** quoting gains over 0.0737 rather than 0.0948 would overstate every
learned model by about 29%.

### 3. What does each part of a factorization machine contribute?

Interaction term **+32.9%**. Parameter sharing **+22.1%**. Logistic regression on all 13,882
features scores **below raw popularity**.

One training loop, four configurations: `order=1` is logistic regression, `blocks=ids` is
matrix factorization, both on is the FM. Poly2 gives each (user, item) pair its own free weight
instead of factorising.

**Why Poly2 fails, measured not assumed:** of the 1.63M pairs scored at test time, **98.3% have
no weight at all** — a candidate is by definition something the user never interacted with. Of
the 1.7% that do fire, 100% came from negative sampling. It collapses to ranking by item bias
(τ = 0.999).

### 4. Which features actually earn their place?

Almost none. **Interaction history alone beats all seven feature blocks together** (0.1159 vs
0.0990). Genre is *actively harmful* by two independent tests.

Blocks added and removed one at a time, three seeds each, at a tuned `k=64`. Both directions
agree: adding genre costs 0.0157, removing it gains 0.0071.

**This phase retracted a Phase 3 claim.** Seed variance for the feature-rich FM is four times
the MF's, so the "+2.9% from side features" reported from single runs was p=0.085, not a result.

**Why genre hurts:** an item's genre is already recoverable from its id, and the id factor is
fitted from that item's own interactions rather than pooled across the 1,603 films tagged Drama.

### 5. Do gradient-boosted trees, which find interactions by splitting, do better?

No configuration reaches the floor. And the two families want **opposite feature sets** — the
signs reverse on both block additions.

**The symmetry:** sparse high-cardinality history is worth +25.7% to the FM and +0.8% to the
tree; dense low-cardinality content is worth −14.6% to the FM and **+14.8%** to the tree.

**Serving cost runs the same way:** the FM scores a candidate in O(k) because its user and item
blocks are disjoint; a tree needs a materialised row per pair. 6–12× more expensive *and* less
accurate.

### 6. What does decoupling the user and item towers cost?

Nothing — it **wins by 43%** (0.1653 vs 0.1159) at 2.2× the coverage. The prediction going in
was that it would lose. Only **10%** of the gap is explained by its larger negative sample.

**Why the prediction was wrong:** each tower is an MLP, so it crosses features freely *within*
each side and gives up crossing only *across* the user/item boundary. The FM crosses the
boundary but is linear in its factors.

**A long-standing question settled:** uniform-unobserved negatives beat unobserved-or-disliked
(0.1653 vs 0.1552) — an item rated 1–3 is one the user *chose* to watch, so training against it
suppresses exactly what should be retrieved.

**Scale doesn't explain it.** Subsampling users 10×, the advantage is flat on Recall@100 and
*shrinks* on NDCG.

### 7. Two models disagree on 73% of their picks. Can that be exploited?

An oracle over their pooled lists shows **+98.3% headroom**. Every realizable combination
*lost* to the better model alone. **Complementary outputs are not combinable outputs.**

**Then everything lost.** Score blending −1.1%; a LightGBM reranker −4.3%; the same reranker
given ids −10.5%. Even choosing the blend weight *on test* caps out at +1.5%.

**Why the bound misled.** It answers "does the union *contain* better items?" Exploiting it
requires knowing *which* ones — the ranking problem restated. An oracle gap is a **necessary**
condition for combination to help, never a sufficient one.

The reranker took **78.7% of its split gain from user and item ids**, memorising validation
rather than combining scores. *Phase 7 runs are single-seed.*

### 8. Is the tower's lead really about the loss function?

No. Softmax and logistic are **indistinguishable** on the FM (p = 0.33), closing none of the
gap. And **the best model on RMSE is the worst ranker.**

Four losses on one FM backbone, sharing one gradient path; BPR falls out of the algebra as the
softmax with a single negative. Pushing the softmax from 1 to 32 negatives peaks at 4 and then
*falls*, while coverage collapses 0.204 → 0.117.

**Listwise losses are fragile to false negatives.** An unfiltered uniform draw makes 9.4% of
negatives real ratings and alone costs the softmax 14% — it concentrates gradient on the
highest-scoring negatives, which is exactly where a mislabelled positive sits.

**The table with holes.** Only regression can produce an RMSE; BPR and softmax learn an order
with no scale. Regression beats the mean baseline by 18% on RMSE and ranks **below raw
popularity** — Phase 1's selection bias, visible as a 55% ranking deficit.

**Threshold.** Training on ≥3 and ≥4 is indistinguishable, with ≥3 reaching more of the
catalogue; ≥5 loses 10.6%.

### 9. Where, exactly, do the models fail?

**29.4% of users get nothing from any model.** Taste drift hurts the best model *nearly twice*
as much as the others. The tower owns the long tail; on blockbusters the order flips.

**Confounds first.** Every descriptor is tangled with how many relevant items a user has, so
effects come from a multivariate model. Raw taste drift correlates with test-set size at
ρ = −0.533 purely through sampling noise; subtracting each user's expected divergence brings it
to +0.058.

**What predicts failure.** Excess drift is the tower's largest effect after target count
(−0.032 per sd, vs −0.017 for the FM); its NDCG falls 38% from the most stable fifth of users to
the most drifted. Activity *reverses* once targets are held fixed, and the apparent penalty for
mainstream taste was confounding.

**The mechanism behind Phase 7.** On the least popular half of the catalogue the tower
retrieves 3.6% of relevant items, the FM 0.5%, cosine 0.0%; on the top decile it is cosine
43.3%, FM 35.4%, tower 25.9%. Items only the tower finds have a median of 703 training ratings;
items only the FM finds, 1,305.

## Nine results that weren't the expected ones

- A tuned **damped mean rating loses to raw popularity**. Exposure predicts a test positive
  better than merit.
- **Logistic regression on 13,882 features also loses to popularity.** What's learnable lives
  almost entirely in the interactions.
- **Refusing to share parameters costs 22.1%** — nearly as much as the interaction term itself.
- **The best FM drops four of its seven feature blocks.** More features made it measurably worse.
- **No gradient-boosted configuration reaches the memory-based floor**, and it is 6–12× costlier
  to serve.
- **The two-tower beats the FM by 43%**, inverting the expectation that decoupled scoring costs
  accuracy.
- A **+98.3% oracle headroom yielded no usable gain.**
- **The best model on RMSE is the worst ranker**, below even raw popularity.
- **The best model is the most fragile** when a user's taste changes.

## Limits, stated as such

- **Everything here is offline.** Offline NDCG is a proxy for engagement and a known imperfect
  one.
- **Scores are not calibrated probabilities.** The FM trains on one sampled negative per
  positive, so it learns in a world where half of all pairs are positive; at serving time a user
  likes a handful of films among ~2,000 candidates. A score of 0.5 is not a 50% chance of
  liking. Harmless for ranking, but any yes/no product decision — whether to send a
  recommendation email at all — would need recalibration to the true base rate and a threshold
  chosen under the serving distribution. Not studied.
- **The latency case for retrieval cannot be shown at this scale.** With 3,883 items an ANN
  index is pointless and both models score in O(k); the tower's structural advantage needs a
  catalogue north of 10⁵ items.
- **The tower's lead is eliminated to architecture as a group, not separated.** The clean
  single-factor test — the tower with its MLPs removed — was not run.
- **"Cold start" here means relatively light.** The dataset was filtered to users with ≥20
  ratings; the coldest user has 10 training interactions. A dedicated cold-start phase and a
  serving phase were dropped for that reason and because the serving infrastructure already
  exists in earlier projects.
- **Untested directions** the error analysis points to: recency-weighted history for drift, a
  popularity-conditioned combination, and tail exposure as an explicit objective.

---

Every figure is computed under one frozen protocol; hyperparameters are selected on a
validation target that never touches test. Nine notebooks, 18 protocol tests, per-phase result
tables. Failed predictions are recorded where they happened rather than rewritten.
