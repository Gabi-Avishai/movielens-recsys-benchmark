# Formulas: the non-learned baselines

Exact definitions for every model in the Phase 2 ladder, stated so that each can be
reimplemented from this page alone. Each formula has been checked against the shipped
code by independent reimplementation (see [Verification](#verification)).

## Notation

| Symbol | Meaning |
|---|---|
| $\mathcal{U}, \mathcal{I}$ | users ($6{,}040$) and catalogue items ($3{,}883$) |
| $\mathcal{D}$ | the **training** interactions $(u, i, r)$, $r \in \{1,\dots,5\}$ |
| $n_i$ | number of training ratings of item $i$, **any rating value** |
| $s_i$ | sum of training ratings of item $i$ |
| $\bar r_i = s_i / n_i$ | mean training rating of item $i$ |
| $m = \sum_i s_i \big/ \sum_i n_i$ | global mean training rating ($3.635$) |
| $R \in \{0,1\}^{\|\mathcal{U}\| \times \|\mathcal{I}\|}$ | $R_{ui} = 1$ iff $u$ rated $i$ **at least 4** in training |
| $\mathcal{U}_i = \{u : R_{ui}=1\}$ | users who liked item $i$ |
| $\mathcal{I}_u = \{i : R_{ui}=1\}$ | items user $u$ liked |
| $G \in \{0,1\}^{\|\mathcal{I}\| \times 18}$ | genre flags; $g_i$ is row $i$ |
| $Y \in \{0,1\}^{\|\mathcal{I}\| \times 10}$ | release-year bin, one-hot |
| $\hat s(u,i)$ | the score; only its **order** is ever used |

### Two different training signals, on purpose

The popularity models use **every** training rating; the neighbourhood and content models
use **only positives** ($R$). This is deliberate, not an oversight:

- Popularity is a proxy for **exposure** — how likely the user was to encounter the item at
  all. A 1-star rating is still evidence that someone watched it, so it counts.
- A neighbourhood model propagates **approval** from one item to another. A 1-star rating is
  not something to spread, so it is excluded.

Phase 1 established why this distinction matters: a test positive requires the user to have
watched the film, and exposure predicts that better than merit does.

---

## 0a · `pop_count` — raw popularity

$$\hat s(u,i) = n_i$$

No user term at all: every user receives the same ranking. Counts all training ratings, so
it measures exposure rather than approval.

## 0b · `damped_mean` — Bayesian-shrunk mean rating

$$\hat s(u,i) = \frac{C\,m + s_i}{C + n_i}, \qquad \hat s(u,i) = 0 \text{ if } n_i = 0$$

The posterior mean of item $i$'s rating under a prior worth $C$ pseudo-ratings at the global
mean $m$. An item needs roughly $C$ ratings of its own before its evidence outweighs the
prior. Items with no training evidence are scored $0$ rather than $m$, so they are never
recommended.

**What it converges to.** Expanding for large $C$,

$$\frac{C m + s_i}{C + n_i} = m + \frac{n_i(\bar r_i - m)}{C} + O(C^{-2}),$$

so the *ordering* tends to $n_i(\bar r_i - m)$ — popularity weighted by **signed** quality
margin, not popularity. Measured Kendall $\tau$ against $n_i(\bar r_i - m)$ at $C = 10^5$ is
$+1.000$; against raw $n_i$ it is $-0.042$, because below-average items are driven to the
bottom. The two nearly coincide at the head (37 of the top 50), which is why NDCG approaches
`pop_count` without overtaking it.

**Selected** $C = 10^5$ on validation, the largest value swept; NDCG rises monotonically in
$C$ across five orders of magnitude.

## 0c · `genre_popularity` — personalised, unlearned

User taste is an $L_1$-normalised histogram of the genres of the items they liked:

$$a_u = \frac{(RG)_u}{\max\!\big(\textstyle\sum_g (RG)_{ug},\; 1\big)}$$

Note the denominator sums *genre occurrences*, not liked items: an item carrying three
genres contributes three. Items are compared by unit-norm genre vector $\hat g_i = g_i /
\lVert g_i \rVert_2$, and the affinity is multiplied by a popularity prior:

$$\hat s(u,i) = \big(a_u \cdot \hat g_i\big) \cdot \log(1 + n_i)^{w}, \qquad w = 1$$

Zero parameters: a per-user histogram times a global prior, no fitting of any kind.

## 0d · `item_item_cooc` — raw co-occurrence counts

$$C_{ij} = |\mathcal{U}_i \cap \mathcal{U}_j| = (R^{\top}R)_{ij}, \qquad C_{ii} := 0$$

$$\hat s(u,i) = \sum_{j \in \mathcal{I}_u} C_{ji}$$

Deliberately unnormalised. A popular item co-occurs with everything, so this is largely
popularity in disguise — which is the point of including it as a separate rung.

## 1i · `item_item_cosine` — normalised item neighbourhood

$$S_{ij} = \cos(i,j) = \frac{|\mathcal{U}_i \cap \mathcal{U}_j|}{\sqrt{|\mathcal{U}_i|\,|\mathcal{U}_j|}}, \qquad S_{ii} := 0$$

Each row is then truncated to its strongest neighbours. Let $\tau_i$ be the $k$-th largest
value in row $i$:

$$S_{ij} := 0 \quad \text{whenever } S_{ij} < \tau_i$$

The comparison is strict, so ties at the threshold are **kept** and a row may retain more
than $k$ entries. Truncation is applied per row, which makes $S$ asymmetric.

$$\hat s(u,i) = \sum_{j \in \mathcal{I}_u} S_{ji}$$

Note the index order: the sum reads **column** $i$ of $S$, so an item $j$ the user liked
contributes to $i$ only when $i$ is among $j$'s own top-$k$ neighbours.

An optional shrinkage factor (disabled by default, $\lambda = 0$) damps similarities
supported by few co-raters:

$$S_{ij} \leftarrow S_{ij}\cdot\frac{c_{ij}}{c_{ij} + \lambda}, \qquad c_{ij} = (R^{\top}R)_{ij}$$

**Selected** $k = 500$ on validation. The sweep is nearly flat — the whole range from $k=10$
to unrestricted spans $0.006$ NDCG — while coverage falls from $0.165$ at $k=10$ to $0.118$
at $k=500$.

## 1u · `user_user_cosine` — the transpose view

$$S^{\mathcal{U}}_{uv} = \frac{|\mathcal{I}_u \cap \mathcal{I}_v|}{\sqrt{|\mathcal{I}_u|\,|\mathcal{I}_v|}}, \qquad S^{\mathcal{U}}_{uu} := 0$$

with the same per-row top-$k$ truncation, then

$$\hat s(u,i) = \sum_{v \in \mathcal{U}} S^{\mathcal{U}}_{uv} R_{vi}$$

**Selected** $k = 100$ on validation.

## 2 · `content_knn` — content cosine, no interaction signal

Build the item content matrix and weight each column by inverse document frequency, so that
a rare genre (Film-Noir, 44 items) counts for more than one carried by half the catalogue
(Drama, 1,603 items). With $d_f$ the number of items having feature $f$:

$$F^{(0)} = [\,G \mid Y\,] \in \{0,1\}^{|\mathcal{I}| \times 28}, \qquad
w_f = \ln\!\frac{|\mathcal{I}|}{\max(d_f, 1)} + 1, \qquad
F^{(1)} = F^{(0)}\operatorname{diag}(w)$$

Rows are $L_2$-normalised to give the item vector $f_i$, and a user's profile is the
normalised sum of the items they liked:

$$f_i = \frac{F^{(1)}_i}{\lVert F^{(1)}_i \rVert_2}, \qquad
p_u = \frac{\sum_{i \in \mathcal{I}_u} f_i}{\bigl\lVert \sum_{i \in \mathcal{I}_u} f_i \bigr\rVert_2}$$

$$\hat s(u,i) = p_u \cdot f_i = \cos(p_u, f_i)$$

Because both vectors are unit-norm the score *is* the cosine. Normalising the sum is
equivalent to normalising the mean — the factor $1/|\mathcal{I}_u|$ cancels — so "profile =
mean content vector" and this definition give identical rankings.

This is the only rung that can score an item with no interaction history, which is why it is
on the ladder despite the weakest accuracy: 237 items (6.1% of the catalogue) have zero
training interactions and are invisible to every model above.

---

## Candidate masking

The four precomputed models set $\hat s(u,i) = -\infty$ wherever $R_{ui} = 1$, so a model
never re-recommends an item the user already liked in training.

This is **redundant** on the standard evaluation path: `Dataset.candidates()` already
excludes the user's entire train(+val) history regardless of rating, which is a strictly
larger exclusion set. It is kept because it makes the models correct when scored outside
that path. `pop_count` and `damped_mean` do not mask, relying on the candidate set alone.

---

## Verification

Each formula above was reimplemented directly from this page and compared against the
shipped model's score matrix on 4,000 random $(u,i)$ pairs:

| Model | Agreement | Max abs. difference |
|---|---|---|
| `pop_count` | exact | $0$ |
| `damped_mean` | exact | $0$ |
| `item_item_cooc` | exact | $0$ |
| `item_item_cosine` | to tolerance | $4.2\times10^{-5}$ |
| `user_user_cosine` | see below | $4.9\times10^{-1}$ |
| `content_knn` | to tolerance | $1.9\times10^{-7}$ |
| `genre_popularity` | to tolerance | $2.0\times10^{-7}$ |

**The `user_user_cosine` discrepancy is a precision artefact, not a formula error.** The
similarity matrices agree before truncation. Top-$k$ selection is a *discontinuous*
operation: the shipped code computes in `float32`, where the 100th-largest similarity in row
0 is $0.21437323$ against `float64`'s $0.21437325$. That eighth-decimal difference flips
which neighbour survives the cut for **324 of 6,040 rows (5.4%)**.

It does not move the results. Evaluated end to end, `float32` and `float64` give
NDCG@10 $= 0.09322$ for both, with recall differing by $2\times10^{-5}$. The sensitivity is
real at the neighbour-set level and immaterial at the metric level — worth knowing, because
it means exact reproduction of the *neighbour lists* requires matching the dtype, while
reproduction of the *reported numbers* does not.

---

## Appendix: metric definitions

For a user $u$ with ranked recommendations $\hat{\mathcal{R}}_u$ (best first, restricted to
that user's candidate pool) and ground-truth set $T_u$, let $h_p = \mathbb{1}[\hat{\mathcal{R}}_u[p] \in T_u]$.

$$\text{Precision@}k = \frac{1}{k}\sum_{p=1}^{k} h_p
\qquad
\text{Recall@}k = \frac{1}{|T_u|}\sum_{p=1}^{k} h_p
\qquad
\text{HitRate@}k = \mathbb{1}\Big[\textstyle\sum_{p \le k} h_p > 0\Big]$$

$$\text{NDCG@}k = \frac{\sum_{p=1}^{k} h_p / \log_2(p+1)}{\sum_{p=1}^{\min(k,\,|T_u|)} 1 / \log_2(p+1)}
\qquad
\text{AP@}k = \frac{1}{\min(|T_u|, k)}\sum_{p=1}^{k} h_p \cdot \frac{1}{p}\sum_{q \le p} h_q$$

The ideal DCG is capped at $\min(k, |T_u|)$, so a user with three relevant items can still
score $1.0$ at $k=10$. Without the cap the mean would measure test-set size rather than
ranking quality.

All are averaged over users with $|T_u| > 0$; users with no reachable ground truth are
**skipped and counted**, never scored as zero.

**Recall ceiling.** With $T^{\text{reach}}_u$ the ground truth actually present in the
candidate pool:

$$\text{ceiling@}k = \frac{1}{|\mathcal{U}^{*}|}\sum_{u} \frac{\min(k,\,|T^{\text{reach}}_u|)}{|T_u|}$$

Recall@10 cannot reach $1.0$ here — users average $13.3$ relevant test items against $k=10$ —
so recall is uninterpretable without it. Measured: $0.787$ filtered, $0.804$ unfiltered.

**Beyond-accuracy.** Let $c_i$ be the number of top-$k$ lists containing item $i$, over the
whole catalogue (items never served count as zeros):

$$\text{Coverage} = \frac{1}{|\mathcal{I}|}\sum_i \mathbb{1}[c_i > 0]
\qquad
\text{Gini} = \frac{\sum_{t=1}^{n} (2t - n - 1)\,c_{(t)}}{n \sum_i c_i}$$

with $c_{(t)}$ the counts sorted ascending. Gini is $0$ when every item is served equally
and approaches $1$ when one item takes everything.

**Mean popularity percentile.** With $\text{rank}(i) \in \{0,\dots,|\mathcal{I}|-1\}$ the
ascending rank of item $i$ by training popularity:

$$\text{pop pct} = \frac{\sum_i c_i \cdot \text{rank}(i)/(|\mathcal{I}|-1)}{\sum_i c_i}$$

$1.0$ means the single most popular item is served every time; $0.5$ means the median item is
served on average.
