# Seven Ways to Rank a Film

`MovieLens-1M · Phase 2 baselines`

Seven recommenders with no learned parameters, each scoring the same film for the same person.
They disagree by more than a thousand ranking positions. Here is exactly what each one
computes, and why they end up so far apart.

## The worked example — user 76

Seven films liked during training. Six of the seven are comedies.

| Film | Ratings | Average | Genres |
|---|---|---|---|
| Election (1999) | 1,248 | 3.94 | Comedy |
| Austin Powers: The Spy Who Shagged Me (1999) | 1,063 | 3.41 | Comedy |
| Dick (1999) | 320 | 3.36 | Comedy |
| American Beauty (1999) | 3,024 | 4.33 | Comedy\|Drama |
| Being John Malkovich (1999) | 1,877 | 4.15 | Comedy |
| Gladiator (2000) | 1,396 | 4.16 | Action\|Drama |
| Chicken Run (2000) | 925 | 3.98 | Animation\|Children's\|Comedy |

**The candidate being scored:** *Star Wars: Episode V — The Empire Strikes Back* (1980). 2,574
ratings, ratings summing to 11,082, average 4.305, genres Action | Adventure | Drama | Sci-Fi |
War. The catalogue's global average rating is 3.634.

---

## 0a. Raw popularity (`pop_count`)

> *"What is everyone watching?"*

Count how many people rated each film and rank by that. There is no user term at all — every
single person receives an identical list. It counts *every* rating, including one-star ones,
because the thing being measured is exposure: did this film cross your path?

| | |
|---|---|
| ratings for Empire Strikes Back | 2,574 |
| **score** | **2,574** |

## 0b. Damped average rating (`damped_mean`)

> *"What is genuinely good — and how sure are we?"*

Rank by average rating, but distrust films with few ratings. Imagine every film starts with
**C** invented ratings sitting exactly at the catalogue average, then add its real ones. A film
with thousands of ratings barely notices the invented ones; a film with three is dragged almost
all the way back to average.

| | |
|---|---|
| C × global average | 25 × 3.634 = 90.9 |
| + sum of real ratings | 11,082 |
| ÷ (C + number of ratings) | 25 + 2,574 = 2,599 |
| **score** | **4.299** |

**C is a dial, not a constant.** At C = 0 this is the raw average, and the winner is an obscure
film with a single five-star rating. Push C up and it slides toward ranking by
`count × (average − 3.634)`. Swept properly, bigger was always better — and even at its best it
still **loses to plain popularity**. Being well-liked predicts less than being seen.

## 0c. Popularity within your genres (`genre_popularity`)

> *"What is popular in the corner of the catalogue this person lives in?"*

Build a histogram of the genres the user actually watches, then score each film by how well it
matches that histogram, multiplied by how popular it is. Personalised, but nothing is learned —
it is a tally and a multiplication.

| | |
|---|---|
| user 76's genre mix | Comedy .545 · Drama .182 · Action .091 |
| match with Empire's genres | 0.122 |
| × log(1 + 2,574) | 7.854 |
| **score** | **0.958** |

The match is only **0.122** — a comedy watcher against a war/sci-fi film. Popularity is doing
nearly all the work, and it still is not enough to rescue the rank.

## 0d. Raw co-occurrence (`item_item_cooc`)

> *"How often does this film turn up alongside what they already like?"*

For each film the user liked, count the people who liked *both* that film and the candidate.
Add the counts up. No division anywhere — and that omission is the whole lesson.

| Liked film | Co-likes with Empire |
|---|---|
| Election | 408 |
| Austin Powers | 225 |
| Dick | 59 |
| American Beauty | 984 |
| Being John Malkovich | 633 |
| Gladiator | 560 |
| Chicken Run | 304 |
| **score** | **3,173** |

**American Beauty alone supplies 31% of the score.** Not because it is especially predictive of
Star Wars, but because 2,527 people liked it and a popular film overlaps with everything. This
model is largely popularity wearing a disguise — which is precisely why it is on the ladder as
its own rung.

## 1i. Cosine between films (`item_item_cosine`)

> *"Is this overlap surprising, or just two popular films colliding?"*

The same overlaps as above, each divided by the square root of how popular the two films are.
An overlap of 400 is remarkable between two small films and unremarkable between two
blockbusters. Dividing is the only change, and it is the change that matters.

| Liked film | Calculation | Similarity |
|---|---|---|
| Election | 408 ÷ √(932 × 2,165) | 0.287 |
| Austin Powers | 225 ÷ √(540 × 2,165) | 0.208 |
| Dick | 59 ÷ √(152 × 2,165) | 0.103 |
| American Beauty | 984 ÷ √(2,527 × 2,165) | 0.421 |
| Being John Malkovich | 633 ÷ √(1,493 × 2,165) | 0.352 |
| Gladiator | 560 ÷ √(1,119 × 2,165) | 0.360 |
| Chicken Run | 304 ÷ √(700 × 2,165) | 0.247 |
| **score** | | **1.978** |

American Beauty falls from **31% of the score to 21%**, and Gladiator — half its size, but an
actual action film — nearly catches it. That single division buys **13% more accuracy and three
times the catalogue coverage**. Each film also only votes for its 500 nearest neighbours, so
thousands of faint similarities cannot pile up into noise.

## 1u. Cosine between people (`user_user_cosine`)

> *"Did people with this taste like it?"*

The same arithmetic rotated ninety degrees. Find the 100 people whose likes overlap most with
this user — again normalised, so overlapping with a very active user counts for less — then add
up the similarity of every neighbour who liked the candidate.

| | |
|---|---|
| neighbours kept | 100 |
| … of whom liked Empire | 6 |
| user 4038 — 5 shared likes of their 28 | 0.357 |
| user 3331 — 5 shared likes of their 32 | 0.334 |
| user 2192 — 3 shared likes of their 12 | 0.327 |
| … and three more | 0.887 |
| **score** | **1.905** |

Only **6 of 100** similar people liked it. That is a much more sceptical verdict than
popularity's, and it comes from asking a small, relevant crowd instead of the whole audience.

## 2. Content similarity (`content_knn`)

> *"Does this simply look like the films they like?"*

Forget the crowd entirely. Describe each film by its genres and its decade, describe the user as
the average of everything they liked, and measure the angle between the two. Rare genres count
for more than common ones — matching on War (44 films in the catalogue) is real evidence,
matching on Drama (1,603 films) is almost none.

| | |
|---|---|
| user 76 looks like | 1998–2000 .785 · Comedy .572 |
| Empire looks like | War .530 · Sci-Fi .449 |
| rarity weights | War 4.30 · Sci-Fi 3.64 · Comedy 2.17 |
| **score — the two nearly perpendicular** | **0.078** |

The weakest scorer on the whole ladder, and kept anyway: it is the **only** model here that can
score a film nobody has rated yet. 237 films — 6.1% of the catalogue — are completely invisible
to every other model on this page.

---

## The same film, seven verdicts

Scores are on incompatible scales, so what matters is the *rank* each model gives Empire Strikes
Back among user 76's 2,135 candidate films.

| Model | Score | Rank |
|---|---|---|
| `pop_count` | 2,574 | 1 |
| `item_item_cooc` | 3,173 | 1 |
| `item_item_cosine` | 1.978 | 18 |
| `damped_mean` | 4.299 | 41 |
| `user_user_cosine` | 1.905 | 51 |
| `genre_popularity` | 0.958 | 873 |
| `content_knn` | 0.078 | 1,262 |

**Two models rank it first. One ranks it 1,262nd. And user 76 did not, in fact, watch *The
Empire Strikes Back* in the test period — they watched four comedies.**

The two models that put it first are the two that cannot tell the difference between *relevant*
and *famous*. They are still hard to beat on aggregate scores, because across six thousand
users, guessing famous is usually right. That is the tension the whole benchmark exists to
measure: a model can win on accuracy while serving the same fifty films to everybody, which is
why catalogue coverage is reported beside every accuracy number rather than buried in an
appendix.

---

Every figure on this page is computed from the MovieLens-1M training split under the project's
frozen evaluation protocol. Formal definitions, including the metric formulas, are in
[`FORMULAS.md`](../FORMULAS.md); each was verified against the shipped code by independent
reimplementation.
