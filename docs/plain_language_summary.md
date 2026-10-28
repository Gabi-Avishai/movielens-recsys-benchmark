# Picking the Next Film

*Recommender systems · a research project*

When a streaming service suggests ten films, how should it choose them? This project built
thirteen different ways of doing it, from "show what's popular" to a neural network, and tested
them all on exactly the same problem — to learn not just which works best, but *why*.

## How it was tested

A million real ratings from about 6,000 people and 3,900 films. Every approach faced the same
exam, under the same rules.

- **Predict the future.** Each approach learned from a person's earlier ratings and was scored
  on the films they went on to watch and like later.
- **Change one thing at a time.** Approaches were built so each differed from the next in a
  single respect, so any difference in results has a known cause.
- **Rule out luck.** Key results were repeated several times. Two early conclusions were
  overturned this way, and the corrections are kept on record.

## The results, in two questions

Being accurate is not enough. A service that recommends the same fifty blockbusters to everyone
can look accurate while being useless, so every approach was judged on both.

| Approach | People who got at least one film they went on to like, in their top 10 | Share of the film catalogue that got recommended to anyone |
|---|---|---|
| **Recommend what's popular** — the simplest possible rule | 33% | 3% |
| **"People who liked this also liked…"** — counting films watched together | 40% | 12% |
| **A learned taste model** — factorization machine | 47% | 20% |
| **Two-part neural model** — the best approach tested | 56% | 45% |

The best approach reached more people *and* showed thirteen times more of the catalogue than
recommending what's popular.

## What was learned

### Being seen predicts liking better than being good

Ranking films by their average rating did *worse* than simply ranking by how many people watched
them. People mostly rate films they chose to watch, so a film's reach says more about whether
someone will like it than its score does.

### More information made things worse

Giving the models extra detail — each film's genre and year, each person's age and job — made
one of the best models measurably less accurate. What a person has already watched turned out
to be the only information that consistently helped.

### The best design won for a structural reason

The winning model has two halves: one learns to describe people, the other to describe films. It
beat the next best by 43%. Suspecting it had simply been trained more favourably, the project
tested its training advantages one by one — and they explained almost none of the gap. The
difference is in how it is built.

### Two good models that disagree can't simply be merged

The two strongest models agreed on fewer than three of every ten films they recommended, which
looked like a chance to combine them. Every attempt made results worse. The reason: one is
better at blockbusters, the other at lesser-known films, and there is no way to know in advance
which kind of film a particular person wants next.

### Changing tastes are the hardest case — especially for the best model

Every approach learns from the past, so all of them struggle when someone's taste shifts. The
best model struggles most, and about three in ten people received nothing useful from any
approach.

| Earlier ratings | What they liked next |
|---|---|
| Mostly horror films | Almost entirely drama |

For this person every model failed — the best one kept suggesting horror classics, faithfully
following a taste the person had left behind.

## What this does not show

- Everything was tested on past data, not with real people using a live service, and results
  offline don't always carry over.
- The ratings come from around 2000, and 3,900 films is small next to a modern catalogue. Some
  advantages of the winning design only appear at a much larger scale.
- Promising next steps — giving recent viewing more weight, and treating blockbusters and
  lesser-known films differently — were identified but not built.

---

Based on the MovieLens-1M dataset. Figures come from a single, fixed testing procedure applied
identically to every approach; a [technical write-up](technical_summary.md) covers the methods
and full results.
