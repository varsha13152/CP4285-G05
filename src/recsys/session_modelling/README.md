# Session Modelling (Stage 2 — short-term)

Models **what the user is doing right now** — their in-session momentum — so
Stage 3 can blend the immediate next-item signal with the long-term taste vector
from `u_long`. Companion to [`user_modelling`](../user_modelling/README.md) (the
long-term `u_long` ALS model): this is the short-term signal that sits *under*
`u_long`, never overwriting it.

```
User's timeline ──→ [Sessioniser] ──→ ordered per-session sequences
                                              │
                                              ▼
                       [MarkovSessionModel] ──→ next-item scores
                                              │
                [Stage 3: Candidate Generation] ←──┘ (SESSION lane)
```

## What this implements

A **first-order Markov next-item baseline** over *constructed* sessions. Two pieces:

* **`Sessioniser`** — cuts each user's interaction timeline into ordered
  sessions by thresholding the idle gap between consecutive interactions.
* **`MarkovSessionModel`** — a transition-count table estimating
  `P(next = j | current = i)` from within-session `(i → j)` counts, backing off
  to global popularity for cold items or an empty session.

**How it scores.** It pools every within-session `(i → j)` adjacency into counts,
then normalises **per source item**, so each item's next-item probabilities sum to
1. To recommend, it takes the **last** watched video, looks up that distribution,
and returns the top-`k` by probability. An item that only ever *ended* sessions
(no outgoing transitions), an unseen item, or an empty session **backs off to
global popularity** — so a ranked list always comes back. Because only the last
item is used, cross-session boundary pairs are never counted; the ablation below
shows dropping them changes nothing.

It is the **smallest-sufficient temporal model** — the baseline that GRU4Rec /
SASRec must beat later.

### Why a count model (no new dependency)
This *is* the standard implementation. The session-rec benchmark
(Ludewig & Jannach, 2018) implements its strong baselines — Markov Chain,
Association Rules, Sequential Rules — as exactly this kind of counting, and its
evaluation found these often match or beat neural models. Counting needs only
pandas; adding a heavy framework would clash with this package's lean dependency
set (which pulls in `implicit` only for the ALS `u_long`).

## The three sessionisation strategies

KuaiRec has **no natural sessions**, so we construct them. Empirically (all
12.5M interactions of `big_matrix`) the inter-interaction gap distribution has
**no valley** — density decays monotonically (median gap ~1.8 min; p90 ~19 min;
p95 ~49 min). So a gap threshold is a *convention / hyperparameter*, not a
natural boundary; the right value is the one that maximises downstream next-item
accuracy.

| Strategy | Rule | Rationale |
|----------|------|-----------|
| `fixed` *(default, 30 min)* | one global gap for everyone | Sits above the typical within-session pause, below genuine breaks; matches the standard web-analytics convention. |
| `per_user_pct` | each user's own gap quantile (default p90), floored (15 min) with a global fallback for sparse users | The boundary-relevant statistic — a user's *tail* gap — varies widely (per-user p90 spans ~9–175 min), so a per-user threshold can fit better than one global value. |
| `single` | one session per user (no cutting) | The **ablation** that measures whether sessionisation helps at all. |

## Decisions + rationale

1. **Markov is the baseline** — the smallest-sufficient temporal model
   (last-item Markov assumption). GRU4Rec / SASRec are the future *models to
   beat*, not the starting point.
2. **No new dependency** — the count model *is* the standard (session-rec
   precedent); adding a framework would bloat the lean dep set.
3. **`fixed` 30 min is the default cut** — the gap distribution has **no valley**
   (monotonic decay), so 30 min is a data-backed *convention*, not a natural
   boundary.
4. **Threshold compares against the inter-interaction gap** (idle time *between*
   interactions), never against watch time (`play_duration`).
5. **Metrics: HR@10 (+@20).** With a single true next item, `HR@k = Recall@k` is
   the headline "did we get it" number. MRR / NDCG are computed but not featured
   for now; `precision@k` is dropped (capped at `1/k` with one target — no extra
   signal).

## Usage

`SessionModeller` is the pipeline entry point — it sessionises and fits the Markov
model **on construction**, then exposes what the next stage needs:

```python
from recsys.session_modelling import SessionModeller

sm = SessionModeller(strategy="fixed")             # trains on data/raw/big_matrix.csv
sm.predict_next(current_session_video_ids, k=10)   # SESSION candidates: [(video_id, score), ...]
sm.avg_session_length_for(user_id)                 # real avg session length (a ranking feature)
```

The low-level pieces are usable directly too (the eval script and future GRU4Rec
training input go through these):

```python
from recsys.session_modelling import Sessioniser, MarkovSessionModel

sequences = Sessioniser(strategy="fixed").to_sequences(interactions_df)  # per-session id lists
model = MarkovSessionModel().fit(sequences)
```

## What's exposed & how the next stage uses it

Stage 3 (candidate generation) blends several *lanes*; this module owns the
**SESSION** lane. What it hands over:

| Exposed | Returns | How the next stage uses it |
|---------|---------|----------------------------|
| `SessionModeller.predict_next(recent, k)` | ranked `[(video_id, score), …]` | the SESSION candidate list for the user's current session |
| `SessionModeller.avg_session_length_for(user_id)` | `float` | real `EngagementProfile.avg_session_length` (a ranking feature, not candidates) |
| `RetrievalStrategy.SESSION` + its router weight | enum + weight inside `QueryRetrievalRequest.strategies` | how much to trust the SESSION lane per query intent (high for browsing, low for search) |
| `Sessioniser.to_sequences()` | ordered id lists | training input for GRU4Rec later |

**No ANN for this lane.** `u_long` produces dense vectors, so it needs a MIPS/ANN
index over `item_factors` to search them. The session model is a `dict[item →
ranked next-items]`: candidate generation is a **hash lookup on the last watched
video that is already sorted** — the fitted table *is* the index. Stage 3 just
calls `predict_next(...)`; there is nothing extra to build.

**Not wired end-to-end yet.** No retriever consumes `RetrievalStrategy.SESSION`
— the weight rides in `QueryRetrievalRequest.strategies` awaiting a Stage 3 that
doesn't exist. `UserModeller.build_session_profile` already swaps in the real
`avg_session_length`, but folding the session next-item scores into the profile
as a query-time lens is still deferred.

## Knobs (`Sessioniser.__init__`)

| Param | Default | Notes |
|-------|---------|-------|
| `strategy` | `"fixed"` | `fixed` / `per_user_pct` / `single` |
| `gap_minutes` | 30.0 | global threshold (`fixed`); fallback for sparse users (`per_user_pct`) |
| `pct` | 0.9 | per-user gap quantile used as threshold (`per_user_pct`) |
| `floor_minutes` | 15.0 | lower bound on a per-user threshold, so within-session bursts are never cut |
| `min_interactions` | 20 | users below this fall back to `gap_minutes` (too few gaps for a stable quantile) |
| `cap_minutes` | `None` | optional upper bound on a per-user threshold; `None` disables |

## Eval results

Full data — all **7,176 users / 12.5M interactions**; `chronological_session_split`,
`test_ratio=0.2`. **HR@k** = fraction of test steps where the true next video is in
the model's top-`k` predictions (`k` = 10, 20); MRR / NDCG are computed but not featured.

| Strategy | Sessions | Avg len | HR@10 | HR@20 | Eval pairs | Fit+eval |
|----------|----------|---------|-------|-------|------------|----------|
| `fixed` (30 min) | 887,482 | 14.1 | **0.1056** | **0.1145** | 1,113,382 | 70.5 s |
| `per_user_pct` (p90) | 966,726 | 13.0 | 0.0531 | 0.0634 | 964,261 | 75.4 s |
| `single` (ablation) | 7,176 | 1746.2 | 0.1368 † | 0.1584 † | 2,512,115 | 55.4 s |

*~3.4 min total compute for all three strategies after a one-time CSV load,
reproduced with `python scripts/evaluate_session_baseline.py` (no `--sample-users`).*
**† `single`'s score is not a fair comparison — see the caveat below.**

Protocol: sessionise → split sessions **chronologically** (earlier train, later
test) → fit Markov on train → score next-item over every `(context, target)`
step of the test sessions. No future leakage by construction.

**The `single` caveat (carry forward).** `chronological_session_split` splits
**by session**. For `single` (1 session/user) this degenerates into holding out
**whole users** — a different, easier test set — so its higher HR (0.1368 vs
`fixed` 0.1056) is **not** a fair cross-strategy comparison. `fixed` vs
`per_user_pct` *is* clean under this split (both held out per session), and
**`fixed` wins** (0.1056 vs 0.0531 HR@10; the 2k-user sample showed the same
ordering, 0.098 vs 0.048). For an apples-to-apples comparison of **all three**
strategies on one **shared** test set, see the ablation (`--ablation` /
`split_by_timestamp`) below.

### Ablation — does sessionisation help? (fair, shared test set)

`--ablation` (`split_by_timestamp`) puts all three strategies on **one shared
test set**: a single global `timestamp` cutoff (q=0.8) — train is every
interaction before it, test is the **raw consecutive** next-item pairs after it,
**identical** across strategies. Only the *train* transition table changes with
the sessionisation. Full data — 2,499,221 test pairs across 6,941 users:

| Strategy | Train sessions | HR@10 | HR@20 |
|----------|----------------|-------|-------|
| `fixed` (30 min) | 625,942 | 0.1320 | 0.1406 |
| `per_user_pct` (p90) | 699,457 | 0.1321 | 0.1407 |
| `single` (no cutting) | 7,176 | 0.1320 | 0.1401 |

**Verdict: sessionisation neither helps nor hurts first-order Markov here** — all
three land at HR@10 ≈ 0.132, within 0.0001 of each other. Cutting the timeline
into sessions only drops the cross-boundary `(i → j)` pairs (~7% of all pairs); a
last-item model barely notices, because those boundary pairs carry little
next-item signal. The apparent `fixed` > `per_user_pct` gap under
`chronological_session_split` was a **test-set artifact** of splitting by session,
not a real modelling effect. (These HR values sit above the by-session table
because the shared test set is a different, larger pair population — each table is
internally consistent but the two are not directly comparable.) Practical
takeaway: keep `fixed` 30 min as the default (simplest, standard), and spend the
modelling budget on the sequence models (GRU4Rec / SASRec), not on tuning the cut.

### Reproducing

Full runs over `data/raw/big_matrix.csv` (12.5M interactions, 7,176 users),
single-threaded pandas:

```bash
python scripts/evaluate_session_baseline.py              # main eval (by-session split)
python scripts/evaluate_session_baseline.py --ablation   # fair shared-test-set ablation
```

Runtime (fit + eval **per strategy**; excludes the one-time ~1 GB CSV load of
~0.5–1 min):

| Run | `fixed` | `per_user_pct` | `single` |
|-----|---------|----------------|----------|
| main eval | 70.5 s | 75.4 s | 55.4 s |
| ablation | 89.4 s | 63.6 s | 32.9 s |

≈3–4 min of compute per run after the load. Add `--sample-users N` for a fast
subset (e.g. `--sample-users 2000` ≈ 15–35 s/strategy).

## Deferred

* **Query-conditioned blend** — `build_session_profile` swaps in the real session
  length but does not yet fold the session next-item scores into the profile as a
  query-time lens.
* **Stage 3 consumption** — a retriever that actually reads the SESSION lane.
* **GRU4Rec / SASRec** — the neural sequence *models to beat*.
* **Per-user tuning** — sweeping `gap_minutes` / `pct` against downstream accuracy.
* **MRR / NDCG reporting** — computed already; not featured while HR is the headline.

*Done in Phase 2:* `SessionModeller`, `RetrievalStrategy.SESSION` + router weights,
and the real `avg_session_length` in `build_session_profile`.

## Tests

`tests/test_sessioniser.py` + `tests/test_session_markov.py` cover the three
strategies, boundary rules, sequence extraction, session stats, transition
counting, popularity backoff, and `exclude`. Pure in-memory DataFrames — no disk
I/O.
