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

It is the **smallest-sufficient temporal model** — it conditions on the **last
item only** (the Markov assumption) — the baseline that GRU4Rec / SASRec must
beat later.

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

```python
from recsys.session_modelling import Sessioniser, MarkovSessionModel

sess = Sessioniser(strategy="fixed", gap_minutes=30.0)
sequences = sess.to_sequences(interactions_df)   # ordered video_id lists, one per session
model = MarkovSessionModel().fit(sequences)
model.predict_next([current_video_id], k=10)     # [(video_id, score), ...] top-10
```

## Contract touchpoints — what the next stage consumes

`Sessioniser` / `MarkovSessionModel` stop at producing these artifacts;
retrieval (Stage 3) and profile assembly consume them.

| Artifact | Returns | Downstream use |
|----------|---------|----------------|
| `MarkovSessionModel.predict_next(recent)` | `[(video_id, score), …]` | the **SESSION** retrieval lane + a ranking feature |
| `Sessioniser.session_stats()` | per-user session lengths | fills `EngagementProfile.avg_session_length` (replaces the daily-intensity proxy) |
| `Sessioniser.to_sequences()` | ordered id lists | training input for GRU4Rec later |

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
`test_ratio=0.2`. HR only (MRR / NDCG are computed but not featured).

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

<!-- ABLATION_TABLE -->

## Deferred

* **GRU4Rec / SASRec** — the neural sequence *models to beat*.
* **Per-user tuning** — sweeping `gap_minutes` / `pct` against downstream accuracy.
* **Phase-2 wiring** — `RetrievalStrategy.SESSION`, router weights, and the
  `SessionModeller` that `build_session_profile` delegates to.
* **MRR / NDCG reporting** — computed already; not featured while HR is the headline.

## Tests

`tests/test_sessioniser.py` + `tests/test_session_markov.py` cover the three
strategies, boundary rules, sequence extraction, session stats, transition
counting, popularity backoff, and `exclude`. Pure in-memory DataFrames — no disk
I/O.
