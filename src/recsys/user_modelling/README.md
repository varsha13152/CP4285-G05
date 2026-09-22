# User & Session Modelling (Stage 2)

Models **who the user is** — their long-term taste — so Stage 3 can combine
*what they asked for* (`QueryIntent`) with *who they are* (`EngagementProfile`).

```
User query ──→ [Stage 1: Intent Understanding] ──→ QueryIntent
                                                       │
User ID ──→ [Stage 2: User Modelling (this)]  ──→ EngagementProfile
                                                       │
              [Stage 3: Candidate Generation]  ←───────┘ (QueryRetrievalRequest)
```

## What this implements: `u_long`

The long-term user model is **matrix factorization** via implicit ALS
(the [`implicit`](https://github.com/benfred/implicit) library) trained on
KuaiRec's sparse `big_matrix`, using `watch_ratio` as the per-interaction
**confidence**. Each user's learned latent factor vector is stored on
`EngagementProfile.user_vector` and consumed downstream **purely as a ranking
signal** — powering the personalised/collaborative candidate lane and a
taste-alignment feature in ranking. It is never used as a hard filter.

### Why ALS over BPR
KuaiRec's informative signal is the graded `watch_ratio`. ALS exploits it
directly as confidence (`C_ui`); BPR would flatten it to binary preference and
throw that signal away.

### Sparsity
Trained **observed-only** over a `scipy` CSR matrix — no `fillna(0)`
densification. The fully-observed `small_matrix` is intentionally **not** touched
here; it is reserved for evaluation.

## Usage

```python
from recsys.user_modelling import UserModeller
from recsys.routing import QueryRouter

modeller = UserModeller()                        # trains on data/raw/big_matrix.csv
profile = modeller.build_profile(user_id=14)     # EngagementProfile w/ user_vector
request = QueryRouter().route(intent, user_id=14, engagement_profile=profile)
```

## Output contract: `EngagementProfile`

Defined in `recsys.contracts.models`. Stage 2 **produces** this object; the
router carries it and Stage 3 / ranking **consume** it. Every field is populated
by `build_profile`. The table gives what each field means, how it is derived,
and how the next stage is expected to use it.

| Field | Meaning | Derived from | Intended use downstream |
|-------|---------|--------------|-------------------------|
| `user_vector` | The `u_long` ALS latent-factor row (length = `factors`) — the user's position in taste-space. **Empty** for cold-start users. | a slice of the ALS user matrix, `U[row]` | Dot with an item vector (`U[u]·V[i]`) to drive the collaborative candidate lane and a personalisation feature in ranking. It is a ranking score, **not** a probability or a predicted watch_ratio. |
| `avg_watch_ratio` | Typical fraction of a video the user watches (`>1` = rewatches). Engagement level. | mean of the user's `watch_ratio` | Ranking feature / calibration — a finish-everything user vs a picky one. |
| `watch_ratio_std` | Consistency of engagement (low = predictable). | sample std of `watch_ratio` (`0.0` for 1-interaction users) | Ranking feature; separates steady from erratic tastes. |
| `total_interactions` | Volume of history — a **confidence** proxy for the whole profile. | row count | Trust weighting: how much to lean on personalisation vs popularity for thin-history users. |
| `active_days` | Distinct days active — spread / tenure. | `nunique(date)` | Activity / trust weighting alongside `total_interactions`. |
| `top_category_ids` | Categories ranked most→least watched (explicit taste). | count of each watched video's `feat` categories, sorted desc | Category-affinity boost; blend with the query's category filter. |
| `category_distribution` | Normalised category shares (sums to `1.0`). | category counts ÷ total | Soft re-rank toward habitual categories; diversity control. |
| `avg_session_length` | **Proxy**: average interactions per active day. Not a true session (no time-gap segmentation yet). | `total_interactions / active_days` | Placeholder until the session model lands; read as daily intensity, not sessions. |
| `recency_days` | Freshness — active vs dormant. Relative to the **dataset's** last timestamp, not "today". | `(dataset_last_ts − user_last_ts) / 86400` | Freshness weighting / decay for dormant users. |

**Current wiring reality.** Downstream stages do **not** read these fields yet —
`QueryRouter.route()` carries the whole `EngagementProfile` through unchanged
inside `QueryRetrievalRequest`, and Stage 3 (retrieval) + ranking are where the
"intended use" column takes effect once those modules exist. The field built to
be the primary signal is `user_vector`; the rest are side-signals the same
profile conveniently carries.

> **Note — item factors (`V`).** One ALS fit learns *both* a user matrix `U`
> (`user_factors`) and an item matrix `V` (`item_factors`). Only the per-user
> `user_vector` = `U[row]` is exposed on the profile today; `V` is held on the
> `UserModeller` instance but not surfaced. A predicted score or candidate list
> needs both (`U[u]·V[i]`), so item-side scoring is planned to live behind a
> `UserModeller.recommend()` / `score()` method (wrapping `implicit`'s
> `model.recommend`), keeping the `video_id ↔ column` mapping encapsulated
> rather than returning raw `V` through the contract.

## Knobs (`UserModeller.__init__`)

| Param | Default | Notes |
|-------|---------|-------|
| `factors` | 16 | latent dimensionality `k` (try 16–32) |
| `regularization` | 0.05 | ALS L2 |
| `iterations` | 15 | ALS sweeps |
| `confidence_alpha` | `None` | `None` = raw `watch_ratio`; float = `1 + α·watch_ratio` |
| `interactions`, `item_categories` | `None` | in-memory tables (disk-free testing seam) |

## Deferred

`build_session_profile(user_id, query_intent)` currently delegates to
`build_profile`. The query-conditioned **session model** (the QCSAR
differentiator — a temporary lens that does not mutate long-term preferences),
`watch_ratio` debiasing, a BPR alternative, and a NeuMF upgrade are all future
work.

## Tests

`tests/test_user_modelling.py` trains on the synthetic `sample_interactions`
fixture (conftest) — no disk I/O, ALS fits in milliseconds.
