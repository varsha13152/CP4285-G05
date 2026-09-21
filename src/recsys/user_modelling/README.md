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

## Output: `EngagementProfile`

Defined in `recsys.contracts.models`. Fields populated here:

| Field | Source |
|-------|--------|
| `user_vector` | ALS latent factors (length = `factors`); **empty** for cold-start users |
| `avg_watch_ratio`, `watch_ratio_std` | per-user `watch_ratio` mean / std |
| `total_interactions`, `active_days` | row count / distinct `date` count |
| `avg_session_length` | interactions per active day |
| `recency_days` | days between the user's last interaction and the dataset's |
| `top_category_ids`, `category_distribution` | join with `item_categories.feat` |

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
