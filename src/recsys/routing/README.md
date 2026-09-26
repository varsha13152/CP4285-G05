# Query Routing

Owned by **Varsha** (lane weights removed for the query-first cascade — see the
change log in the top-level README).

`QueryRouter` turns a Stage 1 `QueryIntent` (plus the optional Stage 2
`EngagementProfile`) into the `QueryRetrievalRequest` that Stage 3 consumes.
It only encodes **what the query asked for** — personalisation is a Stage 4
ranking concern, so there are no retrieval lanes to weight.

| Request field | Derived from | Used by Stage 3 as |
|---------------|--------------|--------------------|
| `category_filter` | `QueryIntent.topics` with confidence ≥ `min_category_confidence` (0.3) | query topic ids for the CATEGORY source — a relevance signal, **not** a hard filter |
| `duration_range_ms` | `QueryIntent.duration` (SHORT <15 s, MEDIUM 15–60 s, LONG 60–315 s, ANY → `None`) | a **soft** target: out-of-range videos are down-weighted, never dropped |
| `num_candidates` | `default_num_candidates` (100) | pool size handed to Stage 4 |
| `engagement_profile`, `user_id` | passed through untouched | Stage 3 reads only `seen_video_ids` (as a flag); Stage 4 reads the rest |

## Usage

```python
from recsys.routing import QueryRouter

router = QueryRouter(default_num_candidates=100)
request = router.route(intent, user_id=14, engagement_profile=profile)
```
