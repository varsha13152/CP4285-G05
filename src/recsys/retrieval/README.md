# Candidate Retrieval (Stage 3)

## Input from Stage 1

You receive a `QueryRetrievalRequest` with:

| Field | Type | Description |
|-------|------|-------------|
| `query_intent.topics` | `list[CategoryMatch]` | Matched KuaiRec categories with confidence scores |
| `query_intent.query_embedding` | `list[float] \| None` | Dense vector for semantic retrieval (optional, may be `None`) |
| `query_intent.style` | `ContentStyle` | Desired content tone (FUNNY, SERIOUS, RELAXING, etc.) |
| `query_intent.duration` | `DurationPreference` | SHORT / MEDIUM / LONG / ANY |
| `query_intent.intent_type` | `QueryIntentType` | ENTERTAINMENT / LEARNING / INSPIRATION / SOCIAL / SPECIFIC_SEARCH / EXPLORATION |
| `query_intent.negative_constraints` | `list[str]` | Things to exclude (e.g., "not too long") |
| `query_intent.semantic_summary` | `str` | LLM-generated one-line summary of user intent |
| `category_filter` | `list[int] \| None` | Hard filter — only category IDs above confidence threshold |
| `duration_range_ms` | `tuple[int, int] \| None` | `(min_ms, max_ms)` from duration preference |
| `strategies` | `list[StrategyWeight]` | Weighted retrieval strategies (CONTENT_BASED, COLLABORATIVE, POPULARITY, SOCIAL) |
| `user_id` | `int \| None` | `None` for anonymous queries |
| `engagement_profile` | `EngagementProfile \| None` | User context if available |
| `num_candidates` | `int` | Target number of candidates (default 100) |

## Your contract

Implement the `Retriever` protocol from `recsys.pipeline.pipeline`:

```python
from recsys.pipeline.pipeline import Retriever
from recsys.contracts import CandidateSet, QueryRetrievalRequest

class MyRetriever:
    def retrieve(self, request: QueryRetrievalRequest) -> CandidateSet:
        ...
```

Return a `CandidateSet` containing `Candidate` objects:

```python
@dataclass
class Candidate:
    video_id: int
    score: float               # Retrieval relevance score
    source_strategy: str       # Which strategy produced this candidate
    category_ids: list[int]    # Video's category IDs
    features: dict = ...       # Optional extra features for ranking
```

## Data available

| File | Contents | Key columns |
|------|----------|-------------|
| `data/raw/item_categories.csv` | video → category IDs | `video_id`, `feat` (list of category_ids) |
| `data/raw/item_daily_features.csv` | Daily engagement stats | `video_id`, `date`, `video_duration`, `play_count`, `like_count`, etc. |
| `data/raw/kuairec_caption_category.csv` | Captions + hierarchical categories | `video_id`, `caption`, `first_level_category_id`, `first_level_category_name` |
| `data/raw/small_matrix.csv` | User-video interactions (99.6% dense) | `user_id`, `video_id`, `watch_ratio`, `is_click`, `is_like`, etc. |

## Strategy guidance

The `strategies` weights tell you how to blend results:

- **CONTENT_BASED**: Filter by `category_filter` and `duration_range_ms`, rank by category/style match
- **COLLABORATIVE**: Use `small_matrix.csv` — find similar users, surface their liked videos
- **POPULARITY**: Rank by engagement metrics (play_count, like_count) within filtered categories
- **SOCIAL**: Trending/viral signals from recent engagement data

## Quick start

```python
from recsys.contracts import QueryRetrievalRequest, CandidateSet, Candidate

class SimpleRetriever:
    def retrieve(self, request: QueryRetrievalRequest) -> CandidateSet:
        candidates = []
        # 1. Filter videos by category_filter
        # 2. Filter by duration_range_ms
        # 3. Score and rank using strategies
        # 4. Return top num_candidates
        return CandidateSet(
            candidates=candidates,
            request=request,
        )
```
