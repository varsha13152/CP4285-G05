# User & Session Modelling (Stage 2)

## Your role in the pipeline

```
User query ──→ [Stage 1: Intent Understanding] ──→ QueryIntent
                                                       │
User ID ──→ [Stage 2: User & Session Modelling] ──→ EngagementProfile + session context
                                                       │
              [Stage 3: Candidate Generation]  ←───────┘ (QueryRetrievalRequest)
```

Stage 1 produces a `QueryIntent` (what the user asked for).
**Your job is to model who the user is** — their long-term preferences, viewing history,
and current session context — so that Stage 3 can combine *what they want* with *who they are*.

## Input from Stage 1

You receive the `QueryIntent` and a `user_id`. The query intent tells you:

| Field | Type | Description |
|-------|------|-------------|
| `query_intent.intent_type` | `QueryIntentType` | ENTERTAINMENT / LEARNING / INSPIRATION / SOCIAL / SPECIFIC_SEARCH / EXPLORATION |
| `query_intent.topics` | `list[CategoryMatch]` | Matched KuaiRec categories with confidence |
| `query_intent.style` | `ContentStyle` | Desired content tone |
| `query_intent.duration` | `DurationPreference` | SHORT / MEDIUM / LONG / ANY |
| `query_intent.semantic_summary` | `str` | One-line summary of user intent |

## Your output: EngagementProfile

The `EngagementProfile` dataclass is already defined in `recsys.contracts.models`:

```python
from recsys.contracts import EngagementProfile

profile = EngagementProfile(
    user_id=42,
    avg_watch_ratio=0.73,      # How much of videos the user typically watches
    total_interactions=1500,
    like_rate=0.12,            # Fraction of watched videos they like
    follow_rate=0.03,          # Fraction of watched videos leading to follow
    category_distribution={12: 0.25, 17: 0.15, 9: 0.12},  # User's category preferences
)
```

## What you can build

### 1. User preference model
Build `EngagementProfile` from `small_matrix.csv`:
- Compute per-user engagement stats (avg watch ratio, like rate, follow rate)
- Build category distribution from their interaction history
- Identify strong preferences vs casual interests

### 2. Session context (the QCSAR differentiator)
The key insight of QCSAR: **the search query creates a temporary session** that shouldn't
alter long-term preferences. You should model:

- **Session intent**: derived from `QueryIntent` — what are they looking for *right now*?
- **Session vs long-term blend**: how much should we weight the query vs their history?
  - A SPECIFIC_SEARCH intent → lean heavily on the query, less on history
  - An EXPLORATION intent → blend query with their historical preferences
- **Session boundaries**: this is a temporary context, not a profile update

### 3. User-query alignment
Combine user preferences with query intent to produce better routing signals:
- If user historically watches Food (cat 12) and searches "cooking tips" → high confidence
- If user never watches Sports (cat 7) but searches "football highlights" → query-driven, not history-driven

## Your contract

You can extend the pipeline by providing an `EngagementProfile` to the router:

```python
from recsys.contracts import EngagementProfile, QueryIntent
from recsys.routing import QueryRouter

class UserModeller:
    def build_profile(self, user_id: int) -> EngagementProfile:
        """Build engagement profile from user's interaction history."""
        ...

    def build_session_profile(
        self, user_id: int, query_intent: QueryIntent
    ) -> EngagementProfile:
        """Build session-aware profile that blends user history with query intent."""
        ...

# Integration:
modeller = UserModeller()
profile = modeller.build_session_profile(user_id=42, query_intent=intent)
request = QueryRouter().route(intent, user_id=42, engagement_profile=profile)
# → request now carries both query intent AND user context for Stage 3
```

## Data available

| File | Contents | Key columns |
|------|----------|-------------|
| `data/raw/small_matrix.csv` | User-video interactions (99.6% dense, 1411 users × 3327 videos) | `user_id`, `video_id`, `watch_ratio`, `is_click`, `is_like`, `is_follow`, `is_comment` |
| `data/raw/user_features.csv` | User demographic features | `user_id`, `user_active_degree`, `follow_user_num`, `fans_user_num`, `friend_user_num`, `register_days` |
| `data/raw/item_categories.csv` | Video → category IDs | `video_id`, `feat` |
| `data/raw/item_daily_features.csv` | Daily engagement stats per video | `video_id`, `date`, `video_duration`, `play_count`, etc. |

## Key design principle

**Session-aware, not session-permanent.** The query creates a temporary lens on recommendations.
When the session ends, the user's long-term profile is unchanged. Think of it as a filter
overlay, not a preference update.

## How it connects

```python
# Full pipeline flow:
intent = parser.parse("cute cat videos")                         # Stage 1 (Varsha)
profile = modeller.build_session_profile(42, intent)             # Stage 2 (you)
request = router.route(intent, user_id=42, engagement_profile=profile)
candidates = retriever.retrieve(request)                         # Stage 3 (Teammate B)
ranked = ranker.rank(candidates)                                 # Stage 4-6 (Varsha, later)
```
