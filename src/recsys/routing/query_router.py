"""Query-conditioned routing for the QCSAR pipeline.

Maps a ``QueryIntent`` to a ``QueryRetrievalRequest``, selecting retrieval
strategies and filters based on the parsed query rather than user behavior.

Usage::

    from recsys.routing import QueryRouter

    router = QueryRouter()
    request = router.route(query_intent, user_id=14)
"""

from __future__ import annotations

from recsys.contracts.enums import (
    DurationPreference,
    QueryIntentType,
    RetrievalStrategy,
)
from recsys.contracts.models import (
    EngagementProfile,
    QueryIntent,
    QueryRetrievalRequest,
    StrategyWeight,
)


# ---------------------------------------------------------------------------
# Strategy weights per query intent type
# ---------------------------------------------------------------------------

_STRATEGY_MAP: dict[QueryIntentType, list[StrategyWeight]] = {
    QueryIntentType.ENTERTAINMENT: [
        StrategyWeight(RetrievalStrategy.CONTENT_BASED, 0.4),
        StrategyWeight(RetrievalStrategy.POPULARITY, 0.4),
        StrategyWeight(RetrievalStrategy.COLLABORATIVE, 0.2),
    ],
    QueryIntentType.LEARNING: [
        StrategyWeight(RetrievalStrategy.CONTENT_BASED, 0.7),
        StrategyWeight(RetrievalStrategy.COLLABORATIVE, 0.2),
        StrategyWeight(RetrievalStrategy.POPULARITY, 0.1),
    ],
    QueryIntentType.SPECIFIC_SEARCH: [
        StrategyWeight(RetrievalStrategy.CONTENT_BASED, 0.8),
        StrategyWeight(RetrievalStrategy.COLLABORATIVE, 0.1),
        StrategyWeight(RetrievalStrategy.POPULARITY, 0.1),
    ],
    QueryIntentType.SOCIAL: [
        StrategyWeight(RetrievalStrategy.POPULARITY, 0.5),
        StrategyWeight(RetrievalStrategy.SOCIAL, 0.3),
        StrategyWeight(RetrievalStrategy.CONTENT_BASED, 0.2),
    ],
    QueryIntentType.EXPLORATION: [
        StrategyWeight(RetrievalStrategy.POPULARITY, 0.3),
        StrategyWeight(RetrievalStrategy.CONTENT_BASED, 0.3),
        StrategyWeight(RetrievalStrategy.COLLABORATIVE, 0.2),
        StrategyWeight(RetrievalStrategy.SOCIAL, 0.2),
    ],
    QueryIntentType.INSPIRATION: [
        StrategyWeight(RetrievalStrategy.CONTENT_BASED, 0.5),
        StrategyWeight(RetrievalStrategy.POPULARITY, 0.3),
        StrategyWeight(RetrievalStrategy.COLLABORATIVE, 0.2),
    ],
}

# ---------------------------------------------------------------------------
# Duration → millisecond ranges (from item_daily_features.csv stats)
# ---------------------------------------------------------------------------

_DURATION_RANGES: dict[DurationPreference, tuple[int, int] | None] = {
    DurationPreference.SHORT: (0, 15_000),
    DurationPreference.MEDIUM: (15_000, 60_000),
    DurationPreference.LONG: (60_000, 315_000),
    DurationPreference.ANY: None,
}

# ---------------------------------------------------------------------------
# Default fallback strategies
# ---------------------------------------------------------------------------

_FALLBACK_STRATEGIES: list[StrategyWeight] = [
    StrategyWeight(RetrievalStrategy.POPULARITY, 0.4),
    StrategyWeight(RetrievalStrategy.CONTENT_BASED, 0.3),
    StrategyWeight(RetrievalStrategy.COLLABORATIVE, 0.2),
    StrategyWeight(RetrievalStrategy.SOCIAL, 0.1),
]


class QueryRouter:
    """Route a ``QueryIntent`` into a ``QueryRetrievalRequest``.

    Parameters
    ----------
    default_num_candidates : int
        Number of candidates to request from retrieval.
    min_category_confidence : float
        Only include topics with confidence at or above this threshold
        in the hard category filter.
    """

    def __init__(
        self,
        default_num_candidates: int = 100,
        min_category_confidence: float = 0.3,
    ) -> None:
        self._num_candidates = default_num_candidates
        self._min_confidence = min_category_confidence

    def route(
        self,
        query_intent: QueryIntent,
        user_id: int | None = None,
        engagement_profile: EngagementProfile | None = None,
    ) -> QueryRetrievalRequest:
        """Map a parsed query intent to retrieval parameters."""
        # Strategy weights
        strategies = _STRATEGY_MAP.get(
            query_intent.intent_type, _FALLBACK_STRATEGIES,
        )

        # Duration range
        duration_range = _DURATION_RANGES.get(query_intent.duration)

        # Category filter — only high-confidence topic matches
        confident_topics = [
            t.category_id
            for t in query_intent.topics
            if t.confidence >= self._min_confidence
        ]
        category_filter = confident_topics if confident_topics else None

        return QueryRetrievalRequest(
            query_intent=query_intent,
            user_id=user_id,
            engagement_profile=engagement_profile,
            strategies=strategies,
            num_candidates=self._num_candidates,
            category_filter=category_filter,
            duration_range_ms=duration_range,
            min_category_confidence=self._min_confidence,
        )
