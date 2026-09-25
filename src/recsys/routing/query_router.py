"""Query-conditioned routing for the QCSAR pipeline.

Maps a ``QueryIntent`` to a ``QueryRetrievalRequest``: which query topics feed
the category retrieval source, the soft duration target, and the candidate
pool size. In the query-first cascade there are no retrieval lanes to weight —
personalisation happens in Stage 4 ranking — so routing is purely about what
the query asked for.

Usage::

    from recsys.routing import QueryRouter

    router = QueryRouter()
    request = router.route(query_intent, user_id=14, engagement_profile=profile)
"""

from __future__ import annotations

from recsys.contracts.enums import DurationPreference
from recsys.contracts.models import (
    EngagementProfile,
    QueryIntent,
    QueryRetrievalRequest,
)

# ---------------------------------------------------------------------------
# Duration → millisecond ranges (from item_daily_features.csv stats)
# KuaiRec durations: SHORT 85% / MEDIUM 14% / LONG 1.1% of videos, which is
# why Stage 3 treats this range as a soft target rather than a filter.
# ---------------------------------------------------------------------------

_DURATION_RANGES: dict[DurationPreference, tuple[int, int] | None] = {
    DurationPreference.SHORT: (0, 15_000),
    DurationPreference.MEDIUM: (15_000, 60_000),
    DurationPreference.LONG: (60_000, 315_000),
    DurationPreference.ANY: None,
}


class QueryRouter:
    """Route a ``QueryIntent`` into a ``QueryRetrievalRequest``.

    Parameters
    ----------
    default_num_candidates : int
        Number of candidates Stage 3 hands to Stage 4 (default 100).
    min_category_confidence : float
        Only topics with confidence at or above this threshold feed the
        category retrieval source.
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
        confident_topics = [
            t.category_id
            for t in query_intent.topics
            if t.confidence >= self._min_confidence
        ]

        return QueryRetrievalRequest(
            query_intent=query_intent,
            user_id=user_id,
            engagement_profile=engagement_profile,
            num_candidates=self._num_candidates,
            category_filter=confident_topics or None,
            duration_range_ms=_DURATION_RANGES.get(query_intent.duration),
            min_category_confidence=self._min_confidence,
        )
