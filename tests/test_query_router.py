"""Tests for the query-conditioned router."""

import pytest

from recsys.contracts.enums import (
    ContentStyle,
    DurationPreference,
    QueryIntentType,
)
from recsys.contracts.models import CategoryMatch, QueryIntent, QueryRetrievalRequest
from recsys.routing.query_router import QueryRouter


def _make_intent(
    intent_type: QueryIntentType = QueryIntentType.ENTERTAINMENT,
    style: ContentStyle = ContentStyle.NEUTRAL,
    duration: DurationPreference = DurationPreference.ANY,
    topics: list[CategoryMatch] | None = None,
) -> QueryIntent:
    """Helper to build a QueryIntent with sensible defaults."""
    return QueryIntent(
        raw_query="test",
        topics=topics or [],
        style=style,
        duration=duration,
        intent_type=intent_type,
        negative_constraints=[],
        semantic_summary="test",
    )


@pytest.fixture
def router() -> QueryRouter:
    return QueryRouter()


class TestQueryRouter:
    def test_returns_query_retrieval_request(self, router: QueryRouter) -> None:
        intent = _make_intent()
        result = router.route(intent)
        assert isinstance(result, QueryRetrievalRequest)

    def test_user_id_passthrough(self, router: QueryRouter) -> None:
        intent = _make_intent()
        result = router.route(intent, user_id=42)
        assert result.user_id == 42

    def test_user_id_none(self, router: QueryRouter) -> None:
        intent = _make_intent()
        result = router.route(intent)
        assert result.user_id is None

    def test_default_pool_is_100(self, router: QueryRouter) -> None:
        assert router.route(_make_intent()).num_candidates == 100

    def test_profile_passthrough(self, router: QueryRouter, sample_engagement_profile) -> None:
        result = router.route(_make_intent(), user_id=1, engagement_profile=sample_engagement_profile)
        assert result.engagement_profile is sample_engagement_profile

    def test_duration_short(self, router: QueryRouter) -> None:
        intent = _make_intent(duration=DurationPreference.SHORT)
        result = router.route(intent)
        assert result.duration_range_ms == (0, 15_000)

    def test_duration_medium(self, router: QueryRouter) -> None:
        intent = _make_intent(duration=DurationPreference.MEDIUM)
        result = router.route(intent)
        assert result.duration_range_ms == (15_000, 60_000)

    def test_duration_long(self, router: QueryRouter) -> None:
        intent = _make_intent(duration=DurationPreference.LONG)
        result = router.route(intent)
        assert result.duration_range_ms == (60_000, 315_000)

    def test_duration_any(self, router: QueryRouter) -> None:
        intent = _make_intent(duration=DurationPreference.ANY)
        result = router.route(intent)
        assert result.duration_range_ms is None

    def test_category_filter_high_confidence(self, router: QueryRouter) -> None:
        topics = [
            CategoryMatch(12, "美食", "Food", 0.9),
            CategoryMatch(9, "喜剧", "Comedy", 0.8),
        ]
        intent = _make_intent(topics=topics)
        result = router.route(intent)
        assert result.category_filter == [12, 9]

    def test_category_filter_below_threshold(self, router: QueryRouter) -> None:
        topics = [
            CategoryMatch(12, "美食", "Food", 0.1),  # Below default 0.3
        ]
        intent = _make_intent(topics=topics)
        result = router.route(intent)
        assert result.category_filter is None

    def test_custom_threshold(self) -> None:
        router = QueryRouter(min_category_confidence=0.8)
        topics = [
            CategoryMatch(12, "美食", "Food", 0.9),
            CategoryMatch(9, "喜剧", "Comedy", 0.5),  # Below 0.8
        ]
        intent = _make_intent(topics=topics)
        result = router.route(intent)
        assert result.category_filter == [12]

    def test_custom_num_candidates(self) -> None:
        router = QueryRouter(default_num_candidates=50)
        intent = _make_intent()
        result = router.route(intent)
        assert result.num_candidates == 50
