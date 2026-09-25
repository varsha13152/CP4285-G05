"""Tests for contract dataclasses — verify they construct correctly."""

from recsys.contracts.enums import QueryIntentType
from recsys.contracts.models import (
    CategoryMatch,
    QueryRetrievalRequest,
)


def test_engagement_profile_creation(sample_engagement_profile):
    p = sample_engagement_profile
    assert p.user_id == 1
    assert 0 <= p.avg_watch_ratio <= 5.0
    assert len(p.top_category_ids) > 0


def test_query_intent_creation(sample_query_intent):
    qi = sample_query_intent
    assert "fun cooking videos" in qi.raw_query
    assert isinstance(qi.intent_type, QueryIntentType)
    assert 0 <= qi.confidence <= 1.0
    assert len(qi.topics) > 0
    assert isinstance(qi.topics[0], CategoryMatch)


def test_retrieval_request_defaults(sample_query_intent):
    req = QueryRetrievalRequest(query_intent=sample_query_intent)
    assert req.num_candidates == 100
    assert req.category_filter is None
    assert req.user_id is None


def test_profile_context_defaults_are_empty(sample_engagement_profile):
    # Cold-start / no-context users: empty means "unavailable", not "disliked".
    p = sample_engagement_profile
    assert p.session_context is None
    assert p.social_scores == {} and p.cf_neighbour_scores == {}


def test_category_match_fields():
    m = CategoryMatch(12, "美食", "Food", 0.9)
    assert m.category_id == 12
    assert m.chinese_name == "美食"
    assert m.english_name == "Food"
    assert m.confidence == 0.9
