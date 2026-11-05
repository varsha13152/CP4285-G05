"""Tests for the simplified QCSARRanker (α-blended long-term / short-term).

Uses tiny synthetic data (6 videos × 4-dim factors) so dot products
can be verified by hand.  No disk I/O, no LLM calls.
"""

from __future__ import annotations

import numpy as np
import pytest

from recsys.contracts.enums import (
    ContentStyle,
    DurationPreference,
    QueryIntentType,
    RetrievalSource,
)
from recsys.contracts.models import (
    Candidate,
    CandidateSet,
    CategoryMatch,
    EngagementProfile,
    QueryIntent,
    QueryRetrievalRequest,
    RankedList,
    SessionContext,
)
from recsys.ranking import QCSARRanker


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def item_factors() -> tuple[np.ndarray, np.ndarray]:
    """6 videos, 4-dim ALS factors.  Rows are NOT normalised on purpose —
    the ranker normalises at construction time."""
    ids = np.array([10, 20, 30, 40, 50, 60])
    factors = np.array([
        [1.0, 0.0, 0.0, 0.0],   # vid 10
        [0.0, 1.0, 0.0, 0.0],   # vid 20
        [1.0, 1.0, 0.0, 0.0],   # vid 30 — similar to 10 and 20
        [0.0, 0.0, 1.0, 0.0],   # vid 40
        [0.0, 0.0, 0.0, 1.0],   # vid 50
        [0.5, 0.5, 0.5, 0.5],   # vid 60 — moderate similarity to everything
    ])
    return factors, ids


@pytest.fixture
def ranker(item_factors: tuple[np.ndarray, np.ndarray]) -> QCSARRanker:
    factors, ids = item_factors
    return QCSARRanker(factors, ids, top_k=20)


def _query_intent(**overrides) -> QueryIntent:
    defaults = dict(
        raw_query="test query",
        topics=[CategoryMatch(12, "美食", "Food", 0.9)],
        style=ContentStyle.FUNNY,
        duration=DurationPreference.SHORT,
        intent_type=QueryIntentType.ENTERTAINMENT,
        negative_constraints=[],
        semantic_summary="test",
        confidence=0.9,
    )
    defaults.update(overrides)
    return QueryIntent(**defaults)


def _candidate(video_id: int, score: float, *, cat: int = 1, **features) -> Candidate:
    feats = {"sem": 0.5, "cat": 0.5, "dur_fit": 1.0, "has_text": 1.0,
             "seen": 0.0, "popularity": 0.5, "trending": 0.5}
    feats.update(features)
    return Candidate(
        video_id=video_id,
        score=score,
        source=RetrievalSource.BOTH,
        category_ids=[cat],
        features=feats,
    )


def _candidate_set(
    candidates: list[Candidate],
    profile: EngagementProfile | None = None,
    intent: QueryIntent | None = None,
    user_id: int | None = 1,
) -> CandidateSet:
    qi = intent or _query_intent()
    req = QueryRetrievalRequest(
        query_intent=qi,
        user_id=user_id,
        engagement_profile=profile,
        category_filter=[12],
    )
    return CandidateSet(user_id=user_id, request=req, candidates=candidates)


def _profile(
    *,
    cf: dict[int, float] | None = None,
    social: dict[int, float] | None = None,
    cat_dist: dict[int, float] | None = None,
    recent: list[int] | None = None,
    next_item: dict[int, float] | None = None,
    session_vector: list[float] | None = None,
) -> EngagementProfile:
    ctx = None
    if recent is not None:
        ctx = SessionContext(
            recent_video_ids=recent,
            next_item_scores=next_item or {},
            session_vector=session_vector or [],
            recent_category_ids=[],
        )
    return EngagementProfile(
        user_id=1,
        avg_watch_ratio=0.5,
        watch_ratio_std=0.2,
        total_interactions=10,
        active_days=5,
        top_category_ids=[1],
        category_distribution=cat_dist if cat_dist is not None else {1: 1.0},
        avg_session_length=3.0,
        recency_days=1,
        cf_neighbour_scores=cf or {},
        social_scores=social or {},
        session_context=ctx,
    )


# ---------------------------------------------------------------------------
# Tests — contract
# ---------------------------------------------------------------------------

class TestRankerContract:
    """Basic contract: return type, field alignment, edge cases."""

    def test_rank_returns_ranked_list(self, ranker: QCSARRanker) -> None:
        cands = [_candidate(10, 0.9), _candidate(20, 0.7), _candidate(30, 0.5)]
        cs = _candidate_set(cands)
        result = ranker.rank(cs)

        assert isinstance(result, RankedList)
        assert result.user_id == 1
        assert result.num_candidates_considered == 3
        assert result.intent_type == QueryIntentType.ENTERTAINMENT
        assert len(result.items) == 3

    def test_ranks_are_one_indexed_and_contiguous(self, ranker: QCSARRanker) -> None:
        cands = [_candidate(10, 0.9), _candidate(20, 0.7), _candidate(30, 0.5)]
        result = ranker.rank(_candidate_set(cands))
        for i, item in enumerate(result.items):
            assert item.rank == i + 1

    def test_empty_candidate_set(self, ranker: QCSARRanker) -> None:
        result = ranker.rank(_candidate_set([]))
        assert isinstance(result, RankedList)
        assert result.items == []
        assert result.num_candidates_considered == 0

    def test_top_k_truncation(self, item_factors: tuple[np.ndarray, np.ndarray]) -> None:
        factors, ids = item_factors
        ranker = QCSARRanker(factors, ids, top_k=3)
        cands = [_candidate(vid, 1.0 - i * 0.1) for i, vid in enumerate(ids)]
        result = ranker.rank(_candidate_set(cands))
        assert len(result.items) == 3


# ---------------------------------------------------------------------------
# Tests — long-term signals
# ---------------------------------------------------------------------------

class TestLongTermSignals:
    """Long-term personal signals: CF, category affinity, social."""

    def test_anonymous_user_ranks_by_query_relevance(
        self, ranker: QCSARRanker,
    ) -> None:
        """No profile → only query relevance contributes."""
        cands = [
            _candidate(10, 0.9),
            _candidate(20, 0.5),
        ]
        result = ranker.rank(_candidate_set(cands, profile=None))
        assert result.items[0].video_id == 10

    def test_cf_scores_boost_candidates(self, ranker: QCSARRanker) -> None:
        """Higher CF score lifts a candidate when query scores are tied."""
        profile = _profile(cf={20: 1.0, 10: 0.0})
        cands = [
            _candidate(10, 0.5),
            _candidate(20, 0.5),
        ]
        result = ranker.rank(_candidate_set(cands, profile=profile))
        assert result.items[0].video_id == 20

    def test_category_affinity_boosts_preferred_categories(
        self, ranker: QCSARRanker,
    ) -> None:
        """Videos in the user's preferred categories rank higher."""
        profile = _profile(cf={}, social={}, cat_dist={2: 0.8, 1: 0.2})
        cands = [
            _candidate(10, 0.5, cat=1),
            _candidate(20, 0.5, cat=2),
        ]
        result = ranker.rank(_candidate_set(cands, profile=profile))
        assert result.items[0].video_id == 20

    def test_social_scores_boost_candidates(self, ranker: QCSARRanker) -> None:
        """Higher social (friends') scores lift a candidate."""
        profile = _profile(social={20: 1.0, 10: 0.0})
        cands = [
            _candidate(10, 0.5),
            _candidate(20, 0.5),
        ]
        result = ranker.rank(_candidate_set(cands, profile=profile))
        assert result.items[0].video_id == 20


# ---------------------------------------------------------------------------
# Tests — short-term signals
# ---------------------------------------------------------------------------

class TestShortTermSignals:
    """Short-term personal signals: Markov, item-item sim, session vector."""

    def test_markov_transition_scores_used(self, ranker: QCSARRanker) -> None:
        profile = _profile(
            recent=[],  # empty recent → item-item sims are zero
            next_item={20: 1.0, 30: 0.0},
        )
        cands = [
            _candidate(20, 0.5),
            _candidate(30, 0.5),
        ]
        result = ranker.rank(_candidate_set(cands, profile=profile))
        assert result.items[0].video_id == 20

    def test_item_similar_to_recent_gets_boosted(
        self, item_factors: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Candidate closer to recent videos in ALS space ranks higher."""
        factors, ids = item_factors
        ranker = QCSARRanker(factors, ids, top_k=20)

        # Recent video: 10 (factor [1,0,0,0]).
        # vid 30 ([1,1,0,0]) is closer than vid 40 ([0,0,1,0]).
        profile = _profile(
            recent=[10],
            session_vector=[1.0, 0.0, 0.0, 0.0],
        )
        cands = [
            _candidate(30, 0.5),
            _candidate(40, 0.5),
        ]
        result = ranker.rank(_candidate_set(cands, profile=profile))
        assert result.items[0].video_id == 30

    def test_no_session_context_means_no_short_term(
        self, ranker: QCSARRanker,
    ) -> None:
        """Without session context, short-term composite is zero."""
        profile = _profile()  # no recent → session_context is None
        cands = [_candidate(10, 0.9), _candidate(20, 0.7)]
        result = ranker.rank(_candidate_set(cands, profile=profile))
        # Query relevance decides (10 has higher retrieval score)
        assert result.items[0].video_id == 10


# ---------------------------------------------------------------------------
# Tests — alpha blending
# ---------------------------------------------------------------------------

class TestAlphaBlending:
    """The α parameter controls the long-term / short-term balance."""

    def test_alpha_one_ignores_short_term(
        self, item_factors: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """With α=1.0, short-term signals have no effect."""
        factors, ids = item_factors
        ranker = QCSARRanker(factors, ids, alpha=1.0, top_k=20)

        # Strong short-term signal for vid 40, strong long-term for vid 20.
        profile = _profile(
            cf={20: 1.0, 40: 0.0},
            recent=[40],  # makes vid 40 similar in short-term
            session_vector=[0.0, 0.0, 1.0, 0.0],  # points toward vid 40
            next_item={40: 1.0},
        )
        cands = [_candidate(20, 0.5), _candidate(40, 0.5)]
        result = ranker.rank(_candidate_set(cands, profile=profile))
        # α=1 → pure long-term → CF favours vid 20
        assert result.items[0].video_id == 20

    def test_alpha_zero_ignores_long_term(
        self, item_factors: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """With α=0.0, long-term signals have no effect."""
        factors, ids = item_factors
        ranker = QCSARRanker(factors, ids, alpha=0.0, top_k=20)

        # Strong long-term for vid 20, strong short-term for vid 40.
        profile = _profile(
            cf={20: 1.0, 40: 0.0},
            recent=[40],
            session_vector=[0.0, 0.0, 1.0, 0.0],
            next_item={40: 1.0},
        )
        cands = [_candidate(20, 0.5), _candidate(40, 0.5)]
        result = ranker.rank(_candidate_set(cands, profile=profile))
        # α=0 → pure short-term → session/markov favour vid 40
        assert result.items[0].video_id == 40

    def test_w_query_one_ignores_personal(
        self, item_factors: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """With w_query=1.0, personal signals don't matter."""
        factors, ids = item_factors
        ranker = QCSARRanker(factors, ids, w_query=1.0, top_k=20)

        # Strong CF for vid 20 but vid 10 has higher query score.
        profile = _profile(cf={20: 1.0, 10: 0.0})
        cands = [_candidate(10, 0.9), _candidate(20, 0.5)]
        result = ranker.rank(_candidate_set(cands, profile=profile))
        # w_query=1 → pure query relevance → vid 10 wins
        assert result.items[0].video_id == 10
