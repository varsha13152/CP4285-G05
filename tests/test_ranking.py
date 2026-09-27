"""Tests for the QCSARRanker (Stages 4-6).

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
    query_embedding: list[float] | None = None,
) -> CandidateSet:
    qi = intent or _query_intent()
    if query_embedding is not None:
        qi = _query_intent(query_embedding=query_embedding)
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
        category_distribution={1: 1.0},
        avg_session_length=3.0,
        recency_days=1,
        cf_neighbour_scores=cf or {},
        social_scores=social or {},
        session_context=ctx,
    )


# ---------------------------------------------------------------------------
# Tests
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


class TestStage4Signals:
    """Stage 4: multi-signal blending."""

    def test_anonymous_user_ranks_by_query_and_popularity(
        self, ranker: QCSARRanker,
    ) -> None:
        # No profile → only query relevance + popularity contribute.
        cands = [
            _candidate(10, 0.9, popularity=0.9, trending=0.9),
            _candidate(20, 0.5, popularity=0.1, trending=0.1),
        ]
        result = ranker.rank(_candidate_set(cands, profile=None))
        assert result.items[0].video_id == 10

    def test_cf_scores_boost_candidates(self, ranker: QCSARRanker) -> None:
        # Two candidates identical except CF score.
        profile = _profile(cf={20: 1.0, 10: 0.0})
        cands = [
            _candidate(10, 0.5, popularity=0.5, trending=0.5),
            _candidate(20, 0.5, popularity=0.5, trending=0.5),
        ]
        result = ranker.rank(_candidate_set(cands, profile=profile))
        assert result.items[0].video_id == 20

    def test_query_caption_similarity_boosts_candidates(self) -> None:
        """Direct query↔caption cosine similarity lifts semantically closer items."""
        # 4-dim caption embeddings aligned to video IDs [10, 20].
        # vid 10 embedding is close to the query; vid 20 is orthogonal.
        cap_embs = np.array([
            [1.0, 0.0, 0.0, 0.0],  # vid 10 — aligned with query
            [0.0, 0.0, 0.0, 1.0],  # vid 20 — orthogonal to query
        ], dtype=np.float64)
        cap_ids = np.array([10, 20])
        als_factors = np.eye(4, dtype=np.float64)[:2]
        als_ids = np.array([10, 20])

        ranker = QCSARRanker(
            als_factors, als_ids,
            caption_embeddings=cap_embs, caption_video_ids=cap_ids,
            top_k=20,
        )
        # Both candidates have identical retrieval scores.
        cands = [
            _candidate(10, 0.5, popularity=0.5, trending=0.5),
            _candidate(20, 0.5, popularity=0.5, trending=0.5),
        ]
        # Query embedding close to vid 10's caption.
        cs = _candidate_set(cands, query_embedding=[1.0, 0.0, 0.0, 0.0])
        result = ranker.rank(cs)
        assert result.items[0].video_id == 10

    def test_category_affinity_boosts_preferred_categories(
        self, ranker: QCSARRanker,
    ) -> None:
        """Videos in the user's preferred categories get a boost."""
        # User watches 80% cat 2, 20% cat 1.  Both candidates identical
        # except category — vid 20 (cat 2) should rank higher.
        profile = _profile(cf={}, social={})
        profile.category_distribution = {2: 0.8, 1: 0.2}
        cands = [
            _candidate(10, 0.5, cat=1, popularity=0.5, trending=0.5),
            _candidate(20, 0.5, cat=2, popularity=0.5, trending=0.5),
        ]
        result = ranker.rank(_candidate_set(cands, profile=profile))
        assert result.items[0].video_id == 20

    def test_markov_transition_scores_used(self, ranker: QCSARRanker) -> None:
        # No recent videos → Stage 5 is a no-op, so only Stage 4 matters.
        profile = _profile(
            next_item={20: 1.0, 30: 0.0},
        )
        # Give session_context without recent_video_ids so Stage 5 skips.
        profile.session_context = SessionContext(
            recent_video_ids=[],
            next_item_scores={20: 1.0, 30: 0.0},
        )
        cands = [
            _candidate(20, 0.5, popularity=0.5, trending=0.5),
            _candidate(30, 0.5, popularity=0.5, trending=0.5),
        ]
        result = ranker.rank(_candidate_set(cands, profile=profile))
        assert result.items[0].video_id == 20


class TestStage5ItemItem:
    """Stage 5: item-item similarity re-ranking."""

    def test_item_similar_to_recent_gets_boosted(
        self, item_factors: tuple[np.ndarray, np.ndarray],
    ) -> None:
        factors, ids = item_factors
        ranker = QCSARRanker(factors, ids, top_k=20)

        # Recent video: 10 (factor [1,0,0,0]).
        # Candidate 30 (factor [1,1,0,0]) is more similar than 40 ([0,0,1,0]).
        # Give them identical Stage 4 scores so Stage 5 decides.
        profile = _profile(
            recent=[10],
            session_vector=[1.0, 0.0, 0.0, 0.0],
        )
        cands = [
            _candidate(30, 0.5, popularity=0.5, trending=0.5),
            _candidate(40, 0.5, popularity=0.5, trending=0.5),
        ]
        result = ranker.rank(_candidate_set(cands, profile=profile))
        assert result.items[0].video_id == 30

    def test_skips_when_no_session_context(self, ranker: QCSARRanker) -> None:
        """Stage 5 is a no-op when there are no recent videos."""
        profile = _profile()  # no recent_video_ids → session_context is None
        cands = [_candidate(10, 0.9), _candidate(20, 0.7)]
        result = ranker.rank(_candidate_set(cands, profile=profile))
        # Should still rank by Stage 4 (query relevance dominates)
        assert result.items[0].video_id == 10


class TestStage6Constraints:
    """Stage 6: demotion, diversity, negative constraints."""

    def test_seen_videos_are_demoted(self, ranker: QCSARRanker) -> None:
        # vid 10 (seen) has slightly higher query score than vid 20 (unseen).
        # Extra anchors reduce rank spacing so the 0.5× demotion flips ordering.
        # With 6 candidates, rank-normalised values are 1.0, 0.8, 0.6, 0.4, 0.2, 0.0
        # so the gap between vid 10 (1.0) and vid 20 (0.8) is small enough.
        cands = [
            _candidate(10, 0.91, seen=1.0, popularity=0.5, trending=0.5),
            _candidate(20, 0.90, seen=0.0, popularity=0.5, trending=0.5),
            _candidate(30, 0.50, seen=0.0, popularity=0.5, trending=0.5),
            _candidate(40, 0.30, seen=0.0, popularity=0.5, trending=0.5),
            _candidate(50, 0.20, seen=0.0, popularity=0.5, trending=0.5),
            _candidate(60, 0.10, seen=0.0, popularity=0.5, trending=0.5),
        ]
        result = ranker.rank(_candidate_set(cands))
        # vid 10 seen → score × 0.5, which drops it below vid 20.
        assert result.items[0].video_id == 20

    def test_seen_video_preserved_for_specific_search(
        self, item_factors: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """SPECIFIC_SEARCH has seen_demotion=1.0 — seen videos keep their rank."""
        factors, ids = item_factors
        ranker = QCSARRanker(factors, ids, top_k=20)

        intent = _query_intent(intent_type=QueryIntentType.SPECIFIC_SEARCH)
        cands = [
            _candidate(10, 0.91, seen=1.0, popularity=0.5, trending=0.5),
            _candidate(20, 0.90, seen=0.0, popularity=0.5, trending=0.5),
            _candidate(30, 0.50, seen=0.0, popularity=0.5, trending=0.5),
            _candidate(40, 0.30, seen=0.0, popularity=0.5, trending=0.5),
            _candidate(50, 0.20, seen=0.0, popularity=0.5, trending=0.5),
            _candidate(60, 0.10, seen=0.0, popularity=0.5, trending=0.5),
        ]
        result = ranker.rank(_candidate_set(cands, intent=intent))
        # No demotion → vid 10 stays on top despite being seen.
        assert result.items[0].video_id == 10

    def test_seen_video_aggressively_demoted_for_exploration(
        self, item_factors: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """EXPLORATION has seen_demotion=0.3 — seen videos drop harder than default."""
        factors, ids = item_factors
        ranker = QCSARRanker(factors, ids, top_k=20)

        intent = _query_intent(intent_type=QueryIntentType.EXPLORATION)
        # vid 10 (seen) scores higher, but 0.3× demotion is harsh.
        cands = [
            _candidate(10, 0.91, seen=1.0, popularity=0.5, trending=0.5),
            _candidate(20, 0.90, seen=0.0, popularity=0.5, trending=0.5),
            _candidate(30, 0.50, seen=0.0, popularity=0.5, trending=0.5),
            _candidate(40, 0.30, seen=0.0, popularity=0.5, trending=0.5),
            _candidate(50, 0.20, seen=0.0, popularity=0.5, trending=0.5),
            _candidate(60, 0.10, seen=0.0, popularity=0.5, trending=0.5),
        ]
        result = ranker.rank(_candidate_set(cands, intent=intent))
        # 0.3× demotion drops vid 10 well below vid 20.
        assert result.items[0].video_id == 20
        # vid 10 should also be below vid 30 (0.3× is very aggressive)
        video_ids = [item.video_id for item in result.items]
        assert video_ids.index(10) > video_ids.index(30)

    def test_category_diversity_penalises_repeats(
        self, ranker: QCSARRanker,
    ) -> None:
        # 4 same-category candidates + 1 different-category + 1 anchor.
        # The anchor creates a normalization range. Same-cat candidates have
        # similar scores; the 15% diversity penalty on repeated categories
        # should push vid 50 (cat 2) above some penalised cat-1 entries.
        cands = [
            _candidate(10, 0.90, cat=1, popularity=0.5, trending=0.5),
            _candidate(20, 0.89, cat=1, popularity=0.5, trending=0.5),
            _candidate(30, 0.88, cat=1, popularity=0.5, trending=0.5),
            _candidate(40, 0.87, cat=1, popularity=0.5, trending=0.5),
            _candidate(50, 0.87, cat=2, popularity=0.5, trending=0.5),
            _candidate(60, 0.10, cat=3, popularity=0.5, trending=0.5),  # anchor
        ]
        result = ranker.rank(_candidate_set(cands))
        video_ids = [item.video_id for item in result.items]
        # vid 50 (cat 2, same score as 40) has no diversity penalty.
        # vid 40 (cat 1, same score) has repeated-category penalty.
        # So vid 50 should rank above vid 40.
        assert video_ids.index(50) < video_ids.index(40)

    def test_negative_constraints_penalise_duration(
        self, item_factors: tuple[np.ndarray, np.ndarray],
    ) -> None:
        factors, ids = item_factors
        ranker = QCSARRanker(factors, ids, top_k=20)

        intent = _query_intent(negative_constraints=["too long"])
        # vid 10 has slightly higher score but poor dur_fit → penalised.
        # With rank normalization, need ≥11 candidates so spacing ≤0.1 —
        # the 0.8× penalty then flips adjacent items.
        cands = [
            _candidate(10, 0.91, dur_fit=0.1, popularity=0.5, trending=0.5),
            _candidate(20, 0.90, dur_fit=0.9, popularity=0.5, trending=0.5),
        ] + [
            _candidate(100 + i, 0.10 + i * 0.08, dur_fit=0.5,
                       popularity=0.5, trending=0.5)
            for i in range(9)  # 9 anchors → 11 total → spacing 0.1
        ]
        result = ranker.rank(_candidate_set(cands, intent=intent))
        assert result.items[0].video_id == 20
