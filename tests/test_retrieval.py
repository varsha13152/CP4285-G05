"""Tests for Stage 3 HybridRetriever on a hand-computable 5-video catalogue."""

import numpy as np
import pandas as pd
import pytest

from recsys.contracts.enums import (
    ContentStyle,
    DurationPreference,
    QueryIntentType,
    RetrievalSource,
)
from recsys.contracts.models import (
    CategoryMatch,
    EngagementProfile,
    QueryIntent,
    QueryRetrievalRequest,
)
from recsys.retrieval import HybridRetriever

# video: category, duration_ms, has_text, 2-d caption embedding
#   1: Food(12)   8 s  text  [1, 0]    exact semantic match, in duration range
#   2: Food(12)  90 s  text  [0, 1]    category match, far outside duration range
#   3: Comedy(9)  8 s  text  [1, 0]    semantic match, other category
#   4: Food(12)   5 s  none  [0, 0]    no text: only the category source can find it
#   5: unknown   n/a   text  [.6, .8]  partial semantic match, unknown duration
ITEMS = pd.DataFrame(
    {
        "category_id": [12, 12, 9, 12, -1],
        "duration_ms": [8000, 90000, 8000, 5000, 0],
        "has_text": [True, True, True, False, True],
        "popularity": [10.0, 20.0, 30.0, 40.0, 50.0],
        "trending": [0.0, 0.1, 0.2, 0.3, 0.4],
    },
    index=pd.Index([1, 2, 3, 4, 5], name="video_id"),
)
EMB = np.array([[1, 0], [0, 1], [1, 0], [0, 0], [0.6, 0.8]], dtype=np.float32)


def _request(embedding=(1.0, 0.0), num_candidates=100, profile=None):
    intent = QueryIntent(
        raw_query="short cooking videos",
        topics=[CategoryMatch(12, "美食", "Food", 0.9)],
        style=ContentStyle.NEUTRAL,
        duration=DurationPreference.SHORT,
        intent_type=QueryIntentType.SPECIFIC_SEARCH,
        negative_constraints=[],
        semantic_summary="short cooking videos",
        query_embedding=list(embedding) if embedding else None,
    )
    return QueryRetrievalRequest(
        query_intent=intent,
        user_id=None if profile is None else profile.user_id,
        engagement_profile=profile,
        num_candidates=num_candidates,
        category_filter=[12],
        duration_range_ms=(0, 15_000),
    )


def _ids(mode, **kw):
    out = HybridRetriever(ITEMS, EMB, mode=mode).retrieve(_request(**kw))
    return [c.video_id for c in out.candidates]


def test_semantic_mode_needs_text_and_breaks_ties_by_id():
    # 1 and 3 tie at 1.2 -> id order; 4 has no text; 2 is last (far outside duration)
    assert _ids("semantic") == [1, 3, 5, 2]


def test_category_mode_finds_textless_videos():
    # 1 and 4 tie at 0.45 + 0.2; 2 is kept (soft duration) but ranked last
    assert _ids("category") == [1, 4, 2]


def test_hybrid_mode_unions_both_sources():
    assert _ids("hybrid") == [1, 3, 5, 4, 2]


def test_num_candidates_truncates():
    assert _ids("hybrid", num_candidates=2) == [1, 3]


def test_missing_query_embedding_falls_back_to_category():
    assert _ids("semantic", embedding=None) == _ids("category")


def test_scores_features_and_sources():
    out = HybridRetriever(ITEMS, EMB, mode="hybrid").retrieve(_request())
    by_id = {c.video_id: c for c in out.candidates}
    assert by_id[1].score == pytest.approx(1.0 + 0.5 * 0.9 + 0.2)
    assert by_id[1].source == RetrievalSource.BOTH
    assert by_id[4].source == RetrievalSource.CATEGORY
    assert by_id[3].source == RetrievalSource.SEMANTIC
    assert by_id[2].features["dur_fit"] == pytest.approx(np.exp(-75 / 15))
    assert by_id[5].features["dur_fit"] == 1.0  # unknown duration is not penalised
    assert by_id[5].category_ids == [] and by_id[1].category_ids == [12]
    assert by_id[3].features["popularity"] == 30.0  # attached for Stage 4, not used here
    assert out.source_counts == {"BOTH": 2, "SEMANTIC": 2, "CATEGORY": 1}


def test_seen_videos_are_flagged_not_removed():
    profile = EngagementProfile(
        user_id=7, avg_watch_ratio=1.0, watch_ratio_std=0.0, total_interactions=1,
        active_days=1, top_category_ids=[], category_distribution={},
        avg_session_length=1.0, recency_days=0, seen_video_ids={1},
    )
    out = HybridRetriever(ITEMS, EMB).retrieve(_request(profile=profile))
    by_id = {c.video_id: c for c in out.candidates}
    assert by_id[1].features["seen"] == 1.0 and by_id[3].features["seen"] == 0.0
    assert out.user_id == 7


def test_cold_start_query_only_still_works():
    out = HybridRetriever(ITEMS, EMB).retrieve(_request())  # no user, no profile
    assert out.user_id is None and len(out.candidates) == 5


def test_rejects_bad_mode_and_misaligned_embeddings():
    with pytest.raises(ValueError):
        HybridRetriever(ITEMS, EMB, mode="popularity")
    with pytest.raises(ValueError):
        HybridRetriever(ITEMS, EMB[:3])
