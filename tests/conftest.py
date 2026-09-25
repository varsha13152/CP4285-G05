"""Shared test fixtures — sample DataFrames and QCSAR types for unit tests."""

import pytest
import pandas as pd

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
    RankedItem,
    RankedList,
)


@pytest.fixture
def sample_interactions():
    """Tiny interaction DataFrame for unit tests (no disk I/O)."""
    return pd.DataFrame({
        "user_id": [1, 1, 1, 1, 2, 2, 2, 3, 3, 3],
        "video_id": [10, 20, 30, 40, 10, 40, 50, 20, 30, 60],
        "watch_ratio": [0.9, 0.1, 0.5, 0.8, 0.8, 0.3, 0.95, 0.2, 0.4, 0.1],
        "play_duration": [5000, 1000, 3000, 4000, 4000, 2000, 6000, 1500, 2500, 800],
        "video_duration": [5500, 10000, 6000, 5000, 5000, 7000, 6200, 8000, 6000, 9000],
        "timestamp": [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
        "date": [20200101, 20200102, 20200103, 20200104,
                 20200101, 20200102, 20200103,
                 20200101, 20200102, 20200103],
    })


@pytest.fixture
def sample_item_categories():
    return pd.DataFrame({
        "video_id": [10, 20, 30, 40, 50, 60],
        "feat": [[1, 5], [2, 5], [1, 3], [2, 7], [1, 5], [3, 7]],
    })


@pytest.fixture
def sample_social_network():
    return pd.DataFrame({
        "user_id": [1, 2, 3],
        "friend_list": [[2, 3], [1], [1, 2]],
    })


@pytest.fixture
def sample_engagement_profile():
    return EngagementProfile(
        user_id=1,
        avg_watch_ratio=0.575,
        watch_ratio_std=0.34,
        total_interactions=4,
        active_days=4,
        top_category_ids=[1, 5, 2, 7],
        category_distribution={1: 0.3, 5: 0.3, 2: 0.2, 7: 0.2},
        avg_session_length=1.0,
        recency_days=1,
    )


@pytest.fixture
def sample_query_intent():
    """A sample parsed QueryIntent for routing / downstream tests."""
    return QueryIntent(
        raw_query="fun cooking videos",
        topics=[
            CategoryMatch(12, "美食", "Food", 0.9),
            CategoryMatch(9, "喜剧", "Comedy", 0.7),
        ],
        style=ContentStyle.FUNNY,
        duration=DurationPreference.SHORT,
        intent_type=QueryIntentType.ENTERTAINMENT,
        negative_constraints=[],
        semantic_summary="Fun short cooking videos with comedic tone",
        confidence=0.9,
    )


@pytest.fixture
def sample_retrieval_request(sample_query_intent):
    return QueryRetrievalRequest(
        query_intent=sample_query_intent,
        user_id=1,
        num_candidates=50,
        category_filter=[12, 9],
    )


@pytest.fixture
def sample_candidate_set(sample_retrieval_request):
    candidates = [
        Candidate(video_id=10, score=0.9, source=RetrievalSource.BOTH,
                  category_ids=[1, 5]),
        Candidate(video_id=20, score=0.7, source=RetrievalSource.SEMANTIC,
                  category_ids=[2, 5]),
        Candidate(video_id=30, score=0.5, source=RetrievalSource.BOTH,
                  category_ids=[1, 3]),
    ]
    return CandidateSet(
        user_id=1,
        request=sample_retrieval_request,
        candidates=candidates,
    )


@pytest.fixture
def sample_ranked_list():
    items = [
        RankedItem(video_id=10, rank=1, score=0.95,
                   source=RetrievalSource.BOTH, category_ids=[1, 5]),
        RankedItem(video_id=20, rank=2, score=0.80,
                   source=RetrievalSource.SEMANTIC, category_ids=[2, 5]),
        RankedItem(video_id=30, rank=3, score=0.60,
                   source=RetrievalSource.BOTH, category_ids=[1, 3]),
    ]
    return RankedList(
        user_id=1,
        items=items,
        num_candidates_considered=10,
        intent_type=QueryIntentType.ENTERTAINMENT,
    )
