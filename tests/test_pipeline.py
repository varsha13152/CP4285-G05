"""Tests for the pipeline orchestrator's Stage 1 → 2 → router → 3 wiring."""

import pytest

from recsys.contracts.models import CandidateSet
from recsys.pipeline.pipeline import RecommendationPipeline
from recsys.routing import QueryRouter


class _Parser:
    def __init__(self, intent):
        self.intent = intent

    def parse(self, query):
        return self.intent


class _Profiler:
    def __init__(self, profile):
        self.profile, self.calls = profile, []

    def build_session_profile(self, user_id, query_intent, recent_video_ids=None):
        self.calls.append((user_id, recent_video_ids))
        return self.profile


class _Retriever:
    def retrieve(self, request):
        self.request = request
        return CandidateSet(user_id=request.user_id, request=request, candidates=[])


def _pipeline(sample_query_intent, sample_engagement_profile):
    profiler, retriever = _Profiler(sample_engagement_profile), _Retriever()
    pipe = RecommendationPipeline(
        _Parser(sample_query_intent), QueryRouter(), retriever, user_profiler=profiler
    )
    return pipe, profiler, retriever


def test_retrieve_passes_stage2_profile_to_retrieval(
    sample_query_intent, sample_engagement_profile
):
    pipe, profiler, retriever = _pipeline(sample_query_intent, sample_engagement_profile)
    out = pipe.retrieve("fun cooking videos", user_id=1, recent_video_ids=[10, 20])
    assert profiler.calls == [(1, [10, 20])]
    assert retriever.request.engagement_profile is sample_engagement_profile
    assert out.user_id == 1


def test_anonymous_search_skips_stage2(sample_query_intent, sample_engagement_profile):
    pipe, profiler, retriever = _pipeline(sample_query_intent, sample_engagement_profile)
    pipe.retrieve("fun cooking videos")
    assert profiler.calls == []
    assert retriever.request.engagement_profile is None


def test_recommend_requires_a_ranker(sample_query_intent, sample_engagement_profile):
    pipe, _, _ = _pipeline(sample_query_intent, sample_engagement_profile)
    with pytest.raises(NotImplementedError):
        pipe.recommend("fun cooking videos")
