"""QCSAR search pipeline — protocol interfaces and orchestrator.

Defines the contracts each pipeline stage must satisfy and provides
a ``RecommendationPipeline`` that wires them together (query-first cascade)::

    query            →  QueryIntentPredictor.parse()            →  QueryIntent
    user_id + recent →  UserProfiler.build_session_profile()    →  EngagementProfile
                     →  QueryConditionedRouter.route()          →  QueryRetrievalRequest
                     →  Retriever.retrieve()  (query-only)      →  CandidateSet
                     →  Ranker.rank()  (personalised, Stage 4)  →  RankedList
"""

from __future__ import annotations

from typing import Protocol

from recsys.contracts.models import (
    CandidateSet,
    EngagementProfile,
    QueryIntent,
    QueryRetrievalRequest,
    RankedList,
)

# ---------------------------------------------------------------------------
# Stage protocols — each teammate implements one
# ---------------------------------------------------------------------------

class QueryIntentPredictor(Protocol):
    """Stage 1: parse a natural-language query into structured intent."""

    def parse(self, query: str) -> QueryIntent: ...


class UserProfiler(Protocol):
    """Stage 2: long-term profile + user-side lookups for Stage 4."""

    def build_session_profile(
        self,
        user_id: int,
        query_intent: QueryIntent,
        recent_video_ids: list[int] | None = None,
    ) -> EngagementProfile: ...


class QueryConditionedRouter(Protocol):
    """Route a QueryIntent into a retrieval request."""

    def route(
        self,
        query_intent: QueryIntent,
        user_id: int | None = None,
        engagement_profile: EngagementProfile | None = None,
    ) -> QueryRetrievalRequest: ...


class Retriever(Protocol):
    """Stage 3: generate query-relevant candidate videos from a retrieval request."""

    def retrieve(self, request: QueryRetrievalRequest) -> CandidateSet: ...


class Ranker(Protocol):
    """Stages 4-6: rank, re-rank, and apply constraints to candidates."""

    def rank(self, candidates: CandidateSet) -> RankedList: ...


# ---------------------------------------------------------------------------
# Pipeline orchestrator
# ---------------------------------------------------------------------------

class RecommendationPipeline:
    """QCSAR pipeline: query → intent → profile → routing → retrieval → ranking.

    Parameters
    ----------
    parser : QueryIntentPredictor
        LLM query parser (Stage 1).
    router : QueryConditionedRouter
        Query-conditioned router.
    retriever : Retriever
        Candidate retrieval (Stage 3).
    user_profiler : UserProfiler | None
        Stage 2 (e.g. ``UserModeller``). ``None`` → no profile; retrieval is
        query-only anyway, so anonymous search still works.
    ranker : Ranker | None
        Ranking and re-ranking (Stages 4-6). Not implemented yet — without it
        use :meth:`retrieve`; :meth:`recommend` requires one.
    """

    def __init__(
        self,
        parser: QueryIntentPredictor,
        router: QueryConditionedRouter,
        retriever: Retriever,
        user_profiler: UserProfiler | None = None,
        ranker: Ranker | None = None,
    ) -> None:
        self._parser = parser
        self._router = router
        self._retriever = retriever
        self._user_profiler = user_profiler
        self._ranker = ranker

    def retrieve(
        self,
        query: str,
        user_id: int | None = None,
        recent_video_ids: list[int] | None = None,
    ) -> CandidateSet:
        """Run Stages 1–3 and return the candidate pool for Stage 4.

        Parameters
        ----------
        query : str
            Natural-language search query (e.g. "fun cooking videos").
        user_id : int | None
            Optional user ID; enables the Stage 2 profile.
        recent_video_ids : list[int] | None
            The user's current-session videos (oldest → newest) for
            ``EngagementProfile.session_context``.
        """
        intent = self._parser.parse(query)
        profile = None
        if user_id is not None and self._user_profiler is not None:
            profile = self._user_profiler.build_session_profile(
                user_id, intent, recent_video_ids=recent_video_ids
            )
        request = self._router.route(intent, user_id=user_id, engagement_profile=profile)
        return self._retriever.retrieve(request)

    def recommend(
        self,
        query: str,
        user_id: int | None = None,
        recent_video_ids: list[int] | None = None,
    ) -> RankedList:
        """Run the full pipeline (Stages 1–4)."""
        if self._ranker is None:
            raise NotImplementedError(
                "No Stage 4 ranker configured; use retrieve() for the candidate pool."
            )
        return self._ranker.rank(self.retrieve(query, user_id, recent_video_ids))
