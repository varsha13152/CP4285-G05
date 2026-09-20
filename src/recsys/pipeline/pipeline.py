"""QCSAR recommendation pipeline — protocol interfaces and orchestrator.

Defines the contracts each pipeline stage must satisfy and provides
a ``RecommendationPipeline`` that wires them together::

    query  →  QueryIntentPredictor.parse()   →  QueryIntent
           →  QueryConditionedRouter.route() →  QueryRetrievalRequest
           →  Retriever.retrieve()           →  CandidateSet
           →  Ranker.rank()                  →  RankedList
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


class QueryConditionedRouter(Protocol):
    """Route a QueryIntent into a retrieval request."""

    def route(
        self,
        query_intent: QueryIntent,
        user_id: int | None = None,
        engagement_profile: EngagementProfile | None = None,
    ) -> QueryRetrievalRequest: ...


class Retriever(Protocol):
    """Stage 3: generate candidate videos from a retrieval request."""

    def retrieve(self, request: QueryRetrievalRequest) -> CandidateSet: ...


class Ranker(Protocol):
    """Stages 4-6: rank, re-rank, and apply constraints to candidates."""

    def rank(self, candidates: CandidateSet) -> RankedList: ...


# ---------------------------------------------------------------------------
# Pipeline orchestrator
# ---------------------------------------------------------------------------

class RecommendationPipeline:
    """QCSAR pipeline: query → intent → routing → retrieval → ranking.

    Parameters
    ----------
    parser : QueryIntentPredictor
        LLM query parser (Stage 1).
    router : QueryConditionedRouter
        Query-conditioned router.
    retriever : Retriever
        Candidate generation (Stage 3).
    ranker : Ranker
        Ranking and re-ranking (Stages 4-6).
    """

    def __init__(
        self,
        parser: QueryIntentPredictor,
        router: QueryConditionedRouter,
        retriever: Retriever,
        ranker: Ranker,
    ) -> None:
        self._parser = parser
        self._router = router
        self._retriever = retriever
        self._ranker = ranker

    def recommend(
        self,
        query: str,
        user_id: int | None = None,
        context: dict | None = None,
    ) -> RankedList:
        """Run the full pipeline for a natural-language query.

        Parameters
        ----------
        query : str
            Natural-language search query (e.g. "fun cooking videos").
        user_id : int | None
            Optional user ID for personalisation.
        context : dict | None
            Extra context (time of day, device, etc.) — reserved for future use.
        """
        intent = self._parser.parse(query)
        request = self._router.route(intent, user_id=user_id)
        candidates = self._retriever.retrieve(request)
        return self._ranker.rank(candidates)
