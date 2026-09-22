"""Dataclass contracts for data flowing between pipeline stages.

Every stage imports its input/output types from here. Do not define
pipeline-level types anywhere else.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from recsys.contracts.enums import (
    ContentStyle,
    DurationPreference,
    QueryIntentType,
    RetrievalStrategy,
)


# ---------------------------------------------------------------------------
# User modelling (consumed by Stage 2)
# ---------------------------------------------------------------------------

@dataclass
class EngagementProfile:
    """Summary of a user's historical engagement patterns."""

    user_id: int
    avg_watch_ratio: float
    watch_ratio_std: float
    total_interactions: int
    active_days: int
    top_category_ids: list[int]
    category_distribution: dict[int, float]
    avg_session_length: float
    recency_days: int
    # Learned long-term (u_long) latent factor vector from matrix factorization
    # (implicit ALS over big_matrix). Empty for unknown/cold-start users.
    # Consumed downstream purely as a ranking signal, never as a hard filter.
    user_vector: list[float] = field(default_factory=list)
    # Distinct video ids the user has already interacted with — for filtering
    # already-seen items out of generated candidates. Empty for cold-start users.
    seen_video_ids: set[int] = field(default_factory=set)


# ---------------------------------------------------------------------------
# Stage 1 output: LLM Intent Understanding
# ---------------------------------------------------------------------------

@dataclass
class CategoryMatch:
    """A matched KuaiRec category with confidence."""

    category_id: int
    chinese_name: str
    english_name: str
    confidence: float  # 0.0-1.0


@dataclass
class QueryIntent:
    """Stage 1 output: structured understanding of a natural-language query.

    This is the primary output consumed by Stage 2 (user/session modelling)
    and Stage 3 (candidate generation).
    """

    raw_query: str
    topics: list[CategoryMatch]
    style: ContentStyle
    duration: DurationPreference
    intent_type: QueryIntentType
    negative_constraints: list[str]
    semantic_summary: str
    query_embedding: list[float] | None = None
    confidence: float = 0.0
    llm_raw_response: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Routing output
# ---------------------------------------------------------------------------

@dataclass
class StrategyWeight:
    """A retrieval strategy and its relative weight for a request."""

    strategy: RetrievalStrategy
    weight: float  # 0.0-1.0, weights should sum to 1.0


@dataclass
class QueryRetrievalRequest:
    """Output of query-conditioned routing — tells retrieval how to find candidates."""

    query_intent: QueryIntent
    user_id: int | None = None
    engagement_profile: EngagementProfile | None = None
    strategies: list[StrategyWeight] = field(default_factory=list)
    num_candidates: int = 100
    category_filter: list[int] | None = None
    duration_range_ms: tuple[int, int] | None = None
    min_category_confidence: float = 0.3


# ---------------------------------------------------------------------------
# Stage 3 output: Candidate Retrieval
# ---------------------------------------------------------------------------

@dataclass
class Candidate:
    """A single retrieved candidate video."""

    video_id: int
    score: float
    source_strategy: RetrievalStrategy
    category_ids: list[int] = field(default_factory=list)
    video_duration_ms: int = 0
    features: dict[str, float] = field(default_factory=dict)


@dataclass
class CandidateSet:
    """Output of candidate retrieval — a pool of candidates to rank."""

    user_id: int
    request: QueryRetrievalRequest
    candidates: list[Candidate]
    strategy_counts: dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Stage 4-6 output: Ranking
# ---------------------------------------------------------------------------

@dataclass
class RankedItem:
    """A single item in the final ranked list."""

    video_id: int
    rank: int
    score: float
    source_strategy: RetrievalStrategy
    category_ids: list[int] = field(default_factory=list)


@dataclass
class RankedList:
    """Output of ranking — the final recommendation."""

    user_id: int
    items: list[RankedItem]
    num_candidates_considered: int
    intent_type: QueryIntentType
