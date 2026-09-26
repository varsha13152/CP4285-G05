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
    RetrievalSource,
)

# ---------------------------------------------------------------------------
# User modelling (Stage 2 output, consumed by Stage 4 ranking)
# ---------------------------------------------------------------------------

@dataclass
class SessionContext:
    """What the user is doing right now — candidate-independent lookups that
    Stage 2 precomputes so Stage 4 can score each candidate with a dict lookup."""

    recent_video_ids: list[int]
    # Markov P(next = video | last watched) — sparse; mostly signals rewatches.
    next_item_scores: dict[int, float] = field(default_factory=dict)
    # Mean ALS item vector of the recent videos (compare with a candidate's item vector).
    session_vector: list[float] = field(default_factory=list)
    # Caption first-level category ids of the recent videos, most recent first.
    recent_category_ids: list[int] = field(default_factory=list)


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
    # Distinct video ids the user has already interacted with. Search does NOT
    # filter these out (users re-find videos); Stage 3 flags them as a feature.
    # Empty for cold-start users.
    seen_video_ids: set[int] = field(default_factory=set)
    # Short-term context, filled by build_session_profile when recent videos are given.
    session_context: SessionContext | None = None
    # {video_id: mean watch_ratio among the user's friends}. Empty for the ~93% of
    # users with no friends in KuaiRec — "unavailable", not "disliked".
    social_scores: dict[int, float] = field(default_factory=dict)
    # {video_id: Σ cosine(user, neighbour) · neighbour watch_ratio} over the user's
    # kNN neighbours in ALS space — the Stage 4 CF ranking signal. Empty for cold start.
    cf_neighbour_scores: dict[int, float] = field(default_factory=dict)


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
class QueryRetrievalRequest:
    """Output of query-conditioned routing — tells retrieval how to find candidates."""

    query_intent: QueryIntent
    user_id: int | None = None
    # Carried through untouched for Stage 4; Stage 3 only reads seen_video_ids (as a flag).
    engagement_profile: EngagementProfile | None = None
    num_candidates: int = 100
    # Query topic ids (caption first-level, confidence >= min_category_confidence)
    # that feed the CATEGORY retrieval source. Name kept for compatibility — it is
    # a relevance signal, NOT a hard filter.
    category_filter: list[int] | None = None
    # Soft duration target: out-of-range videos are down-weighted, never dropped.
    duration_range_ms: tuple[int, int] | None = None
    min_category_confidence: float = 0.3


# ---------------------------------------------------------------------------
# Stage 3 output: Candidate Retrieval
# ---------------------------------------------------------------------------

@dataclass
class Candidate:
    """A single retrieved candidate video."""

    video_id: int
    score: float  # Stage 3 query-relevance score (not a final ranking score)
    source: RetrievalSource
    category_ids: list[int] = field(default_factory=list)
    video_duration_ms: int = 0
    # Ranking features for Stage 4: sem, cat, dur_fit, has_text, seen,
    # popularity, trending (see recsys.retrieval.retriever).
    features: dict[str, float] = field(default_factory=dict)


@dataclass
class CandidateSet:
    """Output of candidate retrieval — a pool of candidates to rank."""

    user_id: int | None
    request: QueryRetrievalRequest
    candidates: list[Candidate]
    source_counts: dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Stage 4-6 output: Ranking
# ---------------------------------------------------------------------------

@dataclass
class RankedItem:
    """A single item in the final ranked list."""

    video_id: int
    rank: int
    score: float
    source: RetrievalSource
    category_ids: list[int] = field(default_factory=list)


@dataclass
class RankedList:
    """Output of ranking — the final recommendation."""

    user_id: int | None
    items: list[RankedItem]
    num_candidates_considered: int
    intent_type: QueryIntentType
