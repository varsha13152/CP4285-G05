"""Shared interface types for all pipeline stages."""

from recsys.contracts.enums import (
    ContentStyle,
    DurationPreference,
    QueryIntentType,
    RetrievalStrategy,
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
    StrategyWeight,
)

__all__ = [
    "ContentStyle",
    "DurationPreference",
    "QueryIntentType",
    "RetrievalStrategy",
    "Candidate",
    "CandidateSet",
    "CategoryMatch",
    "EngagementProfile",
    "QueryIntent",
    "QueryRetrievalRequest",
    "RankedItem",
    "RankedList",
    "StrategyWeight",
]
