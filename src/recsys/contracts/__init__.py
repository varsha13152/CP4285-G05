"""Shared interface types for all pipeline stages."""

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
    SessionContext,
)

__all__ = [
    "Candidate",
    "CandidateSet",
    "CategoryMatch",
    "ContentStyle",
    "DurationPreference",
    "EngagementProfile",
    "QueryIntent",
    "QueryIntentType",
    "QueryRetrievalRequest",
    "RankedItem",
    "RankedList",
    "RetrievalSource",
    "SessionContext",
]
