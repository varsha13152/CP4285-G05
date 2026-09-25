"""Enumerations shared across all pipeline stages."""

from enum import Enum, auto


class QueryIntentType(Enum):
    """What the user's search query is trying to accomplish."""

    ENTERTAINMENT = auto()
    LEARNING = auto()
    INSPIRATION = auto()
    SOCIAL = auto()
    SPECIFIC_SEARCH = auto()
    EXPLORATION = auto()


class ContentStyle(Enum):
    """Desired content style / tone."""

    FUNNY = auto()
    SERIOUS = auto()
    RELAXING = auto()
    ENERGETIC = auto()
    INFORMATIVE = auto()
    DRAMATIC = auto()
    CUTE = auto()
    AESTHETIC = auto()
    NEUTRAL = auto()


class DurationPreference(Enum):
    """Desired video length."""

    SHORT = auto()      # < 15 seconds
    MEDIUM = auto()     # 15-60 seconds
    LONG = auto()       # > 60 seconds
    ANY = auto()        # No preference


class RetrievalSource(Enum):
    """How Stage 3 found a candidate (the query-first cascade has no lanes).

    Personalisation (CF, session, social, popularity) is a Stage 4 ranking
    concern, so every candidate comes from query relevance alone.
    """

    SEMANTIC = auto()  # caption embedding matched the query embedding
    CATEGORY = auto()  # caption category matched a query topic; video has no text
    BOTH = auto()      # category match and the video has text for semantic scoring
