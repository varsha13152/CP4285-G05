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


class RetrievalStrategy(Enum):
    """Available candidate retrieval strategies."""

    COLLABORATIVE = auto()
    CONTENT_BASED = auto()
    POPULARITY = auto()
    SOCIAL = auto()
    SESSION = auto()  # short-term in-session next-item lane (see session_modelling)
