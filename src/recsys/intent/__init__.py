"""Intent understanding — parse natural-language queries into structured intent."""

from recsys.intent.embeddings import NullEmbedder, QueryEmbedder
from recsys.intent.query_parser import QueryParser

__all__ = ["NullEmbedder", "QueryEmbedder", "QueryParser"]
