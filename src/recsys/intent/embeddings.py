"""Pluggable query embedding for semantic retrieval.

Provides a ``QueryEmbedder`` protocol and implementations:

- ``NullEmbedder`` — returns empty vectors (default when no provider is available).
- Future: ``SoclaasEmbedder`` if the SOCLaaS ``/embeddings`` endpoint is supported.
"""

from __future__ import annotations

from typing import Protocol


class QueryEmbedder(Protocol):
    """Protocol for query embedding providers."""

    def embed(self, text: str) -> list[float]:
        """Embed a single text string into a dense vector."""
        ...

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts."""
        ...


class NullEmbedder:
    """Returns empty vectors — used when no embedding provider is available."""

    def embed(self, text: str) -> list[float]:
        return []

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [[] for _ in texts]
