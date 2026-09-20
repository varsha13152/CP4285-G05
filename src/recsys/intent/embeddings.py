"""Pluggable query embedding for semantic retrieval.

Provides a ``QueryEmbedder`` protocol and implementations:

- ``NullEmbedder`` — returns empty vectors (default when no provider is available).
- ``BgeM3Embedder`` — uses the SOCLaaS ``bge-m3`` model (1024-dim, multilingual).

The ``bge-m3`` model is especially well-suited for QCSAR because it handles
cross-lingual similarity — English queries produce embeddings that are close
to Chinese video captions / category names in the same vector space.
"""

from __future__ import annotations

import logging
import os
from typing import Protocol

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "https://soclaas-api.comp.nus.edu.sg/v1"
_DEFAULT_EMBEDDING_MODEL = "bge-m3"


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


class BgeM3Embedder:
    """Embed text via SOCLaaS ``bge-m3`` (1024-dim multilingual embeddings).

    Uses the same SOCLaaS API credentials as the query parser, but targets
    the ``bge-m3`` model which supports the ``/embeddings`` endpoint.

    Parameters
    ----------
    model : str | None
        Embedding model name. Defaults to ``bge-m3``.
    base_url : str | None
        API base URL. Falls back to ``SOCLAAS_BASE_URL`` env var.
    api_key : str | None
        API key. Falls back to ``SOCLAAS_API_KEY`` env var.

    Examples
    --------
    >>> embedder = BgeM3Embedder()
    >>> vec = embedder.embed("cute cat videos")
    >>> len(vec)
    1024
    """

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self._base_url = base_url or os.environ.get("SOCLAAS_BASE_URL", _DEFAULT_BASE_URL)
        self._api_key = api_key or os.environ.get("SOCLAAS_API_KEY", "")
        self._model = model or _DEFAULT_EMBEDDING_MODEL

        if not self._api_key:
            raise ValueError("No API key provided. Set SOCLAAS_API_KEY or pass api_key=.")

        self._client = OpenAI(base_url=self._base_url, api_key=self._api_key)

    def embed(self, text: str) -> list[float]:
        """Embed a single text string into a 1024-dim dense vector."""
        try:
            response = self._client.embeddings.create(
                model=self._model,
                input=[text],
            )
            return response.data[0].embedding
        except Exception:
            logger.warning("Embedding failed for text %r, returning empty", text[:50])
            return []

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts in a single API call."""
        if not texts:
            return []
        try:
            response = self._client.embeddings.create(
                model=self._model,
                input=texts,
            )
            return [d.embedding for d in response.data]
        except Exception:
            logger.warning("Batch embedding failed for %d texts, returning empties", len(texts))
            return [[] for _ in texts]
