"""LLM-based query parser for QCSAR Stage 1.

Parses natural-language search queries into structured ``QueryIntent``
using the SOCLaaS LLM API (OpenAI-compatible).

Environment variables (loaded from ``.env`` via python-dotenv):
    SOCLAAS_BASE_URL  — API endpoint
    SOCLAAS_API_KEY   — API key
    SOCLAAS_MODEL     — Model name

Usage::

    from recsys.intent import QueryParser

    parser = QueryParser()
    intent = parser.parse("fun cooking videos for a student dinner")
    print(intent.topics)           # [CategoryMatch(12, "美食", "Food", 0.9)]
    print(intent.style)            # ContentStyle.FUNNY
    print(intent.intent_type)      # QueryIntentType.ENTERTAINMENT
"""

from __future__ import annotations

import json
import logging
import os

from dotenv import load_dotenv
from openai import OpenAI

from recsys.contracts.enums import ContentStyle, DurationPreference, QueryIntentType
from recsys.contracts.models import CategoryMatch, QueryIntent
from recsys.data.category_index import CategoryIndex

load_dotenv()

logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "https://soclaas-api.comp.nus.edu.sg/v1"

# Maps enum name strings to Python enum values for safe parsing
_STYLE_MAP = {s.name: s for s in ContentStyle}
_DURATION_MAP = {d.name: d for d in DurationPreference}
_INTENT_MAP = {i.name: i for i in QueryIntentType}

SYSTEM_PROMPT = """\
You are a query understanding system for a Chinese short-video platform (Kuaishou).
The user types a search query. Parse it into structured intent.

## Available Categories (ID | Chinese | English)
{category_index_string}

## Output Schema
Respond with valid JSON only — no markdown, no code fences:
{{
  "topics": [{{"category_id": <int>, "confidence": <0.0-1.0>}}],
  "style": "<FUNNY|SERIOUS|RELAXING|ENERGETIC|INFORMATIVE|DRAMATIC|CUTE|AESTHETIC|NEUTRAL>",
  "duration": "<SHORT|MEDIUM|LONG|ANY>",
  "intent_type": "<ENTERTAINMENT|LEARNING|INSPIRATION|SOCIAL|SPECIFIC_SEARCH|EXPLORATION>",
  "negative_constraints": ["<things to exclude>"],
  "semantic_summary": "<one-line summary of what the user wants>"
}}

## Rules
1. Map query to 1-5 categories from the list. Use category IDs.
2. Clear category match → confidence >= 0.8. Vague → lower confidence.
3. Style = desired TONE, not topic.
4. Duration: SHORT = quick clips, MEDIUM = normal, LONG = deep content.
5. negative_constraints: only explicit exclusions user mentioned.
6. semantic_summary: clear one-sentence description for retrieval.
"""


def _default_query_intent(query: str) -> QueryIntent:
    """Return a safe fallback ``QueryIntent`` when LLM parsing fails."""
    return QueryIntent(
        raw_query=query,
        topics=[],
        style=ContentStyle.NEUTRAL,
        duration=DurationPreference.ANY,
        intent_type=QueryIntentType.EXPLORATION,
        negative_constraints=[],
        semantic_summary=query,
        confidence=0.0,
    )


class QueryParser:
    """Parse natural-language queries into structured ``QueryIntent`` via SOCLaaS.

    Parameters
    ----------
    model : str | None
        LLM model name.  Falls back to ``SOCLAAS_MODEL`` env var, then ``"default"``.
    base_url : str | None
        API base URL.  Falls back to ``SOCLAAS_BASE_URL`` env var.
    api_key : str | None
        API key.  Falls back to ``SOCLAAS_API_KEY`` env var.
    category_index : CategoryIndex | None
        Bilingual category index.  Created automatically if not supplied.
    """

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        category_index: CategoryIndex | None = None,
    ) -> None:
        self._base_url = base_url or os.environ.get("SOCLAAS_BASE_URL", _DEFAULT_BASE_URL)
        self._api_key = api_key or os.environ.get("SOCLAAS_API_KEY", "")
        self._model = model or os.environ.get("SOCLAAS_MODEL", "default")

        if not self._api_key:
            raise ValueError("No API key provided. Set SOCLAAS_API_KEY or pass api_key=.")

        self._client = OpenAI(base_url=self._base_url, api_key=self._api_key)
        self._index = category_index or CategoryIndex()
        self._system_prompt = SYSTEM_PROMPT.format(
            category_index_string=self._index.to_prompt_string(),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse(self, query: str) -> QueryIntent:
        """Parse a single natural-language query into a ``QueryIntent``."""
        response = self._client.chat.completions.create(
            model=self._model,
            max_tokens=512,
            messages=[
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": query},
            ],
        )

        text = response.choices[0].message.content or "{}"

        try:
            raw = json.loads(text)
        except json.JSONDecodeError:
            logger.warning("LLM returned invalid JSON for query %r", query)
            return _default_query_intent(query)

        return self._build_intent(query, raw)

    def parse_batch(self, queries: list[str]) -> list[QueryIntent]:
        """Parse multiple queries sequentially."""
        return [self.parse(q) for q in queries]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_intent(self, query: str, raw: dict) -> QueryIntent:
        """Validate and convert raw LLM JSON into a ``QueryIntent``."""
        # --- topics ---
        topics: list[CategoryMatch] = []
        valid_ids = self._index.valid_ids()
        for t in raw.get("topics", []):
            cat_id = t.get("category_id")
            if cat_id not in valid_ids:
                logger.debug("Dropping unknown category_id %s", cat_id)
                continue
            entry = self._index.get_by_id(cat_id)
            if entry is None:
                continue
            confidence = max(0.0, min(1.0, float(t.get("confidence", 0.5))))
            topics.append(CategoryMatch(
                category_id=cat_id,
                chinese_name=entry.chinese_name,
                english_name=entry.english_name,
                confidence=confidence,
            ))

        # --- enums with fallbacks ---
        style = _STYLE_MAP.get(
            raw.get("style", "").upper(), ContentStyle.NEUTRAL,
        )
        duration = _DURATION_MAP.get(
            raw.get("duration", "").upper(), DurationPreference.ANY,
        )
        intent_type = _INTENT_MAP.get(
            raw.get("intent_type", "").upper(), QueryIntentType.EXPLORATION,
        )

        # --- other fields ---
        negative_constraints: list[str] = [
            str(c) for c in raw.get("negative_constraints", []) if c
        ]
        semantic_summary: str = raw.get("semantic_summary", query)

        return QueryIntent(
            raw_query=query,
            topics=topics,
            style=style,
            duration=duration,
            intent_type=intent_type,
            negative_constraints=negative_constraints,
            semantic_summary=semantic_summary,
            confidence=0.9 if topics else 0.5,
            llm_raw_response=raw,
        )
