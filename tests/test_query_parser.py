"""Tests for the LLM query parser.

These are offline tests — they mock the SOCLaaS API so no real LLM
calls are made during CI/local testing.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from recsys.contracts.enums import ContentStyle, DurationPreference, QueryIntentType
from recsys.contracts.models import QueryIntent
from recsys.intent.embeddings import BgeM3Embedder, NullEmbedder
from recsys.intent.query_parser import QueryParser, _default_query_intent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_llm_response(raw_dict: dict) -> MagicMock:
    """Create a mock OpenAI chat completion response."""
    choice = MagicMock()
    choice.message.content = json.dumps(raw_dict)
    response = MagicMock()
    response.choices = [choice]
    return response


@pytest.fixture
def parser() -> QueryParser:
    """Return a QueryParser with a mocked OpenAI client."""
    with patch("recsys.intent.query_parser.OpenAI") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        p = QueryParser(api_key="test-key")
        p._client = mock_client
        return p


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestDefaultQueryIntent:
    def test_returns_neutral_fallback(self) -> None:
        result = _default_query_intent("hello")
        assert result.raw_query == "hello"
        assert result.style == ContentStyle.NEUTRAL
        assert result.duration == DurationPreference.ANY
        assert result.intent_type == QueryIntentType.EXPLORATION
        assert result.confidence == 0.0
        assert result.topics == []


class TestQueryParser:
    def test_parses_valid_response(self, parser: QueryParser) -> None:
        raw = {
            "topics": [{"category_id": 12, "confidence": 0.9}],
            "style": "FUNNY",
            "duration": "SHORT",
            "intent_type": "ENTERTAINMENT",
            "negative_constraints": [],
            "semantic_summary": "fun cooking videos",
        }
        parser._client.chat.completions.create.return_value = _mock_llm_response(raw)

        result = parser.parse("fun cooking videos")
        assert isinstance(result, QueryIntent)
        assert result.raw_query == "fun cooking videos"
        assert len(result.topics) == 1
        assert result.topics[0].category_id == 12
        assert result.topics[0].english_name == "Food"
        assert result.topics[0].chinese_name == "美食"
        assert result.style == ContentStyle.FUNNY
        assert result.duration == DurationPreference.SHORT
        assert result.intent_type == QueryIntentType.ENTERTAINMENT
        assert result.confidence == 0.9

    def test_multiple_topics(self, parser: QueryParser) -> None:
        raw = {
            "topics": [
                {"category_id": 12, "confidence": 0.9},
                {"category_id": 9, "confidence": 0.7},
            ],
            "style": "FUNNY",
            "duration": "ANY",
            "intent_type": "ENTERTAINMENT",
            "negative_constraints": [],
            "semantic_summary": "funny food content",
        }
        parser._client.chat.completions.create.return_value = _mock_llm_response(raw)

        result = parser.parse("funny food comedy")
        assert len(result.topics) == 2
        ids = {t.category_id for t in result.topics}
        assert ids == {12, 9}

    def test_invalid_json_falls_back(self, parser: QueryParser) -> None:
        choice = MagicMock()
        choice.message.content = "this is not json"
        response = MagicMock()
        response.choices = [choice]
        parser._client.chat.completions.create.return_value = response

        result = parser.parse("broken")
        assert result.confidence == 0.0
        assert result.style == ContentStyle.NEUTRAL

    def test_unknown_category_id_dropped(self, parser: QueryParser) -> None:
        raw = {
            "topics": [
                {"category_id": 9999, "confidence": 0.9},
                {"category_id": 12, "confidence": 0.8},
            ],
            "style": "NEUTRAL",
            "duration": "ANY",
            "intent_type": "EXPLORATION",
            "negative_constraints": [],
            "semantic_summary": "test",
        }
        parser._client.chat.completions.create.return_value = _mock_llm_response(raw)

        result = parser.parse("test")
        assert len(result.topics) == 1
        assert result.topics[0].category_id == 12

    def test_unknown_style_falls_back_neutral(self, parser: QueryParser) -> None:
        raw = {
            "topics": [],
            "style": "NONEXISTENT",
            "duration": "ANY",
            "intent_type": "EXPLORATION",
            "negative_constraints": [],
            "semantic_summary": "test",
        }
        parser._client.chat.completions.create.return_value = _mock_llm_response(raw)

        result = parser.parse("test")
        assert result.style == ContentStyle.NEUTRAL

    def test_confidence_clamped(self, parser: QueryParser) -> None:
        raw = {
            "topics": [{"category_id": 1, "confidence": 5.0}],
            "style": "NEUTRAL",
            "duration": "ANY",
            "intent_type": "EXPLORATION",
            "negative_constraints": [],
            "semantic_summary": "test",
        }
        parser._client.chat.completions.create.return_value = _mock_llm_response(raw)

        result = parser.parse("test")
        assert result.topics[0].confidence == 1.0

    def test_negative_constraints_preserved(self, parser: QueryParser) -> None:
        raw = {
            "topics": [],
            "style": "NEUTRAL",
            "duration": "ANY",
            "intent_type": "EXPLORATION",
            "negative_constraints": ["no ads", "nothing political"],
            "semantic_summary": "test",
        }
        parser._client.chat.completions.create.return_value = _mock_llm_response(raw)

        result = parser.parse("test")
        assert result.negative_constraints == ["no ads", "nothing political"]

    def test_parse_batch(self, parser: QueryParser) -> None:
        raw = {
            "topics": [],
            "style": "NEUTRAL",
            "duration": "ANY",
            "intent_type": "EXPLORATION",
            "negative_constraints": [],
            "semantic_summary": "test",
        }
        parser._client.chat.completions.create.return_value = _mock_llm_response(raw)

        results = parser.parse_batch(["q1", "q2", "q3"])
        assert len(results) == 3
        assert all(isinstance(r, QueryIntent) for r in results)

    def test_no_api_key_raises(self) -> None:
        with patch("recsys.intent.query_parser.OpenAI"):
            with patch.dict("os.environ", {}, clear=True):
                with pytest.raises(ValueError, match="No API key"):
                    QueryParser(api_key="")

    def test_embedder_attaches_embedding(self, parser: QueryParser) -> None:
        """When an embedder returns a non-empty vector, it's attached to QueryIntent."""
        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = [0.1, 0.2, 0.3]
        parser._embedder = mock_embedder

        raw = {
            "topics": [{"category_id": 12, "confidence": 0.9}],
            "style": "FUNNY",
            "duration": "SHORT",
            "intent_type": "ENTERTAINMENT",
            "negative_constraints": [],
            "semantic_summary": "fun food",
        }
        parser._client.chat.completions.create.return_value = _mock_llm_response(raw)

        result = parser.parse("fun food")
        assert result.query_embedding == [0.1, 0.2, 0.3]
        mock_embedder.embed.assert_called_once_with("fun food")

    def test_null_embedder_leaves_none(self, parser: QueryParser) -> None:
        """NullEmbedder returns [], so query_embedding stays None."""
        parser._embedder = NullEmbedder()

        raw = {
            "topics": [],
            "style": "NEUTRAL",
            "duration": "ANY",
            "intent_type": "EXPLORATION",
            "negative_constraints": [],
            "semantic_summary": "test",
        }
        parser._client.chat.completions.create.return_value = _mock_llm_response(raw)

        result = parser.parse("test")
        assert result.query_embedding is None


class TestBgeM3Embedder:
    def test_embed_single(self) -> None:
        with patch("recsys.intent.embeddings.OpenAI") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client

            # Mock the embeddings response
            mock_data = MagicMock()
            mock_data.embedding = [0.1] * 1024
            mock_response = MagicMock()
            mock_response.data = [mock_data]
            mock_client.embeddings.create.return_value = mock_response

            embedder = BgeM3Embedder(api_key="test-key")
            result = embedder.embed("hello")

            assert len(result) == 1024
            mock_client.embeddings.create.assert_called_once_with(
                model="bge-m3", input=["hello"],
            )

    def test_embed_batch(self) -> None:
        with patch("recsys.intent.embeddings.OpenAI") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client

            mock_data_1 = MagicMock()
            mock_data_1.embedding = [0.1] * 1024
            mock_data_2 = MagicMock()
            mock_data_2.embedding = [0.2] * 1024
            mock_response = MagicMock()
            mock_response.data = [mock_data_1, mock_data_2]
            mock_client.embeddings.create.return_value = mock_response

            embedder = BgeM3Embedder(api_key="test-key")
            result = embedder.embed_batch(["a", "b"])

            assert len(result) == 2
            assert len(result[0]) == 1024
            assert len(result[1]) == 1024

    def test_embed_failure_returns_empty(self) -> None:
        with patch("recsys.intent.embeddings.OpenAI") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            mock_client.embeddings.create.side_effect = RuntimeError("API down")

            embedder = BgeM3Embedder(api_key="test-key")
            result = embedder.embed("hello")
            assert result == []

    def test_no_api_key_raises(self) -> None:
        with patch("recsys.intent.embeddings.OpenAI"):
            with patch.dict("os.environ", {}, clear=True):
                with pytest.raises(ValueError, match="No API key"):
                    BgeM3Embedder(api_key="")
