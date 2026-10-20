"""Integrity checks for the 100-query synthetic benchmark (no data or API needed)."""

from collections import Counter

from recsys.contracts.enums import ContentStyle, DurationPreference, QueryIntentType
from recsys.data.category_index import CategoryIndex
from tests.fixtures.synthetic_queries_v2 import SYNTHETIC_QUERIES_V2 as Q


def test_size_ids_and_split():
    assert len(Q) == 100
    assert len({q["id"] for q in Q}) == 100
    assert Counter(q["split"] for q in Q) == {"dev": 30, "test": 70}


def test_labels_are_valid():
    valid = CategoryIndex().valid_ids()
    for q in Q:
        assert q["query"] and q["query_zh"]
        assert q["expected_topics"] and set(q["expected_topics"]) <= valid
        assert q["expected_style"] in ContentStyle.__members__
        assert q["expected_duration"] in DurationPreference.__members__
        assert q["expected_intent"] in QueryIntentType.__members__
        assert q["query_type"] in {"broad", "specific", "multi"}
        assert (len(q["expected_topics"]) == 2) == (q["query_type"] == "multi")
        assert (q["expected_subcategory"] is not None) == (q["query_type"] == "specific")


def test_constraints_match_labels():
    for q in Q:
        if q["constraint"] == "duration":
            assert q["expected_duration"] in {"SHORT", "LONG"}
        else:
            assert q["expected_duration"] == "ANY"
        if q["constraint"] == "exclusion":
            assert q["expected_negative"]


def test_every_query_is_evaluable():
    # enough small_matrix videos match each query's labels to score recall
    assert all(q["n_small"] >= 5 for q in Q)
    assert len({t for q in Q for t in q["expected_topics"]}) >= 25


def test_unique_queries():
    assert len({q["query"] for q in Q}) == 100
    assert len({q["query_zh"] for q in Q}) == 100
