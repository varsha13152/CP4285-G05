"""Integrity checks for the v3 synthetic benchmark (no data or API needed)."""

import json
import pathlib
from collections import Counter

from recsys.contracts.enums import ContentStyle, DurationPreference, QueryIntentType
from recsys.data.category_index import CategoryIndex
from tests.fixtures.synthetic_queries_v3 import SYNTHETIC_QUERIES_V3 as Q

BASE = [q for q in Q if not q["refines"]]
REFS = [q for q in Q if q["refines"]]
PROFILE = json.loads((pathlib.Path(__file__).parents[1] / "results/kuaisar_query_profile.json")
                     .read_text(encoding="utf-8"))


def test_size_ids_and_split():
    assert len(BASE) == 100 and len(REFS) == 15
    assert len({q["id"] for q in Q}) == len(Q)
    assert Counter(q["split"] for q in BASE) == {"dev": 30, "test": 70}


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


def test_constraints_match_labels():
    for q in Q:
        if q["constraint"] == "duration":
            assert q["expected_duration"] == "SHORT"
        else:
            assert q["expected_duration"] == "ANY"
        assert bool(q["expected_negative"]) == (q["constraint"] == "exclusion")


def test_every_query_has_answers():
    for q in Q:
        assert q["answer_video_ids"] and all(isinstance(v, int) for v in q["answer_video_ids"])


def test_refinements_narrow_their_base():
    base = {q["id"]: q for q in BASE}
    for r in REFS:
        b = base[r["refines"]]
        assert r["split"] == b["split"] and r["expected_topics"] == b["expected_topics"]
        assert set(r["query"].split()) & set(b["query"].split())
        assert len(r["query"].split()) > len(b["query"].split())


def test_mix_matches_kuaisar_profile():
    types = Counter(q["query_type"] for q in BASE)
    assert 65 <= types["specific"] <= 80 and 12 <= types["multi"] <= 25
    assert sum(q["anchor"] is not None for q in BASE) >= 40
    words = Counter(min(len(q["query"].split()), 4) for q in BASE)
    ours = {"1": words[1] / 100, "2-3": (words[2] + words[3]) / 100, "4+": words[4] / 100}
    for k, share in PROFILE["length"]["share"].items():
        assert abs(ours[k] - share) <= 0.10, k


def test_unique_queries():
    assert len({q["query"] for q in Q}) == len(Q)
    assert len({q["query_zh"] for q in Q}) == len(Q)
