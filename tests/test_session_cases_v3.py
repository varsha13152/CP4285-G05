"""Checks for the leak-free v3 session cases (no data or API needed)."""

import random
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import pytest

from tests.fixtures.session_cases_v3 import CONDITION_MIX, SESSION_CASES_V3 as C
from tests.fixtures.synthetic_queries_v3 import SYNTHETIC_QUERIES_V3 as Q

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from build_session_cases_v3 import MAX_RECENT, pick_context  # noqa: E402

TOPICS = {q["id"]: set(q["expected_topics"]) for q in Q}


def test_conditions_and_users_per_query():
    assert {c["condition"] for c in C} == {"anonymous", "none", "related", "unrelated"}
    assert set(CONDITION_MIX) == {"none", "related", "unrelated"}
    assert sum(CONDITION_MIX.values()) == pytest.approx(1.0, abs=1e-3)
    per_q = Counter((c["query_id"], c["condition"]) for c in C)
    for qid in TOPICS:
        assert per_q[(qid, "anonymous")] == 1
        assert per_q[(qid, "none")] == per_q[(qid, "related")] == per_q[(qid, "unrelated")] == 5


def test_context_matches_condition():
    for c in C:
        cats = set(c["recent_categories"])
        assert len(c["recent_video_ids"]) == len(c["recent_categories"]) <= MAX_RECENT
        if c["condition"] in ("anonymous", "none"):
            assert not c["recent_video_ids"]
        elif c["condition"] == "related":
            assert c["recent_categories"][-1] in TOPICS[c["query_id"]]
        else:
            assert c["recent_video_ids"] and not cats & TOPICS[c["query_id"]]


def test_pick_context():
    sessions = pd.DataFrame({"videos": [[1, 2, 3], [4, 5]], "cats": [[9, 7, 9], [8, 8]]})
    rng = random.Random(0)
    assert pick_context(sessions, {7}, True, rng) == ([1, 2], [9, 7])  # ends on the on-topic video
    assert pick_context(sessions, {7}, False, rng) == ([4, 5], [8, 8])
    assert pick_context(sessions, {6}, True, rng) is None
