#!/usr/bin/env python3
"""Build leak-free session test cases for the v3 queries (Stage 4 personalisation).

For every v3 query, pick ``USERS_PER_QUERY`` small_matrix users who loved
(``watch_ratio >= 2``) at least one video in the query's categories — their
small_matrix videos are the personal ground truth. Their *recent viewing* comes
from a real session in their **big_matrix** history (cut by
``recsys.session_modelling.Sessioniser``, 30-minute gaps), never from
small_matrix, so context can't leak the answers (the two matrices share no
(user, video) pairs; asserted anyway).

Conditions per (query, user):

* ``none``      — the user, no recent viewing (a cold search)
* ``related``   — last <= 6 videos of a session ending on a video in the query's categories
* ``unrelated`` — last <= 6 videos of a session with no video in the query's categories

plus one ``anonymous`` case per query (no user). ``CONDITION_MIX`` holds the
real-world weights of none/related/unrelated measured on KuaiSAR answered
searches (results/kuaisar_query_profile.json) for a realistic blended score.

Usage:
    python scripts/build_session_cases_v3.py
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys

import pandas as pd

_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

from recsys.data.item_features import caption_category_table
from recsys.session_modelling import Sessioniser

USERS_PER_QUERY = 5
MAX_RECENT = 6           # KuaiSAR: median 6 feed views in the 30 min before a search
LIKED_RATIO = 2.0
SEED = 13


def session_table(big: pd.DataFrame, video_cat: dict[int, int]) -> pd.DataFrame:
    """One row per (user, session): ordered video ids and their categories."""
    df = Sessioniser(strategy="fixed").assign(big)
    df["cat"] = df["video_id"].map(video_cat).fillna(-1).astype(int)
    return df.groupby(["user_id", "session_id"]).agg(
        videos=("video_id", list), cats=("cat", list)).reset_index()


def pick_context(sessions: pd.DataFrame, topics: set[int], related: bool,
                 rng: random.Random) -> tuple[list[int], list[int]] | None:
    """Recent videos for one user: a related window ending on an on-topic video,
    or the tail of a session with no on-topic video. None if no such session."""
    rows = []
    for videos, cats in zip(sessions["videos"], sessions["cats"]):
        on = [i for i, c in enumerate(cats) if c in topics]
        if related and on:
            end = on[-1] + 1
            rows.append((videos[max(0, end - MAX_RECENT):end], cats[max(0, end - MAX_RECENT):end]))
        elif not related and not on and videos:
            rows.append((videos[-MAX_RECENT:], cats[-MAX_RECENT:]))
    if not rows:
        return None
    videos, cats = rng.choice(rows)
    return [int(v) for v in videos], [int(c) for c in cats]


def build_cases(queries: list[dict], small: pd.DataFrame, sessions: pd.DataFrame,
                video_cat: dict[int, int], rng: random.Random) -> list[dict]:
    liked = small[small["watch_ratio"] >= LIKED_RATIO].assign(
        cat=lambda d: d["video_id"].map(video_cat))
    by_user = {u: g for u, g in sessions.groupby("user_id")}
    cases = []
    for q in queries:
        topics = set(q["expected_topics"])
        truth = liked[liked["cat"].isin(topics)].groupby("user_id")["video_id"].agg(set)
        candidates = sorted(u for u in truth.index if u in by_user)
        rng.shuffle(candidates)
        cases.append({"query_id": q["id"], "user_id": None, "condition": "anonymous",
                      "recent_video_ids": [], "recent_categories": []})
        picked = 0
        for u in candidates:
            rel = pick_context(by_user[u], topics, True, rng)
            unrel = pick_context(by_user[u], topics, False, rng)
            if rel is None or unrel is None:
                continue
            for cond, ctx in (("none", ([], [])), ("related", rel), ("unrelated", unrel)):
                assert not set(ctx[0]) & truth[u], "context leaks ground truth"
                cases.append({"query_id": q["id"], "user_id": int(u), "condition": cond,
                              "recent_video_ids": ctx[0], "recent_categories": ctx[1]})
            picked += 1
            if picked == USERS_PER_QUERY:
                break
        if picked < USERS_PER_QUERY:
            print(f"  warning: only {picked} users for {q['id']} {q['query']!r}")
    for i, c in enumerate(cases, 1):
        c["case_id"] = f"c{i:04d}"
    return cases


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="data/raw")
    ap.add_argument("--profile", default="results/kuaisar_query_profile.json")
    ap.add_argument("--out", default="tests/fixtures/session_cases_v3.py")
    args = ap.parse_args()
    from tests.fixtures.synthetic_queries_v3 import SYNTHETIC_QUERIES_V3

    root = pathlib.Path(args.data_dir)
    cap = pd.read_csv(root / "kuairec_caption_category.csv", engine="python", on_bad_lines="skip")
    table = caption_category_table(cap)
    video_cat = {int(v): (f[0] if f else -1) for v, f in zip(table["video_id"], table["feat"])}
    small = pd.read_csv(root / "small_matrix.csv", usecols=["user_id", "video_id", "watch_ratio"])
    big = pd.read_csv(root / "big_matrix.csv", usecols=["user_id", "video_id", "timestamp"])
    big = big[big["user_id"].isin(set(small["user_id"]))]
    shared = big.merge(small[["user_id", "video_id"]], on=["user_id", "video_id"])
    assert shared.empty, "big_matrix and small_matrix unexpectedly share (user, video) pairs"
    sessions = session_table(big, video_cat)
    print(f"{len(sessions):,} big_matrix sessions for {sessions['user_id'].nunique():,} small_matrix users")

    cases = build_cases(SYNTHETIC_QUERIES_V3, small, sessions, video_cat, random.Random(SEED))
    ctx = json.loads(pathlib.Path(args.profile).read_text(encoding="utf-8"))["session_context"]
    mix = {"none": ctx["mix_none"], "related": ctx["mix_related"], "unrelated": ctx["mix_unrelated"]}
    order = ["case_id", "query_id", "user_id", "condition", "recent_video_ids", "recent_categories"]
    body = ",\n".join("    " + repr({k: c[k] for k in order}) for c in cases)
    pathlib.Path(args.out).write_text(
        '"""Leak-free session test cases for the v3 queries (scripts/build_session_cases_v3.py).\n\n'
        "Each v3 query has one `anonymous` case and, for 5 small_matrix users who loved a video\n"
        "in its categories, `none` / `related` / `unrelated` cases. Recent videos come from a\n"
        "real big_matrix session (never small_matrix). Users and sessions are real; pairing them\n"
        "with a query is synthetic. Grading (evaluate_ranking.py --cases): query relevance =\n"
        "P@5/MRR on answer_video_ids; personal preference = nDCG@5 on the user's small_matrix\n"
        "watch_ratio (unlabelled videos skipped). CONDITION_MIX = real-world share of each\n"
        "condition among answered KuaiSAR searches, for a realistic blended score.\n"
        '"""\n\n'
        f"CONDITION_MIX: dict[str, float] = {mix!r}\n\n"
        f"SESSION_CASES_V3: list[dict] = [\n{body},\n]\n", encoding="utf-8")
    counts = pd.Series([c["condition"] for c in cases]).value_counts().to_dict()
    print(f"wrote {args.out}: {len(cases)} cases {counts}")


if __name__ == "__main__":
    main()
