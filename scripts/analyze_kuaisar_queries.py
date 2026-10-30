#!/usr/bin/env python3
"""Profile real Kuaishou search behaviour from KuaiSAR, to shape synthetic queries.

KuaiSAR's query text is tokenised (word ids, not words), and its ids don't join to
KuaiRec, so it can't supply queries directly. It can supply the *shape* of real
searches. This script measures that shape and writes a small JSON profile
(committed), so later steps don't need the 2 GB raw data:

* category demand  — share of clicked searches per level-1 category
* length           — query length in words
* specificity      — how many categories a search's clicks span
* head vs tail     — how often the exact same query recurs
* refinement       — follow-up searches within 5 minutes that re-use a word
* session context  — feed viewing in the 30 minutes before a search, related or not
* viewing          — results shown and clicks per search

Data: KuaiSAR-small (https://zenodo.org/record/8181109, KuaiSAR.zip) unpacked to
data/external/kuaisar/ (gitignored).

Usage:
    python scripts/analyze_kuaisar_queries.py
"""

from __future__ import annotations

import argparse
import ast
import json
import pathlib

import numpy as np
import pandas as pd

UNCATEGORISED = "空"
REFINE_WINDOW_MS = 5 * 60 * 1000
CONTEXT_WINDOW_MS = 30 * 60 * 1000
CONTEXT_MAX_VIEWS = 10


def _round(d: dict, n: int = 4) -> dict:
    return {k: round(float(v), n) for k, v in d.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="data/external/kuaisar")
    ap.add_argument("--out", default="results/kuaisar_query_profile.json")
    ap.add_argument("--context-sample", type=int, default=30000,
                    help="searches sampled for the session-context measurement")
    args = ap.parse_args()
    root = pathlib.Path(args.data_dir)

    src = pd.read_csv(root / "src_inter.csv")
    src["tokens"] = src["keyword"].map(lambda k: tuple(ast.literal_eval(k)))
    sess = src.groupby("search_session_id").agg(
        user=("user_id", "first"), tokens=("tokens", "first"),
        ts=("search_session_timestamp", "first"), source=("search_source", "first"),
        n_results=("item_id", "size"), n_clicked=("click_cnt", lambda c: int((c > 0).sum())),
    )
    sess["n_words"] = sess["tokens"].map(len)
    profile: dict = {"n_searches": len(sess), "n_users": int(sess["user"].nunique())}

    # length
    words = sess["n_words"]
    profile["length"] = {
        "median_words": float(words.median()),
        "share": _round({"1": (words == 1).mean(), "2-3": words.between(2, 3).mean(),
                         "4+": (words >= 4).mean()}),
        "median_by_entry_point": {k: float(v) for k, v in
                                  sess.groupby("source")["n_words"].median().items()},
    }

    # head vs tail
    freq = sess["tokens"].map(sess["tokens"].value_counts())
    counts = sess["tokens"].value_counts()
    profile["head_tail"] = _round({
        "one_off_share": (freq == 1).mean(),
        "head_share_10plus": (freq >= 10).mean(),
        "top_1pct_query_coverage": counts.head(max(1, len(counts) // 100)).sum() / len(sess),
    })

    # viewing
    profile["viewing"] = {
        "median_results_per_search": float(sess["n_results"].median()),
        "share_searches_with_click": round(float((sess["n_clicked"] > 0).mean()), 4),
        "median_clicks_when_clicked": float(sess.loc[sess["n_clicked"] > 0, "n_clicked"].median()),
    }

    # topics and specificity (categories of clicked results)
    items = pd.read_csv(root / "item_features.csv",
                        usecols=["item_id", "first_level_category_name", "third_level_category_name"])
    l1 = items.set_index("item_id")["first_level_category_name"]
    clicked = src.loc[src["click_cnt"] > 0, ["search_session_id", "item_id"]].merge(
        items, on="item_id", how="left")
    known = clicked[clicked["first_level_category_name"].notna()
                    & (clicked["first_level_category_name"] != UNCATEGORISED)]
    pairs = known.drop_duplicates(["search_session_id", "first_level_category_name"])
    demand = pairs["first_level_category_name"].value_counts(normalize=True)
    profile["category_demand"] = _round(demand.to_dict())
    profile["uncategorised_click_share"] = round(float(
        (clicked["first_level_category_name"] == UNCATEGORISED).mean()), 4)
    spread = known.groupby("search_session_id").agg(
        l1=("first_level_category_name", "nunique"), l3=("third_level_category_name", "nunique"))
    single = spread[spread["l1"] == 1]
    profile["specificity"] = _round({
        "single_l1": (spread["l1"] == 1).mean(), "two_l1": (spread["l1"] == 2).mean(),
        "three_plus_l1": (spread["l1"] >= 3).mean(),
        "single_l3_given_single_l1": (single["l3"] == 1).mean(),
    })

    # refinement: same user's next search within 5 minutes
    s = sess.sort_values(["user", "ts"])
    nxt = s.groupby("user").shift(-1)
    within = (nxt["ts"] - s["ts"]).le(REFINE_WINDOW_MS)
    shares = [bool(set(a) & set(b)) if isinstance(b, tuple) else False
              for a, b in zip(s["tokens"], nxt["tokens"])]
    refine = within & pd.Series(shares, index=s.index)
    profile["refinement"] = _round({
        "followup_within_5min": within.mean(),
        "reuse_word_given_followup": refine.sum() / max(int(within.sum()), 1),
        "narrowing_given_refinement": (nxt["n_words"] > s["n_words"])[refine].mean(),
    })

    # session context: feed viewing in the 30 minutes before a clicked search
    qcats = known.groupby("search_session_id")["first_level_category_name"].agg(set)
    sample = sess.loc[sess.index.isin(qcats.index)].sample(
        min(args.context_sample, len(qcats)), random_state=0)
    rec = pd.read_csv(root / "rec_inter.csv", usecols=["user_id", "item_id", "timestamp"])
    rec = rec[rec["user_id"].isin(set(sample["user"]))].sort_values("timestamp")
    rec["cat"] = rec["item_id"].map(l1)
    by_user = {u: g for u, g in rec.groupby("user_id")}
    outcome, n_views = [], []
    for sid, row in sample.iterrows():
        g = by_user.get(row["user"])
        if g is None:
            outcome.append("none")
            continue
        hi = np.searchsorted(g["timestamp"].to_numpy(), row["ts"])
        win = g.iloc[max(0, hi - CONTEXT_MAX_VIEWS):hi]
        win = win[(row["ts"] - win["timestamp"]) <= CONTEXT_WINDOW_MS]
        if win.empty:
            outcome.append("none")
            continue
        n_views.append(len(win))
        cats = set(win["cat"].dropna()) - {UNCATEGORISED}
        if not cats:
            outcome.append("recent_uncategorised")
        else:
            outcome.append("related" if cats & qcats[sid] else "unrelated")
    out = pd.Series(outcome)
    has_cat = out.isin(["related", "unrelated"])
    related_rate = (out == "related").sum() / max(int(has_cat.sum()), 1)
    recent = 1 - (out == "none").mean()
    # For comparison: all searches, including ones without a click. Searches made
    # straight after feed browsing mostly end without a click; answered searches
    # (what our synthetic queries model) are mostly standalone.
    feed_ts = {u: np.sort(g["timestamp"].to_numpy()) for u, g in
               pd.read_csv(root / "rec_inter.csv", usecols=["user_id", "timestamp"])
               .groupby("user_id")}
    every = sess.sample(min(args.context_sample, len(sess)), random_state=0)

    def _had_recent(user: int, ts: float) -> bool:
        times = feed_ts.get(user)
        if times is None:
            return False
        i = np.searchsorted(times, ts)
        return bool(i > 0 and ts - times[i - 1] <= CONTEXT_WINDOW_MS)

    recent_all = np.mean([_had_recent(u, t) for u, t in zip(every["user"], every["ts"])])
    profile["session_context"] = _round({
        "share_with_recent_viewing_all_searches": recent_all,
        "share_with_recent_viewing_30min": recent,
        "related_given_recent": related_rate,
        "median_recent_views": float(np.median(n_views)) if n_views else 0.0,
        # condition mix used to weight evaluation: none / related / unrelated
        "mix_none": 1 - recent,
        "mix_related": recent * related_rate,
        "mix_unrelated": recent * (1 - related_rate),
    })
    profile["session_context"]["scope"] = "clicked searches with a categorised result"

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(profile, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in profile.items() if k != "category_demand"},
                     ensure_ascii=False, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
