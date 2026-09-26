#!/usr/bin/env python3
"""Benchmark Stage 3 retrieval modes: semantic vs category vs hybrid.

For each of the 27 synthetic queries (tests/fixtures/synthetic_queries.py):

* **Gold intent** (default): topics = the query's expected topics (confidence
  1.0) and its expected duration, plus a live bge-m3 embedding of the raw query.
  This isolates Stage 3 from Stage 1 errors. ``--use-parser`` runs the live
  Stage 1 parser instead (end-to-end).
* **Category hit-rate@N** — share of the N candidates whose caption category is
  one of the query's expected topics (full 10.7k catalogue).
* **small_matrix Recall@N** — catalogue restricted to small_matrix's 3,327
  videos; for every user, relevant = videos they watched with
  ``watch_ratio >= 2`` whose category is an expected topic; recall =
  |candidates ∩ relevant| / min(|relevant|, N), averaged over users with ≥1
  relevant item, then over queries. Stage 3 is not personalised, so one
  candidate list per query serves every user.
* **Latency** — retrieval compute per query (ms) and the embedding API call.

Caveat: both accuracy labels come from category ids, which favours the
category mode; read the semantic numbers as a lower bound.

Usage:
    python scripts/evaluate_stage3.py
    python scripts/evaluate_stage3.py --use-parser --w-cat 0.3
"""

from __future__ import annotations

import argparse
import datetime as dt
import pathlib
import sys
import time

import numpy as np
import pandas as pd

_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

from dotenv import load_dotenv

load_dotenv()

from tests.fixtures.synthetic_queries import SYNTHETIC_QUERIES

from recsys.contracts.enums import (
    ContentStyle,
    DurationPreference,
    QueryIntentType,
)
from recsys.contracts.models import CategoryMatch, QueryIntent
from recsys.data.category_index import CategoryIndex
from recsys.intent import BgeM3Embedder, QueryParser
from recsys.retrieval import MODES, HybridRetriever, load_catalogue
from recsys.routing import QueryRouter


def _gold_intent(q: dict, embedding: list[float]) -> QueryIntent:
    index = CategoryIndex()
    topics = [
        CategoryMatch(c, index.get_by_id(c).chinese_name, index.get_by_id(c).english_name, 1.0)
        for c in q["expected_topics"]
    ]
    return QueryIntent(
        raw_query=q["query"],
        topics=topics,
        style=ContentStyle[q["expected_style"]],
        duration=DurationPreference[q["expected_duration"]],
        intent_type=QueryIntentType[q["expected_intent"]],
        negative_constraints=[],
        semantic_summary=q["query"],
        query_embedding=embedding,
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default="data/raw")
    p.add_argument("--embeddings", default="data/processed/caption_embeddings.npz")
    p.add_argument("--num-candidates", type=int, default=100)
    p.add_argument("--use-parser", action="store_true", help="live Stage 1 instead of gold intents")
    p.add_argument("--w-sem", type=float, default=1.0)
    p.add_argument("--w-cat", type=float, default=0.5)
    p.add_argument("--w-dur", type=float, default=0.2)
    today = dt.datetime.now().astimezone().date()
    p.add_argument("--out", default=f"results/stage3_eval_{today}.txt")
    args = p.parse_args()
    weights = {"w_sem": args.w_sem, "w_cat": args.w_cat, "w_dur": args.w_dur}
    n = args.num_candidates

    items, emb = load_catalogue(args.data_dir, args.embeddings)
    small = pd.read_csv(
        pathlib.Path(args.data_dir) / "small_matrix.csv",
        usecols=["user_id", "video_id", "watch_ratio"],
    )
    small_ids = np.sort(small["video_id"].unique())
    small_rows = items.index.get_indexer(small_ids)
    items_small, emb_small = items.iloc[small_rows], emb[small_rows]
    liked = small[small["watch_ratio"] >= 2.0].merge(
        items_small[["category_id"]], left_on="video_id", right_index=True
    )

    # --- intents (embedding latency measured per query) --------------------
    router = QueryRouter(default_num_candidates=n)
    embedder, parser = BgeM3Embedder(), None
    if args.use_parser:
        parser = QueryParser(embedder=embedder)
    requests, embed_ms = [], []
    for q in SYNTHETIC_QUERIES:
        t0 = time.perf_counter()
        if parser:
            intent = parser.parse(q["query"])
        else:
            intent = _gold_intent(q, embedder.embed(q["query"]))
        embed_ms.append((time.perf_counter() - t0) * 1000)
        requests.append(router.route(intent))

    # --- evaluate each mode --------------------------------------------------
    rows = []
    for mode in MODES:
        full = HybridRetriever(items, emb, mode=mode, **weights)
        sub = HybridRetriever(items_small, emb_small, mode=mode, **weights)
        hit, rec, lat = [], [], []
        for q, req in zip(SYNTHETIC_QUERIES, requests):
            expected = set(q["expected_topics"])
            t0 = time.perf_counter()
            cands = full.retrieve(req).candidates
            lat.append((time.perf_counter() - t0) * 1000)
            hit.append(np.mean([bool(expected & set(c.category_ids)) for c in cands])
                       if cands else 0.0)

            got = {c.video_id for c in sub.retrieve(req).candidates}
            rel = liked[liked["category_id"].isin(expected)].groupby("user_id")["video_id"]
            per_user = [len(got & set(v)) / min(len(v), n) for _, v in rel]
            rec.append(np.mean(per_user) if per_user else 0.0)
        rows.append({
            "mode": mode,
            f"hit_rate@{n}": round(float(np.mean(hit)), 4),
            f"small_recall@{n}": round(float(np.mean(rec)), 4),
            "retrieval_ms": round(float(np.mean(lat)), 2),
        })

    table = pd.DataFrame(rows).to_string(index=False)
    source = "live Stage 1 parser" if args.use_parser else "gold intents + live bge-m3 embedding"
    report = "\n".join([
        f"Stage 3 retrieval benchmark — {today}",
        f"{len(SYNTHETIC_QUERIES)} synthetic queries, N={n}, intents: {source}",
        f"weights: {weights}",
        (
            f"catalogue: {len(items):,} videos ({int(items['has_text'].sum()):,} with text); "
            f"small_matrix subset {len(items_small):,}"
        ),
        "",
        table,
        "",
        (
            f"query {'parse+embed' if parser else 'embedding'} latency: "
            f"mean {np.mean(embed_ms):.0f} ms (API call, dominates retrieval compute)"
        ),
        "caveat: labels come from category ids, which favours the category mode.",
    ])
    print(report)
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report + "\n")
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
