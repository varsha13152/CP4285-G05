#!/usr/bin/env python3
"""Sweep α across session cases — loads data once, re-ranks cheaply.

Caches CandidateSets from Stages 1-3 (the expensive LLM + retrieval part),
then re-ranks each with different α values (fast, ~0.01s each).

Usage:
    python scripts/sweep_alpha.py
    python scripts/sweep_alpha.py --alphas 0.0 0.25 0.5 0.75 1.0
"""

from __future__ import annotations

import argparse
import ast
import json
import pathlib
import sys
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from dotenv import load_dotenv

load_dotenv()

WATCH_RATIO_CAP = 5.0


def main() -> None:
    ap = argparse.ArgumentParser(description="Sweep α for session cases")
    ap.add_argument("--data-dir", default="data/raw")
    ap.add_argument("--embeddings", default="data/processed/caption_embeddings.npz")
    ap.add_argument("--num-candidates", type=int, default=200)
    ap.add_argument("--split", default="test", choices=["dev", "test", "all"])
    ap.add_argument("--alphas", type=float, nargs="+",
                    default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ap.add_argument("--w-query", type=float, default=0.5)
    args = ap.parse_args()

    from tests.fixtures.query_sets import load_queries
    from tests.fixtures.session_cases_v3 import CONDITION_MIX, SESSION_CASES_V3

    from recsys.data.item_features import caption_category_table
    from recsys.data.loader import load_kuairec
    from recsys.evaluation.metrics import mrr, ndcg_at_k, precision_at_k
    from recsys.intent import BgeM3Embedder, QueryParser
    from recsys.pipeline.pipeline import RecommendationPipeline
    from recsys.ranking import QCSARRanker
    from recsys.retrieval import HybridRetriever
    from recsys.routing import QueryRouter
    from recsys.session_modelling import SessionModeller
    from recsys.user_modelling import UserModeller

    queries = load_queries("v3", args.split, "en")

    # ------------------------------------------------------------------
    # 1. Load data & build pipeline (without ranker)
    # ------------------------------------------------------------------
    print("Loading data and training models ...", flush=True)
    t0 = time.time()
    data = load_kuairec(load_big=True)
    small = data.small_matrix

    profiler = UserModeller(
        interactions=data.big_matrix,
        item_categories=caption_category_table(data.caption_categories),
        social_network=data.social_network,
        session_modeller=SessionModeller(interactions=data.big_matrix),
    )

    pipeline = RecommendationPipeline(
        parser=QueryParser(embedder=BgeM3Embedder()),
        router=QueryRouter(default_num_candidates=args.num_candidates),
        retriever=HybridRetriever.from_disk(mode="hybrid"),
        user_profiler=profiler,
    )
    print(f"  Setup done in {time.time() - t0:.0f}s")

    # Ranker building blocks
    ranker_base: dict = dict(
        item_factors=profiler.item_factors,
        item_ids=profiler.item_ids,
    )
    emb_path = pathlib.Path(args.embeddings)
    if emb_path.exists():
        emb_data = np.load(str(emb_path))
        ranker_base["caption_embeddings"] = emb_data["embeddings"]
        ranker_base["caption_video_ids"] = emb_data["video_ids"]

    from recsys.data.schema import FEAT, USER_ID, VIDEO_ID, WATCH_RATIO

    # Ground truth
    user_gt: dict[int, dict[int, float]] = {}
    for row in small.itertuples(index=False):
        uid = int(getattr(row, USER_ID))
        vid = int(getattr(row, VIDEO_ID))
        wr = float(getattr(row, WATCH_RATIO))
        user_gt.setdefault(uid, {})[vid] = wr

    vid_cats: dict[int, set[int]] = {}
    df_cats = pd.read_csv(f"{args.data_dir}/item_categories.csv")
    for row in df_cats.itertuples(index=False):
        feat_val = getattr(row, FEAT)
        cats = ast.literal_eval(feat_val) if isinstance(feat_val, str) else feat_val
        vid_cats[int(getattr(row, VIDEO_ID))] = set(cats)

    # ------------------------------------------------------------------
    # 2. Cache CandidateSets (Stages 1-3) — the expensive part
    # ------------------------------------------------------------------
    cases_by_q: dict[str, list[dict]] = {}
    for c in SESSION_CASES_V3:
        cases_by_q.setdefault(c["query_id"], []).append(c)
    # Ensure anonymous cases are processed first (anonymous* baseline depends on it)
    for qid in cases_by_q:
        cases_by_q[qid].sort(key=lambda c: (c["user_id"] is not None, c["condition"]))

    print(f"\nCaching CandidateSets for {len(queries)} queries × session cases ...", flush=True)
    # Each entry: (case_dict, query_dict, CandidateSet)
    cached: list[tuple[dict, dict, object]] = []
    t0 = time.time()

    def retrieve_safe(*a, **kw):
        for attempt in range(5):
            try:
                return pipeline.retrieve(*a, **kw)
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(15 * 2 ** attempt)

    for qi, q in enumerate(queries, 1):
        print(f"  [{qi:3d}/{len(queries)}] {q['query']!r}", flush=True)
        for c in cases_by_q.get(q["id"], []):
            cs = retrieve_safe(
                q["query"],
                user_id=c["user_id"],
                recent_video_ids=c["recent_video_ids"] or None,
            )
            cached.append((c, q, cs))

    print(f"  Cached {len(cached)} CandidateSets in {time.time() - t0:.0f}s\n")

    # ------------------------------------------------------------------
    # 3. Sweep α — just re-rank (fast)
    # ------------------------------------------------------------------
    results_all: list[dict] = []

    for alpha in args.alphas:
        ranker = QCSARRanker(**ranker_base, alpha=alpha, w_query=args.w_query, top_k=100)

        conds = ["anonymous", "anonymous*", "none", "related", "unrelated"]
        m1 = {c: {"P@5": [], "MRR@3": []} for c in conds}
        m2 = {c: {"nDCG@5": [], "judged@5": []} for c in conds}

        anon_rankings: dict[str, object] = {}  # query_id -> anonymous RankedList

        for case, q, cs in cached:
            ranked = ranker.rank(cs)
            topics = set(q["expected_topics"])
            answers = set(q["answer_video_ids"])
            scored_personal = q.get("anchor") is None

            if case["user_id"] is None:
                # Anonymous
                m1["anonymous"]["P@5"].append(precision_at_k(ranked, answers, 5))
                m1["anonymous"]["MRR@3"].append(mrr(ranked, answers, k=3))
                anon_rankings[q["id"]] = ranked
            else:
                cond = case["condition"]
                m1[cond]["P@5"].append(precision_at_k(ranked, answers, 5))
                m1[cond]["MRR@3"].append(mrr(ranked, answers, k=3))

                if scored_personal:
                    labels = user_gt.get(case["user_id"], {})
                    rel = {v: min(wr, WATCH_RATIO_CAP)
                           if vid_cats.get(v, set()) & topics else 0.0
                           for v, wr in labels.items()}
                    judged = [i for i in ranked.items if i.video_id in rel]
                    m2[cond]["nDCG@5"].append(
                        ndcg_at_k(SimpleNamespace(items=judged), rel, 5))
                    m2[cond]["judged@5"].append(
                        sum(i.video_id in rel for i in ranked.items[:5]) / 5)

                    # anonymous* baseline
                    if cond == "none" and q["id"] in anon_rankings:
                        anon_r = anon_rankings[q["id"]]
                        judged_a = [i for i in anon_r.items if i.video_id in rel]
                        m2["anonymous*"]["nDCG@5"].append(
                            ndcg_at_k(SimpleNamespace(items=judged_a), rel, 5))
                        m2["anonymous*"]["judged@5"].append(
                            sum(i.video_id in rel for i in anon_r.items[:5]) / 5)

        def avg(acc, c, m):
            return float(np.mean(acc[c][m])) if acc[c][m] else float("nan")

        row = {"alpha": alpha}
        for cond in ["anonymous", "none", "related", "unrelated"]:
            row[f"{cond}_P@5"] = avg(m1, cond, "P@5")
            row[f"{cond}_MRR@3"] = avg(m1, cond, "MRR@3")
        for cond in ["anonymous*", "none", "related", "unrelated"]:
            row[f"{cond}_nDCG@5"] = avg(m2, cond, "nDCG@5")
        results_all.append(row)

        print(f"  α={alpha:.1f}  "
              f"anon MRR@3={row['anonymous_MRR@3']:.4f}  "
              f"none MRR@3={row['none_MRR@3']:.4f}  "
              f"none nDCG@5={row['none_nDCG@5']:.4f}  "
              f"related nDCG@5={row['related_nDCG@5']:.4f}")

    # ------------------------------------------------------------------
    # 4. Save
    # ------------------------------------------------------------------
    out_path = "results/alpha_sweep.json"
    pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results_all, f, indent=2)
    print(f"\nSaved to {out_path}")

    # Also print a table
    print(f"\n{'α':>5}  {'anon MRR@3':>10}  {'none MRR@3':>10}  {'rel MRR@3':>10}  "
          f"{'unrel MRR@3':>11}  {'anon* nDCG':>10}  {'none nDCG':>10}  "
          f"{'rel nDCG':>10}  {'unrel nDCG':>10}")
    for r in results_all:
        print(f"{r['alpha']:5.1f}  "
              f"{r['anonymous_MRR@3']:10.4f}  {r['none_MRR@3']:10.4f}  "
              f"{r['related_MRR@3']:10.4f}  {r['unrelated_MRR@3']:11.4f}  "
              f"{r['anonymous*_nDCG@5']:10.4f}  {r['none_nDCG@5']:10.4f}  "
              f"{r['related_nDCG@5']:10.4f}  {r['unrelated_nDCG@5']:10.4f}")


if __name__ == "__main__":
    main()
