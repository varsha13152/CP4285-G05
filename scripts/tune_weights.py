#!/usr/bin/env python3
"""Sweep α (long-term / short-term ratio) for the simplified QCSARRanker.

Caches Stage 1-3 CandidateSet objects, then evaluates the fast ranking
stage across a grid of α and w_query values, optimising MRR@3 and nDCG@5.

Usage:
    python scripts/tune_weights.py
    python scripts/tune_weights.py --max-users 15 --max-queries 10
    python scripts/tune_weights.py --alpha-steps 11 --wq-steps 5
"""

from __future__ import annotations

import argparse
import ast
import pathlib
import sys
import time

import numpy as np
import pandas as pd

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from dotenv import load_dotenv

load_dotenv()

WATCH_RATIO_CAP = 5.0


def _load_video_categories(data_dir: str) -> dict[int, set[int]]:
    df = pd.read_csv(f"{data_dir}/item_categories.csv")
    vid_cats: dict[int, set[int]] = {}
    for row in df.itertuples(index=False):
        cats = ast.literal_eval(row.feat) if isinstance(row.feat, str) else row.feat
        vid_cats[int(row.video_id)] = set(cats)
    return vid_cats


def main() -> None:
    ap = argparse.ArgumentParser(description="Sweep α for QCSARRanker")
    ap.add_argument("--data-dir", default="data/raw")
    ap.add_argument("--embeddings", default="data/processed/caption_embeddings.npz")
    ap.add_argument("--relevance-threshold", type=float, default=0.5)
    ap.add_argument("--max-users", type=int, default=20)
    ap.add_argument("--max-queries", type=int, default=10)
    ap.add_argument("--num-candidates", type=int, default=200)
    ap.add_argument("--lang", default="zh", choices=["en", "zh", "both"])
    ap.add_argument("--query-set", default="legacy", choices=["legacy", "v2", "v3"],
                    help="v2/v3 tune on their dev split")
    ap.add_argument("--alpha-steps", type=int, default=11,
                    help="Grid points for α from 0.0 to 1.0 (default 11)")
    ap.add_argument("--wq-steps", type=int, default=5,
                    help="Grid points for w_query from 0.1 to 0.9 (default 5)")
    args = ap.parse_args()

    from tests.fixtures.query_sets import load_queries

    # v2/v3: tune on the dev split only so reported test numbers aren't overfitted
    all_queries = load_queries(
        args.query_set,
        "all" if args.query_set == "legacy" else "dev",
        args.lang,
    )

    from recsys.data.item_features import caption_category_table
    from recsys.data.loader import load_kuairec
    from recsys.evaluation.metrics import mrr, ndcg_at_k
    from recsys.intent import BgeM3Embedder, QueryParser
    from recsys.pipeline.pipeline import RecommendationPipeline
    from recsys.ranking import QCSARRanker
    from recsys.retrieval import HybridRetriever
    from recsys.routing import QueryRouter
    from recsys.session_modelling import SessionModeller
    from recsys.user_modelling import UserModeller

    # ------------------------------------------------------------------
    # 1. Load data & build pipeline (without ranker — we swap rankers)
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

    # Ranker building blocks (shared across grid points)
    ranker_base: dict = dict(
        item_factors=profiler.item_factors,
        item_ids=profiler.item_ids,
    )
    emb_path = pathlib.Path(args.embeddings)
    if emb_path.exists():
        emb_data = np.load(str(emb_path))
        ranker_base["caption_embeddings"] = emb_data["embeddings"]
        ranker_base["caption_video_ids"] = emb_data["video_ids"]
        print(f"  Caption embeddings loaded ({emb_data['embeddings'].shape})")

    # ------------------------------------------------------------------
    # 2. Build ground truth & cache CandidateSets
    # ------------------------------------------------------------------
    user_gt: dict[int, dict[int, float]] = {}
    for row in small.itertuples(index=False):
        uid, vid, wr = int(row.user_id), int(row.video_id), float(row.watch_ratio)
        user_gt.setdefault(uid, {})[vid] = wr

    vid_cats = _load_video_categories(args.data_dir)
    all_user_ids = sorted(user_gt.keys())

    queries = all_queries[: args.max_queries]
    print(f"\nCaching CandidateSets: {len(queries)} queries × "
          f"up to {args.max_users} users ...", flush=True)

    # Each entry: (CandidateSet, {vid: capped_watch_ratio}, answer_vids_or_None)
    cached_pairs: list[tuple] = []

    for qi, entry in enumerate(queries, 1):
        query = entry["query"]
        query_cats = set(entry["expected_topics"])
        answer_vids = set(entry.get("answer_video_ids", []))
        print(f"  [{qi:2d}/{len(queries)}] {query!r}", flush=True)

        for uid in all_user_ids[: args.max_users]:
            gt = user_gt[uid]
            relevant_gt: dict[int, float] = {}
            for vid, wr in gt.items():
                v_cats = vid_cats.get(vid, set())
                if wr >= args.relevance_threshold and v_cats & query_cats:
                    relevant_gt[vid] = min(wr, WATCH_RATIO_CAP)
            if not relevant_gt:
                continue

            sorted_vids = sorted(gt.items(), key=lambda x: -x[1])[:5]
            recent_ids = [vid for vid, _ in sorted_vids]

            try:
                cs = pipeline.retrieve(query, user_id=uid, recent_video_ids=recent_ids)
            except Exception:
                continue

            cached_pairs.append((cs, relevant_gt, answer_vids or None))

    print(f"  Cached {len(cached_pairs)} (query, user) evaluation pairs\n")

    if not cached_pairs:
        print("No evaluation pairs — check data and API keys.")
        return

    # ------------------------------------------------------------------
    # 3. Grid search over α and w_query
    # ------------------------------------------------------------------
    alpha_vals = np.linspace(0.0, 1.0, args.alpha_steps)
    wq_vals = np.linspace(0.1, 0.9, args.wq_steps)

    print(f"Grid search: {len(alpha_vals)} α × {len(wq_vals)} w_query, "
          f"optimising nDCG@5 + MRR@3")
    print(f"  α values:     {[f'{v:.2f}' for v in alpha_vals]}")
    print(f"  w_query values: {[f'{v:.2f}' for v in wq_vals]}\n")

    best_score = -1.0
    best_params: dict[str, float] = {}
    all_results: list[dict] = []

    for wq in wq_vals:
        for alpha in alpha_vals:
            ranker = QCSARRanker(**ranker_base, alpha=alpha, w_query=wq)

            ndcgs, mrrs = [], []
            for cs, rel_gt, answer_vids in cached_pairs:
                ranked = ranker.rank(cs)
                ndcgs.append(ndcg_at_k(ranked, rel_gt, 5))
                if answer_vids:
                    mrrs.append(mrr(ranked, answer_vids, k=3))

            mean_ndcg = float(np.mean(ndcgs))
            mean_mrr = float(np.mean(mrrs)) if mrrs else 0.0
            combined = mean_ndcg + mean_mrr  # joint objective

            all_results.append({
                "alpha": float(alpha), "w_query": float(wq),
                "nDCG@5": mean_ndcg, "MRR@3": mean_mrr,
                "combined": combined,
            })

            if combined > best_score:
                best_score = combined
                best_params = {"alpha": float(alpha), "w_query": float(wq),
                               "nDCG@5": mean_ndcg, "MRR@3": mean_mrr}

    # ------------------------------------------------------------------
    # 4. Report
    # ------------------------------------------------------------------
    print("=" * 60)
    print(" α Sweep Results (sorted by combined nDCG@5 + MRR@3)")
    print("=" * 60)
    all_results.sort(key=lambda r: -r["combined"])
    print(f"  {'α':>5} {'w_q':>5} {'nDCG@5':>8} {'MRR@3':>8} {'combined':>10}")
    for r in all_results[:20]:
        print(f"  {r['alpha']:5.2f} {r['w_query']:5.2f} "
              f"{r['nDCG@5']:8.4f} {r['MRR@3']:8.4f} {r['combined']:10.4f}")

    print(f"\n  Best: α={best_params['alpha']:.2f}  w_query={best_params['w_query']:.2f}")
    print(f"        nDCG@5={best_params['nDCG@5']:.4f}  MRR@3={best_params['MRR@3']:.4f}")

    # Compare with default (α=0.5, w_query=0.5)
    ranker_default = QCSARRanker(**ranker_base)
    def_ndcgs, def_mrrs = [], []
    for cs, rel_gt, answer_vids in cached_pairs:
        ranked = ranker_default.rank(cs)
        def_ndcgs.append(ndcg_at_k(ranked, rel_gt, 5))
        if answer_vids:
            def_mrrs.append(mrr(ranked, answer_vids, k=3))
    def_ndcg = float(np.mean(def_ndcgs))
    def_mrr = float(np.mean(def_mrrs)) if def_mrrs else 0.0
    print(f"\n  Default (α=0.5, w_query=0.5): nDCG@5={def_ndcg:.4f}  MRR@3={def_mrr:.4f}")

    # Save
    out_path = "results/tune_alpha.txt"
    pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(f"Best: alpha={best_params['alpha']:.4f}  w_query={best_params['w_query']:.4f}\n")
        f.write(f"nDCG@5={best_params['nDCG@5']:.4f}  MRR@3={best_params['MRR@3']:.4f}\n\n")
        f.write(f"Default: nDCG@5={def_ndcg:.4f}  MRR@3={def_mrr:.4f}\n\n")
        f.write(f"{'alpha':>5} {'w_query':>7} {'nDCG@5':>8} {'MRR@3':>8} {'combined':>10}\n")
        for r in all_results:
            f.write(f"{r['alpha']:5.2f} {r['w_query']:7.2f} "
                    f"{r['nDCG@5']:8.4f} {r['MRR@3']:8.4f} {r['combined']:10.4f}\n")
    print(f"\n  Saved to {out_path}")


if __name__ == "__main__":
    main()
