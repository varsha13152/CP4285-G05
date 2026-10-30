#!/usr/bin/env python3
"""Automated weight tuning for QCSARRanker (Stages 4-6).

Caches the output of Stages 1-3 (CandidateSet objects), then grid-searches
over Stage 4 weight combinations evaluating only the fast ranking stage
(~0.01s per evaluation).

Usage:
    python scripts/tune_weights.py
    python scripts/tune_weights.py --max-users 15 --max-queries 10
    python scripts/tune_weights.py --lang zh --grid-steps 4
"""

from __future__ import annotations

import argparse
import ast
import itertools
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
    ap = argparse.ArgumentParser(description="Tune QCSARRanker Stage 4 weights")
    ap.add_argument("--data-dir", default="data/raw")
    ap.add_argument("--embeddings", default="data/processed/caption_embeddings.npz")
    ap.add_argument("--k", type=int, default=10, help="Optimise NDCG@k (default 10)")
    ap.add_argument("--relevance-threshold", type=float, default=0.5)
    ap.add_argument("--max-users", type=int, default=20)
    ap.add_argument("--max-queries", type=int, default=10)
    ap.add_argument("--num-candidates", type=int, default=200)
    ap.add_argument("--lang", default="zh", choices=["en", "zh", "both"])
    ap.add_argument("--query-set", default="legacy", choices=["legacy", "v2", "v3"],
                    help="v2/v3 tune on their dev split")
    ap.add_argument("--grid-steps", type=int, default=5,
                    help="Number of grid values per weight (default 5)")
    args = ap.parse_args()

    from tests.fixtures.query_sets import load_queries

    # v2/v3: tune on the dev split only, so reported test numbers aren't overfitted
    all_queries = load_queries(args.query_set, "all" if args.query_set == "legacy" else "dev",
                               args.lang)

    from recsys.data.item_features import caption_category_table
    from recsys.data.loader import load_kuairec
    from recsys.evaluation.metrics import ndcg_at_k
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

    # Each entry: (CandidateSet, {vid: capped_watch_ratio} for relevant vids)
    cached_pairs: list[tuple] = []

    for qi, entry in enumerate(queries, 1):
        query = entry["query"]
        query_cats = set(entry["expected_topics"])
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

            cached_pairs.append((cs, relevant_gt))

    print(f"  Cached {len(cached_pairs)} (query, user) evaluation pairs\n")

    if not cached_pairs:
        print("No evaluation pairs — check data and API keys.")
        return

    # ------------------------------------------------------------------
    # 3. Grid search over Stage 4 weights + Stage 5 alpha
    # ------------------------------------------------------------------
    k = args.k
    steps = args.grid_steps
    grid_vals = np.linspace(0.05, 0.50, steps)
    alpha_vals = np.linspace(0.3, 0.9, 4)  # Stage 5 blending

    # Stage 4 has 7 weights: query, query_sem, popularity, cf, cat_affinity,
    # social, markov.  Grid-search the first 5; social+markov split remainder.
    print(f"Grid search: {steps} values × 5 primary weights + "
          f"{len(alpha_vals)} alpha, optimising NDCG@{k} ...")
    print(f"  Grid values: {[f'{v:.2f}' for v in grid_vals]}")
    print(f"  Alpha values: {[f'{v:.2f}' for v in alpha_vals]}")

    best_ndcg = -1.0
    best_weights: dict[str, float] = {}
    combos_evaluated = 0
    t0 = time.time()

    for w_query, w_qsem, w_pop, w_cf, w_cat in itertools.product(
        grid_vals, grid_vals, grid_vals, grid_vals, grid_vals,
    ):
        remainder = 1.0 - (w_query + w_qsem + w_pop + w_cf + w_cat)
        if remainder < 0.02:
            continue  # not enough left for social + markov
        w_social = remainder / 2
        w_markov = remainder / 2

        for alpha in alpha_vals:
            ranker = QCSARRanker(
                **ranker_base,
                w_query=w_query,
                w_query_sem=w_qsem,
                w_popularity=w_pop,
                w_cf=w_cf,
                w_cat_affinity=w_cat,
                w_social=w_social,
                w_session_markov=w_markov,
                stage5_alpha=alpha,
            )

            ndcgs = []
            for cs, rel_gt in cached_pairs:
                ranked = ranker.rank(cs)
                ndcgs.append(ndcg_at_k(ranked, rel_gt, k))

            mean_ndcg = float(np.mean(ndcgs))
            combos_evaluated += 1

            if mean_ndcg > best_ndcg:
                best_ndcg = mean_ndcg
                best_weights = {
                    "w_query": float(w_query),
                    "w_query_sem": float(w_qsem),
                    "w_popularity": float(w_pop),
                    "w_cf": float(w_cf),
                    "w_cat_affinity": float(w_cat),
                    "w_social": float(w_social),
                    "w_session_markov": float(w_markov),
                    "stage5_alpha": float(alpha),
                }
                if combos_evaluated % 200 == 0 or combos_evaluated <= 5:
                    print(f"    [{combos_evaluated:>5d}] new best NDCG@{k}={mean_ndcg:.4f}  "
                          f"q={w_query:.2f} qs={w_qsem:.2f} p={w_pop:.2f} "
                          f"cf={w_cf:.2f} ca={w_cat:.2f} α={alpha:.2f}")

    elapsed = time.time() - t0
    print(f"\n  Evaluated {combos_evaluated} combinations in {elapsed:.1f}s "
          f"({combos_evaluated / max(elapsed, 0.01):.0f}/s)")

    # ------------------------------------------------------------------
    # 4. Report
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(" Best Stage 4 + 5 Weights")
    print("=" * 60)
    print(f"  NDCG@{k} = {best_ndcg:.4f}\n")
    for name, val in best_weights.items():
        print(f"  {name:>20s} = {val:.4f}")
    print()

    # Show what a QCSARRanker constructor call looks like
    print("  # Suggested QCSARRanker kwargs:")
    print("  QCSARRanker(")
    print("      ...,")
    for name, val in best_weights.items():
        print(f"      {name}={val:.4f},")
    print("  )")

    # Also evaluate default weights for comparison
    ranker_default = QCSARRanker(**ranker_base)
    default_ndcgs = [
        ndcg_at_k(ranker_default.rank(cs), rel_gt, k)
        for cs, rel_gt in cached_pairs
    ]
    default_mean = float(np.mean(default_ndcgs))
    print(f"\n  Default weights NDCG@{k} = {default_mean:.4f}")
    print(f"  Improvement: {best_ndcg - default_mean:+.4f} "
          f"({(best_ndcg / max(default_mean, 1e-9) - 1) * 100:+.1f}%)")

    # Save results
    out_path = "results/tune_weights.txt"
    pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(f"NDCG@{k} = {best_ndcg:.4f}\n")
        for name, val in best_weights.items():
            f.write(f"{name} = {val:.4f}\n")
        f.write(f"\nDefault NDCG@{k} = {default_mean:.4f}\n")
        f.write(f"Improvement: {best_ndcg - default_mean:+.4f}\n")
    print(f"\n  Saved to {out_path}")


if __name__ == "__main__":
    main()
