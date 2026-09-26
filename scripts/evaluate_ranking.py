#!/usr/bin/env python3
"""Offline evaluation of the full QCSAR pipeline (Stages 1-6) against
``small_matrix.csv`` ground truth.

Uses **query-aware** ground truth: a user's video is relevant only if
(a) watch_ratio >= threshold AND (b) the video's categories overlap with
the query's expected topics. This measures "did we find on-topic videos
the user enjoyed" rather than "did we find all videos the user enjoyed."

Usage:
    python scripts/evaluate_ranking.py
    python scripts/evaluate_ranking.py --max-users 20 --k 5 10 20
    python scripts/evaluate_ranking.py --verbose --relevance-threshold 0.5
"""

import argparse
import ast
import datetime
import pathlib
import sys
import time

import numpy as np
import pandas as pd

# Ensure project root is on sys.path so we can import from tests/
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from dotenv import load_dotenv

load_dotenv()

# Cap extreme watch_ratios to prevent NDCG denominator distortion
# (some users have watch_ratio > 500; one outlier dominates ideal DCG).
WATCH_RATIO_CAP = 5.0


def _load_video_categories(data_dir: str) -> dict[int, set[int]]:
    """Build {video_id: {first-level category IDs}} from item_categories.csv."""
    df = pd.read_csv(f"{data_dir}/item_categories.csv")
    vid_cats: dict[int, set[int]] = {}
    for row in df.itertuples(index=False):
        cats = ast.literal_eval(row.feat) if isinstance(row.feat, str) else row.feat
        vid_cats[int(row.video_id)] = set(cats)
    return vid_cats


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate QCSAR ranking (Stages 1-6)")
    ap.add_argument("--data-dir", default="data/raw")
    ap.add_argument("--embeddings", default="data/processed/caption_embeddings.npz")
    ap.add_argument("--k", type=int, nargs="+", default=[5, 10, 20])
    ap.add_argument("--relevance-threshold", type=float, default=0.5,
                    help="watch_ratio >= this is considered relevant (default 0.5)")
    ap.add_argument("--max-users", type=int, default=30,
                    help="Max users per query (for speed)")
    ap.add_argument("--max-queries", type=int, default=None,
                    help="Limit synthetic queries to evaluate")
    ap.add_argument("--num-candidates", type=int, default=200,
                    help="Candidate pool size (default 200)")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--lang", default="en", choices=["en", "zh", "both"],
                    help="Query language: en (English), zh (Chinese), both")
    ap.add_argument("--out", default=None,
                    help="Output file (default: results/ranking_eval_{date}.txt)")
    args = ap.parse_args()

    from tests.fixtures.synthetic_queries import SYNTHETIC_QUERIES
    from tests.fixtures.synthetic_queries_zh import SYNTHETIC_QUERIES_ZH

    if args.lang == "zh":
        all_queries = SYNTHETIC_QUERIES_ZH
    elif args.lang == "both":
        all_queries = SYNTHETIC_QUERIES + SYNTHETIC_QUERIES_ZH
    else:
        all_queries = SYNTHETIC_QUERIES
    SYNTHETIC_QUERIES = all_queries  # noqa: F841 — reassign for the rest of main()

    from recsys.data.item_features import caption_category_table
    from recsys.data.loader import load_kuairec
    from recsys.evaluation.metrics import coverage, mrr, ndcg_at_k, precision_at_k, recall_at_k
    from recsys.intent import BgeM3Embedder, QueryParser
    from recsys.pipeline.pipeline import RecommendationPipeline
    from recsys.ranking import QCSARRanker
    from recsys.retrieval import HybridRetriever
    from recsys.routing import QueryRouter
    from recsys.session_modelling import SessionModeller
    from recsys.user_modelling import UserModeller

    # ------------------------------------------------------------------
    # 1. Load data & build models
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

    # Build ranker
    ranker_kwargs: dict = dict(
        item_factors=profiler.item_factors,
        item_ids=profiler.item_ids,
    )
    emb_path = pathlib.Path(args.embeddings)
    if emb_path.exists():
        emb_data = np.load(str(emb_path))
        ranker_kwargs["caption_embeddings"] = emb_data["embeddings"]
        ranker_kwargs["caption_video_ids"] = emb_data["video_ids"]
        print(f"  Caption embeddings loaded ({emb_data['embeddings'].shape})")
    ranker = QCSARRanker(**ranker_kwargs)

    pipeline = RecommendationPipeline(
        parser=QueryParser(embedder=BgeM3Embedder()),
        router=QueryRouter(default_num_candidates=args.num_candidates),
        retriever=HybridRetriever.from_disk(mode="hybrid"),
        user_profiler=profiler,
        ranker=ranker,
    )
    print(f"  Setup done in {time.time() - t0:.0f}s")

    # ------------------------------------------------------------------
    # 2. Build per-user ground truth + per-video categories
    # ------------------------------------------------------------------
    # {user_id: {video_id: watch_ratio}}
    user_gt: dict[int, dict[int, float]] = {}
    for row in small.itertuples(index=False):
        uid, vid, wr = int(row.user_id), int(row.video_id), float(row.watch_ratio)
        user_gt.setdefault(uid, {})[vid] = wr

    vid_cats = _load_video_categories(args.data_dir)
    all_user_ids = sorted(user_gt.keys())
    total_items = small["video_id"].nunique()

    print(f"  {len(all_user_ids)} users, {total_items} items, "
          f"{len(vid_cats)} items with categories\n")

    # ------------------------------------------------------------------
    # 3. Evaluate each query × user
    # ------------------------------------------------------------------
    queries = SYNTHETIC_QUERIES[:args.max_queries] if args.max_queries else SYNTHETIC_QUERIES
    print(f"Evaluating {len(queries)} queries × up to {args.max_users} users each")
    print(f"  relevance: watch_ratio >= {args.relevance_threshold}, "
          f"query-aware (topic-filtered), "
          f"NDCG capped at {WATCH_RATIO_CAP}\n")

    all_ranked_lists: list = []
    prec_acc: dict[int, list[float]] = {k: [] for k in args.k}
    rec_acc: dict[int, list[float]] = {k: [] for k in args.k}
    ndcg_acc: dict[int, list[float]] = {k: [] for k in args.k}
    mrr_acc: list[float] = []
    skipped_no_relevant = 0

    for qi, entry in enumerate(queries, 1):
        query = entry["query"]
        query_cats = set(entry["expected_topics"])
        print(f"[{qi:2d}/{len(queries)}] {query!r}  (cats {query_cats})", flush=True)

        threshold = args.relevance_threshold
        q_t0 = time.time()
        q_pairs = 0

        for uid in all_user_ids[:args.max_users]:
            gt = user_gt[uid]

            # ---- Query-aware ground truth ----
            # A video is relevant only if the user enjoyed it AND it matches
            # the query's topic categories.
            relevant_set = set()
            capped_gt: dict[int, float] = {}
            for vid, wr in gt.items():
                v_cats = vid_cats.get(vid, set())
                if wr >= threshold and v_cats & query_cats:
                    relevant_set.add(vid)
                    capped_gt[vid] = min(wr, WATCH_RATIO_CAP)

            if not relevant_set:
                skipped_no_relevant += 1
                continue

            # Use 5 most-watched videos as recent session context.
            sorted_vids = sorted(gt.items(), key=lambda x: -x[1])[:5]
            recent_ids = [vid for vid, _ in sorted_vids]

            try:
                ranked = pipeline.recommend(query, user_id=uid, recent_video_ids=recent_ids)
            except Exception as exc:
                if args.verbose:
                    print(f"    user {uid}: ERROR {exc}")
                continue

            all_ranked_lists.append(ranked)
            q_pairs += 1

            for k in args.k:
                prec_acc[k].append(precision_at_k(ranked, relevant_set, k))
                rec_acc[k].append(recall_at_k(ranked, relevant_set, k))
                ndcg_acc[k].append(ndcg_at_k(ranked, capped_gt, k))
            mrr_acc.append(mrr(ranked, relevant_set))

        elapsed = time.time() - q_t0
        if args.verbose and q_pairs:
            print(f"    {q_pairs} users with on-topic relevant videos, {elapsed:.1f}s")

    # ------------------------------------------------------------------
    # 4. Report
    # ------------------------------------------------------------------
    lines: list[str] = []

    def p(msg: str = "") -> None:
        print(msg)
        lines.append(msg)

    p()
    p("=" * 60)
    p(" QCSAR Ranking Evaluation (Stages 1-6)")
    p("=" * 60)
    p(f"  Queries: {len(queries)}")
    p(f"  (query, user) pairs evaluated: {len(mrr_acc)}")
    p(f"  Skipped (no on-topic relevant videos): {skipped_no_relevant}")
    p(f"  Relevance: watch_ratio >= {args.relevance_threshold} "
      f"AND category overlap (query-aware)")
    p(f"  NDCG relevance capped at {WATCH_RATIO_CAP}")
    p(f"  Candidate pool: {args.num_candidates}")
    p()

    if not mrr_acc:
        p("  No evaluations completed — check data paths and API keys.")
        _save(lines, args)
        return

    for k in args.k:
        n = len(prec_acc[k])
        avg_p = sum(prec_acc[k]) / n
        avg_r = sum(rec_acc[k]) / n
        avg_n = sum(ndcg_acc[k]) / n
        p(f"  @{k:>2d}  precision={avg_p:.4f}  recall={avg_r:.4f}  NDCG={avg_n:.4f}")

    avg_mrr = sum(mrr_acc) / len(mrr_acc)
    p(f"\n  MRR = {avg_mrr:.4f}")
    cov = coverage(all_ranked_lists, total_items)
    p(f"  Coverage = {cov:.4f} ({int(cov * total_items)}/{total_items} items)")

    _save(lines, args)


def _save(lines: list[str], args) -> None:
    out_path = args.out or f"results/ranking_eval_{datetime.date.today()}.txt"
    pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n  Saved to {out_path}")


if __name__ == "__main__":
    main()
