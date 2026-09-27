#!/usr/bin/env python3
"""A/B test: old (fixed diversity) vs new (intent-conditioned + creator dedup).

Caches LLM query parses and user profiles so both ranker configurations
see exactly the same inputs.  The only difference is the ranker.
"""

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

from recsys.data.item_features import caption_category_table
from recsys.data.loader import load_kuairec
from recsys.evaluation.metrics import (
    coverage, mrr, ndcg_at_k, precision_at_k, recall_at_k,
)
from recsys.intent import BgeM3Embedder, QueryParser
from recsys.pipeline.pipeline import RecommendationPipeline
from recsys.ranking import QCSARRanker
from recsys.retrieval import HybridRetriever
from recsys.routing import QueryRouter
from recsys.session_modelling import SessionModeller
from recsys.user_modelling import UserModeller

WATCH_RATIO_CAP = 5.0

# ---- synthetic queries (Chinese) ----
from tests.fixtures.synthetic_queries_zh import SYNTHETIC_QUERIES_ZH as QUERIES

import ast


def _load_video_categories(data_dir: str) -> dict[int, set[int]]:
    df = pd.read_csv(f"{data_dir}/item_categories.csv")
    vid_cats: dict[int, set[int]] = {}
    for row in df.itertuples(index=False):
        cats = ast.literal_eval(row.feat) if isinstance(row.feat, str) else row.feat
        vid_cats[int(row.video_id)] = set(cats)
    return vid_cats


def evaluate_ranker(
    ranker: QCSARRanker,
    cached_candidate_sets: list[tuple[dict, int, "CandidateSet", set, dict]],
    ks: list[int],
    label: str,
) -> dict:
    """Evaluate a ranker on cached CandidateSets."""
    from recsys.contracts.models import CandidateSet

    prec_acc = {k: [] for k in ks}
    rec_acc = {k: [] for k in ks}
    ndcg_acc = {k: [] for k in ks}
    mrr_acc: list[float] = []
    all_ranked = []

    for entry, uid, cs, relevant_set, capped_gt in cached_candidate_sets:
        try:
            ranked = ranker.rank(cs)
        except Exception as exc:
            print(f"  [{label}] user {uid}: ERROR {exc}")
            continue

        all_ranked.append(ranked)
        for k in ks:
            prec_acc[k].append(precision_at_k(ranked, relevant_set, k))
            rec_acc[k].append(recall_at_k(ranked, relevant_set, k))
            ndcg_acc[k].append(ndcg_at_k(ranked, capped_gt, k))
        mrr_acc.append(mrr(ranked, relevant_set))

    n = len(mrr_acc)
    results = {"label": label, "n_pairs": n}
    for k in ks:
        results[f"P@{k}"] = np.mean(prec_acc[k]) if prec_acc[k] else 0
        results[f"R@{k}"] = np.mean(rec_acc[k]) if rec_acc[k] else 0
        results[f"NDCG@{k}"] = np.mean(ndcg_acc[k]) if ndcg_acc[k] else 0
    results["MRR"] = np.mean(mrr_acc) if mrr_acc else 0
    results["coverage"] = coverage(all_ranked, 3327) if all_ranked else 0
    return results


def main() -> None:
    ks = [5, 10, 20]
    data_dir = "data/raw"
    embeddings_path = "data/processed/caption_embeddings.npz"
    max_users = 30
    num_candidates = 200

    # ---- Load data ----
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

    ranker_kwargs: dict = dict(
        item_factors=profiler.item_factors,
        item_ids=profiler.item_ids,
    )
    emb_path = pathlib.Path(embeddings_path)
    if emb_path.exists():
        emb_data = np.load(str(emb_path))
        ranker_kwargs["caption_embeddings"] = emb_data["embeddings"]
        ranker_kwargs["caption_video_ids"] = emb_data["video_ids"]

    parser = QueryParser(embedder=BgeM3Embedder())
    router = QueryRouter(default_num_candidates=num_candidates)
    retriever = HybridRetriever.from_disk(mode="hybrid")

    print(f"  Setup done in {time.time() - t0:.0f}s")

    # ---- Build ground truth ----
    user_gt: dict[int, dict[int, float]] = {}
    for row in small.itertuples(index=False):
        uid, vid, wr = int(row.user_id), int(row.video_id), float(row.watch_ratio)
        user_gt.setdefault(uid, {})[vid] = wr

    vid_cats = _load_video_categories(data_dir)
    all_user_ids = sorted(user_gt.keys())

    # ---- Cache all CandidateSets (Stages 1-3, shared) ----
    print(f"\nCaching {len(QUERIES)} queries × {max_users} users (Stages 1-3) ...\n")
    cached = []
    skipped = 0

    for qi, entry in enumerate(QUERIES, 1):
        query = entry["query"]
        query_cats = set(entry["expected_topics"])
        print(f"[{qi:2d}/{len(QUERIES)}] {query!r}  (cats {query_cats})", flush=True)

        try:
            intent = parser.parse(query)
        except Exception as exc:
            print(f"  Parse error: {exc}")
            continue

        for uid in all_user_ids[:max_users]:
            gt = user_gt[uid]
            relevant_set = set()
            capped_gt: dict[int, float] = {}
            for vid, wr in gt.items():
                v_cats = vid_cats.get(vid, set())
                if wr >= 0.5 and v_cats & query_cats:
                    relevant_set.add(vid)
                    capped_gt[vid] = min(wr, WATCH_RATIO_CAP)
            if not relevant_set:
                skipped += 1
                continue

            sorted_vids = sorted(gt.items(), key=lambda x: -x[1])[:5]
            recent_ids = [vid for vid, _ in sorted_vids]

            try:
                profile = profiler.build_session_profile(uid, intent, recent_video_ids=recent_ids)
                request = router.route(intent, user_id=uid, engagement_profile=profile)
                cs = retriever.retrieve(request)
                # Attach the profile so the ranker can use it
                cs.request.engagement_profile = profile
            except Exception as exc:
                print(f"  user {uid}: Stage 1-3 error: {exc}")
                continue

            cached.append((entry, uid, cs, relevant_set, capped_gt))

    print(f"\nCached {len(cached)} (query, user) pairs  (skipped {skipped} with no relevant)\n")

    # ---- A: Baseline ranker (fixed diversity, no creator dedup) ----
    ranker_a = QCSARRanker(
        **ranker_kwargs,
        diversity_window=3,  # fixed window, old behaviour
    )
    # Override intent diversity config to disable it (old behaviour)
    ranker_a.INTENT_DIVERSITY_CONFIG = {}
    ranker_a.CREATOR_EXCESS_PENALTY = 1.0  # effectively disable creator dedup
    print("Evaluating A (baseline: fixed diversity, no creator dedup) ...")
    res_a = evaluate_ranker(ranker_a, cached, ks, "Baseline")

    # ---- B: New ranker (intent diversity + creator dedup) ----
    ranker_b = QCSARRanker(**ranker_kwargs)
    print("Evaluating B (intent diversity + creator dedup) ...")
    res_b = evaluate_ranker(ranker_b, cached, ks, "New")

    # ---- Report ----
    print()
    print("=" * 70)
    print(" A/B Comparison: Baseline vs Intent Diversity + Creator Dedup")
    print("=" * 70)
    print(f"  Pairs evaluated: {res_a['n_pairs']} (identical for both)")
    print()
    header = f"{'Metric':<12} {'Baseline':>10} {'New':>10} {'Δ':>10} {'Δ%':>8}"
    print(header)
    print("-" * len(header))

    lines = [header, "-" * len(header)]
    for k in ks:
        for metric_prefix, label in [("P", "precision"), ("R", "recall"), ("NDCG", "NDCG")]:
            key = f"{metric_prefix}@{k}"
            a_val = res_a[key]
            b_val = res_b[key]
            delta = b_val - a_val
            pct = (delta / a_val * 100) if a_val > 0 else 0
            line = f"  {key:<10} {a_val:>10.4f} {b_val:>10.4f} {delta:>+10.4f} {pct:>+7.1f}%"
            print(line)
            lines.append(line)
    # MRR
    a_mrr, b_mrr = res_a["MRR"], res_b["MRR"]
    delta_mrr = b_mrr - a_mrr
    pct_mrr = (delta_mrr / a_mrr * 100) if a_mrr > 0 else 0
    line = f"  {'MRR':<10} {a_mrr:>10.4f} {b_mrr:>10.4f} {delta_mrr:>+10.4f} {pct_mrr:>+7.1f}%"
    print(line)
    lines.append(line)
    # Coverage
    a_cov, b_cov = res_a["coverage"], res_b["coverage"]
    delta_cov = b_cov - a_cov
    pct_cov = (delta_cov / a_cov * 100) if a_cov > 0 else 0
    line = f"  {'Coverage':<10} {a_cov:>10.4f} {b_cov:>10.4f} {delta_cov:>+10.4f} {pct_cov:>+7.1f}%"
    print(line)
    lines.append(line)
    print("=" * 70)

    out = "results/ab_test_diversity.txt"
    pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(out).write_text("\n".join(lines) + "\n")
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
