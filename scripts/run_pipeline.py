#!/usr/bin/env python3
"""Run the QCSAR search pipeline, Stages 1–6 (query → ranked list).

Without ``--user_id``, runs Stages 1–3 only (query → candidate pool).
With ``--user_id``, runs the full pipeline including Stage 4-6 ranking.

Needs ``.env`` (SOCLaaS) and ``data/processed/caption_embeddings.npz``
(``python scripts/build_caption_embeddings.py``).

Usage:
    python scripts/run_pipeline.py --query "cute cat videos"
    python scripts/run_pipeline.py --query "cute cat videos" --user_id 14 --recent 10,20,30
    python scripts/run_pipeline.py --query "home workout routines" --mode semantic

With --user_id, Stage 2 trains ALS + the session model on big_matrix first (~1 min).
"""

import argparse
import pathlib
import time

import numpy as np
from dotenv import load_dotenv

load_dotenv()


def _build_profiler():
    from recsys.data.item_features import caption_category_table
    from recsys.data.loader import load_kuairec
    from recsys.session_modelling import SessionModeller
    from recsys.user_modelling import UserModeller

    data = load_kuairec(load_big=True)  # load big_matrix once for both models
    return UserModeller(
        interactions=data.big_matrix,
        item_categories=caption_category_table(data.caption_categories),
        social_network=data.social_network,
        session_modeller=SessionModeller(interactions=data.big_matrix),
    )


def _build_ranker(profiler, emb_path: str = "data/processed/caption_embeddings.npz"):
    """Construct a QCSARRanker from a trained profiler + optional caption embeddings."""
    from recsys.ranking import QCSARRanker

    kwargs: dict = dict(
        item_factors=profiler.item_factors,
        item_ids=profiler.item_ids,
    )
    if pathlib.Path(emb_path).exists():
        data = np.load(emb_path)
        kwargs["caption_embeddings"] = data["embeddings"]
        kwargs["caption_video_ids"] = data["video_ids"]
    return QCSARRanker(**kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description="QCSAR search pipeline")
    parser.add_argument("--query", required=True, help="Natural-language search query")
    parser.add_argument("--user_id", type=int, default=None, help="Optional user ID")
    parser.add_argument("--recent", default="", help="Comma-separated recent video ids")
    parser.add_argument("--num_candidates", type=int, default=100)
    parser.add_argument("--mode", default="hybrid", choices=["semantic", "category", "hybrid"])
    parser.add_argument("--show", type=int, default=20, help="Items to print")
    args = parser.parse_args()

    from recsys.intent import BgeM3Embedder, QueryParser
    from recsys.pipeline.pipeline import RecommendationPipeline
    from recsys.retrieval import HybridRetriever
    from recsys.routing import QueryRouter

    recent = [int(v) for v in args.recent.split(",") if v.strip()]
    profiler = None
    ranker = None
    if args.user_id is not None:
        t0 = time.time()
        print("Training Stage 2 (ALS + session model) ...", flush=True)
        profiler = _build_profiler()
        print(f"  done in {time.time() - t0:.0f}s")
        ranker = _build_ranker(profiler)
        has_emb = ranker._emb is not None
        print(f"  Stage 4-6 ranker ready (caption embeddings: {has_emb})\n")

    pipeline = RecommendationPipeline(
        parser=QueryParser(embedder=BgeM3Embedder()),
        router=QueryRouter(default_num_candidates=args.num_candidates),
        retriever=HybridRetriever.from_disk(mode=args.mode),
        user_profiler=profiler,
        ranker=ranker,
    )

    t0 = time.time()
    if ranker is not None:
        ranked = pipeline.recommend(args.query, user_id=args.user_id, recent_video_ids=recent)
        elapsed = time.time() - t0
        _print_ranked(args, ranked, elapsed)
    else:
        result = pipeline.retrieve(args.query, user_id=args.user_id, recent_video_ids=recent)
        elapsed = time.time() - t0
        _print_candidates(args, result, elapsed)


def _print_ranked(args, ranked, elapsed: float) -> None:
    """Print the final ranked list (Stages 1-6)."""
    print("=" * 60)
    print(f" Query: {args.query!r}")
    print("=" * 60)
    print(f"  User: {ranked.user_id}  intent: {ranked.intent_type.name}")
    print(f"  Candidates considered: {ranked.num_candidates_considered}")
    print(f"  Pipeline time: {elapsed:.2f}s\n")
    print(f"  {'rank':>4} {'video':>7} {'score':>8} {'source':>8} {'categories'}")
    for item in ranked.items[: args.show]:
        cats = ",".join(str(c) for c in item.category_ids) or "-"
        print(f"  {item.rank:>4} {item.video_id:>7} {item.score:8.4f} "
              f"{item.source.name:>8} {cats}")
    if len(ranked.items) > args.show:
        print(f"  ... ({len(ranked.items) - args.show} more)")


def _print_candidates(args, result, elapsed: float) -> None:
    """Print the Stage 3 candidate pool (no ranking)."""
    intent = result.request.query_intent
    print("=" * 60)
    print(f" Query: {args.query!r}")
    print("=" * 60)
    print(f"  Intent: {intent.intent_type.name}  style {intent.style.name}  "
          f"duration {intent.duration.name}")
    print("  Topics: " + ", ".join(
        f"{t.english_name}({t.category_id}, {t.confidence:.2f})" for t in intent.topics
    ) or "  Topics: (none)")
    print(f"  Category topics: {result.request.category_filter}  "
          f"duration target (soft): {result.request.duration_range_ms}")

    profile = result.request.engagement_profile
    if profile is not None:
        ctx = profile.session_context
        print(f"\n  Stage 2 profile for user {profile.user_id}: "
              f"cf {len(profile.cf_neighbour_scores)} videos, "
              f"social {len(profile.social_scores)} videos, "
              f"session {'-' if ctx is None else len(ctx.recent_video_ids)} recent")

    print(f"\n  Stage 3 ({args.mode}): {len(result.candidates)} candidates "
          f"{result.source_counts} in {elapsed:.2f}s (incl. LLM parse + embedding)\n")
    print(f"  {'video':>7} {'score':>6} {'source':>8} {'cat':>4} {'sem':>5} {'dur':>4} seen")
    for c in result.candidates[: args.show]:
        f = c.features
        print(f"  {c.video_id:>7} {c.score:6.3f} {c.source.name:>8} "
              f"{(c.category_ids or ['-'])[0]:>4} {f['sem']:5.2f} {f['dur_fit']:4.2f} "
              f"{int(f['seen'])}")
    print("\n(Anonymous mode — pass --user_id for personalised Stage 4-6 ranking.)")


if __name__ == "__main__":
    main()
