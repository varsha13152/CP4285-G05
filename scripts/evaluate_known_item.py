#!/usr/bin/env python3
"""Known-item search evaluation — a non-circular metric for QCSAR.

For each sampled target video, construct a query from its metadata (caption,
category name, tags) and run the pipeline.  The primary metric is **MRR**:
if the target video surfaces at rank 3, that query contributes 1/3.

Unlike the category-overlap evaluation in ``evaluate_ranking.py``, this
metric is not circular: the ground-truth is a specific video, not a
category label.

Two modes:
  * Default — build a ``QueryIntent`` directly from video metadata
    (fast, deterministic, no API key).
  * ``--use-parser`` — run the full LLM parser on a natural-language query
    derived from the caption (realistic but slow, requires API key).
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
import textwrap
from datetime import datetime

import numpy as np
import pandas as pd

from recsys.contracts.enums import (
    ContentStyle,
    DurationPreference,
    QueryIntentType,
)
from recsys.contracts.models import (
    CategoryMatch,
    CandidateSet,
    EngagementProfile,
    QueryIntent,
    QueryRetrievalRequest,
)
from recsys.data.category_index import CategoryIndex
from recsys.data.item_features import build_item_table
from recsys.data.loader import load_kuairec
from recsys.evaluation.metrics import hit_at_k, reciprocal_rank
from recsys.retrieval.retriever import HybridRetriever, load_catalogue

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Query construction from video metadata (deterministic, no LLM)
# ---------------------------------------------------------------------------

_CAT_INDEX = CategoryIndex()


def _build_query_text(row: pd.Series) -> str:
    """Derive a short search-like query from a video's metadata."""
    parts: list[str] = []
    # Category name (English) as topic anchor
    cat_id = int(row["category_id"])
    entry = _CAT_INDEX.get_by_id(cat_id)
    if entry is not None:
        parts.append(entry.english_name)
    # First meaningful chunk of the video's text field
    text = str(row.get("text", ""))
    words = [w for w in text.split() if w]
    if words:
        # Take up to 6 words, skip category name if already added
        snippet = " ".join(words[:6])
        parts.append(snippet)
    return " ".join(parts).strip()[:120]  # cap length


def _build_gold_intent(
    row: pd.Series, embedding: np.ndarray | None,
) -> QueryIntent:
    """Build a ``QueryIntent`` directly from the video's metadata."""
    cat_id = int(row["category_id"])
    entry = _CAT_INDEX.get_by_id(cat_id)
    topics = []
    if entry is not None:
        topics.append(
            CategoryMatch(cat_id, entry.chinese_name, entry.english_name, confidence=1.0)
        )
    query_text = _build_query_text(row)
    return QueryIntent(
        raw_query=query_text,
        topics=topics,
        style=ContentStyle.NEUTRAL,
        duration=DurationPreference.ANY,
        intent_type=QueryIntentType.SPECIFIC_SEARCH,
        negative_constraints=[],
        semantic_summary=query_text,
        query_embedding=embedding.tolist() if embedding is not None else None,
        confidence=1.0,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default="data/raw")
    p.add_argument("--embeddings", default="data/processed/caption_embeddings.npz")
    p.add_argument("--num-targets", type=int, default=50,
                   help="Number of target videos to sample")
    p.add_argument("--num-candidates", type=int, default=200,
                   help="Stage 3 retrieval pool size")
    p.add_argument("--max-users", type=int, default=10,
                   help="Users per target query")
    p.add_argument("--k", type=int, nargs="+", default=[5, 10, 20])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use-parser", action="store_true",
                   help="Use the live LLM parser instead of gold intents")
    p.add_argument("--verbose", action="store_true")
    p.add_argument(
        "--out",
        default=f"results/known_item_eval_{datetime.now():%Y%m%d_%H%M}.txt",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    # --- Load data ---
    logger.info("Loading data from %s …", args.data_dir)
    data = load_kuairec(args.data_dir, load_big=True)
    items, emb = load_catalogue(args.data_dir, args.embeddings)

    retriever = HybridRetriever(items, emb, mode="hybrid")

    # --- Optional: full pipeline with ranker ---
    ranker = None
    profiler = None
    try:
        from recsys.user_modelling import UserModeller
        from recsys.session_modelling import SessionModeller
        from recsys.ranking import QCSARRanker

        session_mod = SessionModeller(interactions=data.big_matrix)
        profiler = UserModeller(
            interactions=data.big_matrix,
            item_categories=None,  # loaded from disk via caption_category_table
            session_modeller=session_mod,
            social_network=data.social_network,
        )
        ranker = QCSARRanker(
            profiler.item_factors,
            profiler.item_ids,
        )
        # Load caption embeddings for ranker if available
        try:
            with np.load(args.embeddings) as z:
                ranker = QCSARRanker(
                    profiler.item_factors,
                    profiler.item_ids,
                    caption_embeddings=z["embeddings"],
                    caption_video_ids=z["video_ids"],
                )
        except Exception:
            pass
        logger.info("Full pipeline (Stages 1-6) loaded.")
    except Exception as exc:
        logger.warning("Could not load full pipeline (%s); running retrieval-only.", exc)

    # --- Select target videos ---
    # Videos with text and above-median popularity (likely in interactions)
    median_pop = items["popularity"].median()
    eligible = items[items["has_text"] & (items["popularity"] > median_pop)]
    if len(eligible) < args.num_targets:
        logger.warning(
            "Only %d eligible targets (need %d); using all.",
            len(eligible), args.num_targets,
        )
    rng = random.Random(args.seed)
    target_ids = rng.sample(
        list(eligible.index), min(args.num_targets, len(eligible)),
    )
    logger.info("Sampled %d target videos.", len(target_ids))

    # --- Select users ---
    all_users = sorted(data.small_matrix["user_id"].unique())
    user_sample = rng.sample(all_users, min(args.max_users, len(all_users)))

    # --- Embedding lookup for gold intents ---
    emb_by_vid: dict[int, np.ndarray] = {}
    for i, vid in enumerate(items.index):
        if np.linalg.norm(emb[i]) > 0:
            emb_by_vid[int(vid)] = emb[i]

    # --- Evaluate ---
    rr_acc: list[float] = []
    hit_acc: dict[int, list[float]] = {k: [] for k in args.k}
    rank_acc: list[int] = []  # rank of target among hits only
    n_queries = 0

    for target_vid in target_ids:
        row = items.loc[target_vid]
        target_emb = emb_by_vid.get(int(target_vid))

        if args.use_parser:
            query_text = _build_query_text(row)
            try:
                from recsys.intent import QueryParser
                from recsys.intent.embeddings import BgeM3Embedder
                parser = QueryParser(embedder=BgeM3Embedder())
                intent = parser.parse(query_text)
            except Exception as exc:
                logger.warning("Parser failed for '%s': %s", query_text, exc)
                continue
        else:
            intent = _build_gold_intent(row, target_emb)

        if not intent.topics and intent.query_embedding is None:
            continue  # can't retrieve without at least one signal

        for uid in user_sample:
            n_queries += 1
            profile: EngagementProfile | None = None
            if profiler is not None:
                try:
                    profile = profiler.build_session_profile(uid, intent)
                except Exception:
                    profile = None

            # Route
            from recsys.routing import QueryRouter
            router = QueryRouter(default_num_candidates=args.num_candidates)
            request = router.route(intent, user_id=uid, engagement_profile=profile)

            # Retrieve
            candidate_set = retriever.retrieve(request)

            # Rank (if available)
            if ranker is not None:
                ranked = ranker.rank(candidate_set)
            else:
                # Fallback: use retrieval order as ranking
                from recsys.contracts.models import RankedItem, RankedList
                ranked = RankedList(
                    user_id=uid,
                    items=[
                        RankedItem(
                            video_id=c.video_id,
                            rank=i + 1,
                            score=c.score,
                            source=c.source,
                            category_ids=c.category_ids,
                            creator_id=c.creator_id,
                        )
                        for i, c in enumerate(candidate_set.candidates)
                    ],
                    num_candidates_considered=len(candidate_set.candidates),
                    intent_type=intent.intent_type,
                )

            rr = reciprocal_rank(ranked, int(target_vid))
            rr_acc.append(rr)
            for k in args.k:
                hit_acc[k].append(hit_at_k(ranked, int(target_vid), k))

            if rr > 0:
                rank_acc.append(round(1.0 / rr))

    # --- Report ---
    lines: list[str] = []
    lines.append("=" * 60)
    lines.append("Known-Item Search Evaluation")
    lines.append("=" * 60)
    lines.append(f"Targets sampled:  {len(target_ids)}")
    lines.append(f"Users per target: {len(user_sample)}")
    lines.append(f"Total queries:    {n_queries}")
    lines.append(f"Retrieval pool:   {args.num_candidates}")
    lines.append(f"Mode:             {'LLM parser' if args.use_parser else 'gold intent'}")
    lines.append("")
    mrr_val = float(np.mean(rr_acc)) if rr_acc else 0.0
    lines.append(f"MRR:              {mrr_val:.4f}")
    for k in args.k:
        h = float(np.mean(hit_acc[k])) if hit_acc[k] else 0.0
        lines.append(f"Hit@{k:<3d}           {h:.4f}")
    if rank_acc:
        lines.append(f"Mean rank (hits): {np.mean(rank_acc):.1f}")
        lines.append(f"Median rank:      {np.median(rank_acc):.0f}")
    else:
        lines.append("Mean rank (hits): N/A (no hits)")
    lines.append("=" * 60)

    report = "\n".join(lines)
    print(report)

    import pathlib
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.out).write_text(report + "\n", encoding="utf-8")
    logger.info("Saved to %s", args.out)


if __name__ == "__main__":
    main()
