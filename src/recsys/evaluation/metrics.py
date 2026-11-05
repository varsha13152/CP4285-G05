"""Standard recommendation-system evaluation metrics."""

from __future__ import annotations

import numpy as np

from recsys.contracts.models import RankedList


def precision_at_k(ranked: RankedList, ground_truth: set[int], k: int = 10) -> float:
    """Fraction of top-k items that are relevant."""
    top_k_ids = [item.video_id for item in ranked.items[:k]]
    hits = sum(1 for vid in top_k_ids if vid in ground_truth)
    return hits / k if k > 0 else 0.0


def recall_at_k(ranked: RankedList, ground_truth: set[int], k: int = 10) -> float:
    """Fraction of relevant items that appear in the top-k."""
    if not ground_truth:
        return 0.0
    top_k_ids = [item.video_id for item in ranked.items[:k]]
    hits = sum(1 for vid in top_k_ids if vid in ground_truth)
    return hits / len(ground_truth)


def ndcg_at_k(
    ranked: RankedList, ground_truth: dict[int, float], k: int = 10
) -> float:
    """Normalized Discounted Cumulative Gain at k.

    Parameters
    ----------
    ground_truth : dict[int, float]
        video_id → relevance score (e.g., watch_ratio).
    """
    top_k_ids = [item.video_id for item in ranked.items[:k]]

    # DCG
    dcg = 0.0
    for i, vid in enumerate(top_k_ids):
        rel = ground_truth.get(vid, 0.0)
        dcg += rel / np.log2(i + 2)  # i+2 because rank is 1-indexed

    # Ideal DCG
    ideal_rels = sorted(ground_truth.values(), reverse=True)[:k]
    idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_rels))

    return dcg / idcg if idcg > 0 else 0.0


def mrr(ranked: RankedList, ground_truth: set[int], k: int | None = None) -> float:
    """Mean Reciprocal Rank — 1/rank of the first relevant item.

    Parameters
    ----------
    k : int | None
        If set, only consider the top-k items.  Items beyond rank *k*
        are treated as misses (returns 0.0).
    """
    for item in ranked.items[:k]:
        if item.video_id in ground_truth:
            return 1.0 / item.rank
    return 0.0


def coverage(all_ranked: list[RankedList], total_items: int) -> float:
    """Fraction of all items that appear in at least one recommendation list."""
    recommended = set()
    for ranked in all_ranked:
        for item in ranked.items:
            recommended.add(item.video_id)
    return len(recommended) / total_items if total_items > 0 else 0.0
