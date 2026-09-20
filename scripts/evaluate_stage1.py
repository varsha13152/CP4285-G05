#!/usr/bin/env python3
"""Evaluate Stage 1 (LLM Intent Understanding) against synthetic queries.

Runs QueryParser on each synthetic query in ``tests/fixtures/synthetic_queries.py``
and reports topic-match accuracy, style/duration/intent accuracy, and latency.

Usage:
    python scripts/evaluate_stage1.py
    python scripts/evaluate_stage1.py --limit 5          # first 5 queries only
    python scripts/evaluate_stage1.py --verbose           # print per-query results
"""

import argparse
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path so we can import from tests/
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from dotenv import load_dotenv

load_dotenv()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate QCSAR Stage 1")
    parser.add_argument("--limit", type=int, default=None, help="Max queries to evaluate")
    parser.add_argument("--verbose", action="store_true", help="Print per-query details")
    args = parser.parse_args()

    from tests.fixtures.synthetic_queries import SYNTHETIC_QUERIES
    from recsys.intent import QueryParser

    queries = SYNTHETIC_QUERIES[: args.limit] if args.limit else SYNTHETIC_QUERIES
    print(f"Evaluating {len(queries)} synthetic queries...\n")

    query_parser = QueryParser()

    topic_hits = 0
    topic_total = 0
    style_hits = 0
    duration_hits = 0
    intent_hits = 0
    total = len(queries)
    latencies: list[float] = []

    for i, entry in enumerate(queries, 1):
        query = entry["query"]
        expected_topics = set(entry["expected_topics"])
        expected_style = entry["expected_style"]
        expected_duration = entry["expected_duration"]
        expected_intent = entry["expected_intent"]

        t0 = time.time()
        result = query_parser.parse(query)
        dt = time.time() - t0
        latencies.append(dt)

        # Topic accuracy (Jaccard-style: intersection / union)
        predicted_topics = {t.category_id for t in result.topics}
        if expected_topics or predicted_topics:
            topic_total += 1
            intersection = expected_topics & predicted_topics
            union = expected_topics | predicted_topics
            jaccard = len(intersection) / len(union) if union else 1.0
            topic_hits += jaccard
        else:
            topic_total += 1
            topic_hits += 1.0  # Both empty = correct

        style_ok = result.style.name == expected_style
        duration_ok = result.duration.name == expected_duration
        intent_ok = result.intent_type.name == expected_intent

        if style_ok:
            style_hits += 1
        if duration_ok:
            duration_hits += 1
        if intent_ok:
            intent_hits += 1

        if args.verbose:
            pred_topics_str = ", ".join(
                f"{t.english_name}({t.category_id})" for t in result.topics
            ) or "(none)"
            status = "✓" if (style_ok and duration_ok and intent_ok) else "✗"
            print(f"[{i:2d}] {status} {query!r}")
            print(f"     Topics:   {pred_topics_str}  (expected: {expected_topics})")
            print(f"     Style:    {result.style.name:<12s} {'✓' if style_ok else '✗ expected ' + expected_style}")
            print(f"     Duration: {result.duration.name:<12s} {'✓' if duration_ok else '✗ expected ' + expected_duration}")
            print(f"     Intent:   {result.intent_type.name:<18s} {'✓' if intent_ok else '✗ expected ' + expected_intent}")
            print(f"     Latency:  {dt:.2f}s")
            print()

    # --- Summary ---
    print("=" * 60)
    print(" Stage 1 Evaluation Summary")
    print("=" * 60)
    print(f"  Queries evaluated: {total}")
    print(f"  Topic accuracy (Jaccard):  {topic_hits / topic_total:.1%}")
    print(f"  Style accuracy:            {style_hits / total:.1%}")
    print(f"  Duration accuracy:         {duration_hits / total:.1%}")
    print(f"  Intent type accuracy:      {intent_hits / total:.1%}")
    print(f"  Avg latency:               {sum(latencies) / len(latencies):.2f}s")
    print(f"  Total time:                {sum(latencies):.1f}s")


if __name__ == "__main__":
    main()
