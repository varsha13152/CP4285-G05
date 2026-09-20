#!/usr/bin/env python3
"""Run the QCSAR pipeline — Stage 1 (LLM Intent Understanding).

Usage:
    python scripts/run_pipeline.py --query "fun cooking videos for students"
    python scripts/run_pipeline.py --query "cute cat videos" --user_id 14
    python scripts/run_pipeline.py --query "how to do makeup for beginners"
"""

import argparse
import sys
import time

from dotenv import load_dotenv

load_dotenv()


def main() -> None:
    parser = argparse.ArgumentParser(description="QCSAR Stage 1 — Query Intent Understanding")
    parser.add_argument("--query", required=True, help="Natural-language search query")
    parser.add_argument("--user_id", type=int, default=None, help="Optional user ID")
    parser.add_argument("--num_candidates", type=int, default=100)
    args = parser.parse_args()

    from recsys.intent import QueryParser
    from recsys.routing import QueryRouter

    # --- Stage 1: Parse query ---
    print(f"Query: {args.query!r}\n")

    t0 = time.time()
    query_parser = QueryParser()
    intent = query_parser.parse(args.query)
    parse_time = time.time() - t0

    print("=" * 60)
    print(" Stage 1: LLM Intent Understanding")
    print("=" * 60)
    print(f"  Intent type:   {intent.intent_type.name}")
    print(f"  Style:         {intent.style.name}")
    print(f"  Duration:      {intent.duration.name}")
    print(f"  Confidence:    {intent.confidence:.2f}")
    print(f"  Summary:       {intent.semantic_summary}")
    if intent.negative_constraints:
        print(f"  Constraints:   {intent.negative_constraints}")
    print(f"  Topics:")
    for t in intent.topics:
        print(f"    - {t.english_name} ({t.chinese_name}, id={t.category_id}, conf={t.confidence:.2f})")
    if not intent.topics:
        print("    (none matched)")
    print(f"  Parse time:    {parse_time:.2f}s")

    # --- Routing ---
    router = QueryRouter(default_num_candidates=args.num_candidates)
    request = router.route(intent, user_id=args.user_id)

    print()
    print("=" * 60)
    print(" Routing → QueryRetrievalRequest")
    print("=" * 60)
    strategies_str = ", ".join(
        f"{sw.strategy.name}={sw.weight:.1f}" for sw in request.strategies
    )
    print(f"  Strategies:      {strategies_str}")
    print(f"  Category filter: {request.category_filter}")
    print(f"  Duration range:  {request.duration_range_ms}")
    print(f"  Num candidates:  {request.num_candidates}")
    print(f"  User ID:         {request.user_id}")

    print()
    print("(Stages 2-6 not yet implemented — retrieval and ranking are teammate modules)")


if __name__ == "__main__":
    main()
