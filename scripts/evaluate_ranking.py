#!/usr/bin/env python3
"""Offline evaluation of the QCSAR pipeline against ``small_matrix.csv``
ground truth, using the simplified α-blended ranker.

Ranking formula:
    personal = α × long_term + (1 − α) × short_term
    score    = w_query × query_relevance + (1 − w_query) × personal

Usage:
    python scripts/evaluate_ranking.py --alpha 0.5
    python scripts/evaluate_ranking.py --alpha 0.7 --query-set v3 --split test --cases
    python scripts/evaluate_ranking.py --alpha 0.0 --w-query 0.5 --cases --query-set v3

Without ``--cases``, each user's "recent" videos are their 5 most-watched
small_matrix videos, which are also the ground truth (a leak, kept only so
legacy numbers stay comparable). ``--cases`` (v3) uses
tests/fixtures/session_cases_v3.py instead: recent videos from big_matrix
sessions, reported per condition (anonymous / none / related / unrelated) on
two metrics: query relevance (P@5 / MRR on answer_video_ids) and personal
preference (condensed-list nDCG@5 on small_matrix watch_ratio). See
src/recsys/evaluation/README.md.
"""

import argparse
import datetime
import pathlib
import sys
import time

import numpy as np

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
    """{video_id: {caption first-level category id}} — same ids as the query topics."""
    from recsys.data.item_features import video_topic_sets

    return video_topic_sets(data_dir)


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
    ap.add_argument("--query-set", default="legacy", choices=["legacy", "v2", "v3"])
    ap.add_argument("--split", default="all", choices=["dev", "test", "all"])
    ap.add_argument("--cases", action="store_true",
                    help="v3 only: leak-free session cases, metrics per condition")
    ap.add_argument("--alpha", type=float, default=0.5,
                    help="Long-term / short-term ratio (0=pure ST, 1=pure LT; default 0.5)")
    ap.add_argument("--w-query", type=float, default=0.5,
                    help="Query relevance weight (default 0.5)")
    ap.add_argument("--mf-score", action="store_true",
                    help="Add the direct MF score (user_vector · V) as a long-term signal")
    ap.add_argument("--out", default=None,
                    help="Output file (default: results/ranking_eval_{date}.txt)")
    args = ap.parse_args()

    from tests.fixtures.query_sets import load_queries

    if args.cases and args.query_set != "v3":
        ap.error("--cases needs --query-set v3")
    SYNTHETIC_QUERIES = load_queries(args.query_set, args.split, args.lang)

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
        alpha=args.alpha,
        w_query=args.w_query,
        use_mf_score=args.mf_score,
    )
    emb_path = pathlib.Path(args.embeddings)
    if emb_path.exists():
        emb_data = np.load(str(emb_path))
        ranker_kwargs["caption_embeddings"] = emb_data["embeddings"]
        ranker_kwargs["caption_video_ids"] = emb_data["video_ids"]
        print(f"  Caption embeddings loaded ({emb_data['embeddings'].shape})")
    if args.cases:
        # longer list so the condensed (labelled-only) list has enough items to grade
        ranker_kwargs["top_k"] = 100
    ranker = QCSARRanker(**ranker_kwargs)
    print(f"  Ranker: α={args.alpha}, w_query={args.w_query}, mf_score={args.mf_score}")

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
    if args.cases:
        _evaluate_cases(pipeline, queries, user_gt, vid_cats, args)
        return
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
            mrr_acc.append(mrr(ranked, relevant_set, k=3))

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
    p(f"\n  MRR@3 = {avg_mrr:.4f}")
    cov = coverage(all_ranked_lists, total_items)
    p(f"  Coverage = {cov:.4f} ({int(cov * total_items)}/{total_items} items)")

    _save(lines, args)


def _evaluate_cases(pipeline, queries, user_gt, vid_cats, args) -> None:
    """Score the leak-free v3 session cases with two separate metrics.

    One pipeline run per case (full catalogue), graded twice:

    * **Metric 1, query relevance** (guardrail): P@5 and MRR against the query's
      ``answer_video_ids``. Same key for every user, so the baseline is ``anonymous``.
    * **Metric 2, personal preference** (target): nDCG@5 against *this user's*
      small_matrix ``watch_ratio`` (capped) on on-topic videos; off-topic videos
      score 0. Videos without a small_matrix label for the user are skipped
      (condensed list), never counted as misses. Named queries are excluded:
      their category is too broad to stand in for the query. Baseline
      ``anonymous*`` = the query's anonymous ranking graded against the same user.

    ``judged@5`` = share of the first 5 results that carry a label (diagnostic).
    """
    from types import SimpleNamespace

    from tests.fixtures.session_cases_v3 import CONDITION_MIX, SESSION_CASES_V3

    from recsys.evaluation.metrics import mrr, ndcg_at_k, precision_at_k

    cases_by_q: dict[str, list[dict]] = {}
    for c in SESSION_CASES_V3:
        cases_by_q.setdefault(c["query_id"], []).append(c)
    conds = ["anonymous", "anonymous*", "none", "related", "unrelated"]
    m1 = {c: {"P@5": [], "MRR@3": []} for c in conds}
    m2 = {c: {"nDCG@5": [], "judged@5": []} for c in conds}

    def query_relevance(cond, ranked, answers):
        m1[cond]["P@5"].append(precision_at_k(ranked, answers, 5))
        m1[cond]["MRR@3"].append(mrr(ranked, answers, k=3))

    def personal(cond, ranked, labels, topics):
        rel = {v: min(wr, WATCH_RATIO_CAP) if vid_cats.get(v, set()) & topics else 0.0
               for v, wr in labels.items()}
        judged = [i for i in ranked.items if i.video_id in rel]  # condensed list
        m2[cond]["nDCG@5"].append(ndcg_at_k(SimpleNamespace(items=judged), rel, 5))
        m2[cond]["judged@5"].append(sum(i.video_id in rel for i in ranked.items[:5]) / 5)

    def recommend(*a, **kw):
        # ponytail: plain retry for transient LLM API errors (e.g. SOCLaaS 503s)
        for attempt in range(5):
            try:
                return pipeline.recommend(*a, **kw)
            except Exception:  # noqa: BLE001
                if attempt == 4:
                    raise
                time.sleep(15 * 2 ** attempt)  # 15s, 30s, 60s, 120s

    print(f"Evaluating {len(queries)} queries × session cases (leak-free)\n")
    for qi, q in enumerate(queries, 1):
        print(f"[{qi:3d}/{len(queries)}] {q['query']!r}", flush=True)
        topics, answers = set(q["expected_topics"]), set(q["answer_video_ids"])
        scored_personal = q.get("anchor") is None
        anon = None
        for c in cases_by_q.get(q["id"], []):
            if c["user_id"] is None:
                anon = recommend(q["query"])
                query_relevance("anonymous", anon, answers)
                continue
            ranked = recommend(q["query"], user_id=c["user_id"],
                               recent_video_ids=c["recent_video_ids"] or None)
            query_relevance(c["condition"], ranked, answers)
            if scored_personal:
                labels = user_gt.get(c["user_id"], {})
                personal(c["condition"], ranked, labels, topics)
                if c["condition"] == "none" and anon is not None:
                    personal("anonymous*", anon, labels, topics)

    lines: list[str] = []

    def p(msg: str = "") -> None:
        print(msg)
        lines.append(msg)

    def avg(acc, c, m):
        return float(np.mean(acc[c][m])) if acc[c][m] else float("nan")

    p()
    p("=" * 78)
    p(f" QCSAR ranking — v3 session cases ({args.split} split, {args.lang})")
    p("=" * 78)
    p("  Metric 1 — query relevance (answer_video_ids; guardrail, baseline = anonymous)")
    p(f"  {'condition':<12}{'n':>6}{'P@5':>9}{'MRR@3':>9}   lift P@5 / MRR@3")
    for c in ["anonymous", "none", "related", "unrelated"]:
        row = f"  {c:<12}{len(m1[c]['MRR@3']):>6}{avg(m1, c, 'P@5'):>9.4f}{avg(m1, c, 'MRR@3'):>9.4f}"
        if c in CONDITION_MIX:
            row += "   " + " / ".join(f"{avg(m1, c, m) - avg(m1, 'anonymous', m):+.4f}"
                                       for m in ("P@5", "MRR@3"))
        p(row)
    p()
    p("  Metric 2 — personal preference (user's small_matrix watch_ratio, condensed list;")
    p("  non-named queries; target, baseline = anonymous*)")
    p(f"  {'condition':<12}{'n':>6}{'nDCG@5':>9}{'judged@5':>10}   lift nDCG@5")
    for c in ["anonymous*", "none", "related", "unrelated"]:
        row = (f"  {c:<12}{len(m2[c]['nDCG@5']):>6}{avg(m2, c, 'nDCG@5'):>9.4f}"
               f"{avg(m2, c, 'judged@5'):>10.4f}")
        if c in CONDITION_MIX:
            row += f"   {avg(m2, c, 'nDCG@5') - avg(m2, 'anonymous*', 'nDCG@5'):+.4f}"
        p(row)
    p(f"\n  realistic blend (KuaiSAR mix {CONDITION_MIX}):")
    p(f"    P@5={sum(w * avg(m1, c, 'P@5') for c, w in CONDITION_MIX.items()):.4f}  "
      f"MRR@3={sum(w * avg(m1, c, 'MRR@3') for c, w in CONDITION_MIX.items()):.4f}  "
      f"nDCG@5={sum(w * avg(m2, c, 'nDCG@5') for c, w in CONDITION_MIX.items()):.4f}")
    p("  Personalisation works if metric 2 rises vs anonymous* while metric 1 holds.")
    p("  Metric 2 only sees the ~33% of videos in small_matrix (the popular head).")
    _save(lines, args)


def _save(lines: list[str], args) -> None:
    out_path = args.out or f"results/ranking_eval_{datetime.date.today()}.txt"
    pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n  Saved to {out_path}")


if __name__ == "__main__":
    main()
