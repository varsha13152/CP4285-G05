#!/usr/bin/env python3
"""Cheap ALS-vs-LightFM probe on the **long-term** Stage 2 signals only.

No LLM: gold intents (expected topics at confidence 1.0 + expected style /
duration / intent, as in ``scripts/evaluate_stage3.py``) plus one bge-m3
embedding per query. Retrieval runs once per query; every case of that query
reuses the same ``CandidateSet`` with its own profile swapped in (Stage 3 is
query-only, and the per-candidate ``seen`` flag is not read by the ranker).

Cases: v3 test split, ``anonymous`` and ``none`` only (no recent videos, so
the short-term signal is inactive and α does not matter).

Metrics (definitions copied from ``scripts/evaluate_ranking.py``):

* MRR@3 / P@5 on ``answer_video_ids``: none vs anonymous (guardrail: drop ≤ 0.01)
* nDCG@5 personal (condensed list, watch_ratio capped at 5): none vs anonymous*,
  with a paired bootstrap CI
* rare-answer rank: mean rank of answer videos with < 50 big_matrix viewers,
  none vs anonymous (absent from the top-k list = rank top_k + 1)
* model diagnostic per viewer bucket: median ||V|| and the share of each
  video's top-10 cosine neighbours (on V) in the same first-level category

Usage:
    python scripts/probe_long_term.py --backends als lightfm --w-queries 0.5 0.7
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from dotenv import load_dotenv

load_dotenv()

WATCH_RATIO_CAP = 5.0
MRR3_GUARDRAIL_MAX_DROP = 0.01
RARE_VIEWERS = 50
TOP_K = 100
BUCKETS = [(0, 10, "<10"), (10, 50, "10-49"), (50, 100, "50-99"),
           (100, 500, "100-499"), (500, 1000, "500-999"), (1000, 10**9, "1k+")]


def _gold_intent(q: dict, embedding: list[float]):
    """Copy of ``scripts/evaluate_stage3.py::_gold_intent``."""
    from recsys.contracts.enums import ContentStyle, DurationPreference, QueryIntentType
    from recsys.contracts.models import CategoryMatch, QueryIntent
    from recsys.data.category_index import CategoryIndex

    index = CategoryIndex()
    topics = [
        CategoryMatch(c, index.get_by_id(c).chinese_name, index.get_by_id(c).english_name, 1.0)
        for c in q["expected_topics"]
    ]
    return QueryIntent(
        raw_query=q["query"],
        topics=topics,
        style=ContentStyle[q["expected_style"]],
        duration=DurationPreference[q["expected_duration"]],
        intent_type=QueryIntentType[q["expected_intent"]],
        negative_constraints=[],
        semantic_summary=q["query"],
        query_embedding=embedding,
    )


def _load_video_categories(data_dir: str) -> dict[int, set[int]]:
    """{video_id: {caption first-level category id}} — same ids as the query topics."""
    from recsys.data.item_features import video_topic_sets

    return video_topic_sets(data_dir)


def _diagnostic(V: np.ndarray, item_ids: np.ndarray, viewers: pd.Series,
                first_cat: dict[int, int]) -> list[dict]:
    """Per viewer bucket: median ||V|| and same-category share of top-10 neighbours."""
    norms = np.linalg.norm(V, axis=1)
    unit = V / np.where(norms == 0, 1.0, norms)[:, None]
    cats = np.array([first_cat.get(int(v), -1) for v in item_ids])
    n_view = viewers.reindex(item_ids).fillna(0).to_numpy()
    same = np.full(len(V), np.nan)
    for start in range(0, len(V), 1024):
        sims = unit[start:start + 1024] @ unit.T
        rows = np.arange(sims.shape[0])
        sims[rows, start + rows] = -np.inf  # not your own neighbour
        top = np.argpartition(-sims, 10, axis=1)[:, :10]
        own = cats[start:start + 1024]
        hit = (cats[top] == own[:, None]) & (own[:, None] >= 0)
        same[start:start + 1024] = np.where(own >= 0, hit.mean(axis=1), np.nan)
    out = []
    for lo, hi, label in BUCKETS:
        mask = (n_view >= lo) & (n_view < hi)
        out.append({
            "bucket": label, "n": int(mask.sum()),
            "median_norm": float(np.median(norms[mask])) if mask.any() else float("nan"),
            "same_cat@10": float(np.nanmean(same[mask])) if mask.any() else float("nan"),
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", default="data/raw")
    ap.add_argument("--embeddings", default="data/processed/caption_embeddings.npz")
    ap.add_argument("--num-candidates", type=int, default=200)
    ap.add_argument("--backends", nargs="+", default=["als", "lightfm"],
                    choices=["als", "lightfm"])
    ap.add_argument("--w-queries", type=float, nargs="+", default=[0.5, 0.7])
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to cp1252

    from tests.fixtures.query_sets import load_queries
    from tests.fixtures.session_cases_v3 import SESSION_CASES_V3

    from recsys.contracts.models import CandidateSet
    from recsys.data.item_features import caption_category_table
    from recsys.data.loader import load_kuairec
    from recsys.evaluation.metrics import bootstrap_ci, mrr, ndcg_at_k, precision_at_k
    from recsys.intent import BgeM3Embedder
    from recsys.ranking import QCSARRanker
    from recsys.retrieval import HybridRetriever
    from recsys.routing import QueryRouter
    from recsys.user_modelling import UserModeller

    # ------------------------------------------------------------------
    # 1. data, once
    # ------------------------------------------------------------------
    t0 = time.time()
    data = load_kuairec(args.data_dir, load_big=True)
    big = data.big_matrix
    item_cats = caption_category_table(data.caption_categories)
    first_cat = {int(v): int(f[0]) for v, f in zip(item_cats["video_id"], item_cats["feat"])
                 if isinstance(f, (list, tuple)) and len(f)}
    viewers = big.groupby("video_id")["user_id"].nunique()
    user_gt: dict[int, dict[int, float]] = {}
    for u, v, wr in zip(data.small_matrix["user_id"], data.small_matrix["video_id"],
                        data.small_matrix["watch_ratio"]):
        user_gt.setdefault(int(u), {})[int(v)] = float(wr)
    vid_cats = _load_video_categories(args.data_dir)
    emb = np.load(args.embeddings)
    print(f"Data loaded in {time.time() - t0:.0f}s", flush=True)

    # ------------------------------------------------------------------
    # 2. queries, cases, gold intents, one retrieval per query
    # ------------------------------------------------------------------
    queries = load_queries("v3", "test", "en")
    qids = {q["id"] for q in queries}
    cases_by_q: dict[str, list[dict]] = {}
    for c in SESSION_CASES_V3:
        if c["query_id"] in qids and c["condition"] in ("anonymous", "none"):
            cases_by_q.setdefault(c["query_id"], []).append(c)
    n_cases = sum(len(v) for v in cases_by_q.values())
    print(f"{len(queries)} queries, {n_cases} cases (anonymous + none)", flush=True)

    embedder = BgeM3Embedder()
    router = QueryRouter(default_num_candidates=args.num_candidates)
    retriever = HybridRetriever.from_disk(args.data_dir, args.embeddings, mode="hybrid")
    t0 = time.time()
    per_query: list[tuple[dict, object, CandidateSet]] = []
    for q in queries:
        for attempt in range(5):  # transient SOCLaaS errors
            try:
                vec = embedder.embed(q["query"])
                break
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(15 * 2 ** attempt)
        intent = _gold_intent(q, vec)
        per_query.append((q, intent, retriever.retrieve(router.route(intent))))
    print(f"Embedded + retrieved {len(per_query)} queries in {time.time() - t0:.0f}s",
          flush=True)

    # ------------------------------------------------------------------
    # 3. per backend: train, diagnose, rank, score
    # ------------------------------------------------------------------
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for backend in args.backends:
        lines: list[str] = []

        def p(msg: str = "", lines: list[str] = lines) -> None:
            print(msg, flush=True)
            lines.append(msg)

        modeller = UserModeller(
            interactions=big, item_categories=item_cats,
            social_network=data.social_network,
            caption_categories=data.caption_categories, backend=backend,
        )
        p(f"=== backend: {backend} ===")
        p(f"Training time: {modeller.train_seconds:.1f}s "
          f"(factors={modeller.factors}, 15 iterations/epochs)")
        if backend == "lightfm":
            p(f"LightFM item features: {len(modeller.item_ids)} ids + "
              f"{len(modeller.lightfm_feature_names)} shared (category levels + tags)")

        p("")
        p("Model diagnostic (V rows, by # big_matrix viewers):")
        p(f"  {'bucket':>8}  {'videos':>6}  {'median ||v||':>12}  {'same-cat top-10':>15}")
        for r in _diagnostic(modeller.item_factors, modeller.item_ids, viewers, first_cat):
            p(f"  {r['bucket']:>8}  {r['n']:6d}  {r['median_norm']:12.4f}  "
              f"{r['same_cat@10']:15.3f}")

        # profiles are query-independent here (no recent videos): build once per user
        users = {c["user_id"] for cs in cases_by_q.values() for c in cs if c["user_id"]}
        profiles = {u: modeller.build_profile(u) for u in users}

        ranker_kw = {
            "item_factors": modeller.item_factors, "item_ids": modeller.item_ids,
            "caption_embeddings": emb["embeddings"], "caption_video_ids": emb["video_ids"],
            "use_mf_score": True, "top_k": TOP_K,
        }
        for w_query in args.w_queries:
            ranker = QCSARRanker(**ranker_kw, w_query=w_query)
            m1 = {c: {"P@5": [], "MRR@3": []} for c in ("anonymous", "none")}
            paired = {"none": [], "anon": []}
            rare = {"none": [], "anon": []}

            for q, intent, base_cs in per_query:
                answers = set(q["answer_video_ids"])
                rare_answers = [v for v in answers if viewers.get(v, 0) < RARE_VIEWERS]
                topics = set(q["expected_topics"])
                scored_personal = q.get("anchor") is None
                anon = ranker.rank(base_cs)
                anon_rank = {it.video_id: it.rank for it in anon.items}
                for case in cases_by_q.get(q["id"], []):
                    if case["user_id"] is None:
                        m1["anonymous"]["P@5"].append(precision_at_k(anon, answers, 5))
                        m1["anonymous"]["MRR@3"].append(mrr(anon, answers, k=3))
                        continue
                    uid = case["user_id"]
                    req = router.route(intent, user_id=uid, engagement_profile=profiles[uid])
                    ranked = ranker.rank(CandidateSet(
                        user_id=uid, request=req, candidates=base_cs.candidates,
                        source_counts=base_cs.source_counts,
                    ))
                    m1["none"]["P@5"].append(precision_at_k(ranked, answers, 5))
                    m1["none"]["MRR@3"].append(mrr(ranked, answers, k=3))
                    ranks = {it.video_id: it.rank for it in ranked.items}
                    for v in rare_answers:
                        rare["none"].append(ranks.get(v, TOP_K + 1))
                        rare["anon"].append(anon_rank.get(v, TOP_K + 1))
                    if scored_personal:
                        labels = user_gt.get(uid, {})
                        rel = {v: min(wr, WATCH_RATIO_CAP)
                               if vid_cats.get(v, set()) & topics else 0.0
                               for v, wr in labels.items()}
                        for key, rl in (("none", ranked), ("anon", anon)):
                            judged = [i for i in rl.items if i.video_id in rel]
                            paired[key].append(ndcg_at_k(SimpleNamespace(items=judged), rel, 5))

            def mean(xs: list[float]) -> float:
                return float(np.mean(xs)) if xs else float("nan")

            a_mrr, n_mrr = mean(m1["anonymous"]["MRR@3"]), mean(m1["none"]["MRR@3"])
            drop = a_mrr - n_mrr
            delta, lo, hi, sig = bootstrap_ci(paired["none"], paired["anon"])
            p("")
            p(f"w_query={w_query:.1f}  (use_mf_score=True, top_k={TOP_K}, α irrelevant)")
            p(f"  MRR@3   anonymous={a_mrr:.4f}  none={n_mrr:.4f}  drop={drop:+.4f} "
              f"[{'pass' if drop <= MRR3_GUARDRAIL_MAX_DROP else 'FAIL'} ≤ "
              f"{MRR3_GUARDRAIL_MAX_DROP}]")
            p(f"  P@5     anonymous={mean(m1['anonymous']['P@5']):.4f}  "
              f"none={mean(m1['none']['P@5']):.4f}")
            p(f"  nDCG@5  anonymous*={mean(paired['anon']):.4f}  "
              f"none={mean(paired['none']):.4f}  Δ={delta:+.4f} "
              f"[{lo:+.4f}, {hi:+.4f}] {'*' if sig else 'ns'}  (n={len(paired['none'])})")
            p(f"  rare-answer rank (<{RARE_VIEWERS} viewers, absent={TOP_K + 1}): "
              f"anonymous={mean(rare['anon']):.1f}  none={mean(rare['none']):.1f}  "
              f"Δ={mean(rare['none']) - mean(rare['anon']):+.1f}  "
              f"(n={len(rare['none'])} case×video pairs)")

        path = out_dir / f"probe_long_term_{backend}.txt"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"\nSaved {path}\n", flush=True)


if __name__ == "__main__":
    main()
