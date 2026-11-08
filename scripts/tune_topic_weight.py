#!/usr/bin/env python3
"""Tune the router's topic confidence threshold and Stage 3's topic weight.

The LLM parser often returns a neighbouring category (Fashion for "outfit
transition", Short Drama for a Film & TV query) or adds extra topics at
0.4-0.75 confidence. With ``w_cat = 0.5`` the topic bonus (~0.43) outweighs the
whole semantic-score spread of the pool (~0.13), so a wrong topic pushes the
answers out. This sweeps

* ``min_category_confidence`` (router): topics below it get no bonus, and
* ``w_cat`` (retriever): ``score = w_sem·sem + w_cat·confidence·[topic] + w_dur·dur``

with the **live Stage 1 parser** (parsed once per query, optionally cached),
scored by anonymous MRR@3 of the Stage 3 order. On the dev split it picks the
**smallest change** from today's (0.3, 0.5) — highest ``w_cat``, then lowest
threshold — whose MRR@3 is within one query (1 / n_dev) of the best: dev is
small, so differences below one query are noise. Test reports the choice with a
paired bootstrap CI. Gold intents (expected topics at confidence 1.0) are shown
for reference: they always point at the answers' category, so they favour a
high ``w_cat``.

Usage:
    python scripts/tune_topic_weight.py
    python scripts/tune_topic_weight.py --parse-cache parses.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np

_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

from dotenv import load_dotenv

load_dotenv()

from tests.fixtures.query_sets import load_queries

from recsys.contracts.enums import ContentStyle, DurationPreference, QueryIntentType
from recsys.contracts.models import CategoryMatch, QueryIntent
from recsys.retrieval import HybridRetriever
from recsys.routing import QueryRouter

THRESHOLDS = (0.3, 0.5, 0.6, 0.7, 0.8, 0.9)
W_CATS = (0.5, 0.4, 0.3, 0.2, 0.15, 0.1, 0.05, 0.0)


def _parse_all(queries: list[dict], cache_path: pathlib.Path | None) -> dict[str, dict]:
    """``{query_id: {"topics": [[id, conf], ...], "duration": name, "emb": [...]}}``."""
    cache = {}
    if cache_path and cache_path.exists():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
    missing = [q for q in queries if q["id"] not in cache]
    if missing:
        from recsys.intent import BgeM3Embedder, QueryParser

        parser = QueryParser(embedder=BgeM3Embedder())
        for q in missing:
            for attempt in range(5):  # transient SOCLaaS errors
                try:
                    it = parser.parse(q["query"])
                    break
                except Exception:
                    if attempt == 4:
                        raise
                    time.sleep(15 * 2**attempt)
            cache[q["id"]] = {"topics": [[t.category_id, t.confidence] for t in it.topics],
                              "duration": it.duration.name, "emb": it.query_embedding}
            if cache_path:
                cache_path.write_text(json.dumps(cache), encoding="utf-8")
    return cache


def _intent(q: dict, topics: list[tuple[int, float]], duration: str, emb) -> QueryIntent:
    # Retrieval reads only topics, duration and the embedding.
    return QueryIntent(
        raw_query=q["query"], topics=[CategoryMatch(c, "", "", conf) for c, conf in topics],
        style=ContentStyle[q["expected_style"]], duration=DurationPreference[duration],
        intent_type=QueryIntentType[q["expected_intent"]], negative_constraints=[],
        semantic_summary=q["query"], query_embedding=emb,
    )


def _evaluate(retr, router, intents, queries) -> tuple[np.ndarray, np.ndarray]:
    """Per query: (reciprocal rank of the first answer in the top 3, answer in pool)."""
    rr, hit = [], []
    for q, intent in zip(queries, intents):
        ids = [c.video_id for c in retr.retrieve(router.route(intent)).candidates]
        ans = set(q["answer_video_ids"])
        rr.append(next((1 / r for r, v in enumerate(ids[:3], 1) if v in ans), 0.0))
        hit.append(any(v in ans for v in ids))
    return np.array(rr), np.array(hit, dtype=float)


def _bootstrap_ci(diff: np.ndarray, n: int = 5000) -> tuple[float, float]:
    rng = np.random.default_rng(0)
    means = [rng.choice(diff, len(diff)).mean() for _ in range(n)]
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parse-cache", type=pathlib.Path, default=None,
                    help="JSON file to reuse / store the parser outputs")
    ap.add_argument("--out", default="results/tune_topic_weight.txt")
    args = ap.parse_args()

    splits = {s: load_queries("v3", s, "en") for s in ("dev", "test")}
    parsed = _parse_all(splits["dev"] + splits["test"], args.parse_cache)
    retr = HybridRetriever.from_disk(mode="hybrid")
    pool = QueryRouter().route(_intent(splits["dev"][0], [], "ANY", None)).num_candidates

    def run(split, threshold, w_cat, gold=False):
        retr.w_cat = w_cat
        qs = splits[split]
        intents = [
            _intent(q, [(c, 1.0) for c in q["expected_topics"]], q["expected_duration"],
                    parsed[q["id"]]["emb"]) if gold else
            _intent(q, parsed[q["id"]]["topics"], parsed[q["id"]]["duration"],
                    parsed[q["id"]]["emb"])
            for q in qs
        ]
        return _evaluate(retr, QueryRouter(min_category_confidence=threshold), intents, qs)

    lines = [f"Topic threshold x topic weight — v3 en, pool {pool}, live LLM parser intents",
             f"dev {len(splits['dev'])} queries (choose), test {len(splits['test'])} (report)",
             "metric: anonymous MRR@3 of the Stage 3 order; 'in pool' = an answer in the pool", ""]
    grid = {}
    lines.append("dev, parser intents: MRR@3 (in pool)")
    lines.append("threshold " + "".join(f"  w_cat={w:<12}" for w in W_CATS))
    for t in THRESHOLDS:
        row = []
        for w in W_CATS:
            mrr, hit = run("dev", t, w)
            grid[t, w] = mrr.mean()
            row.append(f"  {mrr.mean():.4f} ({hit.mean():.3f})")
        lines.append(f"{t:<9} " + "".join(f"{c:<20}" for c in row))
    # smallest change (highest w_cat, then lowest threshold) within one dev query of the best
    band = max(grid.values()) - 1 / len(splits["dev"]) - 1e-9
    t_best, w_best = max((k for k, v in grid.items() if v >= band), key=lambda k: (k[1], -k[0]))
    chosen = (f"best dev MRR@3 {max(grid.values()):.4f}; within one query (>= {band:.4f}), "
              f"smallest change: min_category_confidence={t_best}, w_cat={w_best}")
    lines += ["", chosen, "", "test                 parser MRR@3  in pool   gold MRR@3  in pool"]
    res = {}
    for label, t, w in (("before (0.3, 0.5)", 0.3, 0.5),
                        (f"chosen ({t_best}, {w_best})", t_best, w_best)):
        res[label] = run("test", t, w), run("test", t, w, gold=True)
        (p_rr, p_hit), (g_rr, g_hit) = res[label]
        lines.append(f"{label:<20} {p_rr.mean():.4f}        {p_hit.mean():.3f}     "
                     f"{g_rr.mean():.4f}      {g_hit.mean():.3f}")
    (before_p, _), (before_g, _) = res["before (0.3, 0.5)"]
    (after_p, _), (after_g, _) = res[f"chosen ({t_best}, {w_best})"]
    for name, a, b in (("parser", after_p, before_p), ("gold", after_g, before_g)):
        lo, hi = _bootstrap_ci(a - b)
        lines.append(f"change, {name:<6} MRR@3 {(a - b).mean():+.4f} [95% CI {lo:+.4f}, "
                     f"{hi:+.4f}]; queries better {(a > b).sum()}, worse {(a < b).sum()}")
    text = "\n".join(lines) + "\n"
    print(text)
    pathlib.Path(args.out).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
