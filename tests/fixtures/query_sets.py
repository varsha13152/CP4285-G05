"""One loader for the synthetic query sets, shared by the evaluation scripts.

* ``legacy`` — the original 27 queries (English, or the separate Chinese fixture)
* ``v2``     — 100 catalogue-grounded queries (PR #5)
* ``v3``     — 100 KuaiSAR-shaped queries + 15 refinements, each with ``answer_video_ids``

``split`` (dev / test / all) applies to v2 and v3; legacy has no split.
``lang`` zh swaps in ``query_zh``; both returns each query twice.
"""

from __future__ import annotations

QUERY_SETS = ("legacy", "v2", "v3")


def load_queries(query_set: str = "legacy", split: str = "all", lang: str = "en") -> list[dict]:
    if query_set == "legacy":
        from tests.fixtures.synthetic_queries import SYNTHETIC_QUERIES as en
        from tests.fixtures.synthetic_queries_zh import SYNTHETIC_QUERIES_ZH as zh
        return {"en": en, "zh": zh, "both": en + zh}[lang]
    if query_set == "v2":
        from tests.fixtures.synthetic_queries_v2 import SYNTHETIC_QUERIES_V2 as qs
    elif query_set == "v3":
        from tests.fixtures.synthetic_queries_v3 import SYNTHETIC_QUERIES_V3 as qs
    else:
        raise ValueError(f"unknown query set {query_set!r}")
    qs = [q for q in qs if split == "all" or q["split"] == split]
    en = list(qs)
    zh = [{**q, "query": q["query_zh"]} for q in qs]
    return {"en": en, "zh": zh, "both": en + zh}[lang]
