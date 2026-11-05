#!/usr/bin/env python3
"""Generate synthetic search queries v3: shaped like real KuaiSAR searches,
answerable in KuaiRec.

v2 (tests/fixtures/synthetic_queries_v2.py) was grounded in KuaiRec but shaped by
our own choices. v3 takes its *shape* from real Kuaishou searches measured in
KuaiSAR (results/kuaisar_query_profile.json, from scripts/analyze_kuaisar_queries.py):

* **Topics** — KuaiSAR search demand per category, mapped onto KuaiRec categories
  (KUAISAR_TO_KUAIREC) and blended 50/50 with KuaiRec catalogue size, so thin
  categories still get scorable queries.
* **Specificity** — ~72% specific (one sub-category or named thing), ~18%
  multi-topic, ~10% broad/popular (KuaiSAR: 81% of answered searches stay in one
  top-level category, 93% of those in one level-3 sub-category).
* **Long tail** — ~50 queries name something specific (place, person, dish, breed,
  show, game…) taken from KuaiRec caption hashtags; 52% of KuaiSAR queries are
  one-offs, which is what named, specific searches look like.
* **Length** — per-query word-count targets sampled from the real distribution.
* **Refinement pairs** — follow-ups that keep a word and narrow the query
  (KuaiSAR: 32% of searches are followed within 5 minutes, 58% of those refine).

Every query still has an answer in KuaiRec: named queries carry the videos of
their anchor as ``answer_video_ids``; every query is then checked by hand
against real videos (see dylan-docs/synthetic-queries-v3.md).

⚠️  After that hand verification the fixture is canonical: this script refuses to
overwrite it unless ``--force`` is given.

Usage:
    python scripts/generate_synthetic_queries_v3.py --review dylan-docs/synthetic_queries_v3_review.md
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import random
import re
import sys
from collections import Counter

import numpy as np
import pandas as pd

_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "scripts"))

import generate_synthetic_queries as v2  # v2 helpers (that script must not be re-run)
from openai import OpenAI

from recsys.data.category_index import CategoryIndex
from recsys.intent.embeddings import BgeM3Embedder

N_BASE, N_REFINE, N_DEV = 100, 15, 30
TYPE_COUNTS = {"named": 50, "specific": 22, "multi": 18, "broad": 10}
CONSTRAINT_SHARE = 0.25
MIN_SMALL_CATEGORY = 5   # category needs this many small_matrix videos to be scored
MIN_ANCHOR_VIDEOS = 2
SEED = 11

# KuaiSAR (2023) level-1 category -> KuaiRec (2020) caption first-level ids.
# Hand-mapped by meaning; 空 (no category) and 宗教 (religion) have no counterpart.
KUAISAR_TO_KUAIREC = {
    "二次元": [20], "游戏": [3], "影视和短剧": [39, 32], "随手拍": [11], "生活": [11],
    "明星娱乐": [6], "音乐": [2], "情感": [19], "资讯": [28, 23], "美食": [12],
    "才艺": [15], "舞蹈": [1], "读书": [38], "高新数码": [27], "亲子": [25],
    "搞笑": [9], "教育": [14], "穿搭": [5], "运动": [7, 31], "颜值": [8],
    "美妆": [4], "健康": [16], "汽车": [18], "摄影": [26], "动物": [17],
    "三农": [13], "旅游": [10], "房产家居": [36], "奇人异象": [37], "法律": [29],
    "科学": [29], "星座命理": [24], "历史": [21], "财经": [22], "军事": [35],
}

# Promotional / platform hashtags that say nothing about content (36.6% of videos).
BOILERPLATE = ("快手", "热门", "推广", "创作者", "官方", "同城", "作品", "粉丝", "关注",
               "双击", "高清", "主播", "直播", "小助手", "涨粉", "相机", "快影", "正能量",
               "感谢", "点赞", "申请", "抖音", "播放", "流量", "上榜", "合拍", "原创",
               "视频", "记录", "分享", "日常", "生活", "搞笑", "开心", "我要", "老铁")

PROMPT = """You write realistic search queries that users type into Kuaishou, a Chinese short-video app.

Target category: {zh} ({en})
Sub-categories: {subs}
Other categories (for multi-topic slots, pick ONE id): {others}

Write one query per slot:
{slots}

Rules:
- Natural search-box phrasing. Each slot gives an EXACT English word count — match it.
  query_zh is how a Chinese user would search the same thing.
- "named" slots MUST include the given name (English rendering in query_en, the original
  Chinese name in query_zh). "specific" slots target the given sub-category. "broad"
  slots are short, popular, generic searches for the whole category. "multi" slots
  combine this category with ONE other category from the list (set second_category_id).
- Honour constraints: duration SHORT -> say short/quick in English AND 短视频 in Chinese;
  style -> express that tone; exclusion -> "no ..."/"without ..." AND 不要/非/不 in Chinese.
- Vary phrasing by persona. Lower-case English, no punctuation. Safe for all audiences.
- Labels describe only what the query says: duration ANY unless it states a length;
  style NEUTRAL unless it expresses a tone.
Return ONLY a JSON array of objects with keys: "slot", "query_en", "query_zh",
"intent" ({intents}), "style" ({styles}), "duration" ({durations}),
"negative_constraints" (list), "second_category_id" (int or null)."""

ANCHOR_PROMPT = """These are hashtags from Kuaishou videos in the category {zh} ({en}).
Keep ONLY hashtags that name a specific, searchable thing: a place, person/celebrity,
dish/food, animal breed, show/drama/film, game, product/brand, or a named technique/style.
Drop generic words, emotions, slogans and platform tags.
Hashtags: {tags}
Return ONLY a JSON array of objects: {{"tag": <hashtag as given>, "type": <one of place,
person, dish, breed, show, game, product, technique>, "en": <short English rendering>}}."""

REFINE_PROMPT = """A Kuaishou user searched: "{en}" / "{zh}".
Captions of videos that answer it:
{captions}
Write a narrowing follow-up search that keeps at least one word of the original and adds
ONE detail that appears in at least one caption above. Give the added detail as it is
written in that caption (Chinese, 2-6 characters).
Return ONLY a JSON array with one object: [{{"query_en": ..., "query_zh": ..., "detail_zh": ...}}]."""


# ---------------------------------------------------------------------------
# allocation
# ---------------------------------------------------------------------------

def blended_weights(items: pd.DataFrame, profile: dict) -> pd.Series:
    sizes = items.groupby("category_id").size()
    small = items[items.in_small].groupby("category_id").size()
    viable = [c for c in sizes.index
              if c not in v2.EXCLUDED and small.get(c, 0) >= MIN_SMALL_CATEGORY]
    demand = pd.Series(0.0, index=viable)
    for name, share in profile["category_demand"].items():
        targets = [c for c in KUAISAR_TO_KUAIREC.get(name, []) if c in viable]
        if targets:
            split = sizes[targets] / sizes[targets].sum()
            demand[targets] += share * split.to_numpy()
    demand /= demand.sum()
    supply = sizes[viable] / sizes[viable].sum()
    return (0.5 * demand + 0.5 * supply).sort_values(ascending=False)


def allocate(weights: pd.Series, n: int) -> dict[int, int]:
    alloc = {int(c): 1 for c in weights.index}
    quota = weights * (n - len(alloc))
    for c, q in quota.items():
        alloc[int(c)] += int(q)
    for c in (quota - quota.astype(int)).sort_values(ascending=False).index[: n - sum(alloc.values())]:
        alloc[int(c)] += 1
    return alloc


# ---------------------------------------------------------------------------
# named-entity anchors
# ---------------------------------------------------------------------------

def hashtag_candidates(items: pd.DataFrame, cat: int, top: int = 40) -> list[tuple[str, int]]:
    tags: Counter = Counter()
    for cap in items.loc[items.category_id == cat, "caption"]:
        for t in set(re.findall(r"#([^#\s@，。！？,.!?]{2,12})", cap)):
            if not any(b in t for b in BOILERPLATE):
                tags[t] += 1
    return [(t, n) for t, n in tags.most_common(top) if n >= MIN_ANCHOR_VIDEOS]


def anchor_videos(items: pd.DataFrame, cat: int, tag: str) -> list[int]:
    m = (items.category_id == cat) & items.caption.str.contains("#" + tag, regex=False)
    return [int(v) for v in items.index[m]]


def find_anchors(client, model, items, cats: list[int], index) -> dict[int, list[dict]]:
    out = {}
    for c in cats:
        cands = hashtag_candidates(items, c)
        if not cands:
            out[c] = []
            continue
        e = index.get_by_id(c)
        kept = v2.call_llm(client, model, ANCHOR_PROMPT.format(
            zh=e.chinese_name, en=e.english_name, tags=", ".join(t for t, _ in cands)))
        known = dict(cands)
        anchors = []
        for a in kept:
            tag = str(a.get("tag", "")).lstrip("#")
            if tag in known and a.get("en"):
                vids = anchor_videos(items, c, tag)
                if len(vids) >= MIN_ANCHOR_VIDEOS:
                    anchors.append({"tag": tag, "type": a.get("type"), "en": a["en"],
                                    "videos": vids})
        out[c] = sorted(anchors, key=lambda a: -len(a["videos"]))
    return out


# ---------------------------------------------------------------------------
# slots
# ---------------------------------------------------------------------------

def length_target(rng: random.Random, share: dict) -> int:
    r = rng.random()
    if r < share["1"]:
        return 1
    if r < share["1"] + share["2-3"]:
        return rng.choice([2, 3])
    return rng.choice([4, 4, 5])


def plan_slots(items, alloc, anchors, rng, profile) -> list[dict]:
    subs = {c: v2.sub_targets(items, c) for c in alloc}
    slots = [{"category": c} for c, k in alloc.items() for _ in range(k)]
    rng.shuffle(slots)
    quota = dict(TYPE_COUNTS)
    used: dict[int, set] = {c: set() for c in alloc}
    for s in slots:
        c = s["category"]
        free_anchor = [a for a in anchors.get(c, []) if a["tag"] not in used[c]]
        free_sub = [x for x in subs[c] if x[0] not in used[c]]
        if quota["named"] and free_anchor:
            a = free_anchor[0]
            used[c].add(a["tag"])
            s.update(type="named", anchor=a)
            quota["named"] -= 1
        elif quota["specific"] and free_sub:
            name, level = free_sub[0]
            used[c].add(name)
            s.update(type="specific", sub=name, sub_level=level)
            quota["specific"] -= 1
        elif quota["multi"]:
            s.update(type="multi")
            quota["multi"] -= 1
        else:
            s.update(type="broad")
    constrained = set(rng.sample(range(len(slots)), round(CONSTRAINT_SHARE * len(slots))))
    kinds = ["duration", "style", "exclusion"]
    for i, s in enumerate(slots):
        s["slot"] = i
        s["persona"] = rng.choice(v2.PERSONAS)
        s["target_length"] = length_target(rng, profile["length"]["share"])
        s["constraint"] = kinds[i % 3] if i in constrained else "none"
        if s["constraint"] == "duration":
            s["duration_target"] = "SHORT"   # KuaiRec has almost no long videos
            s["target_length"] = max(s["target_length"], 2)
        if s["constraint"] == "style":
            s["style_target"] = rng.choice([x for x in v2.STYLES if x != "NEUTRAL"])
        if s["constraint"] != "none":
            s["target_length"] = max(s["target_length"], 2)
    return slots


def slot_spec(s: dict) -> str:
    parts = [f"slot {s['slot']}: type={s['type']}, EXACTLY {s['target_length']} English word(s)"]
    if s["type"] == "named":
        a = s["anchor"]
        parts.append(f"name={a['en']} ({a['tag']}, a {a['type']})")
    if s["type"] == "specific":
        parts.append(f"sub-category={s['sub']}")
    if s["type"] == "multi":
        parts.append("REQUIRED: set second_category_id and combine both topics")
    parts.append(f"persona={s['persona']}")
    if s["constraint"] == "duration":
        parts.append("constraint=duration SHORT")
    elif s["constraint"] == "style":
        parts.append(f"constraint=style {s['style_target']}")
    elif s["constraint"] == "exclusion":
        parts.append("constraint=exclusion (exclude something plausible)")
    return "- " + ", ".join(parts)


def build_prompt(items, cat, slots, viable, index) -> str:
    e = index.get_by_id(cat)
    sub = items[items.category_id == cat]
    subs = ", ".join(sorted({*sub.l3[sub.l3 != "UNKNOWN"].unique()[:12],
                             *sub.l2[sub.l2 != "UNKNOWN"].unique()[:6]})) or "-"
    others = ", ".join(f"{c}={index.get_by_id(c).english_name}" for c in sorted(viable) if c != cat)
    return PROMPT.format(zh=e.chinese_name, en=e.english_name, subs=subs, others=others,
                         slots="\n".join(slot_spec(s) for s in slots),
                         intents="/".join(v2.INTENTS), styles="/".join(v2.STYLES),
                         durations="/".join(v2.DURATIONS))


# ---------------------------------------------------------------------------
# records + checks
# ---------------------------------------------------------------------------

def to_record(s: dict, out: dict, viable: set) -> dict | None:
    en = str(out.get("query_en", "")).strip().lower()
    zh = str(out.get("query_zh", "")).strip()
    if not en or not zh or abs(len(en.split()) - s["target_length"]) > 1 or len(zh) > 16:
        return None
    if out.get("intent") not in v2.INTENTS or out.get("style") not in v2.STYLES:
        return None
    topics = [s["category"]]
    if s["type"] == "multi":
        second = out.get("second_category_id")
        if not isinstance(second, int) or second not in viable or second == s["category"]:
            return None
        topics.append(second)
    if s["type"] == "named" and s["anchor"]["tag"] not in zh:
        return None   # the Chinese query must carry the anchor name
    negatives = [str(n) for n in out.get("negative_constraints") or [] if n]
    if s["constraint"] == "exclusion" and not negatives:
        return None
    rec = {
        "query": en, "query_zh": zh, "expected_topics": topics,
        "expected_subcategory": s.get("sub"), "subcategory_level": s.get("sub_level"),
        "expected_style": s.get("style_target", out["style"]) if s["constraint"] == "style"
        else out["style"],
        "expected_duration": "SHORT" if s["constraint"] == "duration" else "ANY",
        "expected_intent": out["intent"], "expected_negative": negatives,
        "query_type": "specific" if s["type"] == "named" else s["type"],
        "head_or_tail": "head" if s["type"] == "broad" else "tail",
        "anchor": s["anchor"]["tag"] if s["type"] == "named" else None,
        "answer_video_ids": list(s["anchor"]["videos"]) if s["type"] == "named" else [],
        "constraint": s["constraint"], "persona": s["persona"],
        "target_length": s["target_length"], "refines": None,
    }
    return rec if v2.constraint_visible(rec) else None


def refine(client, model, items, base: dict) -> dict | None:
    pool = base["answer_video_ids"] or [int(v) for v in items.index[
        items.category_id.isin(base["expected_topics"])]][:200]
    caps = items.loc[pool, "caption"]
    caps = caps[caps.str.len().between(6, 80)].head(8)
    if caps.empty:
        return None
    answer = v2.call_llm(client, model, REFINE_PROMPT.format(
        en=base["query"], zh=base["query_zh"], captions="\n".join("- " + c for c in caps)))
    if not answer:
        return None
    out = answer[0]
    detail = str(out.get("detail_zh", "")).strip()
    en, zh = str(out.get("query_en", "")).strip().lower(), str(out.get("query_zh", "")).strip()
    answers = [int(v) for v in caps.index if detail and detail in items.at[v, "caption"]]
    if not answers or not en or not zh or not (set(en.split()) & set(base["query"].split())):
        return None
    if len(en.split()) <= len(base["query"].split()):
        return None   # a refinement narrows: it adds words
    return {**base, "query": en, "query_zh": zh, "answer_video_ids": answers,
            "anchor": base["anchor"], "refines": base["id"], "constraint": "none",
            "expected_negative": [], "expected_duration": "ANY",
            "target_length": len(en.split())}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="data/raw")
    ap.add_argument("--profile", default="results/kuaisar_query_profile.json")
    ap.add_argument("--out", default="tests/fixtures/synthetic_queries_v3.py")
    ap.add_argument("--review", default=None)
    ap.add_argument("--max-rounds", type=int, default=6)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if pathlib.Path(args.out).exists() and not args.force:
        sys.exit(f"{args.out} exists and was verified by hand; refusing to overwrite (use --force).")
    rng = random.Random(SEED)
    profile = json.loads(pathlib.Path(args.profile).read_text(encoding="utf-8"))
    items = v2.load_catalogue(args.data_dir)
    index = CategoryIndex()
    weights = blended_weights(items, profile)
    alloc = allocate(weights, N_BASE)
    viable = set(alloc)
    print(f"{len(alloc)} viable categories; allocation: "
          f"{ {index.get_by_id(c).english_name: k for c, k in alloc.items()} }")

    client = OpenAI(base_url=os.environ.get("SOCLAAS_BASE_URL"),
                    api_key=os.environ["SOCLAAS_API_KEY"])
    model = os.environ.get("SOCLAAS_MODEL", "default")
    anchors = find_anchors(client, model, items, sorted(alloc), index)
    print("anchors per category:", {index.get_by_id(c).english_name: len(a)
                                     for c, a in anchors.items() if a})
    slots = plan_slots(items, alloc, anchors, rng, profile)
    print("slot types:", Counter(s["type"] for s in slots),
          "| constraints:", Counter(s["constraint"] for s in slots))

    embedder = BgeM3Embedder()
    cat_vecs = dict(zip(sorted(viable), v2._unit(embedder.embed_batch(
        [f"{index.get_by_id(c).english_name} {index.get_by_id(c).chinese_name}"
         for c in sorted(viable)]))))
    captions = items.caption.tolist()
    records: dict[int, dict] = {}
    for rnd in range(1, args.max_rounds + 1):
        todo = [s for s in slots if s["slot"] not in records]
        if not todo:
            break
        print(f"round {rnd}: {len(todo)} slots", flush=True)
        by_cat: dict[int, list[dict]] = {}
        for s in todo:
            by_cat.setdefault(s["category"], []).append(s)
        for cat, cs in by_cat.items():
            answers = {a.get("slot"): a for a in v2.call_llm(
                client, model, build_prompt(items, cat, cs, viable, index))}
            for s in cs:
                rec = to_record(s, answers.get(s["slot"], {}), viable)
                if rec is None or v2.copies_caption(rec["query_zh"], captions):
                    continue
                if rec["query_type"] == "multi" and not v2.second_topic_plausible(
                        rec, v2._unit([embedder.embed(rec["query"])])[0], cat_vecs):
                    continue
                rec["n_catalogue"], rec["n_small"] = v2.evaluable_counts(items, rec)
                records[s["slot"]] = rec
        _drop_duplicates(records, embedder)

    base = [records[s["slot"]] for s in slots if s["slot"] in records]
    print(f"{len(base)}/{N_BASE} base queries")
    _split(base, rng)
    base.sort(key=lambda r: (r["expected_topics"][0], r["query_type"], r["query"]))
    for i, r in enumerate(base, 1):
        r["id"] = f"v{i:03d}"
    v2.relabel_intents(client, model, base)

    refined, tried = [], 0
    for r in rng.sample(base, len(base)):
        if len(refined) == N_REFINE or tried > 4 * N_REFINE:
            break
        tried += 1
        if r["query_type"] == "multi":
            continue
        child = refine(client, model, items, r)
        if child:
            child["id"] = f"r{len(refined) + 1:03d}"
            child["n_catalogue"], child["n_small"] = v2.evaluable_counts(items, child)
            refined.append(child)
    print(f"{len(refined)} refinement queries")

    write_fixture(args.out, base + refined, model)
    if args.review:
        write_review(args.review, base + refined, index)
    summary(base, refined)


def _drop_duplicates(records: dict, embedder) -> None:
    keys = sorted(records)
    if len(keys) < 2:
        return
    vecs = embedder.embed_batch([records[k]["query"] for k in keys])
    if not all(vecs):
        return
    sims = v2._unit(vecs) @ v2._unit(vecs).T
    for i, j in zip(*np.triu_indices(len(keys), 1)):
        if sims[i, j] > v2.DUP_COSINE and keys[i] in records and keys[j] in records:
            del records[keys[j]]


def _split(recs: list[dict], rng: random.Random) -> None:
    by: dict[str, list[dict]] = {}
    for r in recs:
        by.setdefault(r["query_type"], []).append(r)
    for group in by.values():
        rng.shuffle(group)
        n_dev = round(N_DEV * len(group) / len(recs))
        for i, r in enumerate(group):
            r["split"] = "dev" if i < n_dev else "test"


FIELDS = ["id", "query", "query_zh", "expected_topics", "expected_subcategory",
          "subcategory_level", "expected_style", "expected_duration", "expected_intent",
          "expected_negative", "query_type", "head_or_tail", "anchor", "answer_video_ids",
          "refines", "constraint", "persona", "target_length", "split", "n_catalogue",
          "n_small"]


def write_fixture(path: str, recs: list[dict], model: str) -> None:
    body = ",\n".join("    " + repr({k: r[k] for k in FIELDS}) for r in recs)
    pathlib.Path(path).write_text(
        '"""Synthetic search benchmark v3: KuaiSAR-shaped, KuaiRec-answerable.\n\n'
        f"Generated by scripts/generate_synthetic_queries_v3.py (model={model}, seed={SEED}),\n"
        "then verified by hand against real videos (dylan-docs/synthetic-queries-v3.md).\n"
        "DO NOT regenerate this file — edit queries by hand and re-verify them instead.\n"
        "Base queries have ids v###; refinement queries r### narrow the base query named in\n"
        "`refines`. answer_video_ids are videos verified to answer the query.\n"
        "split: 30 dev / 70 test (refinements inherit their base query's split).\n"
        '"""\n\n'
        f"SYNTHETIC_QUERIES_V3: list[dict] = [\n{body},\n]\n", encoding="utf-8")
    print(f"wrote {path} ({len(recs)} queries)")


def write_review(path: str, recs: list[dict], index) -> None:
    rows = [
        ("| id | split | type | head/tail | constraint | words | English | 中文 | topics | "
         "anchor / sub-category | answers | refines |"),
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in recs:
        topics = ", ".join(index.get_by_id(t).english_name for t in r["expected_topics"])
        rows.append(f"| {r['id']} | {r['split']} | {r['query_type']} | {r['head_or_tail']} | "
                    f"{r['constraint']} | {len(r['query'].split())} | {r['query']} | "
                    f"{r['query_zh']} | {topics} | {r['anchor'] or r['expected_subcategory'] or '-'} | "
                    f"{len(r['answer_video_ids'])} | {r['refines'] or '-'} |")
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(path).write_text("# Synthetic queries v3 — review\n\n" + "\n".join(rows) + "\n",
                                  encoding="utf-8")
    print(f"wrote {path}")


def summary(base: list[dict], refined: list[dict]) -> None:
    words = Counter(min(len(r["query"].split()), 4) for r in base)
    print("types:", dict(Counter(r["query_type"] for r in base)),
          "| named:", sum(r["anchor"] is not None for r in base),
          "| head/tail:", dict(Counter(r["head_or_tail"] for r in base)))
    print("length shares (1/2/3/4+):", {k: round(v / len(base), 2) for k, v in sorted(words.items())})
    print("split:", dict(Counter(r["split"] for r in base)),
          "| constraints:", dict(Counter(r["constraint"] for r in base)),
          "| refinements:", len(refined))


if __name__ == "__main__":
    main()
