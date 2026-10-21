#!/usr/bin/env python3
"""Generate the 100-query synthetic search benchmark (paired English + Chinese).

⚠️  DO NOT RE-RUN. The committed ``tests/fixtures/synthetic_queries_v2.py`` was
refined by hand after generation: every query was checked against real videos
and 47 were rewritten to match what exists in the catalogue (see
``dylan-docs/synthetic-queries.md``). Re-running would replace those fixes with
fresh, unverified LLM output. The script refuses to overwrite the fixture unless
``--force`` is given; ``--resume --drop <ids>`` replaces only specific queries
(re-verify any replacement against real videos before keeping it).

Queries are grounded in the real KuaiRec catalogue, not invented freely:

1. **Targets from the data.** Viable categories have >= MIN_CATALOGUE searchable
   videos and >= MIN_SMALL small_matrix videos (so recall can be scored);
   Other (34), Selfie (33) and unknown categories are excluded. Each viable
   category gets FLOOR queries, the rest are allocated in proportion to
   catalogue size (largest remainder).
2. **Query types.** ~40% broad (level-1 topic), ~40% specific (a level-3, or
   level-2, sub-category with enough videos), ~20% multi-topic (the LLM picks a
   second viable category). ~25% carry an explicit constraint (duration, style
   or exclusion).
3. **Labels come from the target**, not from an LLM guess: topic ids and the
   sub-category are fixed before generation; duration is ANY unless the query
   states one. The LLM writes the query text (EN + ZH) for each slot, plus
   intent/style labels for what the query expresses.
4. **Automatic checks:** schema and enum validity, length (real Kuaishou
   queries: median 3 words), near-duplicates (bge-m3 cosine), no copied
   captions, and evaluability (enough catalogue / small_matrix videos).
   Failed slots are regenerated.
5. **Split:** 30 dev (tuning) / 70 test (reporting), stratified by query type.

Outputs ``tests/fixtures/synthetic_queries_v2.py`` (the benchmark) and a
human-review table (``--review``). The LLM is not deterministic, so the
committed fixture file — not this script — is the canonical dataset.

Usage:
    python scripts/generate_synthetic_queries.py --review dylan-docs/synthetic_queries_review.md
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pathlib
import random
import re
import sys
import time
from collections import Counter

import numpy as np
import pandas as pd

_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from dotenv import load_dotenv
from openai import OpenAI, OpenAIError

load_dotenv()

from recsys.contracts.enums import ContentStyle, DurationPreference, QueryIntentType
from recsys.data.category_index import CategoryIndex
from recsys.data.item_features import build_item_table
from recsys.intent.embeddings import BgeM3Embedder

N_QUERIES = 100
FLOOR = 2
MIN_CATALOGUE, MIN_SMALL, SUB_MIN_SMALL, SUB_MIN_CATALOGUE = 20, 10, 5, 10
EXCLUDED = {33, 34, -1}  # Selfie, Other, unknown: not meaningful search targets
TYPE_SHARE = {"broad": 0.40, "specific": 0.40, "multi": 0.20}
CONSTRAINT_SHARE = 0.25
N_DEV = 30
DUP_COSINE = 0.93
SEED = 7
PERSONAS = ["student", "parent", "office worker", "older adult", "teenager", "hobbyist"]
INTENTS = [i.name for i in QueryIntentType]
STYLES = [s.name for s in ContentStyle]
DURATIONS = [d.name for d in DurationPreference]

PROMPT = """You write realistic search queries that users type into Kuaishou, a Chinese short-video app.

Target category: {zh} ({en})
Sub-categories in this category: {subs}
Example captions from this category (for flavour only — never copy them): {examples}
Other categories (for multi-topic slots, pick ONE id from here): {others}

Write one query per slot below. Slot specs:
{slots}

Rules:
- Short and natural, like a real search box: mostly 2-4 English words (never more than 6); Chinese 2-8 characters.
- query_zh is the natural Chinese way to search the same thing (not a word-for-word translation).
- "broad" slots target the whole category; "specific" slots must target the given sub-category;
  "multi" slots must combine the target category with the second category you choose.
- Honour the constraint if a slot has one: duration -> state the length naturally ("short", "long"),
  style -> express that tone, exclusion -> say what to exclude ("no ...", "without ...").
- Vary phrasing to fit the persona. Do not always end with "videos". Lower-case English, no punctuation.
- Safe for a general audience: nothing sexual, violent, hateful or medically misleading.
- Labels describe only what the query expresses: duration is ANY unless the query states a length;
  style is NEUTRAL unless the query expresses a tone. Pick the intent that best fits the query.
- query_en and query_zh must mean the same thing, including any constraint.

Return ONLY a JSON array, one object per slot, with keys:
"slot" (int), "query_en", "query_zh", "intent" (one of {intents}), "style" (one of {styles}),
"duration" (one of {durations}), "negative_constraints" (list of strings, [] if none),
"second_category_id" (int for multi slots, else null)."""


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

def load_catalogue(data_dir: str) -> pd.DataFrame:
    cap = pd.read_csv(f"{data_dir}/kuairec_caption_category.csv", engine="python",
                      on_bad_lines="skip")
    items = build_item_table(cap, pd.read_csv(f"{data_dir}/item_daily_features.csv"))
    cap["video_id"] = pd.to_numeric(cap["video_id"], errors="coerce")
    cap = cap.dropna(subset=["video_id"]).drop_duplicates("video_id")
    cap = cap.set_index(cap["video_id"].astype(int))
    for level in ("second", "third"):
        items[f"l{2 if level == 'second' else 3}"] = cap[f"{level}_level_category_name"].reindex(
            items.index).fillna("UNKNOWN")
    items["caption"] = cap["caption"].reindex(items.index).fillna("").astype(str)
    small = set(pd.read_csv(f"{data_dir}/small_matrix.csv", usecols=["video_id"]).video_id)
    items["in_small"] = items.index.isin(small)
    return items


def viable_categories(items: pd.DataFrame) -> pd.DataFrame:
    g = items.groupby("category_id").agg(n=("text", "size"), n_small=("in_small", "sum"))
    g = g[~g.index.isin(EXCLUDED) & (g.n >= MIN_CATALOGUE) & (g.n_small >= MIN_SMALL)]
    return g.sort_values("n", ascending=False)


def allocate(cats: pd.DataFrame) -> dict[int, int]:
    """FLOOR per category, the remainder proportional to size (largest remainder)."""
    alloc = {int(c): FLOOR for c in cats.index}
    rest = N_QUERIES - FLOOR * len(alloc)
    quota = cats.n / cats.n.sum() * rest
    for c, q in quota.items():
        alloc[int(c)] += int(q)
    left = N_QUERIES - sum(alloc.values())
    for c in (quota - quota.astype(int)).sort_values(ascending=False).index[:left]:
        alloc[int(c)] += 1
    return alloc


def sub_targets(items: pd.DataFrame, cat: int) -> list[tuple[str, int]]:
    """Level-3 sub-categories with enough data, else level-2; as (name, level)."""
    sub = items[items.category_id == cat]
    for col, level in (("l3", 3), ("l2", 2)):
        g = sub[sub[col] != "UNKNOWN"].groupby(col).agg(n=("text", "size"), s=("in_small", "sum"))
        g = g[(g.n >= SUB_MIN_CATALOGUE) & (g.s >= SUB_MIN_SMALL)].sort_values("s", ascending=False)
        if len(g):
            return [(name, level) for name in g.index]
    return []


# ---------------------------------------------------------------------------
# slots
# ---------------------------------------------------------------------------

def plan_slots(items, cats, alloc, rng) -> list[dict]:
    subs = {c: sub_targets(items, c) for c in alloc}
    slots = [{"category": c} for c, k in alloc.items() for _ in range(k)]
    rng.shuffle(slots)
    quota = {t: round(s * N_QUERIES) for t, s in TYPE_SHARE.items()}
    used_subs: dict[int, set] = {c: set() for c in alloc}
    for slot in slots:  # specific first where possible, then multi, then broad
        c = slot["category"]
        free = [s for s in subs[c] if s[0] not in used_subs[c]]
        if quota["specific"] and free:
            name, level = free[0]
            used_subs[c].add(name)
            slot.update(type="specific", sub=name, sub_level=level)
            quota["specific"] -= 1
        elif quota["multi"]:
            slot.update(type="multi")
            quota["multi"] -= 1
        else:
            slot.update(type="broad")
    constrained = rng.sample(range(len(slots)), round(CONSTRAINT_SHARE * len(slots)))
    kinds = ["duration", "style", "exclusion"]
    for i, slot in enumerate(slots):
        slot["persona"] = rng.choice(PERSONAS)
        slot["constraint"] = kinds[constrained.index(i) % 3] if i in constrained else "none"
        if slot["constraint"] == "duration":
            # MEDIUM is not a length people type ("medium-length"); test the extremes.
            slot["duration_target"] = rng.choice(["SHORT", "SHORT", "LONG"])
        if slot["constraint"] == "style":
            slot["style_target"] = rng.choice([s for s in STYLES if s != "NEUTRAL"])
    for i, slot in enumerate(slots):
        slot["slot"] = i
    return slots


def slot_spec(slot: dict) -> str:
    parts = [f"slot {slot['slot']}: type={slot['type']}"]
    if slot["type"] == "specific":
        parts.append(f"sub-category={slot['sub']}")
    if slot["type"] == "multi":
        parts.append("REQUIRED: set second_category_id to one id from the other-categories "
                     "list and make the query combine both topics")
    parts.append(f"persona={slot['persona']}")
    if slot["constraint"] == "duration":
        words = ("'short'/'quick' in English AND 短视频/几秒/快速 in Chinese"
                 if slot["duration_target"] == "SHORT"
                 else "'long'/'full' in English AND 长视频/完整版 in Chinese")
        parts.append(f"constraint=duration {slot['duration_target']} "
                     f"(REQUIRED: both queries must state the video length, e.g. {words})")
    elif slot["constraint"] == "style":
        parts.append(f"constraint=style {slot['style_target']}")
    elif slot["constraint"] == "exclusion":
        parts.append("constraint=exclusion (exclude something plausible in this category)")
    return "- " + ", ".join(parts)


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------

def call_llm(client, model, prompt: str) -> list[dict]:
    for attempt in range(1, 4):
        try:
            resp = client.chat.completions.create(
                model=model, max_tokens=2048, temperature=0.9,
                messages=[{"role": "user", "content": prompt}],
            )
            text = (resp.choices[0].message.content or "").strip()
            text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
            out = json.loads(text[text.index("["): text.rindex("]") + 1])
            if isinstance(out, list):
                return out
        except (OpenAIError, ValueError) as exc:  # API error or unparseable JSON — retry
            print(f"    LLM attempt {attempt} failed: {exc}", flush=True)
            time.sleep(2 ** attempt)
    return []


def build_prompt(items, cat, slots, cats, index) -> str:
    entry = index.get_by_id(cat)
    sub = items[items.category_id == cat]
    subs = ", ".join(sorted({*sub.l3[sub.l3 != "UNKNOWN"].unique()[:12],
                             *sub.l2[sub.l2 != "UNKNOWN"].unique()[:6]}))
    caps = sub[sub.caption.str.len().between(6, 60)].caption
    examples = " | ".join(caps.sample(min(4, len(caps)), random_state=cat).str[:40])
    others = ", ".join(f"{int(c)}={index.get_by_id(int(c)).english_name}"
                       for c in cats.index if int(c) != cat)
    return PROMPT.format(zh=entry.chinese_name, en=entry.english_name, subs=subs or "-",
                         examples=examples or "-", others=others,
                         slots="\n".join(slot_spec(s) for s in slots),
                         intents="/".join(INTENTS), styles="/".join(STYLES),
                         durations="/".join(DURATIONS))


# ---------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------

def to_record(slot: dict, out: dict, viable: set) -> dict | None:
    """Validate one LLM answer against its slot; None if it fails."""
    en = str(out.get("query_en", "")).strip().lower()
    zh = str(out.get("query_zh", "")).strip()
    if not en or not zh or len(en.split()) > 6 or len(zh) > 12:
        return None
    if out.get("intent") not in INTENTS or out.get("style") not in STYLES \
            or out.get("duration") not in DURATIONS:
        return None
    topics = [slot["category"]]
    if slot["type"] == "multi":
        second = out.get("second_category_id")
        if not isinstance(second, int) or second not in viable or second == slot["category"]:
            return None
        topics.append(second)
    duration = slot.get("duration_target", "ANY") if slot["constraint"] == "duration" else "ANY"
    style = slot.get("style_target", out["style"]) if slot["constraint"] == "style" else out["style"]
    negatives = [str(n) for n in out.get("negative_constraints") or [] if n]
    if slot["constraint"] == "exclusion" and not negatives:
        return None
    return {
        "query": en,
        "query_zh": zh,
        "expected_topics": topics,
        "expected_subcategory": slot.get("sub"),
        "subcategory_level": slot.get("sub_level"),
        "expected_style": style,
        "expected_duration": duration,
        "expected_intent": out["intent"],
        "expected_negative": negatives,
        "query_type": slot["type"],
        "constraint": slot["constraint"],
        "persona": slot["persona"],
    }


_DURATION_WORDS = {
    "SHORT": (("short", "quick", "brief", "clip", "second", "minute"), ("短", "快", "秒", "分钟")),
    "LONG": (("long", "full", "complete", "in-depth", "detailed", "hour"), ("长", "完整", "全集", "详细", "合集")),
}
_EXCLUDE_EN = ("no ", "without", "not ", "except", "non-")
_EXCLUDE_ZH = ("不", "无", "非", "别", "除了", "没有")


def constraint_visible(rec: dict) -> bool:
    """A constraint label is only valid if the query text actually says it, in both languages."""
    en, zh = f" {rec['query']} ", rec["query_zh"]
    if rec["constraint"] == "duration":
        words_en, words_zh = _DURATION_WORDS.get(rec["expected_duration"], ((), ()))
        return any(w in en for w in words_en) and any(w in zh for w in words_zh)
    if rec["constraint"] == "exclusion":
        about_ads = any("ad" in n.lower().split() or "ads" in n.lower() or "广告" in n
                        for n in rec["expected_negative"])
        return (not about_ads and any(w in en for w in _EXCLUDE_EN)
                and any(w in zh for w in _EXCLUDE_ZH))
    return True


def second_topic_plausible(rec: dict, query_vec: np.ndarray, cat_vecs: dict[int, np.ndarray],
                           top: int = 5) -> bool:
    """Multi-topic: the second category must be among the `top` categories most similar
    to the query (bad pairings rank 9th-23rd, sensible ones 1st-4th on the first draft)."""
    if rec["query_type"] != "multi":
        return True
    ids = list(cat_vecs)
    sims = np.array([cat_vecs[c] @ query_vec for c in ids])
    second = rec["expected_topics"][1]
    return int((sims > sims[ids.index(second)]).sum()) < top


def _unit(vectors) -> np.ndarray:
    arr = np.asarray(vectors, dtype=np.float64)
    return arr / np.linalg.norm(arr, axis=-1, keepdims=True)


INTENT_PROMPT = """Label the search intent of each Kuaishou search query. Definitions:
- ENTERTAINMENT: wants to be entertained or amused
- LEARNING: wants to learn how to do something or understand a topic
- INSPIRATION: wants ideas or motivation (outfits, recipes, designs, life stories)
- SOCIAL: wants what is trending, popular or talked about
- SPECIFIC_SEARCH: looking for a particular named thing, person, show or item
- EXPLORATION: browsing a topic to discover something new
Queries:
{queries}
Return ONLY a JSON array of {{"id": ..., "intent": ...}} objects, one per query."""


def relabel_intents(client, model, recs: list[dict]) -> None:
    """Label intents in one consistent pass (generation-time intents were steered by slot
    hints and came out forced, e.g. 'sports highlights' = LEARNING)."""
    for start in range(0, len(recs), 25):
        batch = recs[start:start + 25]
        lines = "\n".join(f'{r["id"]}: {r["query"]} / {r["query_zh"]}' for r in batch)
        labels = {a.get("id"): a.get("intent") for a in
                  call_llm(client, model, INTENT_PROMPT.format(queries=lines))}
        for r in batch:
            if labels.get(r["id"]) in INTENTS:
                r["expected_intent"] = labels[r["id"]]


def copies_caption(zh: str, captions: list[str]) -> bool:
    return len(zh) >= 8 and any(zh in c for c in captions)


def evaluable_counts(items: pd.DataFrame, rec: dict) -> tuple[int, int]:
    mask = items.category_id.isin(rec["expected_topics"])
    if rec["expected_subcategory"]:
        col = f"l{rec['subcategory_level']}"
        mask &= items[col] == rec["expected_subcategory"]
    return int(mask.sum()), int((mask & items.in_small).sum())


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="data/raw")
    ap.add_argument("--out", default="tests/fixtures/synthetic_queries_v2.py")
    ap.add_argument("--review", default=None, help="write a markdown review table here")
    ap.add_argument("--max-rounds", type=int, default=4)
    ap.add_argument("--resume", action="store_true",
                    help="keep the queries already in --out (if they pass the checks) and "
                         "only fill missing slots")
    ap.add_argument("--drop", default="",
                    help="comma-separated ids to discard on --resume (human-review flags)")
    ap.add_argument("--force", action="store_true",
                    help="allow regenerating from scratch over an existing (hand-refined) fixture")
    args = ap.parse_args()
    if pathlib.Path(args.out).exists() and not (args.resume or args.force):
        sys.exit(f"{args.out} exists and was refined by hand; refusing to overwrite it. "
                 "Use --resume --drop <ids> to replace specific queries, or --force to "
                 "discard the refinements (see the docstring).")
    rng = random.Random(SEED)

    items = load_catalogue(args.data_dir)
    cats = viable_categories(items)
    alloc = allocate(cats)
    print(f"{len(cats)} viable categories; allocation: {alloc}")
    slots = plan_slots(items, cats, alloc, rng)
    print("query types:", Counter(s["type"] for s in slots),
          "| constraints:", Counter(s["constraint"] for s in slots))

    client = OpenAI(base_url=os.environ.get("SOCLAAS_BASE_URL"),
                    api_key=os.environ["SOCLAAS_API_KEY"])
    model = os.environ.get("SOCLAAS_MODEL", "default")
    index, viable = CategoryIndex(), {int(c) for c in cats.index}
    captions = items.caption.tolist()
    embedder = BgeM3Embedder()
    cat_ids = sorted(viable)
    cat_vecs = dict(zip(cat_ids, _unit(embedder.embed_batch(
        [f"{index.get_by_id(c).english_name} {index.get_by_id(c).chinese_name}" for c in cat_ids]))))

    def passes(rec: dict) -> bool:
        if not constraint_visible(rec):
            return False
        if rec["query_type"] == "multi":
            return second_topic_plausible(rec, _unit([embedder.embed(rec["query"])])[0], cat_vecs)
        return True

    records: dict[int, dict] = {}
    if args.resume and pathlib.Path(args.out).exists():
        drop = {d.strip() for d in args.drop.split(",") if d.strip()}
        records = resume_records(args.out, slots, drop)
        failed = [k for k, r in records.items() if not passes(r)]
        for k in failed:
            print(f"    re-check failed, regenerating: {records[k]['query']!r}")
            del records[k]
        print(f"resuming: {len(records)} existing queries kept")
    for rnd in range(1, args.max_rounds + 1):
        todo = [s for s in slots if s["slot"] not in records]
        if not todo:
            break
        print(f"round {rnd}: {len(todo)} slots to generate", flush=True)
        by_cat: dict[int, list[dict]] = {}
        for s in todo:
            by_cat.setdefault(s["category"], []).append(s)
        for cat, cat_slots in by_cat.items():
            answers = {a.get("slot"): a for a in call_llm(
                client, model, build_prompt(items, cat, cat_slots, cats, index))}
            for s in cat_slots:
                rec = to_record(s, answers.get(s["slot"], {}), viable)
                if rec is None or copies_caption(rec["query_zh"], captions) or not passes(rec):
                    continue
                n_cat, n_small = evaluable_counts(items, rec)
                if n_small < SUB_MIN_SMALL:
                    continue
                records[s["slot"]] = {**rec, "n_catalogue": n_cat, "n_small": n_small}
        # near-duplicate check across everything accepted so far
        keys = sorted(records)
        vecs = embedder.embed_batch([records[k]["query"] for k in keys])
        if len(keys) < 2 or not all(vecs):
            continue  # embedding failed this round; the next round re-checks
        vecs = np.array(vecs)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        sims = vecs @ vecs.T
        for i, j in zip(*np.triu_indices(len(keys), 1)):
            if sims[i, j] > DUP_COSINE and keys[i] in records and keys[j] in records:
                print(f"    near-duplicate dropped: {records[keys[j]]['query']!r} "
                      f"~ {records[keys[i]]['query']!r}")
                del records[keys[j]]

    missing = [s["slot"] for s in slots if s["slot"] not in records]
    if missing:
        print(f"WARNING: {len(missing)} slots still missing after {args.max_rounds} rounds")

    # dev/test split, stratified by query type
    final = [records[s["slot"]] for s in slots if s["slot"] in records]
    by_type: dict[str, list[dict]] = {}
    for r in final:
        by_type.setdefault(r["query_type"], []).append(r)
    dev_quota = {t: round(N_DEV * len(v) / len(final)) for t, v in by_type.items()}
    for t, group in by_type.items():
        rng.shuffle(group)
        for i, r in enumerate(group):
            r["split"] = "dev" if i < dev_quota[t] else "test"
    final.sort(key=lambda r: (r["expected_topics"][0], r["query_type"], r["query"]))
    for i, r in enumerate(final, 1):
        r["id"] = f"q{i:03d}"
    relabel_intents(client, model, final)

    write_fixture(args.out, final, model)
    if args.review:
        write_review(args.review, final, index)
    print_summary(final)


def resume_records(path: str, slots: list[dict], drop: set[str]) -> dict[int, dict]:
    """Match existing queries back to slots by (category, type, constraint)."""
    spec = importlib.util.spec_from_file_location("synthetic_queries_v2", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    pool: dict[tuple, list[dict]] = {}
    for r in module.SYNTHETIC_QUERIES_V2:
        if r["id"] in drop:
            print(f"    dropped by review: {r['id']} {r['query']!r}")
            continue
        r = {k: v for k, v in r.items() if k not in ("id", "split")}
        pool.setdefault((r["expected_topics"][0], r["query_type"], r["constraint"]), []).append(r)
    out = {}
    for s in slots:
        group = pool.get((s["category"], s["type"], s["constraint"]))
        if group:
            out[s["slot"]] = group.pop()
    return out


def write_fixture(path: str, recs: list[dict], model: str) -> None:
    order = ["id", "query", "query_zh", "expected_topics", "expected_subcategory",
             "subcategory_level", "expected_style", "expected_duration", "expected_intent",
             "expected_negative", "query_type", "constraint", "persona", "split",
             "n_catalogue", "n_small"]
    body = ",\n".join("    " + repr({k: r[k] for k in order}) for r in recs)
    pathlib.Path(path).write_text(
        '"""Synthetic search benchmark v2: 100 paired English/Chinese queries.\n\n'
        "Generated by scripts/generate_synthetic_queries.py "
        f"(model={model}, seed={SEED}), then\n"
        "refined by hand: every query was checked against real videos and 47 were rewritten\n"
        "so each has at least one genuinely matching video (dylan-docs/synthetic-queries.md).\n"
        "DO NOT regenerate this file — edit queries by hand and re-verify them instead.\n"
        "Labels come from the catalogue target, not the LLM: expected_topics are\n"
        "caption first-level category ids; expected_subcategory is a level-2/3 name.\n"
        "split: 30 dev (tuning) / 70 test (reporting). n_catalogue / n_small = videos\n"
        "matching the labels in the searchable catalogue / small_matrix.\n"
        'The legacy 27-query set stays in synthetic_queries.py.\n"""\n\n'
        f"SYNTHETIC_QUERIES_V2: list[dict] = [\n{body},\n]\n",
        encoding="utf-8",
    )
    print(f"wrote {path} ({len(recs)} queries)")


def write_review(path: str, recs: list[dict], index: CategoryIndex) -> None:
    rows = [
        ("| id | split | type | constraint | English | 中文 | topics | sub-category | "
         "intent | style | duration | exclude | small |"),
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in recs:
        topics = ", ".join(index.get_by_id(t).english_name for t in r["expected_topics"])
        rows.append(
            f"| {r['id']} | {r['split']} | {r['query_type']} | {r['constraint']} | "
            f"{r['query']} | {r['query_zh']} | {topics} | {r['expected_subcategory'] or '-'} | "
            f"{r['expected_intent']} | {r['expected_style']} | {r['expected_duration']} | "
            f"{'; '.join(r['expected_negative']) or '-'} | {r['n_small']} |")
    out = pathlib.Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("# Synthetic queries v2 — review\n\nMark any row that is unnatural, "
                   "mislabelled or unsafe.\n\n" + "\n".join(rows) + "\n", encoding="utf-8")
    print(f"wrote review table {path}")


def print_summary(recs: list[dict]) -> None:
    c = Counter
    print("types:", dict(c(r["query_type"] for r in recs)),
          "| constraints:", dict(c(r["constraint"] for r in recs)),
          "| split:", dict(c(r["split"] for r in recs)))
    print("intents:", dict(c(r["expected_intent"] for r in recs)))
    print("en words: median", int(np.median([len(r["query"].split()) for r in recs])),
          "| categories covered:", len({t for r in recs for t in r["expected_topics"]}))


if __name__ == "__main__":
    main()
