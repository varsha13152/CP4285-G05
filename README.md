# QCSAR — Query-Conditioned Search over KuaiRec

A **video search** pipeline built on the [KuaiRec dataset](https://github.com/chongminggao/KuaiRec)
(Kuaishou short videos). A user types a *specific* search query (plus, optionally,
their user id and the videos they just watched); the system returns the videos
that match the query, ordered for that user.

**Scope.** Search only, for specific queries ("cute cat videos", "learn guitar
basics"). Open-ended browsing ("something funny", "what's popular") is the
main-page recommender's job and out of scope. **Design principle:** the query
decides *which* videos qualify; personalisation only re-orders within them, and a
search never changes the user's long-term profile.

## Architecture — the query-first cascade

```
query ──► Stage 1  QueryParser (LLM) + bge-m3 embedding ──────────► QueryIntent
user_id + recent videos ──► Stage 2  UserModeller (+ SessionModeller) ─► EngagementProfile
        ──► QueryRouter ──► QueryRetrievalRequest (topics, soft duration, pool size)
        ──► Stage 3  HybridRetriever — query relevance only ──────► CandidateSet (100)
        ──► Stage 4  ranking — personalised, pairwise (not built) ─► RankedList ──► user
```

| Stage | Owner | Package | Input → Output |
|-------|-------|---------|----------------|
| 1 Query understanding | Varsha | `recsys/intent/` | query → `QueryIntent` (topics, style, duration, intent, embedding) |
| 2 User & session modelling | Betahaxer | `recsys/user_modelling/`, `recsys/session_modelling/` | user id + recent videos → `EngagementProfile` (+ lookups for Stage 4) |
| Routing | Varsha | `recsys/routing/` | `QueryIntent` + profile → `QueryRetrievalRequest` |
| 3 Candidate retrieval | Dylan | `recsys/retrieval/` | request → `CandidateSet` of query-relevant videos |
| 4–6 Ranking | TBD | — | `CandidateSet` → `RankedList` |

All inter-stage types live in `src/recsys/contracts/` — import them from there.
Stage interfaces are the `Protocol`s in `src/recsys/pipeline/pipeline.py`;
`RecommendationPipeline.retrieve()` runs Stages 1–3 today.

## Design decisions (2026-09-25)

| # | Decision | Rationale |
|---|----------|-----------|
| 1 | **Query-first cascade** replaces the earlier multi-lane design (collaborative / content / popularity / social / session lanes with router weights) | In search the query must gate relevance. Independent lanes let a user's taste or last-watched video pull off-topic videos into a search ("searched cooking, got cats"). |
| 2 | **Stage 3 retrieves on query relevance only** | Keeps every candidate on-topic; cold-start and anonymous users need no special path because retrieval never depends on the user. |
| 3 | **Hybrid retrieval**: bge-m3 text embeddings ∪ caption-category match | Semantic search goes beyond 39 coarse categories and bridges English queries to Chinese text (bge-m3 is multilingual — no translation); the category source guarantees exact-topic matches. Benchmarked below. |
| 4 | **Duration is a soft score**, not a filter | Stage 1 duration is only 70% accurate, and KuaiRec is 85% SHORT / 14% MEDIUM / 1.1% LONG — a hard LONG filter would leave ~115 videos. |
| 5 | **100 candidates**, configurable (`QueryRouter(default_num_candidates=...)`) | Stage 4 plans pairwise ordering: ~5k pairs at N=100 is affordable; ~500k at N=1,000 is not. |
| 6 | **CF is a Stage 4 ranking signal** (kNN users over ALS vectors), not a retrieval filter; benchmark with vs without | As a filter, CF drops relevant niche videos no neighbour watched (big_matrix is 16% dense) and overrides the explicit query. |
| 7 | **Session and social are Stage 4 features**, precomputed by Stage 2 as user-side lookups on `EngagementProfile` | Stage 2 runs before candidates exist, so it builds candidate-independent tables; Stage 4 does one dict lookup per candidate. |
| 8 | **`recent_video_ids` is a Stage 2 input** (`build_session_profile`), not a request field | The session context is built once by Stage 2 and travels on the profile. |
| 9 | **Social = friends first, trending fallback** | Friends' engagement is the truly social signal, but only 6.6% of users have friends (mean 1.4) — trending (a global item feature) covers everyone else. |
| 10 | **Popularity and trending are ranking features only** | Stage 3 has no lanes to back-fill; popularity in retrieval would bias search results toward what is popular rather than what was asked. |
| 11 | **Seen videos are flagged, not filtered** | People search for videos they've already seen; Stage 4 decides whether to demote them. |
| 12 | **One category id space: caption first-level ids** (1–39, `CategoryIndex`) | Stage 1 used caption ids but Stage 2 used `item_categories.csv` `feat` tags (0–30) — a different space, so query topics and user habits could not be compared. |
| 13 | **No vector DB** (e.g. Supabase pgvector) | 10.7k × 1024 float32 ≈ 44 MB fits in memory; exact numpy cosine takes ~2–3 ms vs a 20–100 ms network hop to a hosted DB. The catalogue is static (no inserts), approximate HNSW would blur the benchmark, and a DB adds credentials and test mocking for everyone. Revisit if a deployed multi-service demo is required, embeddings must be shared without a SOCLaaS key, or the catalogue grows ~100×. The `Retriever` protocol is the swap point. |
| 14 | **Vague queries dropped** from `tests/fixtures/synthetic_queries.py` (30 → 27) | They are out of scope for search. |
| 15 | **Embedded text = caption + cover text + topic tags + first/second/third-level category names + `video_tag_name`** | 9.2% of videos have no caption, cover text or tags; the names give every searchable video an embedding, and the third level separates e.g. 宠物猫 (cats) from 宠物狗 (dogs) inside Pets. Trade-off: semantic mode now also carries category signal, so semantic vs category is no longer a clean text-only comparison. |
| 16 | **Only searchable videos are in the catalogue** — drops videos whose latest `visible_status` is `private` / `only friends` (794) and ads (`video_type = AD`, 29): 9,905 of 10,728 remain | Search must never surface private content, and ads are not search results. Raw data and Stage 2 training history are untouched (those views genuinely happened). |

### Cold-start handling

* **Retrieval is unaffected**: Stage 3 uses only the query, so anonymous and new
  users get the same query-relevant candidates.
* **Stage 2 returns empty lookups** for unknown users: `user_vector = []`,
  `cf_neighbour_scores = {}`, `seen_video_ids = set()`, `session_context = None`
  without recent videos. `social_scores` still fills for a new user who has
  friends (it needs only the friends' history).
* **Stage 4 must read empty as "unavailable", not "zero preference"** — e.g. fall
  back to query relevance + popularity/trending when personal signals are missing.

### Requirements for Stage 4 (ranking)

1. **Query relevance stays a ranking feature.** KuaiRec only has user–video
   labels (`watch_ratio`), no query→video labels, so a ranker trained without
   `sem` / `cat` / `dur_fit` would learn what the user likes, not what they searched.
2. **Personal signals come from the profile**: `user_vector · item_factors`,
   `cf_neighbour_scores` (benchmark with vs without), `session_context`
   (`next_item_scores`, `session_vector`, `recent_category_ids`), `social_scores`,
   plus `popularity`, `trending` and `seen` from `Candidate.features`.
3. **Pairwise ordering cost**: comparing all pairs is O(n²) — fine at N=100
   (~5k pairs). The scalable option is to *train* with a pairwise loss
   (RankNet / LambdaMART) and score each item once at serving time.
4. Negative constraints ("no sad songs") belong in Stage 5 re-ranking.

## Stage 3 benchmark (2026-09-25)

27 specific synthetic queries, gold intents + live bge-m3 query embeddings
(`python scripts/evaluate_stage3.py`; raw output in `results/`). Hit-rate =
share of candidates in an expected category; small_matrix recall = share of each
user's liked (`watch_ratio ≥ 2`) videos in the expected categories that were
retrieved (catalogue restricted to the small_matrix videos that are searchable).

**Current (v2):** embedded text = caption + cover text + topic tags + category
names + `video_tag_name`; private videos and ads excluded
(`results/stage3_eval_2026-09-25*_v2.txt`).

| Mode | Hit-rate@100 | Recall@100 | Hit-rate@20 | Recall@20 | Retrieval |
|------|--------------|------------|-------------|-----------|-----------|
| semantic | 0.406 | 0.331 | 0.585 | 0.141 | ~2.3 ms |
| category | 1.000 | 0.861 | 1.000 | 0.363 | ~1.3 ms |
| **hybrid** (default) | 0.867 | 0.859 | 0.996 | **0.398** | ~2.1 ms |

**Before (v1):** caption + cover text + topic tags only, all videos
(`results/stage3_eval_2026-09-25.txt`, `_n20`, `_parser`).

| Mode | Hit-rate@100 | Recall@100 | Hit-rate@20 | Recall@20 |
|------|--------------|------------|-------------|-----------|
| semantic | 0.233 | 0.173 | 0.280 | 0.061 |
| category | 1.000 | 0.850 | 1.000 | 0.361 |
| hybrid | 0.864 | 0.835 | 0.994 | 0.398 |

* **Semantic roughly doubled** (recall@100 0.173 → 0.331, recall@20 0.061 →
  0.141) once category and tag names were embedded — partly because those names
  carry category signal, which the category-derived labels reward.
* Example — "cute cat videos": hybrid's top 20 are **all** level-3 宠物猫 (cats);
  category-only's top 20 are 10 dogs, 4 cats, 4 unknown and 2 other pets.
* **Hybrid improved slightly at N=100** (recall 0.835 → 0.859) and stayed best at
  N=20 (0.398 vs category 0.363), where semantic similarity orders videos
  *within* a category and category-only ties are broken arbitrarily.
* All modes also moved because the catalogue changed (private videos and ads
  removed), so v1 → v2 is not a pure text-only comparison.
* Labels are category-derived, which favours the category mode — read the
  semantic numbers as a lower bound.
* With the live Stage 1 parser (`--use-parser`), recall@100 is 0.391 / 0.883 /
  0.866 (v1: 0.226 / 0.876 / 0.830); parse + embedding takes ~1.6 s per query —
  the API calls, not retrieval, dominate latency.
* Tuning (measured on v1): `w_cat=1.0` (default 0.5) lifted hybrid to 0.844
  recall@100 and 0.408 recall@20 — a candidate default to agree on; re-check on v2.

## How Stage 3 is evaluated — and its limits

There are **no real search queries and no query→video relevance labels** for
KuaiRec (its logs are recommendation-feed views; KuaiSAR's real search logs are
tokenised and not joinable). Every Stage 3 number above is therefore a **proxy**.

### Current method (`scripts/evaluate_stage3.py`)

* **Queries:** the 27 hand-written synthetic queries
  (`tests/fixtures/synthetic_queries.py`), each with expected topics (caption
  category ids) and an expected duration.
* **Intents:** by default the expected ("gold") intent is fed straight into
  Stage 3, so Stage 1 errors are not charged to retrieval; `--use-parser` runs
  the live Stage 1 parser end to end.
* **Metrics** (for candidate generation **recall matters most** — Stage 3 must
  get the right videos into the pool; Stage 4 orders them):

| Metric | Definition | Assumption |
|--------|------------|------------|
| Category hit-rate@N | share of the N candidates whose category is an expected topic | "on-topic" = "in the right category" |
| small_matrix Recall@N | per user, relevant = videos they watched with `watch_ratio ≥ 2` in the expected categories; share of those in the (user-independent) candidate list, averaged over users, then queries | someone making this search would want the videos they already loved in that category |
| Latency | retrieval compute vs API calls | — |

### Limitations

1. **Circular labels.** Both accuracy metrics define relevance by category, so
   category mode scores ~1.0 hit-rate by construction and the metrics cannot see
   relevance *within* a category (cats vs dogs was only caught by hand).
2. **Few, subjective queries:** 27 queries, one author, hand-picked labels.
3. **Watching ≠ searching:** `watch_ratio` comes from the feed; it shows the user
   liked a video, not that it answers a query.
4. **Off-category but relevant videos count as misses** (e.g. a cat video filed
   under 生活 Daily Life).
5. **Some queries have 0 matching small_matrix videos**, which lowers every
   mode's average equally.

### v3 queries and leak-free session cases

Limitations 1, 2 and 5 are addressed by the **v3 set**
(`tests/fixtures/synthetic_queries_v3.py`; design in
`dylan-docs/synthetic-queries-v3.md`). It has 100 queries plus 15 refinements,
shaped like real Kuaishou searches (KuaiSAR profile in
`results/kuaisar_query_profile.json`). Each query carries hand-verified
`answer_video_ids`, so **P@5 / MRR** measure specific answers, not just "right
category". Select it with `--query-set v3 --split test` (tune on `--split dev`).

`scripts/evaluate_ranking.py` used each user's 5 most-watched `small_matrix`
videos as their "recent" context. Those same videos are the ground truth, which
leaks the answers. `--cases` (v3) replaces this with
`tests/fixtures/session_cases_v3.py`: recent videos come from real `big_matrix`
sessions, and results are reported per condition (anonymous / none / related /
unrelated), plus the lift over anonymous.

⚠️ **Use the committed test sets as they are; do not regenerate them.** The
query fixtures were corrected by hand, and the generator scripts exist only to
document how they were built. See the evaluation README.

**Which metric to use for what** (query relevance vs personal preference), and
the known limits of the method, are in
[`src/recsys/evaluation/README.md`](src/recsys/evaluation/README.md).

### Why hybrid scores the way it does

`score = w_sem·sem + w_cat·cat + w_dur·dur` with defaults `1.0 / 0.5 / 0.2`:

* The **category term (≈ +0.48 for a match) acts almost as a gate** — on-topic
  videos rise above off-topic ones.
* The **semantic term orders videos within the category** — category-only gives
  every video in a category the same score and breaks ties by `video_id`
  (arbitrary).
* A few **off-category videos with very close text still get in**
  (≈ 13% of hybrid's top 100).

Effect on the metrics:

* **vs semantic** — the category gate removes look-alikes such as "cute dance"
  for "cute cat": recall@100 0.331 → 0.859.
* **vs category at N=100 — a tie**, because order barely matters: for 16 of the
  27 queries the expected categories hold ≤ 100 small_matrix videos (median 80),
  so category mode returns the whole category and both modes hit the ceiling.
  Hybrid's lower hit-rate (0.867) is the off-category slots, which the labels
  count as misses.
* **vs category at N=20 — hybrid wins** (0.398 vs 0.363): only 6 of 27 queries
  fit in 20 slots, so order within the category decides what gets in; hybrid
  picks the 20 closest in meaning instead of the first 20 by id.

The metrics therefore **hide most of hybrid's real benefit** (ordering within a
category), which only shows at small N or by inspection.

### Future evaluation methods (not implemented)

Two complementary methods with **opposite biases**; using both gives a much
fairer picture than the category proxies. They are needed before deciding the
open tuning questions (`w_cat` default, hashtag stripping, query expansion,
video style labels).

**Method 1 — Known-item search (automatic, large scale).**
1. Sample ~300 searchable videos at random (stratified by category).
2. For each, have the Stage 1 LLM write the English query a user would type to
   find that video, from its caption, tags and category names.
3. Run Stage 3 on each generated query; the **source video is the one known
   relevant item**.
4. Report Recall@N (did the source video make the pool?) and MRR@N (how high?),
   per mode.

* Strength: hundreds of queries, labels **not** derived from categories, fully
  automatic and repeatable (~300 LLM calls, generated once and cached).
* Bias: queries are written from the video's own text, so it **favours the
  semantic mode** — the opposite of the current category bias. Only one relevant
  video per query, so it measures findability, not full relevance.

**Method 2 — Pooled relevance judgments (the standard search-evaluation approach).**
1. Choose 15–20 specific queries (can reuse the synthetic set).
2. **Pool** the top 20 from every mode (semantic, category, hybrid, and future
   variants) → ~40–60 unique videos per query, ~600–1,000 pairs in total.
3. Judge each (query, video) pair on a graded scale: 0 = not relevant,
   1 = partly relevant, 2 = relevant.
4. Report precision@k, nDCG@k, and recall against the pool, per mode.

* Strength: the closest available proxy to real relevance; sees within-category
  quality (cats vs dogs) and rewards relevant off-category videos.
* Cost: judging time. **Method 2b — LLM as judge:** an LLM judges each pair from
  the video's caption, tags and category names, and a human hand-labels a random
  sample (~100 pairs) to measure agreement before the LLM labels are trusted.
  Re-use the saved judgments; only newly pooled videos need judging when a new
  variant is added.
* Bias: pooling only judges videos some mode retrieved; relevant videos no mode
  found stay invisible (recall is relative to the pool).

Once these exist, keep the category metrics only as a sanity check rather than
the headline number.

## Change log for existing work (made on `feat/candidate-generation`)

Each change: **what** changed · **why** · **how it fits the cascade**.

### Contracts (`recsys/contracts/`, shared)
* `RetrievalStrategy` (5 lanes) → `RetrievalSource {SEMANTIC, CATEGORY, BOTH}`;
  `StrategyWeight` and `QueryRetrievalRequest.strategies` removed;
  `Candidate`/`RankedItem.source_strategy` → `source`;
  `CandidateSet.strategy_counts` → `source_counts`.
  · Lanes no longer exist. · A candidate now records *how the query found it*.
* `EngagementProfile` += `session_context`, `social_scores`,
  `cf_neighbour_scores`; new `SessionContext`.
  · Stage 4 needs session / social / CF signals. · Stage 2 precomputes them as
  user-side lookups that ride the existing profile → request → `CandidateSet` path.
* `CandidateSet.user_id` / `RankedList.user_id` accept `None`.
  · Anonymous search. · Retrieval does not need a user.
* `category_filter` / `duration_range_ms` documented as relevance signal / soft
  target (names kept to limit churn).

### Routing — `QueryRouter` (Varsha)
* Removed `_STRATEGY_MAP` and `_FALLBACK_STRATEGIES` (per-intent lane weights).
  · No lanes to weight. · The router now carries only what the query asked for:
  topics (confidence ≥ 0.3), a soft duration target and the pool size.
* `routing/README.md` rewritten — it described an older `IntentRouter` /
  `strategy_map.py` design that no longer existed.
* Tests: the lane-weight tests are replaced by pool-size and profile-passthrough tests.

### Stage 1 fixtures (Varsha)
* Removed the 3 vague queries from `tests/fixtures/synthetic_queries.py` (27 remain).
  · Out of scope for search. · `results/stage1_eval_2026-09-20.txt` is kept as
  the historical 30-query record; re-run `scripts/evaluate_stage1.py` for 27-query numbers.

### Stage 2 — `UserModeller` (Betahaxer)
* Category stats (`top_category_ids`, `category_distribution`) now use caption
  first-level ids via `recsys.data.item_features.caption_category_table`
  (ids outside `CategoryIndex` — UNKNOWN / strays — are skipped).
  · Decision 12. · User habits and query topics are now comparable.
* New `cf_neighbour_scores` (kNN, `k_neighbours=50`), `social_scores`
  (`social_network=` seam), and `build_session_profile(..., recent_video_ids)` →
  `session_context`.
  · Decisions 6–9. · Stage 4 reads these; the query still never mutates the profile.
* `UserModeller` itself is unchanged in method: ALS on big_matrix, default
  `confidence_alpha=None` — **verified** best (Recall@100 on small_matrix 0.090
  vs popularity 0.046; `alpha=10` gives 0.041).

### Stage 2 — session modelling (Betahaxer)
* `evaluate_next_item(..., collapse_repeats=True)`, `PopularityBaseline`, and
  `scripts/evaluate_session_baseline.py --collapse-repeats` plus a popularity row.
  · 10.1% of test steps are consecutive rewatches, which a first-order model
  predicts at rank 1: HR@10 0.106 falls to **0.014** once collapsed (popularity
  0.007). · The Markov table is real but weak beyond rewatches, so it is a
  Stage 4 feature (`SessionContext.next_item_scores`), not a candidate source.
* "SESSION lane" wording removed from the README and docstrings.

### Pipeline
* `RecommendationPipeline` gains a `UserProfiler` (Stage 2) protocol and
  `retrieve(query, user_id, recent_video_ids)`; `recommend()` requires a ranker.
  · Stage 2 was never called before (the profile was always `None`).
* `scripts/run_pipeline.py` now runs Stages 1–3 end to end.

## Verified data facts and known limitations (2026-09-25)

Every file and field is documented in
[`src/recsys/data/README.md`](src/recsys/data/README.md) (dataset reference).


* `small_matrix` ⊂ `big_matrix` in users and videos, with **0** shared
  `(user, video)` pairs — train on big, evaluate on small is valid.
* **9.2%** of videos have no text at all (12.6% have an empty caption); captions
  are short, Chinese and hashtag-heavy — semantic retrieval alone is weak.
* **6.6%** of users (472) have friends — the friends part of SOCIAL is mostly empty.
* Durations: SHORT 85% / MEDIUM 14% / LONG 1.1%.
* Stage 1 accuracy (30-query run): topics 76%, style 90%, duration 70%, intent 73%.
* Evaluation labels are category-derived (no query→video relevance labels exist).
* KuaiRec interactions come from the recommendation feed, not search logs.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,llm]"

# KuaiRec data (Zenodo) -> data/raw/  (scripts/download_data.py only checks the files)
wget https://zenodo.org/records/18164998/files/KuaiRec.zip && unzip KuaiRec.zip
mv "KuaiRec 2.0/data/"*.csv data/raw/
wget -P data/raw https://zenodo.org/records/18164998/files/user_features_raw.csv
python scripts/download_data.py            # verify all 8 files are present

# SOCLaaS credentials: create a gitignored .env with
#   SOCLAAS_BASE_URL=https://soclaas-api.comp.nus.edu.sg/v1
#   SOCLAAS_API_KEY=<your key>
#   SOCLAAS_MODEL=default

python scripts/build_caption_embeddings.py # once, ~5 min -> data/processed/ (gitignored)
```

## Running

```bash
python -m pytest tests/ -q                                         # unit tests (no data needed)
python scripts/run_pipeline.py --query "cute cat videos"          # Stages 1–3
python scripts/run_pipeline.py --query "cute cat videos" --user_id 14 --recent 8077,1778,10713
python scripts/evaluate_stage3.py [--use-parser] [--num-candidates 20]
python scripts/evaluate_session_baseline.py --strategies fixed --collapse-repeats
python scripts/evaluate_stage1.py --verbose
```

## Development

Each member owns their package (see the module READMEs for contracts). Work on
`feat/<module>`, run `python -m pytest tests/test_<module>.py -v`, and open a PR
to `main`. Changes to another member's module are recorded in the change log above.
