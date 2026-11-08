# QCSAR — Query-Conditioned Search over KuaiRec

A **short-video search** pipeline built on the [KuaiRec dataset](https://github.com/chongminggao/KuaiRec)
(Kuaishou). A user types a specific query (optionally with their user id and the
videos they just watched) and gets back matching videos, ordered for them.

**Scope:** specific searches ("cute cat videos", "learn guitar basics"), not
open-ended browsing.

**Core rule:** the query decides *which* videos qualify; personalisation only
re-orders them. A search never changes the user's long-term profile.

## Pipeline

```
query ──► 1 Query understanding ──► QueryIntent ─────────┐
user_id + recent videos ──► 2 User & session model ──► EngagementProfile
                                                         ▼
                                   Router ──► QueryRetrievalRequest
                                                         ▼
                       3 Retrieval (query only) ──► CandidateSet (100)
                                                         ▼
                       4 Ranking (personalised) ──► RankedList (top 20)
```

| Stage | What it does | Package | Owner |
|-------|--------------|---------|-------|
| 1 Query understanding | LLM parses the query into topics, style, duration and intent; bge-m3 embeds it (multilingual, so English queries match Chinese captions) | `recsys/intent/` | Varsha |
| 2 User & session model | **Long-term:** ALS matrix factorisation on `watch_ratio`, plus lookups built from it and the raw data (similar users' videos, friends' videos, category habits, seen videos). **Short-term:** a Markov next-video model over 30-min sessions | `recsys/user_modelling/`, `recsys/session_modelling/` | Betahaxer |
| Routing | Turns the intent into a retrieval request: topics, soft duration target, pool size | `recsys/routing/` | Varsha |
| 3 Retrieval | Hybrid search: embedding similarity + caption-category match + soft duration fit. Uses the query only, so anonymous and new users need no special path. In-memory, ~2 ms | `recsys/retrieval/` | Dylan |
| 4 Ranking | `score = w_query × query relevance + (1 − w_query) × personal`, where `personal = α × long-term + (1 − α) × short-term`. Anonymous users get query relevance only | `recsys/ranking/` | Varsha |

Shared types live in `src/recsys/contracts/`; stage interfaces are the
`Protocol`s in `src/recsys/pipeline/pipeline.py`. Each package has its own
README with the details.

## What each stage passes on

| Stage | Input | Output | How it's calculated |
|-------|-------|--------|---------------------|
| 1 Query understanding | query text | `QueryIntent`: topics (category id + confidence), style, duration (SHORT / MEDIUM / LONG / ANY), intent type, negative constraints, `query_embedding` (1024 numbers) | LLM (SOCLaaS) returns the fields as JSON; bge-m3 embeds the raw query |
| 2 User & session model | `user_id`, recent video ids | `EngagementProfile` (fields below) and the video matrix V (given to the ranker) | Matrix factorization on `big_matrix` (ALS by default, LightFM optional) plus counting over the raw data |
| Router | `QueryIntent`, `EngagementProfile` | `QueryRetrievalRequest`: topics with confidence ≥ 0.3, duration range (SHORT 0–15 s, MEDIUM 15–60 s, LONG 60–315 s), pool size 100, profile passed through unchanged | fixed rules |
| 3 Retrieval | `QueryRetrievalRequest` | `CandidateSet`: 100 candidates, each with `query_score`, category, duration and the flags `seen`, `popularity`, `trending` | Stage 3 score below |
| 4 Ranking | `CandidateSet` (carries the profile) | `RankedList`: top 20 video ids with rank and score | Stage 4 score below |

### Stage 2 output: `EngagementProfile`

U = user vectors, V = video vectors (16 numbers each), trained on `big_matrix`. `UserModeller(backend=...)` picks the method: `"als"` (default, implicit ALS with `watch_ratio` as confidence) or `"lightfm"` (each video vector = its id + category + tag vectors, which gives rarely-watched videos usable vectors). Alternatives tried and results: [Stage 2 README](src/recsys/user_modelling/README.md#status-update-alternatives-tried-2026-11-08).

| Field | Type | How it's calculated | Used by |
|-------|------|---------------------|---------|
| `cf_neighbour_scores` | {video: score} | Take the 50 users with the highest cosine(U_user, U_other). For each video they watched: `Σ similarity × their watch_ratio` | Ranking (long-term) |
| `category_distribution` | {category: share} | Share of the user's views in each category | Ranking (long-term) |
| `social_scores` | {video: score} | Average `watch_ratio` among the user's friends who watched the video (empty for the 93% of users with no friends) | Ranking (long-term) |
| `session_context.next_item_scores` | {video: probability} | Markov: `P(next = j \| last watched = i)` = count(i → j) / count(i → any), counted within 30-min sessions | Ranking (short-term) |
| `session_context.recent_video_ids` | list | The recent videos given as input | Ranking (short-term) |
| `seen_video_ids` | set | Every video in the user's history | Retrieval sets the `seen` flag; not used in ranking |
| `user_vector` | 16 numbers | The user's row of U | Ranking (long-term MF score), only with `use_mf_score=True` (off by default) |
| `session_vector`, `recent_category_ids` | — | Average V of the recent videos; their categories | not used yet |
| Stats | numbers | Mean / std `watch_ratio`, number of views, active days, average session length, days since last view | not used yet |

Unknown users get empty fields; empty means "no data", not "dislikes".

### Stage 3 score: query relevance only

```
sem         = cosine(query_embedding, video text embedding)       # bge-m3
cat         = confidence of the query topic matching the video's category (0 if none)
dur         = 1 inside the duration range, else exp(−seconds outside / 15)
query_score = 1.0·sem + 0.1·cat + 0.2·dur
```

The topic weight was 0.5 until 2026-11-09. At 0.5 the topic bonus outweighed
the semantic differences across the whole pool, so when the parser chose a
neighbouring category (Fashion instead of Attractiveness/Looks) the answers
dropped out. See [`results/tune_topic_weight.txt`](results/tune_topic_weight.txt).

A video qualifies if it has text or a matching category; the top 100 by `query_score` become candidates.

### Stage 4 score: personalised re-ordering

| Group | Signal | Value for candidate *i* |
|-------|--------|-------------------------|
| Query | Query relevance | Stage 3 `query_score` |
| Long-term | CF | `cf_neighbour_scores[i]` |
| Long-term | Category affinity | `category_distribution[category of i]` |
| Long-term | Social | `social_scores[i]` |
| Long-term | MF score (optional) | `user_vector · V_i`, only with `use_mf_score=True` |
| Short-term | Markov | `next_item_scores[i]` |
| Short-term | ALS similarity | average of the top-3 cosine(V_i, V_recent) over the recent videos |
| Short-term | Caption similarity | average of the top-3 cosine(text embedding_i, text embedding_recent) over the recent videos |

A missing value counts as 0.

```
r(x)       = percentile rank of x among the candidates (0–1)
long_term  = average of r(CF), r(category affinity), r(social) [, r(MF score)]
short_term = average of r(Markov), r(ALS similarity), r(caption similarity)   # only with recent videos
             (a signal that is equal for every candidate is left out of its average)

personal   = α·long_term + (1 − α)·short_term   if both are active
           = the active one                     if only one is active
           = 0                                  if neither (anonymous user)

score      = w_query·r(query_score) + (1 − w_query)·personal
```

Defaults: α = 0.5, w_query = 0.5, `use_mf_score = False` (tested: no gain, see
Current results). The 20 highest scores are returned.

## Key design choices

- **Query-first:** retrieval never looks at the user, so taste can't pull
  off-topic videos into a search.
- **Personal signals are ranking features, never filters** (CF, social, session,
  popularity).
- **Seen videos are flagged, not removed:** people search for things they've seen.
- **Duration is a soft score:** only 1% of videos are long, so a hard filter
  would empty the results.
- **Only searchable videos:** private videos and ads are excluded (9,905 of 10,728 remain).

## Evaluation

There are no real query→video labels for KuaiRec, so all numbers are proxies.

### v3 queries and leak-free session cases

Earlier query sets labelled relevance by category and had few, hand-picked
queries. The **v3 set** (`tests/fixtures/synthetic_queries_v3.py`) fixes this.
It has 100 queries plus 15 refinements, shaped like real Kuaishou searches
(KuaiSAR profile in `results/kuaisar_query_profile.json`). Each query carries
hand-verified `answer_video_ids`, so **P@5 / MRR** measure specific answers, not
just "right category". Select it with `--query-set v3 --split test` (tune on
`--split dev`).

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

### Current results

| Stage | Result |
|-------|--------|
| 1 | Topics 76%, style 90%, duration 70%, intent 73% accuracy |
| 2 | ALS Recall@100 0.090 vs popularity 0.046; Markov HR@10 0.014 vs popularity 0.007 (rewatches excluded). Alternatives: LightFM fixes rare-video vectors but shows no significant ranking difference from ALS (the evaluation has almost no rare videos); the three session cuts (30-min, per-user, one per user) give the same HR@10 (0.132); MF score in ranking (`use_mf_score`): no gain, kept off (see Stage 4 row). [Details](src/recsys/user_modelling/README.md#status-update-alternatives-tried-2026-11-08) |
| 3 | Hybrid Recall@100 0.859, Recall@20 0.398 (gold intents, topic weight 0.5). Topic weight lowered to 0.1 (2026-11-09): with the live parser on v3 test, MRR@3 0.148 → 0.177 (3 queries better, 0 worse; not significant) and answer in pool 0.68 → 0.74; with gold topics MRR@3 0.270 → 0.259. Topic confidence threshold kept at 0.3 (no effect on dev). [Details](results/tune_topic_weight.txt) |
| 4 | Personalisation helps **only with session context**: nDCG@5 +0.014 (α=0.4, w_query=0.6, significant), but this did not reproduce in a 2026-11-08 rerun (no significant lift at any setting). History alone lowers query relevance. Adding the MF score (`user_vector · V_i`) as a 4th long-term signal does not help: on the history-only ("none") cases the nDCG@5 lift stays non-significant (−0.005 to −0.008 with it vs −0.001 to −0.002 without) and MRR@3 still drops 0.046–0.050 vs anonymous (guardrail ≤ 0.01 fails at 18/18 settings either way), so `use_mf_score` stays off. [Details](results/RANKING_RESULTS.md#8-option-1-direct-mf-score) |

Raw outputs are in `results/`.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,llm]"

# KuaiRec data (Zenodo) -> data/raw/
wget https://zenodo.org/records/18164998/files/KuaiRec.zip && unzip KuaiRec.zip
mv "KuaiRec 2.0/data/"*.csv data/raw/
wget -P data/raw https://zenodo.org/records/18164998/files/user_features_raw.csv
python scripts/download_data.py            # checks all 8 files are present

# SOCLaaS credentials: create a gitignored .env with
#   SOCLAAS_BASE_URL=https://soclaas-api.comp.nus.edu.sg/v1
#   SOCLAAS_API_KEY=<your key>
#   SOCLAAS_MODEL=default

python scripts/build_caption_embeddings.py # once, ~5 min -> data/processed/
```

## Running

```bash
python -m pytest tests/ -q                                         # unit tests (no data needed)
python scripts/run_pipeline.py --query "cute cat videos"          # anonymous search
python scripts/run_pipeline.py --query "cute cat videos" --user_id 14 --recent 8077,1778,10713
python scripts/evaluate_ranking.py --query-set v3 --split test --cases
python scripts/sweep_alpha.py                                      # α × w_query sweep
```

## Development

Each member owns their package. Work on `feat/<module>`, run
`python -m pytest tests/test_<module>.py -v`, and open a PR to `main`.
