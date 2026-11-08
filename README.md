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
| 2 | ALS Recall@100 0.090 vs popularity 0.046; Markov HR@10 0.014 vs popularity 0.007 (rewatches excluded) |
| 3 | Hybrid Recall@100 0.859, Recall@20 0.398 |
| 4 | Personalisation helps **only with session context**: nDCG@5 +0.014 (α=0.4, w_query=0.6, significant). History alone lowers query relevance |

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
