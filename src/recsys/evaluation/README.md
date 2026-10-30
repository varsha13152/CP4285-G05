# Evaluation

Shared metrics, plus the test sets and method for evaluating the pipeline offline.

## Metrics

| Metric | Function | What it measures |
|--------|----------|------------------|
| Precision@K | `precision_at_k` | Fraction of top-k that are relevant |
| Recall@K | `recall_at_k` | Fraction of relevant items in top-k |
| NDCG@K | `ndcg_at_k` | Ranking quality with graded relevance |
| MRR | `mrr` | 1/rank of first relevant item |
| Coverage | `coverage` | Fraction of catalog recommended |

## Usage

```python
from recsys.evaluation import precision_at_k, ndcg_at_k

ground_truth = {101, 202, 303}  # set of relevant video IDs
score = precision_at_k(ranked_list, ground_truth, k=10)
```

## Data used for evaluation

- **`big_matrix`** (7,176 users × 10,728 videos, 16% observed): used for **training** and for session context.
- **`small_matrix`** (1,411 users × 3,327 videos, **99.6% observed**: nearly every user watched nearly every video): used for **ground truth**.
  - Full observation means a low `watch_ratio` really means "didn't like it", not "never shown".
  - The two matrices share **no** (user, video) pairs.
- Both are feed logs (no search logs, no clicks or likes). `watch_ratio ≥ 2` = "loved".
- **KuaiSAR** (real 2023 Kuaishou searches, tokenised, not joinable) supplies **statistics only**: `results/kuaisar_query_profile.json`.

## Test sets

Load any set with `tests.fixtures.query_sets.load_queries(query_set, split, lang)`.

| Set | Fixture | Notes |
|---|---|---|
| legacy | `synthetic_queries.py` (+ `_zh`) | 27 hand-written queries; kept for comparability |
| v2 | `synthetic_queries_v2.py` | 100 catalogue-grounded queries, 30 dev / 70 test |
| **v3** | `synthetic_queries_v3.py` | 100 queries + 15 refinements, **shaped like real KuaiSAR searches**, each with hand-verified `answer_video_ids` |
| session cases | `session_cases_v3.py` | 1,840 leak-free cases for Stage 4 personalisation |

> ⚠️ **Use the included test sets. Do not regenerate them.**
> The query fixtures (`synthetic_queries_v2.py`, `synthetic_queries_v3.py`) and the session cases (`session_cases_v3.py`) are committed and ready to use. Just run the evaluation commands below.
> - **Don't run** `scripts/generate_synthetic_queries.py`, `scripts/generate_synthetic_queries_v3.py` or `scripts/build_session_cases_v3.py`.
>   - The query sets were **verified and corrected by hand** after generation. Re-running the generators would overwrite that work with new, unverified LLM output (they refuse to overwrite without `--force`; don't pass it).
>   - The session cases are tied to the current v3 query IDs.
> - The scripts are kept only to document how the sets were built.
> - If a query looks wrong, **edit it by hand** in the fixture and tell Dylan (owner), rather than regenerating.

**v3 queries** (`scripts/generate_synthetic_queries_v3.py`):
- **Mix:** 72 specific / 18 multi-topic / 10 broad; 48 name a specific thing (person, game, show…).
- **Shape from KuaiSAR:** lengths follow KuaiSAR's distribution (median 3 words); topics are weighted 50/50 by KuaiSAR search demand and KuaiRec catalogue size.
- **`answer_video_ids`:** videos whose captions were read by hand and judged to answer the query. Named queries start from the videos carrying the named hashtag. The lists are **not exhaustive**.
- **`n_catalogue` / `n_small`:** how many searchable / `small_matrix` videos match the query's category labels (evaluability, not answers).

**Session cases** (`scripts/build_session_cases_v3.py`):
- **Users:** per query, 5 `small_matrix` users who loved ≥1 video in the query's categories.
- **Conditions per user:**
  - `none`: no recent viewing;
  - `related`: the last ≤6 videos of a real `big_matrix` session ending on an on-topic video;
  - `unrelated`: the last ≤6 videos of a session with nothing on-topic.
- Plus one `anonymous` case per query.
- **Recent viewing comes only from `big_matrix`,** so it can't leak the `small_matrix` ground truth. The old `evaluate_ranking.py` context (a user's top-5 `small_matrix` videos) did leak it.
- **Real-world mix** (KuaiSAR answered searches): none 95% / related 2.1% / unrelated 2.9%.

## How to evaluate

```bash
# Stage 3 retrieval (gold intents isolate Stage 1 errors; --use-parser = end to end)
python scripts/evaluate_stage3.py --query-set v3 --split test [--lang zh]
# Stage 4 ranking + personalisation (always use --cases; without it the context leaks)
python scripts/evaluate_ranking.py --query-set v3 --split test --cases
# Tuning: v2/v3 use the dev split automatically
python scripts/tune_weights.py --query-set v3
```

**Tune on dev, report on test.**

## Which metric answers which question

Metrics only **grade** the output. Per case, the pipeline runs once, searching the
**full catalogue** as it would in production, and its one ranked list is graded twice:

| Question | Answer key | Metric | Baseline for lift | Role |
|---|---|---|---|---|
| Stage 3: are the right kinds of videos in the pool? | on-topic videos (category or sub-category) | Recall@100 | — | retrieval diagnostic |
| **Metric 1, query relevance**: does the list answer what was searched? | `answer_video_ids` (binary, same for every user) | **P@5, MRR** | `anonymous` | **guardrail**: must hold |
| **Metric 2, personal preference**: among on-topic videos, are the ones *this user* enjoys near the top? | this user's `small_matrix` `watch_ratio`: on-topic → capped at 5, off-topic → 0, no label → skipped | **nDCG@5** | `anonymous*` (the anonymous ranking graded against the same user) | **target**: should rise |

**Personalisation works if metric 2 rises while metric 1 holds.**
- If metric 1 drops, the ranker is ignoring the query.
- If neither moves, personalisation adds nothing.

Key decisions and why:
- **Two metrics, not one.** Answers are caption-based (they reward text similarity) and identical for every user. Only 6% of (query, user) pairs have the user loving an answer video. Grading personalisation by answers would favour plain lookup, so personalisation is graded on watch behaviour.
- **@5:** real searches show ~6 results and get ~1 click.
- **MRR for metric 1:** answer labels are binary and incomplete, so graded nDCG adds nothing.
- **nDCG for metric 2:** `small_matrix` is fully observed, so each user's ideal order is actually known.
- **No secondary metrics:** nDCG@10 and MRR-on-loved were only robustness checks; add them back if a result is in doubt.
- **Named queries (e.g. "ju jingyi") are graded on metric 1 only.** Their category (all celebrities) is far too broad to stand in for the query.
- **Category matching alone is circular:** category-only retrieval scores hit-rate 1.0 but P@5 0.012 on v3.

### Grading personal preference on the full catalogue (condensed list)
Personal labels exist only for `small_matrix` videos: 33% of the catalogue, and the **popular** third (median 2,832 views per video vs 84 for the rest).
- **Searching only `small_matrix` was rejected.** It isn't the system we'd deploy, it loses the long tail (143 of 273 answers), and it makes retrieval artificially easy.
- **Instead:** the pipeline searches everything, and metric 2 **skips videos without a label** before computing nDCG@5. This is the standard IR treatment of incomplete judgements; treating them as wrong would penalise good long-tail results.
- **What the skipping does *not* do:**
  - it doesn't change what the pipeline recommends;
  - it doesn't favour videos the user watched before. The pipeline learns users only from `big_matrix`; 0.00% of the case users' `big_matrix` views are on `small_matrix` videos, so every graded video is one the pipeline never saw this user interact with.
- **Limit:** metric 2 judges personalisation only among popular videos.
  - The output prints `judged@5`, the share of the first 5 results that carry a label.
  - Compare conditions by **lift**, since the bias is shared.
  - A temporal hold-out on `big_matrix` is the future fix for the long tail.
- **List length:** with `--cases` the ranker returns 100 results (the first 20 are unchanged) so enough labelled videos remain to grade.

## Latest results (v3 test split, 2026-10-30)

- **Stage 3**, gold intents, N = 100:
  - hybrid: P@5 0.111, MRR 0.322;
  - semantic: 0.064 / 0.189;
  - category-only: 0.012 / 0.048.
- **Stage 4** session cases (`results/ranking_eval_v3_test_cases.txt`):

| Condition | P@5 | MRR | nDCG@5 (personal) |
|---|---|---|---|
| anonymous / anonymous* | 0.049 | 0.128 | 0.151 |
| none (history only) | 0.060 | 0.165 | 0.153 |
| related recent viewing | 0.029 | 0.109 | 0.149 |
| unrelated recent viewing | 0.040 | 0.126 | 0.156 |

**Findings:**
- User history slightly improves query relevance but doesn't measurably improve personal preference (nDCG@5 lifts within ±0.005, i.e. noise).
- Related recent viewing pulls results **off-query** without making them more personal.
- Session signals also push unlabelled long-tail videos into the top 5 (`judged@5` falls from 0.59 to 0.30).
- The Stage 4/5 session weights need re-tuning against these cases.

## Known limitations
- No real search judgements: `watch_ratio` shows liking in the feed, not search satisfaction, so metric 2 is a proxy.
- Metric 2 covers only the popular head; the "seen" feature can't be tested, because previously watched videos are never graded.
- Answer lists are incomplete and caption-based (one judge, captions only), so metric 1 is a lower bound. Planned fix: pooled judging of each mode's top 5.
- Category labels are coarse.
- 5 users per query, all with on-topic history. Users with niche or no history are under-tested.
- Session cases pair **real** users and real `big_matrix` sessions with **synthetic** queries: these users never actually ran these searches.
- KuaiSAR statistics (2023) are applied to KuaiRec (2020) via a hand-written category mapping.
- Not represented: photo-set, account and uncategorised searches. Refinement queries are graded standalone.
- `tune_weights.py` still uses the leaky context.

## Next steps
1. Re-tune `tune_weights.py` on the leak-free dev cases (optimise metric 2, keep metric 1 as a constraint).
2. Pool-judge the top results to complete the answer lists.
3. Temporal hold-out on `big_matrix` to grade long-tail personalisation.

Owner of this method: Dylan (Stage 3). The top-level README section "How Stage 3 is evaluated" has the Stage 3 benchmark history.
