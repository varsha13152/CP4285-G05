# Evaluation

Shared metrics, plus the test sets and method for evaluating the pipeline offline.

## Metrics

| Metric | Function | What it measures |
|--------|----------|------------------|
| Precision@K | `precision_at_k` | Fraction of top-k that are relevant |
| Recall@K | `recall_at_k` | Fraction of relevant items in top-k |
| NDCG@K | `ndcg_at_k` | Ranking quality with graded relevance |
| MRR@k | `mrr(ranked, gt, k=3)` | 1/rank of first relevant item in top-k |
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
| **Metric 1, query relevance**: does the list answer what was searched? | `answer_video_ids` (binary, same for every user) | **P@5, MRR@3** | `anonymous` | **guardrail**: must hold |
| **Metric 2, personal preference**: among on-topic videos, are the ones *this user* enjoys near the top? | this user's `small_matrix` `watch_ratio`: on-topic → capped at 5, off-topic → 0, no label → skipped | **nDCG@5** | `anonymous*` (the anonymous ranking graded against the same user) | **target**: should rise |

**Personalisation works if metric 2 rises while metric 1 holds.**
- If metric 1 drops, the ranker is ignoring the query.
- If neither moves, personalisation adds nothing.

Key decisions and why:
- **Two metrics, not one.** Answers are caption-based (they reward text similarity) and identical for every user. Only 6% of (query, user) pairs have the user loving an answer video. Grading personalisation by answers would favour plain lookup, so personalisation is graded on watch behaviour.
- **@5:** real searches show ~6 results and get ~1 click.
- **MRR@3 for metric 1:** answer labels are binary and incomplete, so graded nDCG adds nothing. Cutoff at 3 matches real search behaviour (~1 click in ~6 results).
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

## Ranking formula (simplified)

The ranker was simplified from a 14-knob three-stage cascade to a single α-controlled blend:

```
personal = α × long_term + (1 − α) × short_term
score    = w_query × query_relevance + (1 − w_query) × personal
```

| Composite | Sub-signals (each rank-normalised, then averaged) |
|---|---|
| **Long-term** | CF neighbour scores, category affinity, social scores |
| **Short-term** | Markov transitions, ALS item-item sim, caption sim, session vector sim |

- `alpha` (0–1): the variable of interest. 1.0 = pure long-term, 0.0 = pure short-term.
- `w_query` (default 0.5): fixed weight on query relevance vs personal signals.
- MRR is now **MRR@3** (returns 0 if no answer in top 3).
- Removed from old ranker: popularity/trending, direct query↔caption similarity, Stage 5 item-item cascade, Stage 6 (diversity window, seen-video demotion, negative constraint penalties).

### How to run

```bash
# Single α evaluation
python scripts/evaluate_ranking.py --query-set v3 --split test --cases --alpha 0.5
# α sweep (caches Stages 1-3 once, re-ranks cheaply)
python scripts/sweep_alpha.py --alphas 0.0 0.25 0.5 0.75 1.0
# Tune α + w_query on dev split
python scripts/tune_weights.py --query-set v3
```

## Latest results (v3 test split, simplified ranker, α=0.5)

- **Stage 3**, gold intents, N = 100 (unchanged):
  - hybrid: P@5 0.111, MRR 0.322;
  - semantic: 0.064 / 0.189;
  - category-only: 0.012 / 0.048.
- **Ranking** session cases, α=0.5, w_query=0.5 (`results/ranking_eval_v3_simplified.txt`):

**Metric 1 — query relevance (guardrail):**

| Condition | n | P@5 | MRR@3 | lift P@5 / MRR@3 |
|---|---|---|---|---|
| anonymous (baseline) | 81 | 0.0593 | 0.1296 | — |
| none (history only) | 405 | 0.0464 | 0.0930 | −0.013 / −0.037 |
| related | 405 | 0.0351 | 0.0819 | −0.024 / −0.048 |
| unrelated | 405 | 0.0415 | 0.1025 | −0.018 / −0.027 |

**Metric 2 — personal preference (target):**

| Condition | n | nDCG@5 | judged@5 | lift nDCG@5 |
|---|---|---|---|---|
| anonymous* (baseline) | 185 | 0.1646 | 0.168 | — |
| none | 185 | 0.1775 | 0.061 | +0.013 |
| related | 185 | 0.1617 | 0.071 | −0.003 |
| unrelated | 185 | 0.1696 | 0.087 | +0.005 |

Realistic blend: P@5=0.046 · MRR@3=0.093 · nDCG@5=0.177.

**Findings (α=0.5):**
- Long-term modelling lifts nDCG@5 by +0.013 in the `none` condition (the clearest signal of personalisation helping).
- Query relevance (MRR@3) drops across all personalised conditions — the α=0.5 default gives too much weight to personal signals, pulling results off-query.
- Related recent viewing continues to hurt both metrics — over-personalisation from session context.
- `judged@5` drops sharply (0.17 → 0.06–0.09) when personal signals are active, indicating personal signals promote long-tail videos outside `small_matrix`.
- **Next: sweep α** to find the balance that preserves query relevance while capturing the nDCG@5 lift.

## Known limitations
- No real search judgements: `watch_ratio` shows liking in the feed, not search satisfaction, so metric 2 is a proxy.
- Metric 2 covers only the popular head; the "seen" feature can't be tested, because previously watched videos are never graded.
- Answer lists are incomplete and caption-based (one judge, captions only), so metric 1 is a lower bound. Planned fix: pooled judging of each mode's top 5.
- Category labels are coarse.
- 5 users per query, all with on-topic history. Users with niche or no history are under-tested.
- Session cases pair **real** users and real `big_matrix` sessions with **synthetic** queries: these users never actually ran these searches.
- KuaiSAR statistics (2023) are applied to KuaiRec (2020) via a hand-written category mapping.
- Not represented: photo-set, account and uncategorised searches. Refinement queries are graded standalone.

## Next steps
1. Run α sweep across 0.0–1.0 and find optimal balance for MRR@3 + nDCG@5.
2. Pool-judge the top results to complete the answer lists.
3. Temporal hold-out on `big_matrix` to grade long-tail personalisation.

Owner of this method: Dylan (Stage 3). The top-level README section "How Stage 3 is evaluated" has the Stage 3 benchmark history.
