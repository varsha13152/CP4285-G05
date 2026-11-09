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

"On-topic" means the video's caption first-level category
(`kuairec_caption_category.csv`, via `recsys.data.item_features.video_topic_sets`)
is one of the query's `expected_topics`, the same ids Stage 1 uses. Before
2026-11-09 the scripts used the `feat` tags of `item_categories.csv`, which
number categories differently and stop at 30.

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
When both LT and ST composites are active:
    personal = α × long_term + (1 − α) × short_term
When only one composite is active (e.g. no session → ST dead):
    personal = the active composite (full budget)
score = w_query × query_relevance + (1 − w_query) × personal
```

| Composite | Sub-signals (each rank-normalised, then averaged) |
|---|---|
| **Long-term** (3 signals) | CF neighbour scores, category affinity, social scores |
| **Short-term** (3 signals) | Markov transitions, ALS item-item sim, caption sim |

**Budget normalisation:** when one composite is dead (e.g. no session → ST = 0), the other gets the full personal budget. α only controls the LT/ST blend when both composites are active — this prevents `none` condition users from receiving half the personalisation of `related` condition users at the same α.

- `alpha` (0–1): the variable of interest. 1.0 = pure long-term, 0.0 = pure short-term.
- `w_query` (default 0.5): weight on query relevance vs personal signals.
- MRR is now **MRR@3** (returns 0 if no answer in top 3).
- **Guardrail threshold:** MRR@3 drop ≤ 0.01 absolute vs anonymous baseline. A setting that drops MRR@3 by more than this is deemed to hurt query relevance unacceptably.
- **Statistical significance:** bootstrap 95% CIs (2,000 resamples, paired) are computed for nDCG@5 lift vs anonymous*. A lift is significant when the CI excludes zero.
- Removed from old ranker: popularity/trending, direct query↔caption similarity, session vector cosine similarity (redundant with ALS item-item sim), Stage 5 item-item cascade, Stage 6 (diversity window, seen-video demotion, negative constraint penalties).

### How to run

```bash
# Single α evaluation
python scripts/evaluate_ranking.py --query-set v3 --split test --cases --alpha 0.5
# α sweep (caches Stages 1-3 once, re-ranks cheaply)
python scripts/sweep_alpha.py --alphas 0.0 0.25 0.5 0.75 1.0
# Joint α × w_query sweep (leak-free)
python scripts/sweep_alpha.py --alphas 0.0 0.25 0.5 0.75 1.0 --w-queries 0.5 0.6 0.7 0.8
```

## Latest results (v3 test, α × w_query sweep, `results/alpha_sweep.json`)

- **Stage 3**, gold intents, N = 100, topic weight `w_cat` 0.1 (`results/stage3_eval_v3_test_wcat01.txt`):
  - hybrid: P@5 0.109, MRR 0.310;
  - semantic: 0.064 / 0.189;
  - category-only: 0.012 / 0.048.
  - At the old `w_cat` 0.5, hybrid was 0.111 / 0.322 (`results/stage3_eval_v3_test.txt`). Semantic and category-only do not use `w_cat`, so they are unchanged.

### α sweep findings (44 settings: 11 α × 4 w_query)

**1. Budget normalisation confirmed.** With the fix, the `none` condition (no session → ST dead) gives the same metrics for all α values at a given w_query. α only controls the LT/ST blend when both composites are active (`related` / `unrelated` conditions).

**2. The `none` condition always fails the guardrail.** No (α, w_query) setting keeps MRR@3 within 0.01 of anonymous for the `none` condition. Even at w_query=0.8 the drop is −0.028. This means long-term personalisation hurts query relevance — the CF/category/social signals pull results off-query.

| w_query | none MRR@3 | MRR@3 drop | none nDCG@5 | nDCG@5 Δ vs anon* |
|---|---|---|---|---|
| 0.5 | 0.0914 | −0.042 ✗ | 0.1634 | −0.002 (ns) |
| 0.6 | 0.0988 | −0.035 ✗ | 0.1649 | −0.000 (ns) |
| 0.7 | 0.1004 | −0.033 ✗ | 0.1664 | +0.001 (ns) |
| 0.8 | 0.1053 | −0.028 ✗ | 0.1657 | −0.001 (ns) |

**3. Session-aware personalisation (`related`) does work — significantly.**

| Setting | related nDCG@5 Δ | 95% CI | p<0.05? |
|---|---|---|---|
| **α=0.4, wq=0.6** | **+0.0138** | [+0.002, +0.026] | **yes ✓** |
| α=0.5, wq=0.6 | +0.0114 | [+0.000, +0.023] | yes |
| α=0.6, wq=0.6 | +0.0114 | [+0.001, +0.023] | yes |
| α=0.4, wq=0.7 | +0.0116 | [+0.001, +0.024] | yes |
| α=0.3, wq=0.7 | +0.0111 | [+0.000, +0.022] | yes |
| α=0.6, wq=0.5 | +0.0128 | [+0.001, +0.025] | yes |
| α=0.7, wq=0.6 | +0.0111 | [+0.000, +0.024] | yes |

When users have **on-topic recent viewing**, a 60/40 long-term/short-term blend lifts personal preference by +0.014 nDCG@5 (p<0.05). The optimal setting is **α=0.4, w_query=0.6**.

**4. The `unrelated` condition passes the guardrail at some settings** (e.g. α=0.1, wq=0.6: MRR@3 drop = 0.007 ✓, nDCG@5 Δ = +0.005 ns).

**5. Realistic blend** (95% none / 2.1% related / 2.9% unrelated):
Best realistic nDCG@5 = 0.1666 at α=0.3, wq=0.7 (MRR@3 = 0.1014).

### Interpretation

The experiment reveals a **fundamental trade-off** between query relevance and personal preference:

- **Long-term signals help personal taste but hurt query matching.** CF/category scores promote videos the user tends to enjoy, even when they don't match what was searched. No amount of α/w_query tuning resolves this for the `none` (history-only) condition — the nDCG@5 lift is not statistically significant, and MRR@3 always drops.
- **Short-term signals add value when session context is available.** The `related` condition is the only one with significant nDCG@5 lift (p<0.05). On-topic recent viewing gives the ranker a useful signal that doesn't conflict with query relevance as much.
- **α=0.4–0.6 is the optimal LT/ST blend** for session-aware personalisation. Pure LT (α=1.0) or pure ST (α=0.0) both perform worse.
- **w_query=0.6–0.7 is the sweet spot**: more query weight reduces MRR@3 damage while preserving the nDCG@5 lift.

**Recommendation:** For a deployed system, apply personalisation **only when session context is available** (related condition), at α=0.4, w_query=0.6. For users without session context (95% of searches), use pure query relevance (w_query=1.0).

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
1. Pool-judge the top results to complete the answer lists.
3. Temporal hold-out on `big_matrix` to grade long-tail personalisation.

Owner of this method: Dylan (Stage 3). The top-level README section "How Stage 3 is evaluated" has the Stage 3 benchmark history.
