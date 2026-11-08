# QCSAR Ranking (Stages 4–6) — Evaluation Results

**Date:** 2026-09-26  
**Author:** Varsha  
**Scope:** Stages 4 (initial scoring), 5 (item-item re-ranking), 6 (constraints & diversity)

---

## 1. Evaluation Setup

| Parameter | Value |
|-----------|-------|
| Dataset | KuaiRec (small_matrix ground truth, big_matrix for training) |
| Queries | 20 synthetic Chinese queries across 18 KuaiRec categories |
| Users per query | up to 30 (540 (query, user) pairs evaluated) |
| Candidate pool | 200 (from Stage 3 hybrid retrieval) |
| Relevance threshold | watch_ratio ≥ 0.5 |
| Ground truth | **Query-aware**: video counts as relevant only if (a) user's watch_ratio ≥ 0.5 AND (b) video's category overlaps with the query's expected topic categories |
| NDCG cap | watch_ratio capped at 5.0 to prevent outlier distortion |
| Embedding model | bge-m3 (1024-dim, multilingual) via SOCLaaS |
| CF model | Implicit ALS (16 latent factors) trained on big_matrix |

### Why query-aware ground truth?

Standard offline evaluation counts *all* videos a user liked as relevant, regardless of topic. This penalises a search system for not recommending cooking videos when the user searched for "dance challenges." Our query-aware approach only counts a ground-truth video as relevant if its categories overlap with the query's topic — measuring "did we find on-topic videos the user enjoyed."

---

## 2. Final Results

| Metric | @5 | @10 | @20 |
|--------|-----|------|------|
| **Precision** | 0.473 | 0.434 | 0.375 |
| **Recall** | 0.042 | 0.067 | 0.098 |
| **NDCG** | 0.222 | 0.232 | 0.235 |

| Metric | Value |
|--------|-------|
| **MRR** | 0.679 |
| **Coverage** | 0.317 (1,056 / 3,327 items) |

### Interpretation

- **Precision@5 = 0.473**: Nearly half of the top-5 recommended items are on-topic videos the user genuinely enjoyed (watch_ratio ≥ 0.5). For a search-based recommender on a short-video platform, this is a strong result.
- **MRR = 0.679**: The first relevant result appears at position ~1.5 on average (1/0.679 ≈ 1.47). Users typically find something relevant immediately.
- **NDCG@10 = 0.232**: The ranking captures ~23% of the ideal ordering. This is moderate — limited primarily by retrieval quality (Stage 3 determines the candidate pool).
- **Recall@k is low by design**: With a pool of 200 candidates from 3,327 items and only returning top-20, we can't find all relevant items. Recall measures catalog coverage, not recommendation quality.
- **Coverage = 31.7%**: The system recommends 1,056 distinct items across all evaluations, avoiding the popularity-bias trap of always recommending the same videos.

---

## 3. Ranking Architecture

The ranker implements a three-stage cascade:

```
Stage 3 candidates (200 items)
    │
    ▼
┌─────────────────────────────────────────┐
│  Stage 4: Weighted Multi-Signal Scoring │
│  7 signals, rank-normalised, blended    │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│  Stage 5: Item-Item Similarity          │
│  Re-rank via ALS + caption + session    │
│  α-blend with Stage 4 scores            │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│  Stage 6: Constraints & Diversity       │
│  Seen demotion, category diversity,     │
│  negative constraints, top-k            │
└──────────────────┬──────────────────────┘
                   │
                   ▼
           RankedList (top-20)
```

### Stage 4 — Initial Scoring (7 signals)

Each candidate is scored as a weighted sum of rank-normalised signals:

| Signal | Weight | Source | Description |
|--------|--------|--------|-------------|
| `w_query` | 0.50 | Stage 3 | Composite retrieval score (semantic + category + duration) |
| `w_query_sem` | 0.05 | Stage 1 + embeddings | Direct cosine(query_embedding, caption_embedding) |
| `w_popularity` | 0.20 | Item features | 0.7 × popularity + 0.3 × trending |
| `w_cf` | 0.05 | Stage 2 (ALS) | Collaborative filtering neighbour scores |
| `w_cat_affinity` | 0.05 | Stage 2 (profile) | User's historical category preference distribution |
| `w_social` | 0.075 | Stage 2 (social) | Friends' mean watch_ratio for this video |
| `w_session_markov` | 0.075 | Stage 2 (session) | Markov chain next-item transition probability |

**Normalisation**: All signals are rank-normalised (`scipy.stats.rankdata`) to [0, 1] before blending. This is more robust than min-max normalisation because it treats signals by relative ordering rather than magnitude, preventing outliers from compressing other values.

### Stage 5 — Item-Item Re-Ranking

For logged-in users with session context, Stage 5 computes item-item similarity between each candidate and the user's recent videos:

| Signal | Weight | Description |
|--------|--------|-------------|
| ALS cosine | 0.50 | Mean of top-3 cosine similarities between candidate and recent videos (16-dim ALS factors) |
| Caption cosine | 0.30 | Mean of top-3 cosine similarities between candidate and recent videos (1024-dim bge-m3 embeddings) |
| Session vector | 0.20 | Dot product of candidate's ALS vector with session centroid |

The Stage 5 score is α-blended with the Stage 4 score:

```
final_score = 0.70 × stage4_score + 0.30 × stage5_score
```

### Stage 6 — Constraints & Diversity

Applied sequentially in rank order:

1. **Negative constraint penalties** (×0.8): Duration-based — penalises items with poor `dur_fit` when the query contains "too long" / "too short"
2. **Seen-video demotion** (×0.5): Already-watched videos are penalised but not removed (users may want to re-find videos)
3. **Category diversity** (×0.85): Sliding window of size 3 — if a candidate's primary category appeared in the last 3 selected items, it receives a 15% penalty
4. **Top-k truncation**: Final list capped at 20 items

---

## 4. Weight Tuning

Weights were optimised via grid search using `scripts/tune_weights.py`:

- **Method**: Cache Stage 1–3 outputs (CandidateSet objects), then grid-search over Stage 4 weight combinations + Stage 5 alpha, evaluating only the fast ranking stage (~2 combinations/second)
- **Objective**: Maximise NDCG@10
- **Search space**: 4 values per weight × 5 primary weights × 4 alpha values = 484 valid combinations
- **Data**: 10 Chinese queries × 15 users = 120 (query, user) pairs

### Tuning result

| | NDCG@10 |
|---|---------|
| Before tuning (hand-set weights) | 0.258 |
| After tuning (grid search) | 0.266 |
| **Improvement** | **+3.4%** |

### Key insight from tuning

The retrieval score (`w_query=0.50`) is by far the most important ranking signal. This makes sense: for query-conditioned search, the Stage 3 retrieval score already encodes semantic similarity + category match + duration fitness. The ranking stage's value comes from adding personalisation signals (CF, social, session, category affinity) and enforcing diversity — not from overriding retrieval.

---

## 5. Design Decisions

### Rank-based vs min-max normalisation

Min-max normalisation maps values to [0, 1] based on their magnitude:
```
normalised = (x - min) / (max - min)
```
**Problem**: One extreme outlier compresses all other values near zero. In our data, popularity scores have heavy-tailed distributions — a single viral video makes all others look equally unpopular.

Rank-based normalisation maps values to [0, 1] based on their *position*:
```
normalised = (rank - 1) / (n - 1)
```
This ensures uniform spread regardless of the raw distribution.

### Query-caption semantic similarity as a separate signal

The Stage 3 retrieval score blends semantic similarity with category and duration signals. By extracting a pure `cosine(query_embedding, caption_embedding)` as a separate Stage 4 signal, we let the weight tuner control how much pure semantic relevance matters independently of category/duration matching. The tuner assigned it a small weight (0.05), confirming the blended retrieval score already captures most of the semantic signal.

### User-category affinity

The `category_distribution` from Stage 2's engagement profile gives a normalised histogram of the user's viewing history by category. For example, a user who watches 40% food videos and 10% music videos will see food candidates boosted relative to music candidates — even if both categories are relevant to the query. This is a lightweight personalisation signal that doesn't require collaborative filtering.

---

## 6. Metric Definitions

| Metric | Formula | Measures |
|--------|---------|----------|
| **Precision@k** | \|relevant ∩ top-k\| / k | Fraction of recommendations that are relevant |
| **Recall@k** | \|relevant ∩ top-k\| / \|all relevant\| | Fraction of all relevant items found |
| **NDCG@k** | DCG@k / idealDCG@k, using watch_ratio as graded relevance | Quality of ranking order (higher watch_ratio items should rank first) |
| **MRR** | mean(1 / rank_of_first_relevant) | How quickly the user finds a relevant item |
| **Coverage** | \|unique recommended items\| / \|total items\| | Catalog diversity across all recommendations |

---

## 7. Files

| File | Description |
|------|-------------|
| `src/recsys/ranking/ranker.py` | QCSARRanker — three-stage cascade (Stages 4–6) |
| `src/recsys/ranking/__init__.py` | Package init |
| `tests/test_ranking.py` | 14 unit tests covering all three stages |
| `scripts/evaluate_ranking.py` | Full pipeline evaluation with query-aware ground truth |
| `scripts/tune_weights.py` | Automated weight tuning via grid search |
| `scripts/run_pipeline.py` | Interactive pipeline demo (query → ranked list) |
| `tests/fixtures/synthetic_queries.py` | 20 English synthetic evaluation queries |
| `tests/fixtures/synthetic_queries_zh.py` | 20 Chinese synthetic evaluation queries |

---

## 8. Option 1: direct MF score

**Change.** `QCSARRanker(use_mf_score=True)` adds the plain ALS prediction
`mf[i] = user_vector · V[candidate_i]` (raw, un-normalised item factors) as a
4th long-term signal next to CF, category affinity and social. Like the others
it is rank-normalised inside `_active_mean`, so it is dropped automatically for
cold-start users (empty `user_vector`). Candidates with no ALS row get the
median of the known scores (neutral). A dimension mismatch between
`user_vector` and the item factors logs a warning and skips the signal.
Default is `False`, so rankings are unchanged until results justify switching.

**Evaluation.** Two sweeps over α ∈ {0, 0.2, 0.4, 0.6, 0.8, 1.0} ×
w_query ∈ {0.5, 0.6, 0.7} on the v3 test split, same code except the flag:

```bash
python scripts/sweep_alpha.py --alphas 0.0 0.2 0.4 0.6 0.8 1.0 --w-queries 0.5 0.6 0.7 --out results/alpha_sweep_baseline_rerun.json
python scripts/sweep_alpha.py --alphas 0.0 0.2 0.4 0.6 0.8 1.0 --w-queries 0.5 0.6 0.7 --mf-score --out results/alpha_sweep_mf.json
```

Run-to-run noise: each sweep recomputes Stages 1–3, and the SOCLaaS query
parser is not deterministic, so even the anonymous baseline moved between the
two runs (MRR@3 0.1481 → 0.1502, anonymous* nDCG@5 0.1681 → 0.1711) although
the flag cannot affect it. The deltas below are paired within each run
(personalised vs anonymous on the same candidates), so compare deltas, not
absolute values. "none" does not depend on α (no session → long-term only).

| w_query (α = 0.4) | MF | none ΔnDCG@5 [95% CI] | none MRR@3 drop | none guardrail | related ΔnDCG@5 [95% CI] | related MRR@3 drop | realistic nDCG@5 | realistic MRR@3 |
|---|---|---|---|---|---|---|---|---|
| 0.5 | off | −0.0017 [−0.0177, +0.0129] | 0.061 | ✗ | +0.0024 [−0.0116, +0.0172] | 0.069 | 0.1668 | 0.0878 |
| 0.5 | on  | −0.0062 [−0.0232, +0.0087] | 0.048 | ✗ | −0.0053 [−0.0223, +0.0111] | 0.061 | 0.1650 | 0.1018 |
| 0.6 | off | −0.0010 [−0.0145, +0.0119] | 0.050 | ✗ | +0.0042 [−0.0091, +0.0181] | 0.056 | 0.1675 | 0.0989 |
| 0.6 | on  | −0.0052 [−0.0211, +0.0091] | 0.050 | ✗ | −0.0048 [−0.0215, +0.0110] | 0.042 | 0.1658 | 0.1004 |
| 0.7 | off | −0.0009 [−0.0141, +0.0118] | 0.051 | ✗ | +0.0036 [−0.0085, +0.0165] | 0.049 | 0.1674 | 0.0983 |
| 0.7 | on  | −0.0084 [−0.0242, +0.0053] | 0.046 | ✗ | −0.0033 [−0.0190, +0.0115] | 0.037 | 0.1628 | 0.1053 |

Across all 18 settings, with and without MF:
- "none" passes the MRR@3 guardrail (drop ≤ 0.01) at **0/18** settings and its
  nDCG@5 lift is significant at **0/18**.
- The MF point estimate for "none" nDCG@5 is lower than without it at every
  setting (−0.005 to −0.008 vs −0.001 to −0.002); every CI still contains 0.
- MF shrinks the MRR@3 drop slightly at w_query 0.5 and 0.7 (by 0.005 to
  0.013), still 4–5× over the guardrail. Realistic MRR@3 rises 0.002–0.014,
  realistic nDCG@5 falls 0.002–0.005. Part of the MRR change is the run noise
  above (≈ 0.002 on the anonymous baseline).

**Verdict: don't adopt.** The direct MF score does not make personal
preference measurably better (nDCG@5 lift not significant, point estimate
lower) and does not bring "none" within the MRR@3 guardrail. Keep
`use_mf_score=False`.

---

## 9. LightFM vs ALS (long-term probe)

**Question.** ALS gives rarely-watched videos weak vectors. Does LightFM, which
builds each video vector as the sum of id + category + tag feature vectors, fix
that, and does it help long-term personalisation?

**Setup** (`scripts/probe_long_term.py --backends als lightfm --w-queries 0.5 0.7`).
v3 test split, 81 queries; cases `anonymous` (81) and `none` (405). Gold
intents (no LLM) + one bge-m3 embedding per query; hybrid retrieval once per
query, 200 candidates; ranking with `use_mf_score=True`, `top_k=100`. There are
no recent videos, so short-term is inactive and α does not matter. Metric code
is copied from `scripts/evaluate_ranking.py`. Raw output:
`results/probe_long_term_als.txt`, `results/probe_long_term_lightfm.txt`.

LightFM: 16 components, WARP, 15 epochs, seed 42; item features = identity +
first/second/third-level caption category (UNKNOWN skipped) + `topic_tag`
tags on ≥ 5 videos (831 shared features); user features = identity; biases
dropped.

| Backend | Training time |
|---|---|
| ALS (`implicit`, multi-threaded) | 6.4 s |
| LightFM (single thread, no OpenMP on Windows) | 677 s |

### Model diagnostic (item vectors `V`)

Same-cat top-10 = share of a video's 10 nearest videos (cosine on `V`) with the
same first-level caption category.

| Viewers (big_matrix) | Videos | ALS median ‖v‖ | LightFM median ‖v‖ | ALS same-cat top-10 | LightFM same-cat top-10 |
|---|---|---|---|---|---|
| < 10 | 1338 | 0.0013 | 1.239 | 0.101 | 0.923 |
| 10–49 | 1839 | 0.0035 | 1.079 | 0.129 | 0.886 |
| 50–99 | 1041 | 0.0078 | 1.014 | 0.145 | 0.872 |
| 100–499 | 2134 | 0.0194 | 1.056 | 0.164 | 0.838 |
| 500–999 | 658 | 0.0531 | 1.188 | 0.177 | 0.799 |
| 1k+ | 3718 | 0.1093 | 1.378 | 0.208 | 0.761 |

LightFM removes the norm collapse (rare and popular vectors are now the same
scale). Its same-category share is high **partly by construction**: the
category is one of the summed features, and for rare videos the id vector is
barely trained, so those videos sit near their category vector. It shows the
features are used, not that the vectors are better.

### Ranking (none vs anonymous)

Guardrail: MRR@3 drop ≤ 0.01. nDCG@5 = personal (condensed list, watch_ratio
capped at 5), none vs anonymous*, paired bootstrap 95% CI, n = 185.
Rare-answer rank = mean rank of answer videos with < 50 viewers (absent from
the top 100 = 101), n = 175 case × video pairs; anonymous = 45.6 for both.

| w_query | Backend | MRR@3 none (anon 0.2695) | MRR@3 drop | ΔnDCG@5 [95% CI] | Rare-answer rank none (Δ vs anon) |
|---|---|---|---|---|---|
| 0.5 | ALS | 0.1391 | +0.1305 ✗ | −0.0095 [−0.0232, +0.0033] ns | 67.6 (+22.0) |
| 0.5 | LightFM | 0.1333 | +0.1362 ✗ | −0.0024 [−0.0142, +0.0098] ns | 60.6 (+15.0) |
| 0.7 | ALS | 0.1613 | +0.1082 ✗ | −0.0059 [−0.0162, +0.0046] ns | 54.4 (+8.8) |
| 0.7 | LightFM | 0.1737 | +0.0959 ✗ | +0.0011 [−0.0073, +0.0096] ns | 51.2 (+5.7) |

LightFM is ahead on every point estimate at w_query = 0.7 and buries rare
answers less at both settings (about a third less), but no nDCG@5 difference
is significant and both backends fail the MRR@3 guardrail by ~10× the allowed
drop. That matches the earlier finding: the long-term signal in the "none"
condition never passes the guardrail.

**Verdict: inconclusive.** LightFM fixes the rare-video vectors and buries rare
answers less, but it gives no significant nDCG@5 lift, does not rescue the
MRR@3 guardrail, and trains ~100× slower here. Keep `backend="als"` as the
default.

**Install note (Windows).** `pip install lightfm` (1.17) fails on Python 3.12:
the C code shipped in the package uses CPython internals removed in 3.12
(`curexc_traceback`, `ob_digit`). On Python 3.11 it builds (via
`setup.py bdist_wheel`, since its `setup.py` breaks under pip's isolated
build), but WARP/BPR then crash with an access violation: `rand_r` casts to
`long`, which is 32-bit on Windows, so negative samples get negative indices.
The numbers above use 1.17 with that one cast changed to `long long` (matches
Linux behaviour).

## 10. Evaluation fix: on-topic labels

**Date:** 2026-11-09

Metric 2 (personal nDCG@5) grades a video by the user's `watch_ratio` only if
it is on-topic for the query. The scripts decided "on-topic" with the `feat`
tags of `item_categories.csv`, but the query topics are caption first-level
category ids. The two number categories differently:

* `feat` only has ids 0–30, so topics 31+ (Fitness, Short Drama, Military,
  Real Estate/Home, Oddities, Film & TV) never matched.
* Some ids barely agree (small_matrix videos whose caption topic is matched by
  the same `feat` id: Politics/News 0 of 15, Science & Law 3 of 38,
  Tech/Digital 5 of 21, Fashion 51 of 124).
* `feat` is multi-label, so off-topic videos counted as on-topic (Daily Life:
  426 counted vs 122 by caption).

`evaluate_ranking.py`, `sweep_alpha.py`, `tune_weights.py` and
`probe_long_term.py` now use `video_topic_sets` (caption first-level ids).
Effect, measured with gold intents and a 200-candidate pool (anonymous MRR@3 is
unchanged by the fix):

| | dev, `feat` | dev, caption | test, `feat` | test, caption |
|---|---|---|---|---|
| Personal cases that can never score | 30/225 | 0/225 | 15/555 | 0/555 |
| nDCG@5 personalised, w_query = 0.5 | 0.244 | 0.328 | 0.233 | 0.329 |
| nDCG@5 lift, w_query = 0.5 [95% CI] | +0.001 [−0.010, +0.012] | −0.001 [−0.013, +0.011] | −0.004 [−0.011, +0.003] | −0.008 [−0.016, +0.001] |
| nDCG@5 lift, w_query = 0.7 [95% CI] | −0.005 [−0.013, +0.003] | −0.005 [−0.014, +0.004] | +0.001 [−0.004, +0.005] | +0.004 [−0.000, +0.009] |

The level of nDCG@5 rises because more videos are graded, but no verdict
changes: the pooled personal lift is still not significant, and the only
significant per-condition lifts are negative (test "none" at w_query = 0.5:
−0.016 [−0.032, −0.002]; dev "related" at 0.7: −0.018 [−0.036, −0.001]). Results in sections 2–9
were computed with the old `feat` labels and Stage 3 topic weight 0.5.
