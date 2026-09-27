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
