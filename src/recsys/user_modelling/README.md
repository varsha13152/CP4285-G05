# User & Session Modelling (Stage 2)

Models **who the user is** — their long-term taste — and precomputes the
user-side lookups that **Stage 4 ranking** needs. In the query-first cascade
(see the top-level README) Stage 3 retrieves on the query alone; everything
personal in this profile is a *ranking* signal, never a retrieval filter.

```
User query ──→ [Stage 1: Intent Understanding] ──→ QueryIntent ─────────────┐
User ID + recent videos ──→ [Stage 2 (this)] ──→ EngagementProfile ─────────┤
                                                                            ▼
                  [Router] ──→ QueryRetrievalRequest ──→ [Stage 3: retrieval, query-only]
                                                                            │ CandidateSet
                                                                            ▼
                                                 [Stage 4: ranking — reads this profile]
```

## Status update: alternatives tried (2026-11-08)

### Long-term (user history)

| Alternative | What it changes | Result | Decision |
|---|---|---|---|
| **ALS** (default) | User and video vectors from implicit ALS, `watch_ratio` as confidence | Fast (~6 s). Rare videos (< 50 viewers, 30% of the catalogue) get weak vectors: ~100× shorter, and only 10–13% of their nearest videos share their category. | Keep as default |
| **LightFM** (`backend="lightfm"`) | Each video vector = its id + category + tag vectors, so rare videos borrow from shared features | Fixes rare-video vectors (normal length; nearest videos mostly share the category). In ranking, no significant difference from ALS, but the personal-preference labels contain almost no rare videos (0.3%), so the current evaluation can't show the benefit. Much slower (~11 min). | Available, not default |
| **MF score off** in ranking | Ranker uses CF, category affinity and social only | Personalisation lowers MRR@3 by 0.05–0.06 (allowed: 0.01); no significant nDCG@5 gain. | Baseline |
| **MF score on** in ranking (`use_mf_score=True`) | Adds `user_vector · video_vector` as a 4th long-term signal | Running; results to follow. | Pending |

### Short-term (current session)

Alternatives for splitting viewing history into sessions
([`session_modelling`](../session_modelling/README.md)):

| Alternative | Next-video accuracy (HR@10) | Decision |
|---|---|---|
| 30-minute gap | 0.1320 | Keep (simplest) |
| Per-user gap (each user's 90th-percentile gap) | 0.1321 | – |
| One session per user | 0.1320 | – |

No difference. The session cut only affects one of the three short-term
ranking signals (Markov next-video), so it can't move the final ranking much.

### Known evaluation issues

- The personal nDCG@5 uses video category labels (`item_categories.csv`
  `feat`) that disagree with the query categories for ~7 topics, and 4 topics
  can never score. Fix planned.
- The LLM query parser gives different results between runs, so gains below
  ~0.015 can't be separated from run-to-run noise.

## What this implements: `u_long`

The long-term user model is **matrix factorization** via implicit ALS
(the [`implicit`](https://github.com/benfred/implicit) library) trained on
KuaiRec's sparse `big_matrix`, using `watch_ratio` as the per-interaction
**confidence**. Each user's learned latent factor vector is stored on
`EngagementProfile.user_vector` and consumed downstream **purely as a ranking
signal**. It is never used as a hard filter.

**Verified (2026-09-25):** trained on `big_matrix`, ALS reaches Recall@100 =
0.090 on `small_matrix` (relevant = `watch_ratio ≥ 2`) vs 0.046 for popularity.
The default `confidence_alpha=None` (raw `watch_ratio`) beats `alpha=10` (0.041).

### Why ALS over BPR
KuaiRec's informative signal is the graded `watch_ratio`. ALS exploits it
directly as confidence (`C_ui`); BPR would flatten it to binary preference and
throw that signal away.

### Alternative backend: LightFM (`backend="lightfm"`)
ALS gives rarely-watched videos weak vectors: videos with < 50 viewers (29.6%
of the catalogue) get vectors ~100× shorter than popular ones, and their
nearest neighbours rarely share a category. LightFM builds each video vector
as the **sum of feature vectors** (its id + its first/second/third-level
caption category + its `topic_tag` tags used on ≥ 5 videos), so a rare video
borrows from shared, well-trained features.

* Same users × videos matrix and id maps as ALS; every view is a positive,
  `sample_weight` = summed `watch_ratio` per pair clipped to `[0, 5]`.
* Users: identity features only. `no_components=factors`, WARP loss,
  `iterations` epochs, `random_state` seed. Biases are dropped, so `U · V`
  stays comparable to ALS and popularity stays its own Stage 4 feature.
* Item feature rows are normalised to sum to 1 (as `lightfm.data.Dataset` does).
* Needs the optional extra: `pip install -e ".[lightfm]"`. On Windows the
  1.17 package does not build on Python 3.12, and its negative sampler
  crashes on Windows (a 32-bit `long` cast in `rand_r`), so a patched build is
  needed there — see `results/RANKING_RESULTS.md` ("LightFM vs ALS").
* Compared with ALS by `scripts/probe_long_term.py`.

### Sparsity
Trained **observed-only** over a `scipy` CSR matrix — no `fillna(0)`
densification. The fully-observed `small_matrix` is intentionally **not** touched
here; it is reserved for evaluation (it shares users and videos with
`big_matrix` but has **zero** overlapping `(user, video)` pairs).

## Usage

```python
from recsys.session_modelling import SessionModeller
from recsys.user_modelling import UserModeller

modeller = UserModeller(session_modeller=SessionModeller())    # trains on data/raw
profile = modeller.build_session_profile(14, intent, recent_video_ids=[...])
```

## Output contract: `EngagementProfile`

Defined in `recsys.contracts.models`. Stage 2 **produces** it; the router and
Stage 3 carry it; **Stage 4 consumes it**. Empty values mean *unavailable*
(cold start / no friends / no session), not *disliked*.

| Field | Meaning | Derived from | Intended use in Stage 4 |
|-------|---------|--------------|-------------------------|
| `user_vector` | ALS latent-factor row — the user's position in taste-space. **Empty** for cold start. | `U[row]` | `U[u]·V[i]` taste-alignment feature, now used by the ranker as a 4th long-term signal behind `QCSARRanker(use_mf_score=True)` (off by default). A ranking score, not a probability. |
| `cf_neighbour_scores` | `{video_id: Σ cosine(u, v) · watch_ratio(v, video)}` over the `k_neighbours` (50) nearest users. **Empty** for cold start. | brute-force cosine over ALS user vectors (7k × 16 ≈ 0.1 ms) + neighbours' `big_matrix` rows | the CF ranking signal (kNN users). Benchmark with vs without. |
| `social_scores` | `{video_id: mean watch_ratio among the user's friends}`. **Empty** for the ~93% of users with no friends. | `social_network.csv` friend lists + friends' `big_matrix` rows (works for cold-start users who have friends) | friends-first social signal; fall back to the global **trending** item feature when empty. |
| `session_context` | `SessionContext` — see below. `None` unless recent videos were given. | `build_session_profile(..., recent_video_ids)` | short-term momentum features. |
| `seen_video_ids` | Distinct videos the user already watched. | the user's training rows | **Flag, not filter**: search users often re-find videos; Stage 3 marks `seen`, Stage 4 decides. |
| `top_category_ids` / `category_distribution` | Habitual categories, as **caption first-level ids** (same space as Stage 1's `CategoryIndex`). | caption category of each watched video | affinity between the user's habits and a candidate's category. |
| `avg_watch_ratio`, `watch_ratio_std` | engagement level / consistency | per-user mean / std | calibration features |
| `total_interactions`, `active_days` | history volume — trust in personalisation | row count / `nunique(date)` | weight personal signals vs query relevance |
| `avg_session_length` | real mean session length when a `SessionModeller` is injected; else the interactions-per-active-day proxy | `SessionModeller.avg_session_length_for` | ranking feature |
| `recency_days` | freshness vs the **dataset's** last timestamp | `(dataset_last_ts − user_last_ts)/86400` | decay for dormant users |

### `SessionContext` (from `build_session_profile`)

| Field | Meaning |
|-------|---------|
| `recent_video_ids` | the videos passed in, oldest → newest |
| `next_item_scores` | `P(next = v \| last watched)` from the Markov table — **no popularity backoff** (popularity is its own feature). Sparse, and mostly a rewatch signal (see session_modelling). |
| `session_vector` | mean ALS item vector of the recent videos — compare with a candidate's `item_factors` row |
| `recent_category_ids` | caption categories of the recent videos, most recent first |

The query never changes the profile: a search is a temporary lens, and blending
query relevance with these signals is Stage 4's job.

## Item-side exports

One ALS fit learns both `U` (users) and `V` (items). The item side is exposed on
the `UserModeller` instance (shared across users, so not on the profile):

| Export | Returns | Meaning |
|--------|---------|---------|
| `item_factors` | `np.ndarray` `(n_videos, factors)` | `V` (read-only). Row `col` is `item_ids[col]`. |
| `item_ids` | `np.ndarray` `(n_videos,)` | `video_id` per row |
| `column_for_video_id(video_id)` | `int` or `None` | row of `video_id` in `item_factors` |

Stage 4 uses these to score `user_vector · V[i]` and `session_vector · V[i]`.
Factors are comparable **only within one training run**.

## Knobs (`UserModeller.__init__`)

| Param | Default | Notes |
|-------|---------|-------|
| `backend` | `"als"` | `"als"` (implicit ALS) or `"lightfm"` (content-aware MF, see above). Both fill the same `U` / `V`, so nothing downstream changes. Fit time is on `train_seconds`. |
| `factors` | 16 | latent dimensionality `k` (try 16–32) |
| `regularization` | 0.05 | ALS L2 (unused by LightFM) |
| `iterations` | 15 | ALS sweeps / LightFM epochs |
| `confidence_alpha` | `None` | `None` = raw `watch_ratio` (better on KuaiRec); float = `1 + α·watch_ratio` |
| `k_neighbours` | 50 | users behind `cf_neighbour_scores` |
| `session_modeller` | `None` | enables real session length + `next_item_scores` |
| `interactions`, `item_categories`, `social_network` | `None` | in-memory tables (disk-free testing seam); from disk, categories come from the caption file |
| `caption_categories` | `None` | raw caption table for LightFM item features; from disk, the loader's table. `None` in memory → identity-only features |

## Deferred

A BPR backend, a NeuMF upgrade, and `watch_ratio` debiasing.

## Tests

`tests/test_user_modelling.py` trains on the synthetic fixtures in `conftest.py`
(no disk I/O). It covers the vector, stats, cold start, kNN-CF against a
hand-computed single neighbour, friends' mean scores and session context.
`TestLightFMBackend` (skipped when `lightfm` is not installed) checks shapes
against the ALS id maps, profile lookups, and that a rare video sharing a
category and tag with a popular one ends up closer to it than to an unrelated video.
