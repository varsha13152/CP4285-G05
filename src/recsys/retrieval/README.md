# Candidate Retrieval (Stage 3)

Owned by **Dylan**. Design decisions, rationale and benchmark live in the
[top-level README](../../../README.md); this page is the module reference.

`HybridRetriever` implements the `Retriever` protocol: `QueryRetrievalRequest` →
`CandidateSet` of `num_candidates` (default 100) **query-relevant** videos. The
catalogue holds only **searchable** videos: 9,905 of 10,728 — the 794
private / friends-only videos (latest `visible_status`) and 29 ads are never
returned.
It does not personalise: CF, session, social and popularity are Stage 4 ranking
signals, so a search never pulls in off-topic videos.

## Scoring (numpy, whole catalogue, ~2–3 ms)

| Term | Meaning |
|------|---------|
| `sem` | cosine(query bge-m3 embedding, video text embedding); video text = caption + cover text + topic tags + category names (all three levels) + `video_tag_name`; every searchable video has some |
| `cat` | confidence of the query topic matching the video's caption first-level category |
| `dur` | 1 inside `duration_range_ms`, else `exp(-seconds_outside / 15)` — soft, never a filter |

| Mode | Eligible videos | Score |
|------|-----------------|-------|
| `semantic` | has text | `w_sem·sem + w_dur·dur` |
| `category` | category matches a query topic | `w_cat·cat + w_dur·dur` |
| `hybrid` *(default)* | either | `w_sem·sem + w_cat·cat + w_dur·dur` |

No query embedding → falls back to `category`. Ties are broken by `video_id`.
Defaults: `w_sem=1.0, w_cat=0.5, w_dur=0.2`.

## Output: `Candidate`

`source` is `SEMANTIC` / `CATEGORY` (category match, no text) / `BOTH`.
`features` carries everything Stage 4 needs from the item side:

| Key | Meaning |
|-----|---------|
| `sem`, `cat`, `dur_fit` | the relevance terms above — Stage 4 must keep query relevance as a ranking feature |
| `has_text` | whether the video had text to embed |
| `seen` | 1 if in `EngagementProfile.seen_video_ids` (flagged, **not** filtered) |
| `popularity` | total `play_cnt` |
| `trending` | `log((last-7-day plays + 1) / (previous-7-day plays + 1))` |

## Setup and use

```bash
python scripts/build_caption_embeddings.py   # once: ~5 min over SOCLaaS -> data/processed/caption_embeddings.npz
python scripts/run_pipeline.py --query "cute cat videos" [--user_id 14 --recent 1,2,3]
python scripts/evaluate_stage3.py            # benchmark: semantic vs category vs hybrid
```

The benchmark's metrics are category-based proxies (there are no real query
labels). Their limits, why hybrid scores as it does, and two planned fairer
methods (known-item search, pooled relevance judgments) are documented in the
top-level README under *How Stage 3 is evaluated — and its limits*.

```python
from recsys.retrieval import HybridRetriever

retriever = HybridRetriever.from_disk(mode="hybrid")   # ~1 s
candidates = retriever.retrieve(request)
```

## Tests

`tests/test_retrieval.py` checks a hand-computable 5-video catalogue: mode
orderings, tie-breaks, soft duration, textless videos, the no-embedding
fallback, `seen` flagging and cold start.
