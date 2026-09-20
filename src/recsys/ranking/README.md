# Ranking (Stages 4-6)

## Input

You receive a `CandidateSet` from retrieval. Each `Candidate` has:

| Field | Type | Description |
|-------|------|-------------|
| `video_id` | `int` | KuaiRec video identifier |
| `score` | `float` | Retrieval relevance score |
| `source_strategy` | `str` | Which retrieval strategy produced this candidate |
| `category_ids` | `list[int]` | Video's KuaiRec category IDs |
| `features` | `dict` | Additional features from retrieval (variable) |

## Additional context

The `CandidateSet.request` field contains the full `QueryRetrievalRequest`,
giving access to the original query intent for re-ranking signals:

```python
query_intent = candidates.request.query_intent

# Available re-ranking signals:
query_intent.style           # ContentStyle — boost items matching desired tone
query_intent.negative_constraints  # list[str] — penalize/remove matching items
query_intent.query_embedding       # list[float] | None — for semantic re-ranking
query_intent.semantic_summary      # str — LLM summary of user intent
query_intent.intent_type           # QueryIntentType — may influence ranking strategy
```

## Your contract

Implement the `Ranker` protocol from `recsys.pipeline.pipeline`:

```python
from recsys.pipeline.pipeline import Ranker
from recsys.contracts import CandidateSet, RankedList, RankedItem

class MyRanker:
    def rank(self, candidates: CandidateSet) -> RankedList:
        ranked_items = []
        for c in candidates.candidates:
            # Score each candidate
            ranked_items.append(RankedItem(
                video_id=c.video_id,
                score=final_score,
                rank=position,
                explanation="...",
            ))
        return RankedList(
            items=sorted(ranked_items, key=lambda x: x.score, reverse=True),
            intent_type=candidates.request.query_intent.intent_type,
        )
```

## Ranking stages

| Stage | Purpose | Input |
|-------|---------|-------|
| **Stage 4**: Initial ranking | Score candidates by relevance to query | CandidateSet |
| **Stage 5**: Re-ranking | Apply style/tone matching, negative constraint filtering | Stage 4 output |
| **Stage 6**: Final constraints | Session-aware diversification, dedup, business rules | Stage 5 output |

## Key design principle

This is a **session-based** recommender. The ranking should **not** alter the user's
long-term preference profile. All query-conditioned boosting/penalizing is temporary
and scoped to the current search session.

## Evaluation

Use `data/raw/small_matrix.csv` (99.6% dense) for offline evaluation — it provides
real feedback labels for nearly every user-video pair.
