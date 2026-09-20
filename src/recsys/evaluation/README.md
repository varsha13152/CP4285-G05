# Evaluation

Shared evaluation metrics for measuring recommendation quality.

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
