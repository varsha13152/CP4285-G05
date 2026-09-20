"""Tests for evaluation metrics."""

from recsys.evaluation.metrics import coverage, mrr, ndcg_at_k, precision_at_k, recall_at_k


def test_precision_at_k(sample_ranked_list):
    ground_truth = {10, 30, 99}  # 2 of 3 ranked items are relevant
    p = precision_at_k(sample_ranked_list, ground_truth, k=3)
    assert abs(p - 2 / 3) < 0.01


def test_recall_at_k(sample_ranked_list):
    ground_truth = {10, 30, 99}
    r = recall_at_k(sample_ranked_list, ground_truth, k=3)
    assert abs(r - 2 / 3) < 0.01


def test_mrr_first_hit(sample_ranked_list):
    ground_truth = {20}  # video 20 is at rank 2
    assert abs(mrr(sample_ranked_list, ground_truth) - 0.5) < 0.01


def test_mrr_no_hit(sample_ranked_list):
    ground_truth = {999}
    assert mrr(sample_ranked_list, ground_truth) == 0.0


def test_ndcg_perfect_ranking(sample_ranked_list):
    # If ground truth matches the ranking order, NDCG should be high
    ground_truth = {10: 1.0, 20: 0.8, 30: 0.6}
    score = ndcg_at_k(sample_ranked_list, ground_truth, k=3)
    assert score > 0.9


def test_coverage(sample_ranked_list):
    all_ranked = [sample_ranked_list]
    c = coverage(all_ranked, total_items=100)
    assert abs(c - 3 / 100) < 0.01
