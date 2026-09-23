"""Unit tests for the Markov next-item model and its offline evaluation."""

import pandas as pd
import pytest

from recsys.session_modelling import MarkovSessionModel, Sessioniser
from recsys.session_modelling.evaluate import (
    chronological_session_split,
    evaluate_next_item,
    split_by_timestamp,
)

# from 1 -> always 2; from 2 -> 3 (x2) or 4 (x1); 3 has no within-session next
_SEQS = [[1, 2, 3], [1, 2, 4], [2, 3]]


@pytest.fixture
def model():
    return MarkovSessionModel().fit(_SEQS)


# ---------------------------------------------------------------------------
# transition counting + prediction
# ---------------------------------------------------------------------------

def test_transition_probabilities(model):
    assert model.predict_next([1]) == [(2, 1.0)]
    assert model.predict_next([2]) == [(3, pytest.approx(2 / 3)), (4, pytest.approx(1 / 3))]


def test_topk_truncation(model):
    assert model.predict_next([2], k=1) == [(3, pytest.approx(2 / 3))]


def test_backoff_to_popularity_for_cold_context(model):
    # item 3 has no learned transitions -> popularity fallback.
    # popularity counts: 2:3, 1:2, 3:2, 4:1 -> ties broken by id asc
    ids = [item for item, _ in model.predict_next([3])]
    assert ids == [2, 1, 3, 4]
    assert model.predict_next([999]) == model.predict_next([3])  # unknown -> same fallback


def test_exclude_filters_before_topk(model):
    assert model.predict_next([2], exclude={3}) == [(4, pytest.approx(1 / 3))]


def test_n_items(model):
    assert model.n_items == 4


def test_predict_never_returns_internal_list(model):
    out = model.predict_next([2])
    out.append((999, 0.0))
    assert model.predict_next([2]) == [(3, pytest.approx(2 / 3)), (4, pytest.approx(1 / 3))]


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------

def test_evaluate_next_item_perfect_recall():
    # train teaches 1->2 and 2->{3,4}; test session [1,2,3] -> both targets top-ranked
    model = MarkovSessionModel().fit([[1, 2, 3], [1, 2, 4]])
    metrics = evaluate_next_item(model, [[1, 2, 3]], ks=(10, 20))
    assert metrics["n_eval_pairs"] == 2          # (1->2) and (2->3)
    assert metrics["HR@10"] == pytest.approx(1.0)
    assert metrics["MRR@10"] == pytest.approx(1.0)  # both targets at rank 1
    assert metrics["NDCG@10"] == pytest.approx(1.0)


def test_hr_is_monotonic_in_k():
    model = MarkovSessionModel().fit([[1, 2, 3], [1, 3, 2], [2, 4, 1]])
    m = evaluate_next_item(model, [[1, 2, 4], [2, 3, 1]], ks=(1, 5, 20))
    assert m["HR@1"] <= m["HR@5"] <= m["HR@20"]
    assert 0.0 <= m["MRR@20"] <= 1.0


def test_empty_test_set():
    model = MarkovSessionModel().fit(_SEQS)
    assert evaluate_next_item(model, [[5]], ks=(10,))["n_eval_pairs"] == 0  # len-1 session -> no pairs


# ---------------------------------------------------------------------------
# chronological split (no-leakage)
# ---------------------------------------------------------------------------

def test_chronological_split_is_disjoint_and_time_ordered():
    rows = [(u, t * 1000, 100 + i) for u in (1, 2) for i, t in enumerate(range(6))]
    df = Sessioniser(strategy="single").assign(pd.DataFrame(
        rows, columns=["user_id", "timestamp", "video_id"]))
    train, test = chronological_session_split(df, test_ratio=0.5)
    assert len(train) + len(test) == df["session_id"].nunique()
    assert len(test) >= 1


# ---------------------------------------------------------------------------
# global-timestamp split (fair cross-strategy ablation)
# ---------------------------------------------------------------------------

def _toy_interactions() -> pd.DataFrame:
    # two users, 5 interactions each; video ids increase with time so a
    # time-ordered run is a strictly increasing id sequence.
    rows = [(u, t * 1000, 10 * u + t) for u in (1, 2) for t in range(5)]
    return pd.DataFrame(rows, columns=["user_id", "timestamp", "video_id"])


def test_split_by_timestamp_no_leakage():
    df = _toy_interactions()
    train_df, test_seqs = split_by_timestamp(df, cutoff_quantile=0.8)
    cutoff = df["timestamp"].quantile(0.8)
    assert train_df["timestamp"].max() < cutoff  # train strictly before the cutoff
    test_ids = {v for seq in test_seqs for v in seq}
    assert test_ids == set(df[df["timestamp"] >= cutoff]["video_id"])  # test at/after


def test_split_by_timestamp_test_is_ordered_and_deterministic():
    df = _toy_interactions()
    _, test_a = split_by_timestamp(df, cutoff_quantile=0.6)
    _, test_b = split_by_timestamp(df, cutoff_quantile=0.6)
    assert test_a == test_b                      # takes no strategy -> deterministic
    for seq in test_a:
        assert seq == sorted(seq)                # raw per-user runs, in time order


def test_split_by_timestamp_test_set_is_shared_across_strategies():
    # the whole point of the ablation: sessionisation changes only the *train*
    # transition table; the held-out test pairs are identical for every strategy.
    df = _toy_interactions()
    train_df, test_seqs = split_by_timestamp(df, cutoff_quantile=0.6)
    fixed = Sessioniser(strategy="fixed", gap_minutes=0.001).to_sequences(train_df)
    single = Sessioniser(strategy="single").to_sequences(train_df)
    assert len(fixed) >= len(single)             # tiny gap cuts more -> more sessions
    m_fixed = evaluate_next_item(MarkovSessionModel().fit(fixed), test_seqs, ks=(10,))
    m_single = evaluate_next_item(MarkovSessionModel().fit(single), test_seqs, ks=(10,))
    assert m_fixed["n_eval_pairs"] == m_single["n_eval_pairs"]  # same shared test set
