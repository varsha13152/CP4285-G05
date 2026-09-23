"""Unit tests for SessionModeller (the short-term session model + its wiring).

Trains on the tiny synthetic ``sample_interactions`` fixture (conftest) so there
is no disk I/O. With the default ``fixed`` 30-min strategy every user's few
close-together interactions fall into a single session.
"""

import pytest

from recsys.session_modelling import SessionModeller


@pytest.fixture
def session_modeller(sample_interactions):
    return SessionModeller(interactions=sample_interactions)  # fixed, 30-min default


def test_predict_next_delegates_to_markov(session_modeller):
    # 20 -> 30 is the only transition out of item 20 (users 1 and 3) -> prob 1.0
    assert session_modeller.predict_next([20]) == [(30, pytest.approx(1.0))]


def test_predict_next_topk_and_exclude(session_modeller):
    assert session_modeller.predict_next([20], k=1) == [(30, pytest.approx(1.0))]
    # excluding the only learned next item leaves an empty ranking
    assert session_modeller.predict_next([20], exclude={30}) == []


def test_avg_session_length_for_known_user(session_modeller):
    # user 1's 4 interactions (timestamps 1-4 s) sit in one 30-min session;
    # user 2's 3 interactions in one session.
    assert session_modeller.avg_session_length_for(1) == pytest.approx(4.0)
    assert session_modeller.avg_session_length_for(2) == pytest.approx(3.0)


def test_avg_session_length_for_unknown_user_is_zero(session_modeller):
    assert session_modeller.avg_session_length_for(99999) == 0.0


def test_model_property_exposes_fitted_markov(session_modeller):
    assert session_modeller.model.n_items > 0
