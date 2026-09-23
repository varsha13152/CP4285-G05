"""Unit tests for the Stage 2 long-term user model (``UserModeller``).

Trains on the tiny synthetic ``sample_interactions`` fixture (see conftest) so
the ALS fit runs in milliseconds with no disk I/O.
"""

import math

import pytest

from recsys.contracts.models import EngagementProfile
from recsys.user_modelling import UserModeller

FACTORS = 4


@pytest.fixture
def modeller(sample_interactions, sample_item_categories):
    """A UserModeller trained on the synthetic interactions fixture."""
    return UserModeller(
        interactions=sample_interactions,
        item_categories=sample_item_categories,
        factors=FACTORS,
        iterations=5,
    )


# ---------------------------------------------------------------------------
# user_vector (the u_long latent factors)
# ---------------------------------------------------------------------------

def test_user_vector_has_factors_length(modeller):
    profile = modeller.build_profile(user_id=1)
    assert isinstance(profile, EngagementProfile)
    assert len(profile.user_vector) == FACTORS
    assert all(isinstance(x, float) for x in profile.user_vector)


def test_every_known_user_gets_a_vector(modeller, sample_interactions):
    for user_id in sample_interactions["user_id"].unique():
        profile = modeller.build_profile(int(user_id))
        assert len(profile.user_vector) == FACTORS


def test_factors_knob_controls_vector_length(sample_interactions):
    modeller = UserModeller(interactions=sample_interactions, factors=8, iterations=3)
    assert len(modeller.build_profile(1).user_vector) == 8


# ---------------------------------------------------------------------------
# engagement statistics
# ---------------------------------------------------------------------------

def test_stats_populated_for_known_user(modeller):
    # User 1 watched 4 videos: watch_ratios [0.9, 0.1, 0.5, 0.8] over 4 dates.
    profile = modeller.build_profile(user_id=1)
    assert profile.total_interactions == 4
    assert profile.active_days == 4
    assert profile.avg_watch_ratio == pytest.approx(0.575)
    assert profile.watch_ratio_std > 0.0
    assert profile.avg_session_length == pytest.approx(1.0)  # 4 interactions / 4 days
    assert profile.recency_days >= 0


def test_category_distribution_normalised(modeller):
    profile = modeller.build_profile(user_id=1)
    # User 1's videos map to categories {1,5},{2,5},{1,3},{2,7}.
    assert profile.top_category_ids  # non-empty
    assert set(profile.top_category_ids) == {1, 2, 3, 5, 7}
    assert profile.top_category_ids[0] in (1, 2, 5)  # the three tied top cats
    assert math.isclose(sum(profile.category_distribution.values()), 1.0, abs_tol=1e-9)


def test_category_stats_stubbed_without_item_categories(sample_interactions):
    # No item_categories supplied -> category stats are an empty (valid) stub,
    # but the vector and scalar stats are still populated.
    modeller = UserModeller(interactions=sample_interactions, factors=FACTORS, iterations=3)
    profile = modeller.build_profile(user_id=1)
    assert profile.top_category_ids == []
    assert profile.category_distribution == {}
    assert len(profile.user_vector) == FACTORS
    assert profile.total_interactions == 4


# ---------------------------------------------------------------------------
# cold start / unknown users
# ---------------------------------------------------------------------------

def test_unknown_user_returns_empty_vector(modeller):
    profile = modeller.build_profile(user_id=99999)
    assert profile.user_id == 99999
    assert profile.user_vector == []
    assert profile.total_interactions == 0
    assert profile.active_days == 0
    assert profile.avg_watch_ratio == 0.0
    assert profile.top_category_ids == []
    assert profile.category_distribution == {}


# ---------------------------------------------------------------------------
# session profile (deferred: delegates to build_profile)
# ---------------------------------------------------------------------------

def test_build_session_profile_delegates(modeller, sample_query_intent):
    session = modeller.build_session_profile(user_id=1, query_intent=sample_query_intent)
    base = modeller.build_profile(user_id=1)
    assert session.user_id == base.user_id
    assert session.user_vector == base.user_vector
    assert session.total_interactions == base.total_interactions


def test_build_session_profile_uses_real_session_length(
    sample_interactions, sample_item_categories, sample_query_intent
):
    # With a SessionModeller injected, avg_session_length switches from the
    # daily-intensity proxy to the real session length; the long-term signal
    # (user_vector) is preserved.
    from recsys.session_modelling import SessionModeller

    sm = SessionModeller(interactions=sample_interactions)  # user 1 -> one 4-video session
    modeller = UserModeller(
        interactions=sample_interactions,
        item_categories=sample_item_categories,
        factors=FACTORS,
        iterations=5,
        session_modeller=sm,
    )
    base = modeller.build_profile(user_id=1)
    session = modeller.build_session_profile(user_id=1, query_intent=sample_query_intent)
    assert base.avg_session_length == pytest.approx(1.0)     # proxy: 4 interactions / 4 days
    assert session.avg_session_length == pytest.approx(4.0)  # real: one 4-video session
    assert session.user_vector == base.user_vector          # long-term signal preserved


# ---------------------------------------------------------------------------
# confidence transform knob
# ---------------------------------------------------------------------------

def test_confidence_alpha_knob_trains(sample_interactions, sample_item_categories):
    modeller = UserModeller(
        interactions=sample_interactions,
        item_categories=sample_item_categories,
        factors=FACTORS,
        iterations=3,
        confidence_alpha=40.0,  # confidence = 1 + 40 * watch_ratio
    )
    profile = modeller.build_profile(user_id=1)
    assert len(profile.user_vector) == FACTORS
    assert profile.avg_watch_ratio == pytest.approx(0.575)  # stats independent of knob


# ---------------------------------------------------------------------------
# candidate-generation exports (item factors + id map) and the seen-set
# ---------------------------------------------------------------------------

def test_item_factors_shape(modeller, sample_interactions):
    n_videos = sample_interactions["video_id"].nunique()
    assert modeller.item_factors.shape == (n_videos, FACTORS)


def test_item_ids_align_with_item_factors(modeller):
    ids = modeller.item_ids
    assert len(ids) == modeller.item_factors.shape[0]
    for col, vid in enumerate(ids):                       # round-trip
        assert modeller.column_for_video_id(int(vid)) == col


def test_column_for_unknown_video_is_none(modeller):
    assert modeller.column_for_video_id(999999) is None


def test_seen_video_ids_on_profile(modeller):
    # user 1 watched videos 10, 20, 30, 40 in the fixture
    assert modeller.build_profile(1).seen_video_ids == {10, 20, 30, 40}


def test_seen_video_ids_empty_for_unknown_user(modeller):
    assert modeller.build_profile(99999).seen_video_ids == set()
