"""Unit tests for the Stage 2 long-term user model (``UserModeller``).

Trains on the tiny synthetic ``sample_interactions`` fixture (see conftest) so
the ALS fit runs in milliseconds with no disk I/O.
"""

import math

import numpy as np
import pandas as pd
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


# ---------------------------------------------------------------------------
# Stage 4 lookups: kNN-CF, social, session context
# ---------------------------------------------------------------------------

def test_cf_neighbour_scores_match_single_neighbour(sample_interactions):
    import numpy as np

    m = UserModeller(interactions=sample_interactions, factors=FACTORS, iterations=5,
                     k_neighbours=1)
    u = m._unit_users
    sims = u @ u[0]  # user 1 is row 0
    sims[0] = -np.inf
    best = int(np.argmax(sims))
    got = m.build_profile(1).cf_neighbour_scores
    if sims[best] <= 0:
        assert got == {}
        return
    rows = sample_interactions[sample_interactions.user_id == int(m._row_to_user[best])]
    expected = {int(v): sims[best] * w for v, w in zip(rows.video_id, rows.watch_ratio)}
    assert got.keys() == expected.keys()
    for vid, score in expected.items():
        assert got[vid] == pytest.approx(score)


def test_cf_neighbour_scores_empty_for_cold_start(modeller):
    assert modeller.build_profile(99999).cf_neighbour_scores == {}


def test_social_scores_are_friends_mean_watch_ratio(sample_interactions, sample_social_network):
    m = UserModeller(interactions=sample_interactions, factors=FACTORS, iterations=3,
                     social_network=sample_social_network)
    # user 1's friends are users 2 and 3, who watched disjoint videos
    assert m.build_profile(1).social_scores == pytest.approx(
        {10: 0.8, 40: 0.3, 50: 0.95, 20: 0.2, 30: 0.4, 60: 0.1}
    )
    assert m.build_profile(99999).social_scores == {}  # no friends -> unavailable


def test_session_context_from_recent_videos(
    sample_interactions, sample_item_categories, sample_query_intent
):
    from recsys.session_modelling import SessionModeller

    m = UserModeller(interactions=sample_interactions, item_categories=sample_item_categories,
                     factors=FACTORS, iterations=3,
                     session_modeller=SessionModeller(interactions=sample_interactions))
    ctx = m.build_session_profile(1, sample_query_intent, recent_video_ids=[10, 20]).session_context
    assert ctx.recent_video_ids == [10, 20]
    assert ctx.next_item_scores == {30: 1.0}          # 20 -> 30 in both sessions that contain 20
    assert len(ctx.session_vector) == FACTORS
    assert ctx.recent_category_ids == [2, 5, 1]       # most recent video's categories first


def test_session_context_absent_without_recent_videos(modeller, sample_query_intent):
    assert modeller.build_session_profile(1, sample_query_intent).session_context is None


# ---------------------------------------------------------------------------
# backend switch: ALS (default) vs LightFM
# ---------------------------------------------------------------------------

def test_default_backend_is_als(sample_interactions):
    m = UserModeller(interactions=sample_interactions, factors=FACTORS, iterations=3)
    assert m.backend == "als"
    assert type(m._model).__module__.startswith("implicit")


def test_unknown_backend_rejected(sample_interactions):
    with pytest.raises(ValueError, match="backend"):
        UserModeller(interactions=sample_interactions, backend="bpr")


def _caption_table(video_ids, first, second, tags):
    """Tiny raw caption table in the kuairec_caption_category.csv layout."""
    return pd.DataFrame({
        "video_id": video_ids,
        "topic_tag": tags,
        "first_level_category_id": first,
        "second_level_category_id": second,
        "third_level_category_id": [-124] * len(video_ids),
    })


@pytest.fixture
def sample_caption_categories():
    # "cooking" is on 5 videos (>= 5 -> a shared feature); "travel" on 1 (dropped).
    return _caption_table(
        [10, 20, 30, 40, 50, 60],
        [1, 2, 1, 2, 1, 3],
        [11, 21, 11, 21, 11, -124],
        ["[cooking]", "[cooking]", "[cooking,fun]", "[cooking]", "[cooking]", "[travel]"],
    )


class TestLightFMBackend:
    @pytest.fixture(autouse=True)
    def _needs_lightfm(self):
        pytest.importorskip("lightfm")

    def _fit(self, interactions, captions, **kw):
        return UserModeller(
            interactions=interactions, caption_categories=captions,
            factors=FACTORS, iterations=kw.pop("iterations", 5), backend="lightfm", **kw,
        )

    def test_shapes_match_als_id_maps(self, sample_interactions, sample_caption_categories):
        lfm = self._fit(sample_interactions, sample_caption_categories)
        als = UserModeller(interactions=sample_interactions, factors=FACTORS, iterations=3)
        assert np.array_equal(lfm.item_ids, als.item_ids)
        assert lfm._user_to_row == als._user_to_row
        assert lfm.item_factors.shape == als.item_factors.shape
        assert lfm._user_factors.shape == als._user_factors.shape
        assert np.isfinite(lfm.item_factors).all()
        assert np.isfinite(lfm._user_factors).all()
        # identity + c1:{1,2,3} + c2:{11,21} + tag:cooking; -124 and rare tags skipped
        assert sorted(lfm.lightfm_feature_names) == sorted(
            ["c1:1", "c1:2", "c1:3", "c2:11", "c2:21", "tag:cooking"]
        )

    def test_build_profile_has_vector_and_cf(
        self, sample_interactions, sample_caption_categories
    ):
        m = self._fit(sample_interactions, sample_caption_categories, k_neighbours=2)
        profile = m.build_profile(1)
        assert len(profile.user_vector) == FACTORS
        assert profile.cf_neighbour_scores  # non-empty
        assert all(np.isfinite(v) for v in profile.cf_neighbour_scores.values())

    def test_identity_only_without_captions(self, sample_interactions):
        m = self._fit(sample_interactions, None, iterations=2)
        assert m.lightfm_feature_names == []
        assert m.item_factors.shape == (6, FACTORS)

    def test_rare_video_borrows_from_shared_features(self):
        # Video 1 is watched by 40 users, video 2 by one; they share category
        # 7/77 and tag "cooking". Video 3 is unrelated (category 9, no tags).
        rows = []
        for u in range(40):
            rows.append((u, 1, 1.0))
            rows.append((u, 100 + u % 5, 0.5))  # filler videos 100..104
        rows += [(0, 2, 1.0), (1, 3, 1.0), (2, 3, 0.8)]
        inter = pd.DataFrame(rows, columns=["user_id", "video_id", "watch_ratio"])
        inter["timestamp"] = 1.0
        inter["date"] = 20200101
        captions = _caption_table(
            [1, 2, 3, 100, 101, 102, 103, 104],
            [7, 7, 9, 8, 8, 8, 8, 8],
            [77, 77, 99, 88, 88, 88, 88, 88],
            ["[cooking]", "[cooking]", "[]"] + ["[cooking]"] * 5,
        )
        m = self._fit(inter, captions, iterations=10)

        def cos(a, b):
            va = m.item_factors[m.column_for_video_id(a)]
            vb = m.item_factors[m.column_for_video_id(b)]
            return float(va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb)))

        assert cos(1, 2) > cos(1, 3)
        assert cos(1, 2) > cos(2, 3)
