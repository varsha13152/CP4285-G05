"""Unit tests for :class:`recsys.session_modelling.Sessioniser`.

Self-contained synthetic frames with realistic second-scale timestamps, so the
boundary logic is exercised with known gaps and no disk I/O.
"""

import pandas as pd
import pytest

from recsys.session_modelling import Sessioniser


def _frame(rows: list[tuple[int, float, int]]) -> pd.DataFrame:
    """rows = list of (user_id, timestamp_seconds, video_id)."""
    return pd.DataFrame(rows, columns=["user_id", "timestamp", "video_id"])


# user 1: a 2880 s (48 min) break splits it; user 2: one tight session
_FIXED_FRAME = _frame(
    [
        (1, 0, 10), (1, 60, 11), (1, 120, 12), (1, 3000, 13), (1, 3060, 14),
        (2, 0, 20), (2, 100, 21), (2, 200, 22),
    ]
)


# ---------------------------------------------------------------------------
# fixed
# ---------------------------------------------------------------------------

def test_fixed_splits_on_the_large_gap():
    seqs = Sessioniser(strategy="fixed", gap_minutes=30).to_sequences(_FIXED_FRAME)
    assert seqs == [[10, 11, 12], [13, 14], [20, 21, 22]]


def test_fixed_session_ids_are_contiguous_and_zero_indexed():
    df = Sessioniser(strategy="fixed", gap_minutes=30).assign(_FIXED_FRAME)
    assert df["session_id"].tolist() == [0, 0, 0, 1, 1, 2, 2, 2]


def test_larger_threshold_merges_the_break():
    # 48-min break < 60-min threshold -> user 1 stays a single session
    seqs = Sessioniser(strategy="fixed", gap_minutes=60).to_sequences(_FIXED_FRAME)
    assert seqs == [[10, 11, 12, 13, 14], [20, 21, 22]]


# ---------------------------------------------------------------------------
# single (ablation)
# ---------------------------------------------------------------------------

def test_single_is_one_session_per_user():
    seqs = Sessioniser(strategy="single").to_sequences(_FIXED_FRAME)
    assert seqs == [[10, 11, 12, 13, 14], [20, 21, 22]]


# ---------------------------------------------------------------------------
# per_user_pct
# ---------------------------------------------------------------------------

def test_per_user_pct_floor_prevents_over_splitting():
    # gaps are all 30 s except one 600 s (10 min) gap; floor is 15 min, so the
    # 10-min gap must NOT cut -> a single session.
    rows = [(1, t, 100 + i) for i, t in enumerate([0, 30, 60, 90, 120, 150, 180, 210, 240, 840])]
    seqs = Sessioniser(
        strategy="per_user_pct", pct=0.9, floor_minutes=15, min_interactions=5
    ).to_sequences(_frame(rows))
    assert len(seqs) == 1


def test_per_user_pct_cuts_a_gap_above_the_users_own_scale():
    # tight ~1000 s gaps then a 12000 s outlier -> the outlier exceeds the
    # user's own p90 (well above the 15-min floor) -> exactly two sessions.
    ts = [0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 20000]
    rows = [(1, t, 100 + i) for i, t in enumerate(ts)]
    seqs = Sessioniser(
        strategy="per_user_pct", pct=0.9, floor_minutes=15, min_interactions=5
    ).to_sequences(_frame(rows))
    assert len(seqs) == 2
    assert seqs[1] == [109]  # only the post-break interaction


def test_per_user_pct_sparse_user_uses_global_fallback():
    sess = Sessioniser(
        strategy="per_user_pct", gap_minutes=1000, pct=0.9, floor_minutes=15, min_interactions=5
    )
    # user A: 3 interactions (< min) -> global fallback (1000 min) -> no cut
    # user B: 6 interactions -> own p90 (~20 min) -> the 2000 s gap cuts
    rows = (
        [(1, t, 10 + i) for i, t in enumerate([0, 100, 2100])]
        + [(2, t, 20 + i) for i, t in enumerate([0, 10, 20, 30, 40, 2040])]
    )
    df = sess.assign(_frame(rows))
    a_sessions = df.loc[df.user_id == 1, "session_id"].nunique()
    b_sessions = df.loc[df.user_id == 2, "session_id"].nunique()
    assert a_sessions == 1  # sparse -> global fallback merges
    assert b_sessions == 2  # adaptive -> outlier gap splits


# ---------------------------------------------------------------------------
# session_stats
# ---------------------------------------------------------------------------

def test_session_stats():
    stats = Sessioniser(strategy="fixed", gap_minutes=30).session_stats(_FIXED_FRAME)
    by_user = stats.set_index("user_id")
    assert by_user.loc[1, "num_sessions"] == 2
    assert by_user.loc[1, "avg_session_length"] == pytest.approx(2.5)  # (3 + 2) / 2
    assert by_user.loc[2, "num_sessions"] == 1
    assert by_user.loc[2, "avg_session_length"] == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# guards
# ---------------------------------------------------------------------------

def test_invalid_strategy_raises():
    with pytest.raises(ValueError):
        Sessioniser(strategy="nonsense")


def test_missing_columns_raises():
    bad = pd.DataFrame({"user_id": [1], "timestamp": [0.0]})  # no video_id
    with pytest.raises(KeyError):
        Sessioniser().assign(bad)
