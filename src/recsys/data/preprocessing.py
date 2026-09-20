"""Shared preprocessing utilities for KuaiRec data."""

from __future__ import annotations

import pandas as pd

from recsys.data.schema import DATE, TIMESTAMP, USER_ID, WATCH_RATIO


def clip_watch_ratio(df: pd.DataFrame, max_ratio: float = 5.0) -> pd.DataFrame:
    """Clip watch_ratio outliers. Values above *max_ratio* are capped."""
    out = df.copy()
    out[WATCH_RATIO] = out[WATCH_RATIO].clip(upper=max_ratio)
    return out


def add_positive_label(
    df: pd.DataFrame, threshold: float = 2.0, col_name: str = "is_positive"
) -> pd.DataFrame:
    """Add a binary label: 1 if watch_ratio >= threshold, else 0."""
    out = df.copy()
    out[col_name] = (out[WATCH_RATIO] >= threshold).astype(int)
    return out


def temporal_train_test_split(
    df: pd.DataFrame, test_ratio: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split interactions by timestamp: earliest → train, latest → test."""
    sorted_df = df.sort_values(TIMESTAMP)
    split_idx = int(len(sorted_df) * (1 - test_ratio))
    return sorted_df.iloc[:split_idx].copy(), sorted_df.iloc[split_idx:].copy()


def per_user_leave_one_out(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """For each user, hold out their most recent interaction as test."""
    sorted_df = df.sort_values([USER_ID, TIMESTAMP])
    test = sorted_df.groupby(USER_ID).tail(1)
    train = sorted_df.drop(test.index)
    return train, test
