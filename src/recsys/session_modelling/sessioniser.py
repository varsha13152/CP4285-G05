"""Sessionisation — cut each user's interaction timeline into ordered sessions.

A "session" is a bounded, time-ordered chunk of one user's interactions that
stands in for "one sitting / one task". KuaiRec has **no natural sessions**, so
we construct them by cutting the per-user timeline wherever the gap between
consecutive interactions exceeds a threshold.

Empirically (all 12.5M interactions of ``big_matrix``) the inter-interaction
gap distribution has **no valley** — density decays monotonically (median gap
~1.8 min; p90 ~19 min; p95 ~49 min). So a gap threshold is a *convention /
hyperparameter*, not a natural boundary; the right value is the one that
maximises downstream next-item accuracy. Three strategies are provided:

* ``fixed`` — one global gap for everyone (default 30 min: sits above the
  typical within-session pause, below genuine breaks; ~14-video sessions,
  ~124/user; matches the standard web-analytics convention).
* ``per_user_pct`` — each user's own gap quantile (default p90) as their
  threshold, floored (default 15 min, so bursts are never cut) with a global
  fallback for sparse users. The boundary-relevant statistic (a user's *tail*
  gap) varies widely across users (per-user p90 gap spans ~9–175 min), so this
  can fit better than one global value.
* ``single`` — one session per user (no cutting). The **ablation** that lets you
  measure whether sessionisation helps at all.

Design notes
------------
* The threshold is compared against the **inter-interaction gap** (idle time
  *between* interactions), never against watch time (``play_duration``).
* Pure pandas/numpy — no training, no GPU.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from recsys.data.schema import TIMESTAMP, USER_ID, VIDEO_ID

_SEC_PER_MIN = 60.0
_STRATEGIES = ("fixed", "per_user_pct", "single")
_SESSION_ID = "session_id"


class Sessioniser:
    """Assign a ``session_id`` to each interaction via a pluggable boundary rule.

    Parameters
    ----------
    strategy : {"fixed", "per_user_pct", "single"}
        Boundary rule. See the module docstring.
    gap_minutes : float
        The global inactivity threshold (minutes) for ``fixed``, and the
        fallback for sparse users under ``per_user_pct``.
    pct : float
        Quantile (0–1) of each user's own gaps used as their threshold under
        ``per_user_pct`` (default 0.9 = their p90 gap).
    floor_minutes : float
        Lower bound (minutes) on a per-user threshold, so within-session bursts
        are never cut. ``per_user_pct`` only.
    min_interactions : int
        Users with fewer interactions than this fall back to ``gap_minutes``
        under ``per_user_pct`` (too few gaps to estimate a stable quantile).
    cap_minutes : float | None
        Optional upper bound (minutes) on a per-user threshold, to avoid a
        single giant session for users with very long tail gaps. ``None``
        disables the cap.
    """

    def __init__(
        self,
        strategy: str = "fixed",
        gap_minutes: float = 30.0,
        pct: float = 0.9,
        floor_minutes: float = 15.0,
        min_interactions: int = 20,
        cap_minutes: float | None = None,
    ) -> None:
        if strategy not in _STRATEGIES:
            raise ValueError(f"strategy must be one of {_STRATEGIES}, got {strategy!r}")
        self.strategy = strategy
        self.gap_minutes = float(gap_minutes)
        self.pct = float(pct)
        self.floor_minutes = float(floor_minutes)
        self.min_interactions = int(min_interactions)
        self.cap_minutes = None if cap_minutes is None else float(cap_minutes)

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def assign(self, interactions: pd.DataFrame) -> pd.DataFrame:
        """Return a copy sorted by ``(user_id, timestamp)`` with a new,
        globally-unique, contiguous ``session_id`` column (0-indexed)."""
        missing = {USER_ID, VIDEO_ID, TIMESTAMP} - set(interactions.columns)
        if missing:
            raise KeyError(f"interactions is missing required columns: {sorted(missing)}")

        df = interactions.sort_values(
            [USER_ID, TIMESTAMP], kind="mergesort"
        ).reset_index(drop=True)
        gap = df.groupby(USER_ID, sort=False)[TIMESTAMP].diff()  # seconds; NaN at each user's first row
        threshold = self._row_thresholds(df, gap)

        starts_new = gap.isna().to_numpy() | (gap.to_numpy() > threshold)
        df[_SESSION_ID] = (np.cumsum(starts_new) - 1).astype(np.int64)
        return df

    def to_sequences(self, interactions: pd.DataFrame) -> list[list[int]]:
        """Ordered ``video_id`` lists, one per session — the input to
        :class:`~recsys.session_modelling.markov.MarkovSessionModel`."""
        df = self.assign(interactions)
        return [
            [int(v) for v in g[VIDEO_ID].to_numpy()]
            for _, g in df.groupby(_SESSION_ID, sort=True)
        ]

    def session_stats(self, interactions: pd.DataFrame) -> pd.DataFrame:
        """Per-user session statistics: ``avg_session_length``,
        ``median_session_length`` and ``num_sessions`` (feeds
        ``EngagementProfile.avg_session_length``)."""
        df = self.assign(interactions)
        per_session = df.groupby(_SESSION_ID, sort=False).agg(
            length=(VIDEO_ID, "size"), user_id=(USER_ID, "first")
        )
        stats = per_session.groupby(USER_ID)["length"].agg(
            avg_session_length="mean",
            median_session_length="median",
            num_sessions="size",
        )
        return stats.reset_index()

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    def _row_thresholds(self, df: pd.DataFrame, gap: pd.Series) -> np.ndarray:
        """A per-row gap threshold (seconds); a gap strictly above it starts a
        new session."""
        n = len(df)
        if self.strategy == "single":
            return np.full(n, np.inf)
        if self.strategy == "fixed":
            return np.full(n, self.gap_minutes * _SEC_PER_MIN)

        # per_user_pct ---------------------------------------------------
        global_thr = self.gap_minutes * _SEC_PER_MIN
        per_user_q = gap.groupby(df[USER_ID], sort=False).quantile(self.pct)
        per_user_q = per_user_q.clip(lower=self.floor_minutes * _SEC_PER_MIN)
        if self.cap_minutes is not None:
            per_user_q = per_user_q.clip(upper=self.cap_minutes * _SEC_PER_MIN)

        counts = df.groupby(USER_ID, sort=False).size()
        # sparse users (or users with an undefined quantile) -> global fallback
        per_user_thr = per_user_q.where(counts >= self.min_interactions, global_thr)
        per_user_thr = per_user_thr.fillna(global_thr)
        return df[USER_ID].map(per_user_thr).to_numpy()
