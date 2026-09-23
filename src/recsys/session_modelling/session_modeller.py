"""Short-term session modelling — the in-session companion to ``u_long``.

``SessionModeller`` is the Stage-2 short-term counterpart of
:class:`~recsys.user_modelling.user_modeller.UserModeller`. Where ``UserModeller``
learns *who the user is* (long-term ALS taste vector), this learns *what the user
is doing right now*: it sessionises the interaction history, fits a first-order
:class:`~recsys.session_modelling.markov.MarkovSessionModel`, and exposes two
things the rest of the pipeline consumes:

* :meth:`predict_next` — the **SESSION** retrieval lane (next-item scores given
  the current session), and a ranking feature.
* :meth:`avg_session_length_for` — the real per-user average session length,
  which replaces the daily-intensity *proxy* on
  :attr:`EngagementProfile.avg_session_length` (see
  :meth:`UserModeller.build_session_profile`).

Like ``UserModeller`` it trains **on construction** and offers an in-memory
``interactions`` seam so unit tests avoid disk I/O.
"""

from __future__ import annotations

import pandas as pd

from recsys.data.loader import load_kuairec
from recsys.session_modelling.markov import MarkovSessionModel
from recsys.session_modelling.sessioniser import Sessioniser

_AVG_LEN = "avg_session_length"
_USER_ID = "user_id"


class SessionModeller:
    """Fit a session-based next-item model and per-user session statistics.

    Parameters
    ----------
    data_dir : str
        Directory holding the raw KuaiRec CSVs. Ignored when *interactions* is
        supplied directly (the testing / custom-data seam).
    strategy, gap_minutes, pct, floor_minutes, min_interactions, cap_minutes
        Passed straight through to :class:`Sessioniser` — the sessionisation
        boundary rule. Defaults match ``Sessioniser`` (``fixed``, 30 min).
    interactions : pandas.DataFrame | None
        In-memory interaction table to train on instead of loading from disk.
        Must expose ``user_id``, ``video_id`` and ``timestamp`` columns.
    """

    def __init__(
        self,
        data_dir: str = "data/raw",
        strategy: str = "fixed",
        gap_minutes: float = 30.0,
        pct: float = 0.9,
        floor_minutes: float = 15.0,
        min_interactions: int = 20,
        cap_minutes: float | None = None,
        interactions: pd.DataFrame | None = None,
    ) -> None:
        if interactions is None:
            data = load_kuairec(data_dir, load_big=True)
            interactions = data.big_matrix
            if interactions is None:  # pragma: no cover - defensive
                raise RuntimeError("load_kuairec did not return big_matrix")

        self._sessioniser = Sessioniser(
            strategy=strategy,
            gap_minutes=gap_minutes,
            pct=pct,
            floor_minutes=floor_minutes,
            min_interactions=min_interactions,
            cap_minutes=cap_minutes,
        )
        # Fit the transition table on constructed session sequences.
        self._model = MarkovSessionModel().fit(self._sessioniser.to_sequences(interactions))
        # Per-user session stats -> a plain {user_id: avg_session_length} lookup.
        stats = self._sessioniser.session_stats(interactions)
        self._avg_len: dict[int, float] = {
            int(u): float(v) for u, v in zip(stats[_USER_ID], stats[_AVG_LEN])
        }

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def predict_next(
        self,
        recent_items: list[int],
        k: int | None = None,
        exclude: set[int] | None = None,
    ) -> list[tuple[int, float]]:
        """Score candidate next videos given the current session — the SESSION
        retrieval lane. Delegates to the fitted
        :class:`MarkovSessionModel`."""
        return self._model.predict_next(recent_items, k=k, exclude=exclude)

    def avg_session_length_for(self, user_id: int) -> float:
        """Real average session length for *user_id* (0.0 for unknown users) —
        fills :attr:`EngagementProfile.avg_session_length`, replacing the
        interactions-per-active-day proxy."""
        return self._avg_len.get(int(user_id), 0.0)

    @property
    def model(self) -> MarkovSessionModel:
        """The fitted next-item model (read-only handle for downstream use)."""
        return self._model
