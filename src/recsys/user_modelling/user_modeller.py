"""Long-term user modelling (``u_long``) via implicit ALS matrix factorization.

Trains an implicit-feedback ALS model over KuaiRec's sparse ``big_matrix``,
feeding ``watch_ratio`` as the per-interaction *confidence*. Each user's learned
latent factor vector is exposed on ``EngagementProfile.user_vector`` and consumed
downstream purely as a ranking signal (the personalised/collaborative candidate
lane and a taste-alignment feature in ranking), never as a hard filter.

Design notes
------------
* **ALS over BPR** — KuaiRec's informative signal is the graded ``watch_ratio``,
  which ALS exploits directly as confidence; BPR would flatten it to binary.
* **Sparsity** — trained observed-only over a ``scipy`` CSR matrix; no
  ``fillna(0)`` densification. The fully-observed ``small_matrix`` is reserved
  for evaluation and is intentionally *not* touched here.
* **Confidence transform** (open knob) — ``confidence_alpha=None`` uses the raw
  ``watch_ratio`` as confidence (the default); a float applies
  ``1 + confidence_alpha * watch_ratio``.
* **Session model** (``build_session_profile``) is deferred and currently
  delegates to :meth:`build_profile`.

Usage::

    from recsys.user_modelling import UserModeller
    from recsys.routing import QueryRouter

    modeller = UserModeller()                       # trains on data/raw/big_matrix.csv
    profile = modeller.build_profile(user_id=14)    # EngagementProfile w/ user_vector
    request = QueryRouter().route(intent, user_id=14, engagement_profile=profile)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from recsys.contracts.models import EngagementProfile, QueryIntent
from recsys.data.loader import load_kuairec

if TYPE_CHECKING:
    from recsys.session_modelling import SessionModeller

_SECONDS_PER_DAY = 86_400.0


class UserModeller:
    """Build long-term engagement profiles backed by ALS latent factors.

    Parameters
    ----------
    data_dir : str
        Directory holding the raw KuaiRec CSVs. Ignored when *interactions*
        is supplied directly (the testing / custom-data seam).
    factors : int
        Latent dimensionality ``k`` of the learned user/item vectors
        (``EngagementProfile.user_vector`` has length ``factors``).
    regularization : float
        ALS L2 regularization strength.
    iterations : int
        Number of ALS alternating-minimization sweeps.
    confidence_alpha : float | None
        Confidence transform (open knob). ``None`` feeds the raw
        ``watch_ratio`` as confidence; a float feeds
        ``1 + confidence_alpha * watch_ratio``.
    random_state : int | None
        Seed for ALS factor initialisation, for reproducible vectors.
    interactions : pandas.DataFrame | None
        In-memory interaction table to train on instead of loading from disk.
        Must expose ``user_id``, ``video_id``, ``watch_ratio``, ``date`` and
        ``timestamp`` columns. Used by unit tests to avoid disk I/O.
    item_categories : pandas.DataFrame | None
        ``video_id`` → ``feat`` (list of category ids) table used for the
        category statistics. When ``None`` and loading from disk, the loader's
        ``item_categories`` is used; when ``None`` with in-memory
        *interactions*, category stats are left empty (a valid stub).
    show_progress : bool
        Whether the ALS fit prints a progress bar.
    session_modeller : SessionModeller | None
        Optional short-term session model (``recsys.session_modelling``). When
        set, :meth:`build_session_profile` replaces the ``avg_session_length``
        proxy with its real per-user value (and, later, will blend its next-item
        scores). ``None`` keeps the long-term-only behaviour.
    """

    def __init__(
        self,
        data_dir: str = "data/raw",
        factors: int = 16,
        regularization: float = 0.05,
        iterations: int = 15,
        confidence_alpha: float | None = None,
        random_state: int | None = 42,
        interactions: pd.DataFrame | None = None,
        item_categories: pd.DataFrame | None = None,
        show_progress: bool = False,
        session_modeller: SessionModeller | None = None,
    ) -> None:
        # Lazy import so importing the package does not hard-require `implicit`
        # unless a model is actually trained.
        from implicit.als import AlternatingLeastSquares

        self.factors = factors
        self._session_modeller = session_modeller

        if interactions is None:
            data = load_kuairec(data_dir, load_big=True)
            interactions = data.big_matrix
            if interactions is None:  # pragma: no cover - defensive
                raise RuntimeError("load_kuairec did not return big_matrix")
            if item_categories is None:
                item_categories = data.item_categories

        self._interactions = interactions

        # --- id maps (sorted for determinism) -----------------------------
        unique_users = np.sort(interactions["user_id"].unique())
        unique_videos = np.sort(interactions["video_id"].unique())
        self._user_to_row = {int(u): i for i, u in enumerate(unique_users)}
        self._video_to_col = {int(v): i for i, v in enumerate(unique_videos)}
        self._col_to_video = unique_videos  # column index -> video_id (np.ndarray)
        n_users = len(unique_users)
        n_videos = len(unique_videos)

        # --- confidence matrix (users x videos), observed-only ------------
        rows = interactions["user_id"].map(self._user_to_row).to_numpy()
        cols = interactions["video_id"].map(self._video_to_col).to_numpy()
        conf = interactions["watch_ratio"].to_numpy(dtype=np.float32)
        if confidence_alpha is not None:
            conf = (1.0 + confidence_alpha * conf).astype(np.float32)
        # Duplicate (user, video) pairs are summed by scipy -> cumulative
        # confidence, which is the intended "more views => more sure" behaviour.
        user_items = csr_matrix(
            (conf, (rows, cols)), shape=(n_users, n_videos), dtype=np.float32
        )

        # --- fit ALS ------------------------------------------------------
        self._model = AlternatingLeastSquares(
            factors=factors,
            regularization=regularization,
            iterations=iterations,
            random_state=random_state,
        )
        self._model.fit(user_items, show_progress=show_progress)
        self._user_factors = np.asarray(self._model.user_factors)
        self._item_factors = np.asarray(self._model.item_factors)

        # --- precomputed per-user statistics ------------------------------
        self._global_last_ts = float(interactions["timestamp"].max())
        self._stats = self._compute_user_stats(interactions)
        # positional row indices per user, for cheap per-user category counts
        self._user_indices = {
            int(uid): idx for uid, idx in interactions.groupby("user_id").indices.items()
        }
        self._video_ids = interactions["video_id"].to_numpy()
        self._vid_to_feats = self._build_category_lookup(item_categories)

    # ------------------------------------------------------------------
    # construction helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _compute_user_stats(interactions: pd.DataFrame) -> dict[int, dict]:
        """One-pass per-user scalar aggregates keyed by ``user_id``."""
        grouped = interactions.groupby("user_id").agg(
            avg_watch_ratio=("watch_ratio", "mean"),
            watch_ratio_std=("watch_ratio", "std"),
            total_interactions=("watch_ratio", "size"),
            active_days=("date", "nunique"),
            last_ts=("timestamp", "max"),
        )
        # std is NaN for single-interaction users -> treat as 0 variance.
        grouped["watch_ratio_std"] = grouped["watch_ratio_std"].fillna(0.0)
        return {int(uid): rec for uid, rec in grouped.to_dict("index").items()}

    @staticmethod
    def _build_category_lookup(
        item_categories: pd.DataFrame | None,
    ) -> dict[int, list[int]]:
        """``video_id`` -> list of category ids, from the ``feat`` column."""
        if item_categories is None or "feat" not in getattr(item_categories, "columns", []):
            return {}
        lookup: dict[int, list[int]] = {}
        for vid, feat in zip(item_categories["video_id"], item_categories["feat"]):
            if isinstance(feat, (list, tuple)):
                lookup[int(vid)] = [int(c) for c in feat]
        return lookup

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def build_profile(self, user_id: int) -> EngagementProfile:
        """Build the long-term :class:`EngagementProfile` for *user_id*.

        Returns a zeroed profile with an empty ``user_vector`` for unknown /
        cold-start users (those absent from the training interactions).
        """
        user_id = int(user_id)
        row = self._user_to_row.get(user_id)
        if row is None:
            return self._cold_start_profile(user_id)

        rec = self._stats[user_id]
        top_category_ids, category_distribution = self._category_stats(user_id)
        active_days = int(rec["active_days"])
        total_interactions = int(rec["total_interactions"])
        avg_session_length = (
            total_interactions / active_days if active_days else 0.0
        )
        recency_days = max(
            0, round((self._global_last_ts - float(rec["last_ts"])) / _SECONDS_PER_DAY)
        )

        return EngagementProfile(
            user_id=user_id,
            avg_watch_ratio=float(rec["avg_watch_ratio"]),
            watch_ratio_std=float(rec["watch_ratio_std"]),
            total_interactions=total_interactions,
            active_days=active_days,
            top_category_ids=top_category_ids,
            category_distribution=category_distribution,
            avg_session_length=float(avg_session_length),
            recency_days=recency_days,
            user_vector=self._user_factors[row].tolist(),
            seen_video_ids=self._seen_video_ids(user_id),
        )

    def build_session_profile(
        self, user_id: int, query_intent: QueryIntent
    ) -> EngagementProfile:
        """Session-aware profile: the long-term profile with its
        ``avg_session_length`` proxy replaced by the **real** per-user session
        length from the injected :class:`SessionModeller`.

        With no session model (``session_modeller=None``) this falls back to
        :meth:`build_profile` unchanged. The query-conditioned session *blend*
        (the QCSAR differentiator — folding ``query_intent`` and the session
        next-item scores into the profile) is still deferred; only the
        session-length signal is wired in so far.
        """
        profile = self.build_profile(user_id)
        if self._session_modeller is not None:
            profile.avg_session_length = self._session_modeller.avg_session_length_for(user_id)
        return profile

    # ------------------------------------------------------------------
    # exports for downstream candidate generation
    # ------------------------------------------------------------------
    @property
    def item_factors(self) -> np.ndarray:
        """The learned item latent-factor matrix ``V`` — shape ``(n_videos, factors)``.

        Row ``i`` is the vector for ``self.item_ids[i]``. Index this into an ANN /
        MIPS structure for candidate generation: the user's ``user_vector`` is the
        query and a large dot product predicts affinity. Returns the live array —
        treat as read-only. Comparable only with vectors from *this* training run.
        """
        return self._item_factors

    @property
    def item_ids(self) -> np.ndarray:
        """Video ids aligned to the rows of :attr:`item_factors` (``item_ids[col]``
        is the ``video_id`` for row ``col``) — decodes ANN results back to ids."""
        return self._col_to_video

    def column_for_video_id(self, video_id: int) -> int | None:
        """Row index of *video_id* in :attr:`item_factors`, or ``None`` if unseen."""
        return self._video_to_col.get(int(video_id))

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    def _category_stats(self, user_id: int) -> tuple[list[int], dict[int, float]]:
        """Ranked category ids and normalised distribution for *user_id*."""
        if not self._vid_to_feats:
            return [], {}
        idx = self._user_indices.get(user_id)
        if idx is None:
            return [], {}
        counts: dict[int, int] = {}
        for vid in self._video_ids[idx]:
            for cat in self._vid_to_feats.get(int(vid), ()):
                counts[cat] = counts.get(cat, 0) + 1
        total = sum(counts.values())
        if not total:
            return [], {}
        ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        top_category_ids = [cat for cat, _ in ranked]
        category_distribution = {cat: n / total for cat, n in ranked}
        return top_category_ids, category_distribution

    def _seen_video_ids(self, user_id: int) -> set[int]:
        idx = self._user_indices.get(int(user_id))
        if idx is None:
            return set()
        return {int(v) for v in self._video_ids[idx]}

    @staticmethod
    def _cold_start_profile(user_id: int) -> EngagementProfile:
        return EngagementProfile(
            user_id=user_id,
            avg_watch_ratio=0.0,
            watch_ratio_std=0.0,
            total_interactions=0,
            active_days=0,
            top_category_ids=[],
            category_distribution={},
            avg_session_length=0.0,
            recency_days=0,
            user_vector=[],
            seen_video_ids=set(),
        )
