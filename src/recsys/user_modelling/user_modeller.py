"""Long-term user modelling (``u_long``) via implicit ALS matrix factorization.

Trains an implicit-feedback ALS model over KuaiRec's sparse ``big_matrix``,
feeding ``watch_ratio`` as the per-interaction *confidence*. Each user's learned
latent factor vector is exposed on ``EngagementProfile.user_vector`` and consumed
downstream purely as a Stage 4 ranking signal, never as a retrieval filter.

In the query-first cascade, Stage 2 also precomputes the candidate-independent
lookups Stage 4 needs, so ranking is a dict lookup per candidate:

* ``cf_neighbour_scores`` — kNN users (cosine over ALS user vectors) and a
  ``{video_id: Σ sim · watch_ratio}`` over their history.
* ``social_scores`` — ``{video_id: mean watch_ratio among the user's friends}``
  (only ~6.6% of KuaiRec users have friends; empty otherwise).
* ``session_context`` — built by :meth:`build_session_profile` from the user's
  recent videos.

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
* **Categories** are caption first-level ids (``CategoryIndex``), matching
  Stage 1's query topics — not the ``feat`` tags of ``item_categories.csv``.

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

from recsys.contracts.models import EngagementProfile, QueryIntent, SessionContext
from recsys.data.item_features import caption_category_table
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
        category statistics. When ``None`` and loading from disk, it is built
        from the caption file's first-level ids
        (:func:`~recsys.data.item_features.caption_category_table`); when
        ``None`` with in-memory *interactions*, category stats are left empty
        (a valid stub).
    show_progress : bool
        Whether the ALS fit prints a progress bar.
    session_modeller : SessionModeller | None
        Optional short-term session model (``recsys.session_modelling``). When
        set, :meth:`build_session_profile` uses the real ``avg_session_length``
        and fills ``session_context.next_item_scores``. ``None`` keeps the
        long-term-only behaviour.
    social_network : pandas.DataFrame | None
        ``user_id`` → ``friend_list`` table for ``social_scores``. When ``None``
        and loading from disk, the loader's ``social_network`` is used; when
        ``None`` with in-memory *interactions*, ``social_scores`` stay empty.
    k_neighbours : int
        Number of nearest users (cosine over ALS user vectors) behind
        ``cf_neighbour_scores``.
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
        social_network: pd.DataFrame | None = None,
        k_neighbours: int = 50,
    ) -> None:
        # Lazy import so importing the package does not hard-require `implicit`
        # unless a model is actually trained.
        from implicit.als import AlternatingLeastSquares

        self.factors = factors
        self._session_modeller = session_modeller
        self._k_neighbours = k_neighbours

        if interactions is None:
            data = load_kuairec(data_dir, load_big=True)
            interactions = data.big_matrix
            if interactions is None:  # pragma: no cover - defensive
                raise RuntimeError("load_kuairec did not return big_matrix")
            if item_categories is None:
                item_categories = caption_category_table(data.caption_categories)
            if social_network is None:
                social_network = data.social_network

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

        # --- Stage 4 lookups: kNN-CF and social -----------------------------
        self._watch = interactions["watch_ratio"].to_numpy(dtype=np.float64)
        self._video_cols = cols  # item_factors column per interaction row
        self._row_to_user = unique_users
        norms = np.linalg.norm(self._user_factors, axis=1, keepdims=True)
        self._unit_users = self._user_factors / np.where(norms == 0, 1.0, norms)
        self._friends: dict[int, list[int]] = {}
        if social_network is not None and "friend_list" in social_network.columns:
            self._friends = {
                int(u): [int(f) for f in fl]
                for u, fl in zip(social_network["user_id"], social_network["friend_list"])
                if len(fl)
            }

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
            profile = self._cold_start_profile(user_id)
            # Social needs only the friends' history, so it survives cold start.
            profile.social_scores = self._social_scores(user_id)
            return profile

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
            social_scores=self._social_scores(user_id),
            cf_neighbour_scores=self._cf_neighbour_scores(row),
        )

    def build_session_profile(
        self,
        user_id: int,
        query_intent: QueryIntent,
        recent_video_ids: list[int] | None = None,
    ) -> EngagementProfile:
        """Long-term profile plus the user's short-term context.

        * ``avg_session_length`` — the real per-user value from the injected
          :class:`SessionModeller` (replaces the daily-intensity proxy).
        * ``session_context`` — built from *recent_video_ids* (oldest → newest)
          when given; ``None`` otherwise.

        The query does not alter the profile: blending the query with the
        user's taste is Stage 4's job (query relevance stays a ranking
        feature), so the long-term profile is never mutated by a search.
        *query_intent* is accepted for the pipeline contract.
        """
        profile = self.build_profile(user_id)
        if self._session_modeller is not None:
            profile.avg_session_length = self._session_modeller.avg_session_length_for(user_id)
        if recent_video_ids:
            profile.session_context = self._session_context(list(recent_video_ids))
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

    def _cf_neighbour_scores(self, row: int) -> dict[int, float]:
        """``{video_id: Σ cosine(u, v) · watch_ratio(v, video)}`` over the
        user's ``k_neighbours`` nearest users with positive similarity."""
        # ponytail: exact brute-force kNN (7k users × 16 dims ≈ 0.1 ms); an ANN
        # index only pays off at ~100× the users.
        sims = self._unit_users @ self._unit_users[row]
        sims[row] = -np.inf  # never your own neighbour
        k = min(self._k_neighbours, len(sims) - 1)
        if k <= 0:
            return {}
        top = np.argpartition(-sims, k - 1)[:k]
        top = top[sims[top] > 0]
        if not len(top):
            return {}
        idx_parts, weight_parts = [], []
        for nrow in top:
            idx = self._user_indices[int(self._row_to_user[nrow])]
            idx_parts.append(idx)
            weight_parts.append(np.full(len(idx), sims[nrow]))
        idx = np.concatenate(idx_parts)
        scores = np.bincount(
            self._video_cols[idx],
            weights=np.concatenate(weight_parts) * self._watch[idx],
            minlength=len(self._col_to_video),
        )
        nz = np.flatnonzero(scores)
        return {int(self._col_to_video[c]): float(scores[c]) for c in nz}

    def _social_scores(self, user_id: int) -> dict[int, float]:
        """``{video_id: mean watch_ratio among the user's friends}``."""
        idx = [
            self._user_indices[f] for f in self._friends.get(int(user_id), ())
            if f in self._user_indices
        ]
        if not idx:
            return {}
        idx = np.concatenate(idx)
        means = (
            pd.Series(self._watch[idx]).groupby(self._video_ids[idx]).mean()
        )
        return {int(v): float(s) for v, s in means.items()}

    def _session_context(self, recent: list[int]) -> SessionContext:
        """Short-term lookups from the user's recent videos (oldest → newest)."""
        next_scores: dict[int, float] = {}
        if self._session_modeller is not None:
            # Raw transitions for the last video only — no popularity backoff,
            # since popularity is already its own Stage 4 feature.
            next_scores = dict(self._session_modeller.model.transitions_for(recent[-1]))
        cols = [c for c in (self.column_for_video_id(v) for v in recent) if c is not None]
        session_vector = (
            self._item_factors[cols].mean(axis=0).tolist() if cols else []
        )
        recent_cats: list[int] = []
        for vid in reversed(recent):  # most recent first, de-duplicated
            for cat in self._vid_to_feats.get(int(vid), ()):
                if cat not in recent_cats:
                    recent_cats.append(cat)
        return SessionContext(
            recent_video_ids=recent,
            next_item_scores=next_scores,
            session_vector=session_vector,
            recent_category_ids=recent_cats,
        )

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
