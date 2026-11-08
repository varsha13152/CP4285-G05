"""Simplified QCSARRanker — α-blended long-term / short-term personalisation.

Replaces the original three-stage cascade (Stages 4-6) with a single scoring
stage whose only tuneable axis is the long-term / short-term balance::

    When both LT and ST composites are active:
        personal = α × long_term + (1 − α) × short_term
    When only one composite is active (e.g. no session → ST dead):
        personal = the active composite (full budget)
    score = w_query × query_relevance + (1 − w_query) × personal

This ensures that α only controls the LT/ST *blend*, and users without
a session still receive the full personalisation budget (not α × LT).

Long-term composite (mean of rank-normalised sub-signals):
    • CF neighbour scores  (kNN collaborative filtering)
    • Category affinity    (user's habitual categories)
    • Social scores        (friends' watch ratios)
    • MF score             (``user_vector · V[video]``, the direct ALS
                            prediction; only when ``use_mf_score=True``)

Short-term composite (mean of rank-normalised sub-signals):
    • Markov transition P(next | last watched)
    • ALS item-item similarity to recent videos
    • Caption embedding similarity to recent videos

Anonymous users (no profile) are ranked by query relevance alone.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from scipy.stats import rankdata

from recsys.contracts.models import (
    CandidateSet,
    RankedItem,
    RankedList,
)

if TYPE_CHECKING:
    from recsys.contracts.models import Candidate, EngagementProfile

logger = logging.getLogger(__name__)


class QCSARRanker:
    """Single-stage ranker that blends long-term and short-term signals.

    Parameters
    ----------
    item_factors : np.ndarray
        ALS item latent factor matrix, shape ``(n_videos, k)``.
        Obtained from ``UserModeller.item_factors``.
    item_ids : np.ndarray
        Video IDs aligned to rows of *item_factors*,
        from ``UserModeller.item_ids``.
    caption_embeddings : np.ndarray | None
        Optional ``(n_videos, 1024)`` L2-normalised caption embeddings
        from ``data/processed/caption_embeddings.npz``.
    caption_video_ids : np.ndarray | None
        Video IDs aligned to rows of *caption_embeddings*.
    alpha : float
        Long-term / short-term ratio.  ``1.0`` = pure long-term,
        ``0.0`` = pure short-term.  Default ``0.5``.
    w_query : float
        Weight on query relevance vs personal signals.
        ``1.0`` = rank by query relevance only (ignore personal);
        ``0.0`` = rank by personal signals only.  Default ``0.5``.
    top_k : int
        Maximum items in the final ``RankedList``.
    use_mf_score : bool
        If ``True``, add the direct matrix-factorisation prediction
        ``profile.user_vector · item_factors[video]`` (raw, un-normalised
        factors) as a 4th long-term signal.  Skipped for cold-start users
        (empty ``user_vector``) and when the user vector's dimension does
        not match the item factors.  Default ``False`` (behaviour unchanged).
    """

    def __init__(
        self,
        item_factors: np.ndarray,
        item_ids: np.ndarray,
        caption_embeddings: np.ndarray | None = None,
        caption_video_ids: np.ndarray | None = None,
        *,
        alpha: float = 0.5,
        w_query: float = 0.5,
        top_k: int = 20,
        use_mf_score: bool = False,
    ) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        if not 0.0 <= w_query <= 1.0:
            raise ValueError(f"w_query must be in [0, 1], got {w_query}")

        # Raw ALS factors for the direct MF prediction U[user] · V[video].
        self._als_raw: np.ndarray = np.asarray(item_factors, dtype=np.float64)

        # L2-normalise ALS factors for cosine similarity via dot product.
        norms = np.linalg.norm(item_factors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)  # avoid div-by-zero
        self._als_normed: np.ndarray = item_factors / norms
        self._als_ids: np.ndarray = np.asarray(item_ids)
        self._vid_to_als: dict[int, int] = {
            int(vid): i for i, vid in enumerate(self._als_ids)
        }

        # Optional caption embeddings (already L2-normalised on disk).
        self._emb: np.ndarray | None = None
        self._vid_to_emb: dict[int, int] | None = None
        if caption_embeddings is not None and caption_video_ids is not None:
            self._emb = np.asarray(caption_embeddings)
            self._vid_to_emb = {
                int(vid): i for i, vid in enumerate(caption_video_ids)
            }

        self._alpha = alpha
        self._w_query = w_query
        self._top_k = top_k
        self._use_mf_score = use_mf_score

    # ------------------------------------------------------------------
    # Public API (Ranker protocol)
    # ------------------------------------------------------------------

    def rank(self, candidates: CandidateSet) -> RankedList:
        """Score and rank candidates using α-blended personalisation."""
        if not candidates.candidates:
            return RankedList(
                user_id=candidates.user_id,
                items=[],
                num_candidates_considered=0,
                intent_type=candidates.request.query_intent.intent_type,
            )

        cands = candidates.candidates
        profile = candidates.request.engagement_profile
        n = len(cands)

        # --- Query relevance (from Stage 3 composite score) ---
        query_arr = np.array([c.score for c in cands], dtype=np.float64)
        query_n = self._rank_normalize(query_arr)

        # --- Long-term personal signals ---
        cf_arr = np.zeros(n, dtype=np.float64)
        cat_aff_arr = np.zeros(n, dtype=np.float64)
        social_arr = np.zeros(n, dtype=np.float64)
        mf_arr = np.zeros(n, dtype=np.float64)

        if profile is not None:
            cf_scores = profile.cf_neighbour_scores
            soc_scores = profile.social_scores
            cat_dist = profile.category_distribution
            for i, c in enumerate(cands):
                cf_arr[i] = cf_scores.get(c.video_id, 0.0)
                social_arr[i] = soc_scores.get(c.video_id, 0.0)
                if c.category_ids and cat_dist:
                    cat_aff_arr[i] = cat_dist.get(c.category_ids[0], 0.0)

            if self._use_mf_score and profile.user_vector:
                mf_arr = self._mf_scores(cands, profile.user_vector)

        lt_signals = [cf_arr, cat_aff_arr, social_arr]
        if self._use_mf_score:
            lt_signals.append(mf_arr)
        long_term = self._active_mean(lt_signals)

        # --- Short-term personal signals ---
        markov_arr = np.zeros(n, dtype=np.float64)
        als_sim_arr = np.zeros(n, dtype=np.float64)
        cap_sim_arr = np.zeros(n, dtype=np.float64)

        if profile is not None and profile.session_context is not None:
            ctx = profile.session_context

            # Markov transition P(next | last watched)
            nxt = ctx.next_item_scores
            for i, c in enumerate(cands):
                markov_arr[i] = nxt.get(c.video_id, 0.0)

            recent = ctx.recent_video_ids
            if recent:
                # ALS item-item similarity (mean of top-3 cosines to recent)
                recent_rows = [
                    self._vid_to_als[v] for v in recent if v in self._vid_to_als
                ]
                if recent_rows:
                    recent_mat = self._als_normed[recent_rows]  # (n_recent, k)
                    for i, c in enumerate(cands):
                        row = self._vid_to_als.get(c.video_id, -1)
                        if row == -1:
                            continue
                        sims = recent_mat @ self._als_normed[row]  # (n_recent,)
                        top_k = min(3, len(sims))
                        als_sim_arr[i] = float(np.sort(sims)[-top_k:].mean())

                # Caption embedding similarity (optional)
                if self._emb is not None and self._vid_to_emb is not None:
                    recent_emb_rows = [
                        self._vid_to_emb[v]
                        for v in recent
                        if v in self._vid_to_emb
                    ]
                    if recent_emb_rows:
                        recent_emb = self._emb[recent_emb_rows]  # (n_recent, d)
                        for i, c in enumerate(cands):
                            erow = self._vid_to_emb.get(c.video_id, -1)
                            if erow == -1:
                                continue
                            sims = recent_emb @ self._emb[erow]
                            top_k = min(3, len(sims))
                            cap_sim_arr[i] = float(np.sort(sims)[-top_k:].mean())

        short_term = self._active_mean(
            [markov_arr, als_sim_arr, cap_sim_arr],
        )

        # --- α-blended scoring (budget-normalised) ---
        # When one composite is dead (all equal), the other gets the full
        # personal budget.  α only controls the blend when both contribute.
        lt_active = long_term.max() != long_term.min()
        st_active = short_term.max() != short_term.min()

        if lt_active and st_active:
            personal = self._alpha * long_term + (1 - self._alpha) * short_term
        elif lt_active:
            personal = long_term
        elif st_active:
            personal = short_term
        else:
            personal = np.zeros(n, dtype=np.float64)

        scores = self._w_query * query_n + (1 - self._w_query) * personal

        # Sort descending (stable) and build RankedList
        order = np.argsort(-scores, kind="stable")
        items = [
            RankedItem(
                video_id=cands[i].video_id,
                rank=rank + 1,
                score=float(scores[i]),
                source=cands[i].source,
                category_ids=cands[i].category_ids,
            )
            for rank, i in enumerate(order[: self._top_k])
        ]
        return RankedList(
            user_id=candidates.user_id,
            items=items,
            num_candidates_considered=len(cands),
            intent_type=candidates.request.query_intent.intent_type,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _mf_scores(
        self, cands: list[Candidate], user_vector: list[float],
    ) -> np.ndarray:
        """Direct MF prediction ``user_vector · V[video]`` per candidate.

        Candidates without an ALS row get the median of the known scores
        (neutral: dot products can be negative, so 0 is not neutral).
        Returns all zeros (a dead signal) when the dimensions disagree or
        no candidate has an ALS row.
        """
        n = len(cands)
        u = np.asarray(user_vector, dtype=np.float64)
        if u.shape[0] != self._als_raw.shape[1]:
            logger.warning(
                "user_vector dim %d != item factor dim %d; skipping MF score "
                "(user and item factors come from different ALS runs)",
                u.shape[0], self._als_raw.shape[1],
            )
            return np.zeros(n, dtype=np.float64)

        rows = np.array(
            [self._vid_to_als.get(c.video_id, -1) for c in cands], dtype=np.int64,
        )
        known = rows >= 0
        mf = np.zeros(n, dtype=np.float64)
        if not known.any():
            return mf
        mf[known] = self._als_raw[rows[known]] @ u
        mf[~known] = float(np.median(mf[known]))
        return mf

    def _active_mean(self, raw_signals: list[np.ndarray]) -> np.ndarray:
        """Mean of rank-normalised signals, excluding all-zero signals.

        Signals where every candidate has the same value (e.g. social
        scores when the user has no friends) are dropped so they don't
        dilute the composite.  Returns zeros when *all* signals are dead.
        """
        normed = []
        for arr in raw_signals:
            if arr.max() != arr.min():  # has variance → active
                normed.append(self._rank_normalize(arr))
        if not normed:
            return np.zeros_like(raw_signals[0])
        return np.mean(normed, axis=0)

    @staticmethod
    def _rank_normalize(arr: np.ndarray) -> np.ndarray:
        """Rank-based normalisation to [0, 1].  More robust than min-max
        because it treats signals by relative ordering, not magnitude.
        Ties receive averaged ranks (``scipy.stats.rankdata``).
        Returns zeros when all values are equal."""
        if arr.max() == arr.min():
            return np.zeros_like(arr)
        ranks = rankdata(arr, method="average")
        # Map rank 1..n → 0..1
        return (ranks - 1.0) / (len(ranks) - 1.0)
