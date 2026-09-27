"""QCSARRanker — three-stage cascade: initial ranking, item-item re-ranking,
and constraint enforcement.

Implements the ``Ranker`` protocol from ``recsys.pipeline.pipeline``::

    Search Query  →  content-based (Stage 3 score)
                  →  + CF / social / Markov (Stage 4)
                  →  + item-item similarity re-ranking (Stage 5)
                  →  + diversity / demotion / top-k (Stage 6)
                  →  RankedList
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from scipy.stats import rankdata

from recsys.contracts.enums import QueryIntentType
from recsys.contracts.models import (
    CandidateSet,
    RankedItem,
    RankedList,
)

if TYPE_CHECKING:
    from recsys.contracts.models import Candidate, EngagementProfile

logger = logging.getLogger(__name__)


class QCSARRanker:
    """Stages 4-6 ranker for the QCSAR pipeline.

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
    w_query, w_query_sem, w_popularity, w_cf, w_social, w_session_markov,
    w_cat_affinity : float
        Stage 4 signal weights (should sum to 1.0).
    w_item_als, w_item_caption, w_session_vector : float
        Stage 5 item-item signal weights (should sum to 1.0).
    stage5_alpha : float
        Mixing ratio: ``alpha * stage4 + (1 - alpha) * stage5``.
    seen_demotion : float
        Fallback multiplicative penalty for already-seen videos (Stage 6).
        Overridden per-query by ``INTENT_SEEN_DEMOTION`` when the query's
        intent type has an entry.
    diversity_window : int
        Sliding window size for category-diversity enforcement (Stage 6).
    top_k : int
        Maximum items in the final ``RankedList``.
    """

    # Per-intent seen-video demotion.  SPECIFIC_SEARCH preserves seen
    # videos (the user is re-finding); EXPLORATION aggressively demotes
    # them (novelty is the goal).  Missing intents fall back to the
    # constructor's ``seen_demotion``.
    INTENT_SEEN_DEMOTION: dict[QueryIntentType, float] = {
        QueryIntentType.SPECIFIC_SEARCH: 1.0,   # no penalty — re-finding
        QueryIntentType.LEARNING:        0.8,    # mild — rewatching tutorials is normal
        QueryIntentType.ENTERTAINMENT:   0.5,    # default — fresh content preferred
        QueryIntentType.INSPIRATION:     0.5,    # default
        QueryIntentType.SOCIAL:          0.5,    # default
        QueryIntentType.EXPLORATION:     0.3,    # aggressive — novelty is the point
    }

    def __init__(
        self,
        item_factors: np.ndarray,
        item_ids: np.ndarray,
        caption_embeddings: np.ndarray | None = None,
        caption_video_ids: np.ndarray | None = None,
        *,
        # Stage 4 weights (tuned via scripts/tune_weights.py)
        w_query: float = 0.50,
        w_query_sem: float = 0.05,
        w_popularity: float = 0.20,
        w_cf: float = 0.05,
        w_cat_affinity: float = 0.05,
        w_social: float = 0.075,
        w_session_markov: float = 0.075,
        # Stage 5 weights
        w_item_als: float = 0.50,
        w_item_caption: float = 0.30,
        w_session_vector: float = 0.20,
        stage5_alpha: float = 0.70,
        # Stage 6 knobs
        seen_demotion: float = 0.5,
        diversity_window: int = 3,
        top_k: int = 20,
    ) -> None:
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

        # Stage 4
        self._w_query = w_query
        self._w_query_sem = w_query_sem
        self._w_pop = w_popularity
        self._w_cf = w_cf
        self._w_cat_aff = w_cat_affinity
        self._w_social = w_social
        self._w_markov = w_session_markov

        # Stage 5
        self._w_item_als = w_item_als
        self._w_item_cap = w_item_caption
        self._w_sess_vec = w_session_vector
        self._alpha = stage5_alpha

        # Stage 6
        self._seen_demotion = seen_demotion
        self._div_window = diversity_window
        self._top_k = top_k

    # ------------------------------------------------------------------
    # Public API (Ranker protocol)
    # ------------------------------------------------------------------

    def rank(self, candidates: CandidateSet) -> RankedList:
        """Run the three-stage ranking cascade and return a ``RankedList``."""
        if not candidates.candidates:
            return RankedList(
                user_id=candidates.user_id,
                items=[],
                num_candidates_considered=0,
                intent_type=candidates.request.query_intent.intent_type,
            )

        profile = candidates.request.engagement_profile
        scored = self._stage4_initial_rank(candidates)
        scored = self._stage5_rerank(scored, profile)
        return self._stage6_constrain(scored, candidates)

    # ------------------------------------------------------------------
    # Stage 4 — weighted multi-signal scoring
    # ------------------------------------------------------------------

    def _stage4_initial_rank(
        self,
        candidates: CandidateSet,
    ) -> list[tuple[Candidate, float]]:
        cands = candidates.candidates
        profile = candidates.request.engagement_profile
        n = len(cands)

        # 1. Query relevance (from Stage 3 composite score)
        query_arr = np.array([c.score for c in cands], dtype=np.float64)

        # 2. Direct query↔caption semantic similarity
        qsem_arr = np.zeros(n, dtype=np.float64)
        query_emb = candidates.request.query_intent.query_embedding
        if query_emb is not None and self._emb is not None and self._vid_to_emb is not None:
            qv = np.array(query_emb, dtype=np.float64)
            qv_norm = np.linalg.norm(qv)
            if qv_norm > 0:
                qv = qv / qv_norm
                for i, c in enumerate(cands):
                    row = self._vid_to_emb.get(c.video_id, -1)
                    if row != -1:
                        val = float(self._emb[row] @ qv)
                        if np.isfinite(val):
                            qsem_arr[i] = val

        # 3. Popularity + trending
        pop_arr = np.array(
            [c.features.get("popularity", 0.0) for c in cands], dtype=np.float64,
        )
        trend_arr = np.array(
            [c.features.get("trending", 0.0) for c in cands], dtype=np.float64,
        )
        pop_combined = (
            0.7 * self._rank_normalize(pop_arr) + 0.3 * self._rank_normalize(trend_arr)
        )

        # 4-7. Personalisation signals (zero when anonymous)
        cf_arr = np.zeros(n, dtype=np.float64)
        social_arr = np.zeros(n, dtype=np.float64)
        markov_arr = np.zeros(n, dtype=np.float64)
        cat_aff_arr = np.zeros(n, dtype=np.float64)

        if profile is not None:
            cf_scores = profile.cf_neighbour_scores
            soc_scores = profile.social_scores
            cat_dist = profile.category_distribution
            for i, c in enumerate(cands):
                cf_arr[i] = cf_scores.get(c.video_id, 0.0)
                social_arr[i] = soc_scores.get(c.video_id, 0.0)
                # User-category affinity: how much of the user's history is
                # in this candidate's primary category.
                if c.category_ids and cat_dist:
                    cat_aff_arr[i] = cat_dist.get(c.category_ids[0], 0.0)

            ctx = profile.session_context
            if ctx is not None:
                nxt = ctx.next_item_scores
                for i, c in enumerate(cands):
                    markov_arr[i] = nxt.get(c.video_id, 0.0)

        # Rank-based normalisation — robust to outliers
        query_n = self._rank_normalize(query_arr)
        qsem_n = self._rank_normalize(qsem_arr)
        pop_n = self._rank_normalize(pop_combined)
        cf_n = self._rank_normalize(cf_arr)
        cat_aff_n = self._rank_normalize(cat_aff_arr)
        social_n = self._rank_normalize(social_arr)
        markov_n = self._rank_normalize(markov_arr)

        # Weighted blend
        scores = (
            self._w_query * query_n
            + self._w_query_sem * qsem_n
            + self._w_pop * pop_n
            + self._w_cf * cf_n
            + self._w_cat_aff * cat_aff_n
            + self._w_social * social_n
            + self._w_markov * markov_n
        )

        # Sort descending (stable)
        order = np.argsort(-scores, kind="stable")
        return [(cands[i], float(scores[i])) for i in order]

    # ------------------------------------------------------------------
    # Stage 5 — item-item similarity re-ranking
    # ------------------------------------------------------------------

    def _stage5_rerank(
        self,
        scored: list[tuple[Candidate, float]],
        profile: EngagementProfile | None,
    ) -> list[tuple[Candidate, float]]:
        # Skip when there is no session context to compare against.
        if profile is None or profile.session_context is None:
            return scored
        recent = profile.session_context.recent_video_ids
        if not recent:
            return scored

        n = len(scored)

        # --- ALS item-item similarity ---
        # Gather normalised ALS vectors for recent videos.
        recent_rows = [self._vid_to_als[v] for v in recent if v in self._vid_to_als]
        als_sim = np.zeros(n, dtype=np.float64)
        if recent_rows:
            recent_mat = self._als_normed[recent_rows]  # (n_recent, k)
            cand_rows = [
                self._vid_to_als.get(c.video_id, -1) for c, _ in scored
            ]
            for i, row in enumerate(cand_rows):
                if row == -1:
                    continue
                sims = recent_mat @ self._als_normed[row]  # (n_recent,)
                # Mean of top-3 (or fewer)
                top_k = min(3, len(sims))
                als_sim[i] = float(np.sort(sims)[-top_k:].mean())

        # --- Caption embedding similarity (optional) ---
        cap_sim = np.zeros(n, dtype=np.float64)
        has_cap = False
        if self._emb is not None and self._vid_to_emb is not None:
            recent_emb_rows = [
                self._vid_to_emb[v] for v in recent if v in self._vid_to_emb
            ]
            if recent_emb_rows:
                has_cap = True
                recent_emb = self._emb[recent_emb_rows]  # (n_recent, 1024)
                for i, (c, _) in enumerate(scored):
                    erow = self._vid_to_emb.get(c.video_id, -1)
                    if erow == -1:
                        continue
                    sims = recent_emb @ self._emb[erow]
                    top_k = min(3, len(sims))
                    cap_sim[i] = float(np.sort(sims)[-top_k:].mean())

        # --- Session vector similarity ---
        sess_vec = profile.session_context.session_vector
        sess_sim = np.zeros(n, dtype=np.float64)
        if sess_vec:
            sv = np.array(sess_vec, dtype=np.float64)
            sv_norm = np.linalg.norm(sv)
            if sv_norm > 0:
                sv = sv / sv_norm
                for i, (c, _) in enumerate(scored):
                    row = self._vid_to_als.get(c.video_id, -1)
                    if row == -1:
                        continue
                    sess_sim[i] = float(self._als_normed[row] @ sv)

        # Rank-based normalisation
        als_n = self._rank_normalize(als_sim)
        cap_n = self._rank_normalize(cap_sim)
        sess_n = self._rank_normalize(sess_sim)

        # Compute weights (redistribute caption weight when absent)
        w_als = self._w_item_als
        w_cap = self._w_item_cap if has_cap else 0.0
        w_sess = self._w_sess_vec
        if not has_cap:
            total_other = w_als + w_sess
            if total_other > 0:
                w_als = w_als / total_other
                w_sess = w_sess / total_other

        stage5 = w_als * als_n + w_cap * cap_n + w_sess * sess_n

        # Blend with Stage 4 scores
        stage4_scores = np.array([s for _, s in scored], dtype=np.float64)
        blended = self._alpha * stage4_scores + (1 - self._alpha) * stage5

        order = np.argsort(-blended, kind="stable")
        return [(scored[i][0], float(blended[i])) for i in order]

    # ------------------------------------------------------------------
    # Stage 6 — constraints, diversity, top-k
    # ------------------------------------------------------------------

    def _stage6_constrain(
        self,
        scored: list[tuple[Candidate, float]],
        candidates: CandidateSet,
    ) -> RankedList:
        intent = candidates.request.query_intent

        # 1. Negative constraint penalties (duration heuristics)
        neg = intent.negative_constraints or []
        has_long = any(kw in c.lower() for c in neg for kw in ("long", "lengthy"))
        has_short = any(kw in c.lower() for c in neg for kw in ("short", "brief"))
        adjusted: list[tuple[Candidate, float]] = []
        for cand, score in scored:
            dur_fit = cand.features.get("dur_fit", 1.0)
            if has_long and dur_fit < 0.5:
                score *= 0.8  # penalise long videos (low dur_fit ↔ poor match)
            if has_short and dur_fit < 0.5:
                score *= 0.8
            adjusted.append((cand, score))

        # 2. Seen-video demotion — intent-conditioned
        seen_penalty = self.INTENT_SEEN_DEMOTION.get(
            intent.intent_type, self._seen_demotion,
        )
        for i, (cand, score) in enumerate(adjusted):
            if cand.features.get("seen", 0.0) >= 1.0:
                adjusted[i] = (cand, score * seen_penalty)

        # 3. Category diversity (sliding-window penalty)
        # Sort by score first so the window is applied in rank order.
        adjusted.sort(key=lambda cs: (-cs[1], cs[0].video_id))
        recent_cats: list[int] = []
        diversified: list[tuple[Candidate, float]] = []
        for cand, score in adjusted:
            primary_cat = cand.category_ids[0] if cand.category_ids else -1
            if primary_cat in recent_cats:
                score *= 0.85  # 15 % diversity penalty
            diversified.append((cand, score))
            recent_cats.append(primary_cat)
            if len(recent_cats) > self._div_window:
                recent_cats.pop(0)

        # 4. Final sort, top-k, and build RankedList
        diversified.sort(key=lambda cs: (-cs[1], cs[0].video_id))
        items = [
            RankedItem(
                video_id=c.video_id,
                rank=i + 1,
                score=s,
                source=c.source,
                category_ids=c.category_ids,
            )
            for i, (c, s) in enumerate(diversified[: self._top_k])
        ]
        return RankedList(
            user_id=candidates.user_id,
            items=items,
            num_candidates_considered=len(candidates.candidates),
            intent_type=intent.intent_type,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

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
