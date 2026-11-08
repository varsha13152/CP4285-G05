"""Stage 3 — query-only candidate retrieval for video search.

``HybridRetriever`` implements the ``Retriever`` protocol
(:mod:`recsys.pipeline.pipeline`). It finds the ``num_candidates`` videos most
relevant to the **query** and nothing else: personalisation (CF, session,
social, popularity) is Stage 4's job, so a search never pulls in off-topic
videos just because the user liked something similar.

Relevance per video (all computed with numpy over the whole catalogue)::

    sem  = cosine(query_embedding, caption_embedding)        # bge-m3, 0 if no text
    cat  = confidence of the query topic matching the video's caption category
    dur  = 1 inside duration_range_ms, else exp(-seconds_outside / 15)   # soft
    score = w_sem·sem + w_cat·cat + w_dur·dur                 # per mode, below

Modes (benchmarked against each other in ``scripts/evaluate_stage3.py``):

* ``semantic`` — videos with text, ``w_sem·sem + w_dur·dur``
* ``category`` — videos whose category matches a query topic, ``w_cat·cat + w_dur·dur``
* ``hybrid``   — either of the above, all three terms. The category source is
  exact-topic match (category and tag names are embedded too, so every
  searchable video has text).

A query without an embedding falls back to ``category``. Ties are broken by
``video_id`` so results are deterministic. Popularity and trending are attached
as features for Stage 4 but never used to retrieve or order.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

from recsys.contracts.enums import RetrievalSource
from recsys.contracts.models import Candidate, CandidateSet, QueryRetrievalRequest
from recsys.data.item_features import build_item_table

MODES = ("semantic", "category", "hybrid")


def load_catalogue(
    data_dir: str = "data/raw",
    embeddings_path: str = "data/processed/caption_embeddings.npz",
) -> tuple[pd.DataFrame, np.ndarray]:
    """The item table and its row-aligned caption embeddings."""
    root = pathlib.Path(data_dir)
    # Same read options as recsys.data.loader, without loading the big matrices.
    captions = pd.read_csv(
        root / "kuairec_caption_category.csv", engine="python", on_bad_lines="skip"
    )
    daily = pd.read_csv(root / "item_daily_features.csv")
    items = build_item_table(captions, daily)
    with np.load(embeddings_path) as z:  # read each array once — z[...] re-reads
        cached = pd.DataFrame(z["embeddings"], index=z["video_ids"])
    emb = cached.reindex(items.index, fill_value=0.0).to_numpy(dtype=np.float32)
    # A video only counts as having text if it was actually embedded.
    items = items.assign(has_text=items["has_text"] & np.linalg.norm(emb, axis=1).astype(bool))
    return items, emb


class HybridRetriever:
    """Query-relevance retrieval over caption embeddings and caption categories.

    Parameters
    ----------
    items : pandas.DataFrame
        Output of :func:`~recsys.data.item_features.build_item_table`, indexed
        by ``video_id``.
    embeddings : numpy.ndarray
        ``(len(items), dim)`` L2-normalised caption embeddings, row-aligned
        with *items* (zero rows where ``has_text`` is False).
    mode : {"semantic", "category", "hybrid"}
    w_sem, w_cat, w_dur : float
        Weights of the semantic, category and duration terms. ``w_cat`` is
        0.1 (was 0.5): at 0.5 the topic bonus outweighed the semantic spread of
        the whole pool, so a parser topic that is a neighbouring category pushed
        the answers out (``scripts/tune_topic_weight.py``).
    duration_scale_s : float
        Seconds outside the requested duration range at which the duration fit
        drops to ``1/e`` (soft target, never a filter).
    """

    def __init__(
        self,
        items: pd.DataFrame,
        embeddings: np.ndarray,
        mode: str = "hybrid",
        w_sem: float = 1.0,
        w_cat: float = 0.1,
        w_dur: float = 0.2,
        duration_scale_s: float = 15.0,
    ) -> None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        if len(embeddings) != len(items):
            raise ValueError("embeddings must be row-aligned with items")
        self.mode = mode
        self.w_sem, self.w_cat, self.w_dur = w_sem, w_cat, w_dur
        self._scale_ms = duration_scale_s * 1000.0
        self._ids = items.index.to_numpy(dtype=np.int64)
        self._emb = np.asarray(embeddings, dtype=np.float32)
        self._cat = items["category_id"].to_numpy(dtype=np.int64)
        self._dur = items["duration_ms"].to_numpy(dtype=np.float64)
        self._has_text = items["has_text"].to_numpy(dtype=bool)
        self._pop = items["popularity"].to_numpy(dtype=np.float64)
        self._trend = items["trending"].to_numpy(dtype=np.float64)

    @classmethod
    def from_disk(
        cls,
        data_dir: str = "data/raw",
        embeddings_path: str = "data/processed/caption_embeddings.npz",
        **kwargs,
    ) -> HybridRetriever:
        """Build from the raw CSVs + the cached embeddings
        (``scripts/build_caption_embeddings.py``)."""
        return cls(*load_catalogue(data_dir, embeddings_path), **kwargs)

    # ------------------------------------------------------------------
    def retrieve(self, request: QueryRetrievalRequest) -> CandidateSet:
        intent = request.query_intent
        topics = set(request.category_filter or [])
        conf = {t.category_id: t.confidence for t in intent.topics if t.category_id in topics}

        cat = np.array([conf.get(int(c), 0.0) for c in self._cat])
        dur = self._duration_fit(request.duration_range_ms)
        sem = np.zeros(len(self._ids))
        mode = self.mode
        if intent.query_embedding:
            q = np.asarray(intent.query_embedding, dtype=np.float32)
            sem = self._emb @ (q / (np.linalg.norm(q) or 1.0))
        elif mode != "category":
            mode = "category"  # nothing to compare captions against

        if mode == "semantic":
            score, eligible = self.w_sem * sem + self.w_dur * dur, self._has_text
        elif mode == "category":
            score, eligible = self.w_cat * cat + self.w_dur * dur, cat > 0
        else:
            score = self.w_sem * sem + self.w_cat * cat + self.w_dur * dur
            eligible = self._has_text | (cat > 0)

        idx = np.flatnonzero(eligible)
        # Highest score first; ties broken by video_id for determinism.
        idx = idx[np.lexsort((self._ids[idx], -score[idx]))][: request.num_candidates]

        seen = (
            request.engagement_profile.seen_video_ids if request.engagement_profile else set()
        )
        candidates = [self._candidate(i, score, sem, cat, dur, seen) for i in idx]
        counts: dict[str, int] = {}
        for c in candidates:
            counts[c.source.name] = counts.get(c.source.name, 0) + 1
        return CandidateSet(
            user_id=request.user_id,
            request=request,
            candidates=candidates,
            source_counts=counts,
        )

    # ------------------------------------------------------------------
    def _duration_fit(self, rng: tuple[int, int] | None) -> np.ndarray:
        if rng is None:
            return np.ones(len(self._ids))
        lo, hi = rng
        outside = np.maximum(lo - self._dur, 0) + np.maximum(self._dur - hi, 0)
        fit = np.exp(-outside / self._scale_ms)
        fit[self._dur <= 0] = 1.0  # unknown duration: don't penalise
        return fit

    def _candidate(self, i, score, sem, cat, dur, seen) -> Candidate:
        if cat[i] > 0:
            source = RetrievalSource.BOTH if self._has_text[i] else RetrievalSource.CATEGORY
        else:
            source = RetrievalSource.SEMANTIC
        vid = int(self._ids[i])
        return Candidate(
            video_id=vid,
            score=float(score[i]),
            source=source,
            category_ids=[int(self._cat[i])] if self._cat[i] >= 0 else [],
            video_duration_ms=int(self._dur[i]),
            features={
                "sem": float(sem[i]),
                "cat": float(cat[i]),
                "dur_fit": float(dur[i]),
                "has_text": float(self._has_text[i]),
                "seen": float(vid in seen),
                "popularity": float(self._pop[i]),
                "trending": float(self._trend[i]),
            },
        )
