"""First-order Markov next-item baseline for session-based recommendation.

The model is a **transition-count table**: it estimates
``P(next_item = j | current_item = i)`` from how often ``i`` is immediately
followed by ``j`` *within a session*. At serving time, given the user's current
session, it scores candidate next videos from the last watched one, backing off
to global popularity when the last item was never seen in training.

Why a from-scratch count model (no new dependency)
--------------------------------------------------
This *is* the standard implementation. The session-rec benchmark framework
(Ludewig & Jannach, 2018) implements its simple, strong baselines — Markov
Chain, Association Rules, Sequential Rules — as exactly this kind of counting,
and their evaluation found these often match or beat neural models. Counting
needs only pandas; adding a heavy framework would clash with this package's
lean dependency set (which pulls in ``implicit`` only for the ALS ``u_long``).

This is the smallest-sufficient temporal model: it conditions on the **last
item only** (the Markov assumption). Use it as the baseline to beat with
GRU4Rec / SASRec later.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from itertools import pairwise

_Scored = list[tuple[int, float]]


class MarkovSessionModel:
    """A first-order Markov (item→item) next-item model.

    Fit on ordered per-session ``video_id`` sequences (from
    :meth:`Sessioniser.to_sequences`), then call :meth:`predict_next`.
    """

    def __init__(self) -> None:
        # current_item -> [(next_item, prob), ...] sorted by prob desc, id asc
        self._transitions: dict[int, _Scored] = {}
        # global popularity fallback: [(item, share), ...] sorted desc
        self._popularity: _Scored = []
        self._n_items = 0

    # ------------------------------------------------------------------
    def fit(self, sequences: list[list[int]]) -> MarkovSessionModel:
        """Count within-session ``(current -> next)`` transitions and item
        popularity. Sessions of length 1 contribute popularity but no
        transition (they have no within-session "next")."""
        transition_counts: dict[int, Counter] = defaultdict(Counter)
        popularity = Counter()

        for seq in sequences:
            for item in seq:
                popularity[item] += 1
            for current, nxt in pairwise(seq):
                transition_counts[current][nxt] += 1

        self._n_items = len(popularity)
        self._transitions = {
            current: _normalise(counter) for current, counter in transition_counts.items()
        }
        self._popularity = _normalise(popularity)
        return self

    # ------------------------------------------------------------------
    def predict_next(
        self,
        recent_items: list[int],
        k: int | None = None,
        exclude: set[int] | None = None,
    ) -> _Scored:
        """Rank candidate next videos given the current session.

        First-order: uses ``recent_items[-1]`` as the context. Falls back to
        global popularity when that item has no learned transitions (cold item)
        or the session is empty. ``exclude`` drops items (e.g. already-seen)
        before truncating to ``k``.
        """
        last = recent_items[-1] if recent_items else None
        ranked = self._transitions.get(last) or self._popularity

        if exclude:
            ranked = [(item, score) for item, score in ranked if item not in exclude]
        else:
            ranked = list(ranked)  # defensive copy (never hand out an internal list)

        return ranked[:k] if k is not None else ranked

    # ------------------------------------------------------------------
    @property
    def n_items(self) -> int:
        """Number of distinct items seen during :meth:`fit`."""
        return self._n_items

    def transitions_for(self, item: int) -> _Scored:
        """The learned next-item distribution for *item* (empty if unseen)."""
        return list(self._transitions.get(item, []))


def _normalise(counter: Counter) -> _Scored:
    """Counter -> [(key, share)] sorted by share desc then key asc (deterministic)."""
    total = sum(counter.values())
    if total == 0:
        return []
    ranked = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))
    return [(int(key), count / total) for key, count in ranked]
