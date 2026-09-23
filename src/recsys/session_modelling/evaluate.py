"""Offline next-item evaluation for the session baseline.

No-future-leakage by construction: sessions are split **chronologically**
(earlier sessions train, later sessions test), so the model never sees an
interaction that happened after a test interaction.

Metrics are the single-target ranking family (each test step has exactly one
true next item), reported at cut-offs ``k``:

* ``HR@k``   — Hit Rate (= Recall@k for a single target): was the true next
  item in the top-k? The headline "did we get it" number.
* ``MRR@k``  — Mean Reciprocal Rank: ``1/rank`` of the true item (0 beyond k).
  Position-sensitive — rewards ranking it high.
* ``NDCG@k`` — ``1/log2(rank+1)`` (0 beyond k); for a single target this is a
  smooth, standard position-weighted score.

``precision@k`` is *not* reported: with one relevant item it is capped at
``1/k`` and carries no extra signal over HR@k.

These are the next-item specialisation of :mod:`recsys.evaluation.metrics`
(which operates on full-pipeline ``RankedList`` objects); computed directly here
for speed over many test steps.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from recsys.data.schema import TIMESTAMP, VIDEO_ID
from recsys.session_modelling.markov import MarkovSessionModel

_SESSION_ID = "session_id"


def chronological_session_split(
    df_assigned: pd.DataFrame, test_ratio: float = 0.2
) -> tuple[list[list[int]], list[list[int]]]:
    """Split assigned interactions into (train, test) session sequences by each
    session's **start time** — the latest ``test_ratio`` of sessions are test.

    Parameters
    ----------
    df_assigned : pandas.DataFrame
        Output of :meth:`Sessioniser.assign` (has ``session_id`` + ``timestamp``).
    """
    starts = df_assigned.groupby(_SESSION_ID, sort=False)[TIMESTAMP].min().sort_values()
    n_test = int(len(starts) * test_ratio)
    n_train = len(starts) - n_test
    train_ids = set(starts.index[:n_train])
    test_ids = set(starts.index[n_train:])

    def sequences(ids: set[int]) -> list[list[int]]:
        sub = df_assigned[df_assigned[_SESSION_ID].isin(ids)]
        return [
            [int(v) for v in g[VIDEO_ID].to_numpy()]
            for _, g in sub.groupby(_SESSION_ID, sort=True)
        ]

    return sequences(train_ids), sequences(test_ids)


def evaluate_next_item(
    model: MarkovSessionModel,
    test_sequences: list[list[int]],
    ks: tuple[int, ...] = (10, 20),
    exclude_seen: bool = False,
) -> dict[str, float]:
    """Average HR@k / MRR@k / NDCG@k over every ``(context, target)`` step of
    every test session.

    For each session ``[i0, i1, ...]`` and each position ``t >= 1``, the context
    is ``i[t-1]`` (first-order) and the target is ``i[t]``.
    """
    max_k = max(ks)
    hits = dict.fromkeys(ks, 0)
    mrr = dict.fromkeys(ks, 0.0)
    ndcg = dict.fromkeys(ks, 0.0)
    n_pairs = 0
    cache: dict[int, list[tuple[int, float]]] = {}

    for seq in test_sequences:
        for t in range(1, len(seq)):
            context, target = seq[t - 1], seq[t]
            if exclude_seen:
                ranked = model.predict_next([context], k=max_k, exclude=set(seq[:t]))
            elif context in cache:
                ranked = cache[context]
            else:
                ranked = model.predict_next([context], k=max_k)
                cache[context] = ranked

            rank = next(
                (pos for pos, (item, _) in enumerate(ranked, start=1) if item == target),
                None,
            )
            n_pairs += 1
            if rank is not None:
                for k in ks:
                    if rank <= k:
                        hits[k] += 1
                        mrr[k] += 1.0 / rank
                        ndcg[k] += 1.0 / np.log2(rank + 1)

    if n_pairs == 0:
        return {"n_eval_pairs": 0}

    out: dict[str, float] = {}
    for k in ks:
        out[f"HR@{k}"] = hits[k] / n_pairs
        out[f"MRR@{k}"] = mrr[k] / n_pairs
        out[f"NDCG@{k}"] = ndcg[k] / n_pairs
    out["n_eval_pairs"] = n_pairs
    return out
