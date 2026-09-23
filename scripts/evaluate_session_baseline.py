#!/usr/bin/env python
"""Train + evaluate the Markov session baseline on KuaiRec, per sessionisation strategy.

No-leakage protocol: sessionise the per-user timeline, split sessions
chronologically (earlier train, later test), fit the first-order Markov on train
sessions, and score next-item prediction over every step of the test sessions.

Usage::

    python scripts/evaluate_session_baseline.py                     # all strategies, full big_matrix
    python scripts/evaluate_session_baseline.py --sample-users 2000 # faster sample
    python scripts/evaluate_session_baseline.py --strategies fixed --gap-minutes 30
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from recsys.data.schema import TIMESTAMP, USER_ID, VIDEO_ID
from recsys.session_modelling import MarkovSessionModel, Sessioniser
from recsys.session_modelling.evaluate import (
    chronological_session_split,
    evaluate_next_item,
)


def _load(data_dir: str, sample_users: int) -> pd.DataFrame:
    path = pathlib.Path(data_dir) / "big_matrix.csv"
    print(f"loading {path} ...", flush=True)
    df = pd.read_csv(path, usecols=[USER_ID, VIDEO_ID, TIMESTAMP])
    if sample_users > 0:
        keep = (
            pd.Series(df[USER_ID].unique())
            .sample(n=min(sample_users, df[USER_ID].nunique()), random_state=42)
        )
        df = df[df[USER_ID].isin(set(keep))]
    print(f"  {len(df):,} interactions, {df[USER_ID].nunique():,} users", flush=True)
    return df


def _make(strategy: str, args: argparse.Namespace) -> Sessioniser:
    return Sessioniser(
        strategy=strategy,
        gap_minutes=args.gap_minutes,
        pct=args.pct,
        floor_minutes=args.floor_minutes,
        min_interactions=args.min_interactions,
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default="data/raw")
    p.add_argument("--sample-users", type=int, default=0, help="0 = all users")
    p.add_argument("--test-ratio", type=float, default=0.2)
    p.add_argument("--strategies", default="fixed,per_user_pct,single")
    p.add_argument("--gap-minutes", type=float, default=30.0)
    p.add_argument("--pct", type=float, default=0.9)
    p.add_argument("--floor-minutes", type=float, default=15.0)
    p.add_argument("--min-interactions", type=int, default=20)
    p.add_argument("--ks", default="10,20")
    args = p.parse_args()

    ks = tuple(int(x) for x in args.ks.split(","))
    df = _load(args.data_dir, args.sample_users)

    rows = []
    for strategy in args.strategies.split(","):
        t0 = time.time()
        sess = _make(strategy, args)
        assigned = sess.assign(df)
        n_sessions = assigned["session_id"].nunique()
        avg_len = len(assigned) / n_sessions
        train, test = chronological_session_split(assigned, test_ratio=args.test_ratio)
        model = MarkovSessionModel().fit(train)
        metrics = evaluate_next_item(model, test, ks=ks)
        dt = time.time() - t0
        print(
            f"\n[{strategy}] {n_sessions:,} sessions "
            f"(avg len {avg_len:.1f}, {n_sessions / df[USER_ID].nunique():.1f}/user) "
            f"| train {len(train):,} / test {len(test):,} sessions "
            f"| {metrics.get('n_eval_pairs', 0):,} eval pairs | {dt:.1f}s"
        )
        line = {"strategy": strategy, "avg_session_len": round(avg_len, 1)}
        for k in ks:
            for m in ("HR", "MRR", "NDCG"):
                print(f"    {m}@{k}: {metrics[f'{m}@{k}']:.4f}")
                line[f"{m}@{k}"] = round(metrics[f"{m}@{k}"], 4)
        rows.append(line)

    print("\n=== summary ===")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
