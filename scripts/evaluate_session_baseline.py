#!/usr/bin/env python
"""Train + evaluate the Markov session baseline on KuaiRec, per sessionisation strategy.

No-leakage protocol: sessionise the per-user timeline, split sessions
chronologically (earlier train, later test), fit the first-order Markov on train
sessions, and score next-item prediction over every step of the test sessions.

Usage::

    python scripts/evaluate_session_baseline.py                     # all strategies, full big_matrix
    python scripts/evaluate_session_baseline.py --sample-users 2000 # faster sample
    python scripts/evaluate_session_baseline.py --strategies fixed --gap-minutes 30
    python scripts/evaluate_session_baseline.py --ablation          # fair shared-test-set comparison
    python scripts/evaluate_session_baseline.py --collapse-repeats  # don't score consecutive rewatches

Every strategy is reported next to a context-free popularity baseline fitted on
the same train sessions, so the Markov model's added value is visible.
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
    PopularityBaseline,
    chronological_session_split,
    evaluate_next_item,
    split_by_timestamp,
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


def _run_ablation(df: pd.DataFrame, args: argparse.Namespace, ks: tuple[int, ...]) -> None:
    """Fair cross-strategy ablation: one global time cutoff gives every strategy
    the **same** held-out test pairs; only the train transition table changes."""
    train_df, test_seqs = split_by_timestamp(df, cutoff_quantile=args.cutoff_quantile)
    n_test_pairs = sum(max(0, len(s) - 1) for s in test_seqs)
    print(
        f"\nablation | global timestamp cutoff q={args.cutoff_quantile} "
        f"| train {len(train_df):,} interactions "
        f"| shared test {n_test_pairs:,} pairs across {len(test_seqs):,} users",
        flush=True,
    )
    rows = []
    for strategy in args.strategies.split(","):
        t0 = time.time()
        train_seqs = _make(strategy, args).to_sequences(train_df)
        model = MarkovSessionModel().fit(train_seqs)
        metrics = evaluate_next_item(
            model, test_seqs, ks=ks, collapse_repeats=args.collapse_repeats
        )
        dt = time.time() - t0
        line = {"strategy": strategy, "train_sessions": len(train_seqs)}
        for k in ks:
            line[f"HR@{k}"] = round(metrics[f"HR@{k}"], 4)
        line["secs"] = round(dt, 1)
        hr = "  ".join(f"HR@{k} {metrics[f'HR@{k}']:.4f}" for k in ks)
        print(f"    [{strategy}] {hr}  ({len(train_seqs):,} train sessions, {dt:.1f}s)")
        rows.append(line)
    print("\n=== ablation summary (shared test set, HR only) ===")
    print(pd.DataFrame(rows).to_string(index=False))


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
    p.add_argument(
        "--ablation",
        action="store_true",
        help="fair cross-strategy comparison on one shared, time-based test set",
    )
    p.add_argument(
        "--cutoff-quantile",
        type=float,
        default=0.8,
        help="global timestamp quantile for the ablation train/test split",
    )
    p.add_argument(
        "--collapse-repeats",
        action="store_true",
        help="merge consecutive rewatches in test sequences before scoring",
    )
    args = p.parse_args()

    ks = tuple(int(x) for x in args.ks.split(","))
    df = _load(args.data_dir, args.sample_users)

    if args.ablation:
        _run_ablation(df, args, ks)
        return

    rows = []
    for strategy in args.strategies.split(","):
        t0 = time.time()
        sess = _make(strategy, args)
        assigned = sess.assign(df)
        n_sessions = assigned["session_id"].nunique()
        avg_len = len(assigned) / n_sessions
        train, test = chronological_session_split(assigned, test_ratio=args.test_ratio)
        model = MarkovSessionModel().fit(train)
        metrics = evaluate_next_item(model, test, ks=ks, collapse_repeats=args.collapse_repeats)
        pop = evaluate_next_item(
            PopularityBaseline(model), test, ks=ks, collapse_repeats=args.collapse_repeats
        )
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
        pop_line = {"strategy": f"{strategy} (popularity)", "avg_session_len": round(avg_len, 1)}
        for k in ks:
            for m in ("HR", "MRR", "NDCG"):
                pop_line[f"{m}@{k}"] = round(pop[f"{m}@{k}"], 4)
        rows.append(pop_line)

    print("\n=== summary ===")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
