#!/usr/bin/env python3
"""Embed every public KuaiRec video's text with bge-m3 (SOCLaaS) for Stage 3 semantic retrieval.

One-off, offline job. Text = caption + manual cover text + topic tags +
first/second/third-level category names + ``video_tag_name``
(``recsys.data.item_features.build_item_table``, which also drops private
videos and ads). A video without any text would get a zero row and
``has_text = False`` (none do on KuaiRec).
Re-run (after deleting any ``.partial.npz``) whenever that text definition changes.

Output: ``data/processed/caption_embeddings.npz`` (gitignored) with
``video_ids`` (int64), ``embeddings`` (float32, L2-normalised, 1024-dim) and
``has_text`` (bool), all row-aligned. Progress is saved after every batch to a
``.partial.npz`` next to it, so an interrupted run resumes where it stopped.

Usage:
    python scripts/build_caption_embeddings.py
    python scripts/build_caption_embeddings.py --batch-size 32 --limit 200   # quick trial
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from recsys.data.item_features import build_item_table
from recsys.data.loader import load_kuairec
from recsys.intent.embeddings import BgeM3Embedder

_RETRIES = 3


def _embed(embedder: BgeM3Embedder, texts: list[str]) -> list[list[float]]:
    """embed_batch swallows API errors and returns empty vectors — retry
    instead of silently saving zeros."""
    for attempt in range(1, _RETRIES + 1):
        vecs = embedder.embed_batch(texts)
        if vecs and all(vecs):
            return vecs
        time.sleep(2 ** attempt)
    raise RuntimeError(f"embedding failed {_RETRIES}x for a batch starting {texts[0][:40]!r}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default="data/raw")
    p.add_argument("--out", default="data/processed/caption_embeddings.npz")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--limit", type=int, default=0, help="only embed the first N texts (trial)")
    args = p.parse_args()

    out = pathlib.Path(args.out)
    partial = out.with_suffix(".partial.npz")
    out.parent.mkdir(parents=True, exist_ok=True)

    data = load_kuairec(args.data_dir)
    items = build_item_table(data.caption_categories, data.item_daily_features)
    todo = items[items["has_text"]]
    if args.limit:
        todo = todo.head(args.limit)

    done: dict[int, np.ndarray] = {}
    if partial.exists():
        z = np.load(partial)
        done = dict(zip(z["video_ids"].tolist(), z["embeddings"]))
        print(f"resuming: {len(done):,} already embedded")

    remaining = [(int(v), t) for v, t in zip(todo.index, todo["text"]) if int(v) not in done]
    print(f"{len(items):,} videos, {len(todo):,} with text, {len(remaining):,} to embed")

    embedder = BgeM3Embedder()
    t0 = time.time()
    for start in range(0, len(remaining), args.batch_size):
        batch = remaining[start:start + args.batch_size]
        for (vid, _), vec in zip(batch, _embed(embedder, [t for _, t in batch])):
            done[vid] = np.asarray(vec, dtype=np.float32)
        np.savez(partial, video_ids=np.array(list(done)), embeddings=np.stack(list(done.values())))
        n = start + len(batch)
        print(f"  {n:,}/{len(remaining):,}  ({time.time() - t0:.0f}s)", flush=True)

    dim = len(next(iter(done.values())))
    video_ids = items.index.to_numpy(dtype=np.int64)
    emb = np.zeros((len(video_ids), dim), dtype=np.float32)
    has_text = np.zeros(len(video_ids), dtype=bool)
    for row, vid in enumerate(video_ids):
        if vid in done:
            vec = done[vid]
            emb[row] = vec / (np.linalg.norm(vec) or 1.0)
            has_text[row] = True
    np.savez(out, video_ids=video_ids, embeddings=emb, has_text=has_text)
    partial.unlink(missing_ok=True)
    print(f"saved {out} — {has_text.sum():,}/{len(video_ids):,} videos embedded, dim {dim}")


if __name__ == "__main__":
    main()
