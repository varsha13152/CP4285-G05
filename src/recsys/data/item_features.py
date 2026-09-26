"""Per-video item features shared by Stage 2, Stage 3 and Stage 4.

Categories use the **caption first-level category id** (``kuairec_caption_category.csv``),
the same id space as Stage 1's ``CategoryIndex``. The ``feat`` tags in
``item_categories.csv`` are a different id space (0–30) and are not used.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from recsys.data.category_index import CategoryIndex
from recsys.data.schema import (
    CAPTION,
    DATE,
    FIRST_CATEGORY,
    PLAY_CNT,
    SECOND_CATEGORY,
    THIRD_CATEGORY,
    TOPIC_TAG,
    VIDEO_DURATION,
    VIDEO_ID,
)

_FIRST_LEVEL_ID = "first_level_category_id"
_COVER_TEXT = "manual_cover_text"
_VISIBLE_STATUS = "visible_status"
_VIDEO_TYPE = "video_type"
_VIDEO_TAG_NAME = "video_tag_name"
_TRENDING_WINDOW_DAYS = 7
# Placeholder values KuaiRec uses for "no text".
_EMPTY_TEXT = re.compile(r"^(UNKNOWN|nan|\[\])$")


def _clean(value: object) -> str:
    text = "" if pd.isna(value) else str(value).strip()
    return "" if _EMPTY_TEXT.match(text) else text


def _video_text(caption: object, cover: object, tags: object) -> str:
    """caption + cover text + topic tags (``[a,b]`` → ``a b``), empties dropped."""
    tag_text = _clean(tags).strip("[]").replace(",", " ")
    return " ".join(p for p in (_clean(caption), _clean(cover), tag_text) if p)


def _category_text(*names: object) -> str:
    """Category / tag names, ``UNKNOWN`` dropped, de-duplicated."""
    kept: list[str] = []
    for name in map(_clean, names):
        if name and name not in kept:
            kept.append(name)
    return " ".join(kept)


def build_item_table(
    caption_categories: pd.DataFrame, item_daily_features: pd.DataFrame
) -> pd.DataFrame:
    """One row per video, indexed by ``video_id``:

    * ``category_id`` — caption first-level id (``-1`` if outside ``CategoryIndex``)
    * ``duration_ms`` — median ``video_duration`` (``0`` = unknown)
    * ``text`` / ``has_text`` — what Stage 3 embeds: caption + cover text +
      topic tags + first/second/third-level category names + ``video_tag_name``
      (names de-duplicated, ``UNKNOWN`` dropped). The names give the ~9% of
      videos without any caption text an embedding too.
    * ``popularity`` — total ``play_cnt`` over the dataset
    * ``trending`` — ``log((last 7 days play_cnt + 1) / (previous 7 days + 1))``;
      0 = flat, > 0 rising, < 0 falling

    Only searchable videos are kept: **latest** ``visible_status`` is ``public``
    (search must never surface private or friends-only videos, 7.4% of KuaiRec)
    and ``video_type`` is ``NORMAL`` (ads excluded).

    ``popularity`` and ``trending`` are Stage 4 ranking features only; Stage 3
    never retrieves or orders by them.
    """
    cats = caption_category_table(caption_categories).set_index(VIDEO_ID)["feat"]
    cap = caption_categories.assign(
        **{VIDEO_ID: pd.to_numeric(caption_categories[VIDEO_ID], errors="coerce")}
    ).dropna(subset=[VIDEO_ID]).drop_duplicates(VIDEO_ID)
    cap = cap.set_index(cap[VIDEO_ID].astype(int))

    daily = item_daily_features
    # Latest non-null value per video (status and tags can change day to day).
    latest = daily.sort_values(DATE).groupby(VIDEO_ID)[
        [_VISIBLE_STATUS, _VIDEO_TYPE, _VIDEO_TAG_NAME]
    ].last()
    tag_name = latest[_VIDEO_TAG_NAME].reindex(cap.index)
    text = pd.Series(
        [
            " ".join(p for p in (_video_text(c, m, t), _category_text(l1, l2, l3, tag)) if p)
            for c, m, t, l1, l2, l3, tag in zip(
                cap[CAPTION], cap[_COVER_TEXT], cap[TOPIC_TAG],
                cap[FIRST_CATEGORY], cap[SECOND_CATEGORY], cap[THIRD_CATEGORY], tag_name,
            )
        ],
        index=cap.index,
    )

    grouped = daily.groupby(VIDEO_ID)
    duration = grouped[VIDEO_DURATION].median()
    popularity = grouped[PLAY_CNT].sum()
    days = pd.to_datetime(daily[DATE].astype(str), format="%Y%m%d")
    last = days.max()
    recent = days > last - pd.Timedelta(days=_TRENDING_WINDOW_DAYS)
    previous = ~recent & (days > last - pd.Timedelta(days=2 * _TRENDING_WINDOW_DAYS))
    recent_cnt = daily[recent].groupby(VIDEO_ID)[PLAY_CNT].sum()
    previous_cnt = daily[previous].groupby(VIDEO_ID)[PLAY_CNT].sum()

    ids = text.index.union(duration.index)
    searchable = latest.reindex(ids)
    ids = ids[
        (searchable[_VISIBLE_STATUS].eq("public") & searchable[_VIDEO_TYPE].eq("NORMAL"))
        .to_numpy()
    ]
    table = pd.DataFrame(index=ids.rename(VIDEO_ID))
    table["category_id"] = [cats.get(v, [])[0] if cats.get(v) else -1 for v in ids]
    table["duration_ms"] = duration.reindex(ids).fillna(0).astype(np.int64)
    table["text"] = text.reindex(ids).fillna("")
    table["has_text"] = table["text"] != ""
    table["popularity"] = popularity.reindex(ids).fillna(0).astype(float)
    # Log ratio: a raw ratio explodes (to ~5.7e7) for videos uploaded inside the
    # last week, whose previous-week count is 0.
    table["trending"] = np.log(
        (recent_cnt.reindex(ids).fillna(0) + 1) / (previous_cnt.reindex(ids).fillna(0) + 1)
    )
    return table


def caption_category_table(caption_categories: pd.DataFrame) -> pd.DataFrame:
    """``video_id`` → ``feat`` (a one-element list holding the caption
    first-level category id), in the same shape as ``item_categories`` so it
    drops into code that expects that table.

    Ids outside ``CategoryIndex`` (-124 "UNKNOWN", and the one-video strays
    418 / 667) get an empty list.
    """
    valid = CategoryIndex().valid_ids()
    ids = pd.to_numeric(caption_categories[_FIRST_LEVEL_ID], errors="coerce")
    vids = pd.to_numeric(caption_categories[VIDEO_ID], errors="coerce")
    table = pd.DataFrame({
        VIDEO_ID: vids,
        "feat": [[int(c)] if pd.notna(c) and int(c) in valid else [] for c in ids],
    }).dropna(subset=[VIDEO_ID])
    table[VIDEO_ID] = table[VIDEO_ID].astype(int)
    return table.drop_duplicates(VIDEO_ID).reset_index(drop=True)
