"""Tests for shared per-video item features."""

import math

import pandas as pd

from recsys.data.item_features import (
    build_item_table,
    caption_category_table,
    video_topic_sets,
)


def test_caption_category_table_keeps_only_known_ids():
    cap = pd.DataFrame({
        "video_id": [1, 2, 3, 4],
        "first_level_category_id": [12, -124, 418, None],  # Food, UNKNOWN, stray, missing
    })
    table = caption_category_table(cap)
    assert dict(zip(table.video_id, table.feat)) == {1: [12], 2: [], 3: [], 4: []}


def test_build_item_table():
    cap = pd.DataFrame({
        "video_id": [1, 2, 3, 4, 5],
        "caption": ["猫咪 #萌宠", None, None, "私密", "广告"],
        "manual_cover_text": ["UNKNOWN"] * 5,
        "topic_tag": ["[萌宠,猫]", "[]", "[]", "[]", "[]"],
        "first_level_category_id": [17, -124, 12, 17, 12],
        "first_level_category_name": ["宠物", "UNKNOWN", "美食", "宠物", "美食"],
        "second_level_category_name": ["宠物日常记录", "UNKNOWN", "美食", "UNKNOWN", "美食"],
        "third_level_category_name": ["宠物猫", "UNKNOWN", "UNKNOWN", "UNKNOWN", "UNKNOWN"],
    })
    # video 1: 1 play in the previous week, 3 in the last week; video 2 only old plays;
    # video 4 was public but is private on its latest day, video 5 is an ad -> both excluded
    daily = pd.DataFrame({
        "video_id": [1, 1, 2, 3, 4, 4, 5],
        "date": [20200901, 20200909, 20200825, 20200825, 20200825, 20200901, 20200825],
        "video_duration": [8000.0, 8000.0, 70000.0, 9000.0, 5000.0, 5000.0, 5000.0],
        "play_cnt": [1, 3, 5, 1, 1, 1, 1],
        "visible_status": ["public"] * 4 + ["public", "private", "public"],
        "video_type": ["NORMAL"] * 6 + ["AD"],
        "video_tag_name": ["萌宠", None, None, "美食", None, None, None],
    })
    t = build_item_table(cap, daily)
    assert list(t.index) == [1, 2, 3]
    # caption text, then de-duplicated category + tag names (UNKNOWN dropped);
    # video 1's tag comes from its latest non-null day
    assert t.loc[1, "text"] == "猫咪 #萌宠 萌宠 猫 宠物 宠物日常记录 宠物猫 萌宠" and t.loc[1, "has_text"]
    assert t.loc[3, "text"] == "美食" and t.loc[3, "has_text"]  # no caption; names de-duplicated
    assert t.loc[2, "text"] == "" and not t.loc[2, "has_text"]  # nothing at all
    assert t.loc[1, "category_id"] == 17 and t.loc[2, "category_id"] == -1
    assert t.loc[2, "duration_ms"] == 70000
    assert t.loc[1, "popularity"] == 4
    assert math.isclose(t.loc[1, "trending"], math.log((3 + 1) / (1 + 1)))
    assert t.loc[2, "trending"] == 0.0  # no plays in either window


def test_video_topic_sets_uses_caption_first_level_ids(tmp_path):
    pd.DataFrame({
        "video_id": [1, 2, 3],
        "first_level_category_id": [11, -124, 2],
    }).to_csv(tmp_path / "kuairec_caption_category.csv", index=False)
    # UNKNOWN (-124) has no topic, so video 2 is absent rather than an empty set
    assert video_topic_sets(str(tmp_path)) == {1: {11}, 3: {2}}
