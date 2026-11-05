# Data Layer

Shared loading, preprocessing and item features for KuaiRec, plus a field-level
reference for every file. Profiled from the real files on 2026-09-25.

## Usage

```python
from recsys.data import load_kuairec
from recsys.data.preprocessing import clip_watch_ratio, temporal_train_test_split
from recsys.data.item_features import build_item_table

data = load_kuairec("data/raw")                  # load_big=True adds big_matrix (~1 GB)
clean = clip_watch_ratio(data.small_matrix)      # caps watch_ratio at 5.0
train, test = temporal_train_test_split(clean)
items = build_item_table(data.caption_categories, data.item_daily_features)
```

Import column names from `recsys.data.schema` (`USER_ID`, `VIDEO_ID`,
`WATCH_RATIO`, ...) instead of hardcoding strings.

**KuaiSAR (optional).** Only needed to re-run `scripts/analyze_kuaisar_queries.py`
(the profile it writes, `results/kuaisar_query_profile.json`, is committed).
Download KuaiSAR-small (`KuaiSAR.zip`, https://zenodo.org/record/8181109) and
unpack its CSVs into `data/external/kuaisar/` (gitignored).

## KuaiRec at a glance

KuaiRec 2.0 ([GitHub](https://github.com/chongminggao/KuaiRec), data on
[Zenodo](https://zenodo.org/records/18164998)) — Kuaishou **recommendation-feed**
logs, summer 2020 (see the date caveat below). There are **no search logs**. All ids are
anonymised, dataset-local integers (they do not match KuaiSAR — see the top-level
README).

| File | Rows | Grain | Key |
|------|------|-------|-----|
| `big_matrix.csv` | 12,530,806 | one impression/view (user × video × time) | `user_id`, `video_id` |
| `small_matrix.csv` | 4,676,570 | same, **fully observed** block | `user_id`, `video_id` |
| `user_features.csv` | 7,176 | one user, anonymised features | `user_id` |
| `user_features_raw.csv` | 7,176 | one user, readable features | `user_id` |
| `item_categories.csv` | 10,728 | one video, `feat` tag list | `video_id` |
| `item_daily_features.csv` | 343,341 | one video × day, platform-wide stats | `video_id`, `date` |
| `kuairec_caption_category.csv` | 10,732 | one video, text + category hierarchy | `video_id` |
| `social_network.csv` | 472 | one user with friends | `user_id` |

**How the two matrices relate.** `small_matrix` covers 1,411 users × 3,327 videos
at 99.6% density; `big_matrix` covers 7,176 users × 10,728 videos at 16.3%. The
small block's users and videos are a subset of the big one's, but the two share
**zero** `(user, video)` pairs — train on big, evaluate on small.

**Implicit feedback only.** Interactions carry *watch time* (`play_duration`,
`watch_ratio`). There are **no** per-interaction click / like / follow / comment
flags (engagement counts exist only as platform-wide daily totals per video).
The KuaiRec authors' example positive label is `watch_ratio > 2`
(`preprocessing.add_positive_label` uses `>= 2.0`).

## Interactions — `big_matrix.csv`, `small_matrix.csv`

| Column | Type | Meaning | Notes |
|--------|------|---------|-------|
| `user_id` | int | user | big 0..7175 · small 1,411 of them |
| `video_id` | int | video | big 0..10727 · small 3,327 of them |
| `play_duration` | int (ms) | how long the user watched | median ≈ 7.3 s |
| `video_duration` | int (ms) | video length | median ≈ 9.6 s |
| `time` | str | view time, `YYYY-MM-DD hh:mm:ss.sss` | **small: 3.9% missing** |
| `date` | int `YYYYMMDD` | view date | big: 2020-07-05..09-05 but only 28 distinct values; small: float (NaNs), 63 values |
| `timestamp` | float (s) | Unix view time | big: 2020-06-23..09-09 — **disagrees with `date` for some rows**, prefer `timestamp`; **small: 3.9% missing** |
| `watch_ratio` | float | `play_duration / video_duration` | median ≈ 0.72; **max ≈ 573** → use `clip_watch_ratio` |

A user can view the same video more than once: ~10% of consecutive steps in a
session are rewatches (see `session_modelling`).

## Users — `user_features.csv` and `user_features_raw.csv`

Both have 7,176 rows in the same `user_id` order. Shared columns:

| Column | Meaning | Values |
|--------|---------|--------|
| `user_active_degree` | activity tier | `full_active` 6,092 · `high_active` 938 · `UNKNOWN` 118 · `middle_active` 28 |
| `is_lowactive_period` | in a low-activity period | 0/1 |
| `follow_user_num` (+ `_range`) | accounts the user follows | 0..2,100, median 33 |
| `fans_user_num` (+ `_range`) | the user's followers | 0..11,401, median 2 |
| `friend_user_num` (+ `_range`) | mutual follows | 0..1,425, median 0 |
| `register_days` (+ `_range`) | account age in days | 0..2,245, median 225 · `register_days_range` differs between the two files in 1,117 rows (different bucket edges) |

`user_features.csv` only: `is_live_streamer`, `is_video_author` (0/1) and 18
anonymised `onehot_feat0..17` — encoded versions of raw columns. Verified
one-to-one against `user_features_raw.csv`:

| onehot | = raw column | onehot | = raw column |
|--------|--------------|--------|--------------|
| `onehot_feat0` | `gender` | `onehot_feat9` | `fre_city_level` |
| `onehot_feat1` | `age_range` | `onehot_feat10` | `fre_community_type` |
| `onehot_feat2` | *unidentified* (30 values) | `onehot_feat11` | `platform` |
| `onehot_feat3` | `phone_model` | `onehot_feat12` | `is_install_douyin` |
| `onehot_feat4` | *unidentified* (12 values, 2.8% NaN) | `onehot_feat13` | `is_install_huoshan` |
| `onehot_feat5` | `fre_country` | `onehot_feat14` | `is_install_xigua` |
| `onehot_feat6` | `fre_country_region` | `onehot_feat15` | `is_install_douyu` |
| `onehot_feat7` | `fre_province` | `onehot_feat16` | `is_install_huya` |
| `onehot_feat8` | `fre_city` | `onehot_feat17` | `is_install_yy` |

`user_features_raw.csv` only (readable):

| Group | Columns |
|-------|---------|
| Demographics | `gender` (F/M), `age_range` (8 bands, e.g. `12-17`, `50+`) |
| Creator flags | `is_live_author`, `is_photo_author` (0/1) |
| Device | `phone_brand` (34), `phone_model` (1,076), `mod_price` (phone price), `platform` (`ANDROID_PHONE`/`IPHONE`/`IPAD`), `os_version` |
| Location (frequent) | `fre_country`, `fre_country_region` (南方/北方), `fre_province` (47), `fre_city` (340), `fre_city_level` (一线城市…), `fre_community_type` (城区/乡村…) |
| Other apps installed | `is_install_douyin`, `_huoshan`, `_xigua`, `_douyu`, `_huya`, `_yy` (0/1, ~1% missing) |
| App | `app_download_channel`, `app_version`, `app_major_version`, `app_minor_version` |
| Network | `isp` (carrier) |

## Videos — categories and text

**`item_categories.csv`** — `video_id`, `feat`: a list of **tag ids 0–30**
(e.g. `[27, 9]`; 433 distinct combinations). *Not* the same id space as the
caption categories below; the pipeline does not use it for categories.

**`kuairec_caption_category.csv`** — the text and the category hierarchy the
pipeline uses (read with `engine="python", on_bad_lines="skip"`; `video_id`
loads as text, 10,732 rows for 10,730 ids).

| Column | Meaning | Notes |
|--------|---------|-------|
| `caption` | uploader's caption (Chinese, hashtag-heavy, median ~31 chars) | **12.6% empty** |
| `manual_cover_text` | text on the cover image | 62% `UNKNOWN` |
| `topic_tag` | hashtag list, e.g. `[五爱市场,搞笑]` | 43% `[]` |
| `first_level_category_id` / `_name` | top category | **38 real ids = `CategoryIndex`** (1–39); plus −124 `UNKNOWN` (41 videos), 418 / 667 (1 each) |
| `second_level_category_id` / `_name` | sub-category | 139 values |
| `third_level_category_id` / `_name` | sub-sub-category | 221 values, mostly `UNKNOWN` (−124) |

9.2% of videos have no text in any of caption / cover text / tags. The
pipeline therefore also embeds the three category names and `video_tag_name`,
which gives every searchable video some text.
`item_features.caption_category_table` gives `video_id → [first-level id]`.

## Videos — `item_daily_features.csv` (one row per video per day)

63 days, 2020-07-05 → 2020-09-05. Counts are **platform-wide** (all Kuaishou
users that day), not restricted to KuaiRec users.

| Group | Columns |
|-------|---------|
| Keys | `video_id`, `date` (`YYYYMMDD`) |
| Static video info | `author_id` (0..8,369), `video_type` (`NORMAL` 10,699 / `AD` 29), `upload_dt` (2019-05-18 → 2020-09-05), `upload_type` (18 kinds, e.g. `ShortImport` 5,934, `ShortCamera` 2,022, `Kmovie` 987, `PictureSet` 131), `visible_status` (latest day: `public` 9,934 / `private` 791 / `only friends` 3 — can change day to day), `video_duration` (ms, **3.1% missing**), `video_width`, `video_height`, `music_id` (raw id), `video_tag_id` / `video_tag_name` (another tag scheme; 9.4% name missing) |
| Exposure | `show_cnt`, `show_user_num` |
| Plays | `play_cnt`, `play_user_num`, `play_duration` (total ms), `play_progress` (0–1, mean progress), `complete_play_cnt`/`_user_num`, `valid_play_cnt`/`_user_num`, `long_time_play_cnt`/`_user_num`, `short_time_play_cnt`/`_user_num` |
| Likes | `like_cnt`, `like_user_num`, `click_like_cnt`, `double_click_cnt`, `cancel_like_cnt`, `cancel_like_user_num` |
| Comments | `comment_cnt`, `comment_user_num`, `direct_comment_cnt`, `reply_comment_cnt`, `delete_comment_cnt`, `delete_comment_user_num`, `comment_like_cnt`, `comment_like_user_num`, `comment_stay_duration` (ms spent in comments) |
| Follows | `follow_cnt`, `follow_user_num`, `cancel_follow_cnt`, `cancel_follow_user_num` |
| Share / download | `share_cnt`, `share_user_num`, `download_cnt`, `download_user_num` |
| Negative feedback | `report_cnt`, `report_user_num`, `reduce_similar_cnt`, `reduce_similar_user_num` ("show fewer like this") |
| Collect (favourite) | `collect_cnt`, `collect_user_num`, `cancel_collect_cnt`, `cancel_collect_user_num` (**20.3% missing**) |

`*_cnt` = number of events, `*_user_num` = distinct users. Durations in
KuaiRec: SHORT (< 15 s) 85%, MEDIUM (15–60 s) 14%, LONG (> 60 s) 1.1%.

## Social — `social_network.csv`

`user_id`, `friend_list` (list of `user_id`s). Only **472 of 7,176 users (6.6%)**
have friends, 1.4 on average (max 5); 670 friend pairs in total.

## Derived by this project

| Artifact | Built by | Contents |
|----------|----------|----------|
| item table | `item_features.build_item_table` | **searchable videos only** (9,905: latest `visible_status == public` and not an ad), per `video_id`: `category_id` (caption first-level, −1 unknown), `duration_ms` (median, 0 unknown), `text` (caption + cover text + topic tags + L1/L2/L3 category names + `video_tag_name`), `has_text`, `popularity` (total `play_cnt`), `trending` (`log((last-7-day plays+1)/(previous-7-day plays+1))`) |
| caption embeddings | `scripts/build_caption_embeddings.py` | `data/processed/caption_embeddings.npz` (gitignored): `video_ids`, `embeddings` (1024-d bge-m3, L2-normalised; zero rows for no text), `has_text` — rebuilt for the public catalogue |

## Pitfalls

* Drop rows with missing `timestamp` in `small_matrix` before any time-based split,
  and order by `timestamp`, not `date` (they disagree for some `big_matrix` rows).
* Clip `watch_ratio` outliers (`clip_watch_ratio`, default cap 5.0).
* Use caption first-level ids for categories, never `item_categories.feat`.
* Daily counts are platform-wide popularity, not KuaiRec-user behaviour.
* Interactions come from the recommendation feed, so they reflect what the old
  recommender chose to show (exposure bias), not what users searched for.
