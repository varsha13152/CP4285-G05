"""Column name constants for KuaiRec CSVs.

Import these instead of hardcoding column strings.
"""

# --- Interaction columns (small_matrix / big_matrix) ---
USER_ID = "user_id"
VIDEO_ID = "video_id"
WATCH_RATIO = "watch_ratio"
PLAY_DURATION = "play_duration"
VIDEO_DURATION = "video_duration"
TIMESTAMP = "timestamp"
DATE = "date"
TIME = "time"

# --- User feature columns ---
USER_ACTIVE_DEGREE = "user_active_degree"
IS_LOWACTIVE = "is_lowactive_period"
IS_LIVE_STREAMER = "is_live_streamer"
IS_VIDEO_AUTHOR = "is_video_author"
FANS_NUM = "fans_user_num"
FOLLOW_NUM = "follow_user_num"
FRIEND_NUM = "friend_user_num"
REGISTER_DAYS = "register_days"

# --- Raw user feature columns ---
GENDER = "gender"
AGE_RANGE = "age_range"
PROVINCE = "province"
CITY = "city"
CITY_TIER = "city_tier"

# --- Item category columns ---
FEAT = "feat"  # tag list in item_categories.csv

# --- Caption / category columns ---
CAPTION = "caption"
TOPIC_TAG = "topic_tag"
FIRST_CATEGORY = "first_level_category_name"
SECOND_CATEGORY = "second_level_category_name"
THIRD_CATEGORY = "third_level_category_name"

# --- Daily feature columns ---
SHOW_CNT = "show_cnt"
PLAY_CNT = "play_cnt"
LIKE_CNT = "like_cnt"
SHARE_CNT = "share_cnt"
COMMENT_CNT = "comment_cnt"
FOLLOW_CNT = "follow_cnt"
UPLOAD_DT = "upload_dt"
MUSIC_ID = "music_id"
