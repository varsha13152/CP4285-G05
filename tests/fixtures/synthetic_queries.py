"""Synthetic queries with expected outputs for evaluating Stage 1 (LLM Intent Understanding).

Each entry has:
    query:             Natural-language search string
    expected_topics:   List of KuaiRec first-level category IDs that should match
    expected_style:    Expected ContentStyle name
    expected_duration: Expected DurationPreference name
    expected_intent:   Expected QueryIntentType name
    expected_negative: (optional) Expected negative constraints
"""

SYNTHETIC_QUERIES: list[dict] = [
    # --- Clear single-category queries ---
    {
        "query": "fun cooking videos for a student dinner",
        "expected_topics": [12],          # 美食 (Food)
        "expected_style": "FUNNY",
        "expected_duration": "SHORT",
        "expected_intent": "ENTERTAINMENT",
    },
    {
        "query": "how to do makeup for beginners",
        "expected_topics": [4],           # 美妆 (Beauty/Makeup)
        "expected_style": "INFORMATIVE",
        "expected_duration": "MEDIUM",
        "expected_intent": "LEARNING",
    },
    {
        "query": "cute cat videos",
        "expected_topics": [17],          # 宠物 (Pets)
        "expected_style": "CUTE",
        "expected_duration": "SHORT",
        "expected_intent": "ENTERTAINMENT",
    },
    {
        "query": "trending dance challenges",
        "expected_topics": [1],           # 舞蹈 (Dance)
        "expected_style": "ENERGETIC",
        "expected_duration": "SHORT",
        "expected_intent": "SOCIAL",
    },
    {
        "query": "relaxing travel scenery, not too long",
        "expected_topics": [10],          # 旅游 (Travel)
        "expected_style": "RELAXING",
        "expected_duration": "SHORT",
        "expected_intent": "ENTERTAINMENT",
        "expected_negative": ["too long"],
    },
    {
        "query": "learn guitar basics",
        "expected_topics": [2],           # 音乐 (Music)
        "expected_style": "INFORMATIVE",
        "expected_duration": "MEDIUM",
        "expected_intent": "LEARNING",
    },
    {
        "query": "funny comedy sketches",
        "expected_topics": [9],           # 喜剧 (Comedy)
        "expected_style": "FUNNY",
        "expected_duration": "SHORT",
        "expected_intent": "ENTERTAINMENT",
    },
    {
        "query": "home workout routines",
        "expected_topics": [31],          # 健身 (Fitness)
        "expected_style": "ENERGETIC",
        "expected_duration": "MEDIUM",
        "expected_intent": "LEARNING",
    },
    {
        "query": "anime cosplay highlights",
        "expected_topics": [20],          # 二次元 (Anime/ACG)
        "expected_style": "AESTHETIC",
        "expected_duration": "SHORT",
        "expected_intent": "ENTERTAINMENT",
    },
    {
        "query": "investment tips for beginners",
        "expected_topics": [22],          # 财经 (Finance)
        "expected_style": "INFORMATIVE",
        "expected_duration": "MEDIUM",
        "expected_intent": "LEARNING",
    },
    # --- Multi-category queries ---
    {
        "query": "cooking and fashion hauls",
        "expected_topics": [12, 5],       # 美食 + 时尚
        "expected_style": "NEUTRAL",
        "expected_duration": "ANY",
        "expected_intent": "EXPLORATION",
    },
    {
        "query": "sports highlights and fitness motivation",
        "expected_topics": [7, 31],       # 运动 + 健身
        "expected_style": "ENERGETIC",
        "expected_duration": "SHORT",
        "expected_intent": "INSPIRATION",
    },
    # --- Vague / ambiguous queries ---
    {
        "query": "something funny to watch",
        "expected_topics": [9],           # 喜剧 (Comedy) — most likely
        "expected_style": "FUNNY",
        "expected_duration": "ANY",
        "expected_intent": "ENTERTAINMENT",
    },
    {
        "query": "show me something new",
        "expected_topics": [],            # No specific category
        "expected_style": "NEUTRAL",
        "expected_duration": "ANY",
        "expected_intent": "EXPLORATION",
    },
    {
        "query": "what's popular right now",
        "expected_topics": [],
        "expected_style": "NEUTRAL",
        "expected_duration": "ANY",
        "expected_intent": "SOCIAL",
    },
    # --- Negation queries ---
    {
        "query": "comedy but not slapstick, keep it short",
        "expected_topics": [9],           # 喜剧 (Comedy)
        "expected_style": "FUNNY",
        "expected_duration": "SHORT",
        "expected_intent": "ENTERTAINMENT",
        "expected_negative": ["slapstick"],
    },
    {
        "query": "music videos, no sad songs",
        "expected_topics": [2],           # 音乐 (Music)
        "expected_style": "ENERGETIC",
        "expected_duration": "ANY",
        "expected_intent": "ENTERTAINMENT",
        "expected_negative": ["sad songs"],
    },
    # --- Specific search ---
    {
        "query": "celebrity interviews",
        "expected_topics": [6],           # 明星娱乐 (Celebrity Entertainment)
        "expected_style": "NEUTRAL",
        "expected_duration": "MEDIUM",
        "expected_intent": "SPECIFIC_SEARCH",
    },
    {
        "query": "short drama romance series",
        "expected_topics": [32, 19],      # 短剧 + 情感
        "expected_style": "DRAMATIC",
        "expected_duration": "LONG",
        "expected_intent": "ENTERTAINMENT",
    },
    # --- Style-heavy queries ---
    {
        "query": "aesthetic photography",
        "expected_topics": [26],          # 摄影 (Photography)
        "expected_style": "AESTHETIC",
        "expected_duration": "SHORT",
        "expected_intent": "INSPIRATION",
    },
    {
        "query": "serious news analysis",
        "expected_topics": [23],          # 时政资讯 (Politics/News)
        "expected_style": "SERIOUS",
        "expected_duration": "LONG",
        "expected_intent": "LEARNING",
    },
    {
        "query": "cute baby and kids moments",
        "expected_topics": [25],          # 亲子 (Parenting)
        "expected_style": "CUTE",
        "expected_duration": "SHORT",
        "expected_intent": "ENTERTAINMENT",
    },
    # --- Duration-specific ---
    {
        "query": "quick daily life clips",
        "expected_topics": [11],          # 生活 (Daily Life)
        "expected_style": "RELAXING",
        "expected_duration": "SHORT",
        "expected_intent": "ENTERTAINMENT",
    },
    {
        "query": "long educational science documentaries",
        "expected_topics": [29],          # 科学与法律
        "expected_style": "INFORMATIVE",
        "expected_duration": "LONG",
        "expected_intent": "LEARNING",
    },
    # --- Niche categories ---
    {
        "query": "astrology and zodiac readings",
        "expected_topics": [24],          # 星座命理
        "expected_style": "RELAXING",
        "expected_duration": "MEDIUM",
        "expected_intent": "ENTERTAINMENT",
    },
    {
        "query": "car reviews and test drives",
        "expected_topics": [18],          # 汽车 (Automobiles)
        "expected_style": "INFORMATIVE",
        "expected_duration": "MEDIUM",
        "expected_intent": "LEARNING",
    },
    {
        "query": "weird and unusual things",
        "expected_topics": [37],          # 奇人异象 (Oddities)
        "expected_style": "FUNNY",
        "expected_duration": "SHORT",
        "expected_intent": "ENTERTAINMENT",
    },
    {
        "query": "gaming highlights and esports",
        "expected_topics": [3],           # 游戏 (Gaming)
        "expected_style": "ENERGETIC",
        "expected_duration": "MEDIUM",
        "expected_intent": "ENTERTAINMENT",
    },
    {
        "query": "home decoration inspiration",
        "expected_topics": [36],          # 房产家居 (Real Estate/Home)
        "expected_style": "AESTHETIC",
        "expected_duration": "MEDIUM",
        "expected_intent": "INSPIRATION",
    },
    {
        "query": "health and wellness tips",
        "expected_topics": [16],          # 健康 (Health)
        "expected_style": "INFORMATIVE",
        "expected_duration": "MEDIUM",
        "expected_intent": "LEARNING",
    },
]
