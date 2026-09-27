"""Chinese synthetic queries for evaluating the full QCSAR pipeline.

Same structure as ``synthetic_queries.py`` but in Chinese — eliminates the
cross-lingual gap (Chinese queries ↔ Chinese video captions).  Categories
and expected outputs match the English counterparts.
"""

SYNTHETIC_QUERIES_ZH: list[dict] = [
    # --- Clear single-category queries ---
    {
        "query": "有趣的做饭视频",
        "english": "fun cooking videos",
        "expected_topics": [12],          # 美食 (Food)
        "expected_style": "FUNNY",
        "expected_duration": "SHORT",
        "expected_intent": "ENTERTAINMENT",
    },
    {
        "query": "新手化妆教程",
        "english": "beginner makeup tutorial",
        "expected_topics": [4],           # 美妆 (Beauty/Makeup)
        "expected_style": "INFORMATIVE",
        "expected_duration": "MEDIUM",
        "expected_intent": "LEARNING",
    },
    {
        "query": "可爱的猫咪视频",
        "english": "cute cat videos",
        "expected_topics": [17],          # 宠物 (Pets)
        "expected_style": "CUTE",
        "expected_duration": "SHORT",
        "expected_intent": "ENTERTAINMENT",
    },
    {
        "query": "热门舞蹈挑战",
        "english": "trending dance challenges",
        "expected_topics": [1],           # 舞蹈 (Dance)
        "expected_style": "ENERGETIC",
        "expected_duration": "SHORT",
        "expected_intent": "SOCIAL",
    },
    {
        "query": "放松的旅游风景",
        "english": "relaxing travel scenery",
        "expected_topics": [10],          # 旅游 (Travel)
        "expected_style": "RELAXING",
        "expected_duration": "SHORT",
        "expected_intent": "ENTERTAINMENT",
    },
    {
        "query": "吉他入门教学",
        "english": "guitar basics tutorial",
        "expected_topics": [2],           # 音乐 (Music)
        "expected_style": "INFORMATIVE",
        "expected_duration": "MEDIUM",
        "expected_intent": "LEARNING",
    },
    {
        "query": "搞笑喜剧段子",
        "english": "funny comedy sketches",
        "expected_topics": [9],           # 喜剧 (Comedy)
        "expected_style": "FUNNY",
        "expected_duration": "SHORT",
        "expected_intent": "ENTERTAINMENT",
    },
    {
        "query": "居家健身训练",
        "english": "home workout routines",
        "expected_topics": [31],          # 健身 (Fitness)
        "expected_style": "ENERGETIC",
        "expected_duration": "MEDIUM",
        "expected_intent": "LEARNING",
    },
    {
        "query": "动漫cosplay集锦",
        "english": "anime cosplay highlights",
        "expected_topics": [20],          # 二次元 (Anime/ACG)
        "expected_style": "AESTHETIC",
        "expected_duration": "SHORT",
        "expected_intent": "ENTERTAINMENT",
    },
    {
        "query": "理财投资入门知识",
        "english": "investment tips for beginners",
        "expected_topics": [22],          # 财经 (Finance)
        "expected_style": "INFORMATIVE",
        "expected_duration": "MEDIUM",
        "expected_intent": "LEARNING",
    },
    # --- Multi-category ---
    {
        "query": "美食和穿搭分享",
        "english": "food and fashion hauls",
        "expected_topics": [12, 5],       # 美食 + 时尚
        "expected_style": "NEUTRAL",
        "expected_duration": "ANY",
        "expected_intent": "EXPLORATION",
    },
    {
        "query": "体育集锦和健身励志",
        "english": "sports highlights and fitness motivation",
        "expected_topics": [7, 31],       # 运动 + 健身
        "expected_style": "ENERGETIC",
        "expected_duration": "SHORT",
        "expected_intent": "INSPIRATION",
    },
    # --- Negation ---
    {
        "query": "喜剧视频，不要太长",
        "english": "comedy videos, not too long",
        "expected_topics": [9],           # 喜剧 (Comedy)
        "expected_style": "FUNNY",
        "expected_duration": "SHORT",
        "expected_intent": "ENTERTAINMENT",
        "expected_negative": ["太长"],
    },
    {
        "query": "音乐视频，不要伤感的",
        "english": "music videos, no sad ones",
        "expected_topics": [2],           # 音乐 (Music)
        "expected_style": "ENERGETIC",
        "expected_duration": "ANY",
        "expected_intent": "ENTERTAINMENT",
        "expected_negative": ["伤感"],
    },
    # --- Specific search ---
    {
        "query": "明星访谈节目",
        "english": "celebrity interviews",
        "expected_topics": [6],           # 明星娱乐 (Celebrity Entertainment)
        "expected_style": "NEUTRAL",
        "expected_duration": "MEDIUM",
        "expected_intent": "SPECIFIC_SEARCH",
    },
    # --- Popular categories in KuaiRec ---
    {
        "query": "颜值超高的小哥哥小姐姐",
        "english": "attractive people videos",
        "expected_topics": [8],           # 颜值 (Attractiveness)
        "expected_style": "AESTHETIC",
        "expected_duration": "SHORT",
        "expected_intent": "ENTERTAINMENT",
    },
    {
        "query": "日常生活记录vlog",
        "english": "daily life vlog",
        "expected_topics": [11],          # 生活 (Daily Life)
        "expected_style": "RELAXING",
        "expected_duration": "MEDIUM",
        "expected_intent": "ENTERTAINMENT",
    },
    {
        "query": "感人的情感故事",
        "english": "touching emotional stories",
        "expected_topics": [19],          # 情感 (Emotions)
        "expected_style": "DRAMATIC",
        "expected_duration": "MEDIUM",
        "expected_intent": "ENTERTAINMENT",
    },
    {
        "query": "游戏精彩操作",
        "english": "gaming highlights",
        "expected_topics": [3],           # 游戏 (Gaming)
        "expected_style": "ENERGETIC",
        "expected_duration": "MEDIUM",
        "expected_intent": "ENTERTAINMENT",
    },
    {
        "query": "农村生活田园风光",
        "english": "rural life countryside scenery",
        "expected_topics": [13],          # 三农 (Agriculture)
        "expected_style": "RELAXING",
        "expected_duration": "MEDIUM",
        "expected_intent": "ENTERTAINMENT",
    },
]
