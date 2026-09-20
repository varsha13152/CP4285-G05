"""Bilingual category index for KuaiRec's first-level category taxonomy.

Maps between category IDs, Chinese names, and English translations so the
LLM query parser can bridge English user queries to the Chinese metadata in
KuaiRec.  The mapping is hardcoded (not LLM-generated at runtime) because
the KuaiRec taxonomy is fixed, and human-verified translations avoid
hallucination.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class CategoryEntry:
    """One entry in the bilingual category index."""

    category_id: int
    chinese_name: str
    english_name: str
    english_aliases: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Hardcoded mapping — extracted from kuairec_caption_category.csv
# (first_level_category_id × first_level_category_name) and
# human-verified English translations.
# ---------------------------------------------------------------------------

_RAW_CATEGORIES: list[tuple[int, str, str, list[str]]] = [
    (1, "舞蹈", "Dance", ["dancing", "choreography"]),
    (2, "音乐", "Music", ["songs", "singing"]),
    (3, "游戏", "Gaming", ["games", "video games", "esports"]),
    (4, "美妆", "Beauty/Makeup", ["cosmetics", "skincare", "makeup"]),
    (5, "时尚", "Fashion", ["style", "outfit", "clothing"]),
    (6, "明星娱乐", "Celebrity Entertainment", ["stars", "idol", "celebrity"]),
    (7, "运动", "Sports", ["athletics", "exercise"]),
    (8, "颜值", "Attractiveness/Looks", ["beauty", "good-looking", "selfie"]),
    (9, "喜剧", "Comedy", ["funny", "humor", "jokes"]),
    (10, "旅游", "Travel", ["tourism", "trip", "vacation"]),
    (11, "生活", "Daily Life", ["lifestyle", "everyday"]),
    (12, "美食", "Food", ["cooking", "recipes", "cuisine"]),
    (13, "三农", "Agriculture", ["farming", "rural", "countryside"]),
    (14, "教育", "Education", ["learning", "teaching", "tutorial"]),
    (15, "艺术", "Art", ["crafts", "creative", "painting"]),
    (16, "健康", "Health", ["wellness", "medical"]),
    (17, "宠物", "Pets", ["animals", "dogs", "cats"]),
    (18, "汽车", "Automobiles", ["cars", "vehicles", "driving"]),
    (19, "情感", "Emotions/Relationships", ["love", "feelings", "romance"]),
    (20, "二次元", "Anime/ACG", ["animation", "manga", "cosplay"]),
    (21, "人文", "Humanities", ["culture", "history"]),
    (22, "财经", "Finance", ["money", "investing", "business"]),
    (23, "时政资讯", "Politics/News", ["current affairs", "policy"]),
    (24, "星座命理", "Astrology/Fortune", ["horoscope", "zodiac"]),
    (25, "亲子", "Parenting", ["kids", "family", "children"]),
    (26, "摄影", "Photography", ["camera", "photos"]),
    (27, "高新数码", "Tech/Digital", ["gadgets", "electronics", "phone"]),
    (28, "民生资讯", "People's News", ["local news", "community"]),
    (29, "科学与法律", "Science & Law", ["legal", "scientific"]),
    (31, "健身", "Fitness", ["workout", "gym", "exercise"]),
    (32, "短剧", "Short Drama", ["mini-series", "sketch"]),
    (33, "自拍", "Selfie", ["self-portrait"]),
    (34, "其他", "Other", ["miscellaneous"]),
    (35, "军事", "Military", ["army", "defense"]),
    (36, "房产家居", "Real Estate/Home", ["housing", "interior", "decoration"]),
    (37, "奇人异象", "Oddities", ["strange", "unusual", "weird"]),
    (38, "读书", "Reading", ["books", "literature"]),
    (39, "影视综", "Film & TV", ["movies", "television", "shows"]),
]


class CategoryIndex:
    """Bilingual index of KuaiRec's first-level category taxonomy.

    Usage::

        index = CategoryIndex()
        entry = index.get_by_id(12)
        # CategoryEntry(category_id=12, chinese_name='美食', english_name='Food', ...)

        matches = index.search_english("cooking")
        # [CategoryEntry(12, '美食', 'Food', ...)]

        prompt_text = index.to_prompt_string()
        # "ID | Chinese | English\\n1 | 舞蹈 | Dance\\n..."
    """

    def __init__(self) -> None:
        self._by_id: dict[int, CategoryEntry] = {}
        for cat_id, cn, en, aliases in _RAW_CATEGORIES:
            self._by_id[cat_id] = CategoryEntry(
                category_id=cat_id,
                chinese_name=cn,
                english_name=en,
                english_aliases=list(aliases),
            )

    def get_by_id(self, cat_id: int) -> CategoryEntry | None:
        """Look up a category by its integer ID. Returns None if not found."""
        return self._by_id.get(cat_id)

    def search_english(self, term: str) -> list[CategoryEntry]:
        """Find categories matching an English term (case-insensitive).

        Matches against ``english_name`` and ``english_aliases``.  Returns
        all entries where the search term appears as a substring.
        """
        term_lower = term.lower().strip()
        if not term_lower:
            return []

        results: list[CategoryEntry] = []
        for entry in self._by_id.values():
            searchable = [entry.english_name.lower()] + [
                a.lower() for a in entry.english_aliases
            ]
            if any(term_lower in s for s in searchable):
                results.append(entry)
        return results

    def all_entries(self) -> list[CategoryEntry]:
        """Return all category entries sorted by ID."""
        return sorted(self._by_id.values(), key=lambda e: e.category_id)

    def valid_ids(self) -> set[int]:
        """Return the set of all valid category IDs."""
        return set(self._by_id.keys())

    def to_prompt_string(self) -> str:
        """Format the index as a table for LLM prompt injection.

        Returns a string like::

            ID | Chinese | English | Also known as
             1 | 舞蹈     | Dance   | dancing, choreography
             2 | 音乐     | Music   | songs, singing
            ...
        """
        lines = ["ID | Chinese | English | Also known as"]
        for entry in self.all_entries():
            aliases = ", ".join(entry.english_aliases) if entry.english_aliases else ""
            lines.append(
                f"{entry.category_id} | {entry.chinese_name} | "
                f"{entry.english_name} | {aliases}"
            )
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self._by_id)

    def __repr__(self) -> str:
        return f"CategoryIndex({len(self)} categories)"
