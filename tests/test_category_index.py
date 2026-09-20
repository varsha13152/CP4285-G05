"""Tests for the bilingual category index."""

import pytest

from recsys.data.category_index import CategoryEntry, CategoryIndex


@pytest.fixture
def index() -> CategoryIndex:
    return CategoryIndex()


class TestCategoryIndex:
    def test_length(self, index: CategoryIndex) -> None:
        assert len(index) == 38

    def test_get_by_id_valid(self, index: CategoryIndex) -> None:
        entry = index.get_by_id(12)
        assert entry is not None
        assert entry.category_id == 12
        assert entry.chinese_name == "美食"
        assert entry.english_name == "Food"
        assert "cooking" in entry.english_aliases

    def test_get_by_id_missing(self, index: CategoryIndex) -> None:
        assert index.get_by_id(9999) is None

    def test_search_english_exact(self, index: CategoryIndex) -> None:
        results = index.search_english("Food")
        assert any(e.category_id == 12 for e in results)

    def test_search_english_alias(self, index: CategoryIndex) -> None:
        results = index.search_english("cooking")
        assert any(e.category_id == 12 for e in results)

    def test_search_english_case_insensitive(self, index: CategoryIndex) -> None:
        results = index.search_english("GAMING")
        assert any(e.category_id == 3 for e in results)

    def test_search_english_partial(self, index: CategoryIndex) -> None:
        results = index.search_english("anim")
        # Should match Anime/ACG (animation alias) and/or Pets (animals alias)
        ids = {e.category_id for e in results}
        assert 20 in ids or 17 in ids

    def test_search_english_empty(self, index: CategoryIndex) -> None:
        assert index.search_english("") == []
        assert index.search_english("   ") == []

    def test_search_english_no_match(self, index: CategoryIndex) -> None:
        assert index.search_english("zzznonexistent") == []

    def test_all_entries_sorted(self, index: CategoryIndex) -> None:
        entries = index.all_entries()
        ids = [e.category_id for e in entries]
        assert ids == sorted(ids)

    def test_valid_ids(self, index: CategoryIndex) -> None:
        ids = index.valid_ids()
        assert 1 in ids
        assert 12 in ids
        assert 39 in ids
        # ID 30 is missing from KuaiRec taxonomy
        assert 30 not in ids

    def test_to_prompt_string(self, index: CategoryIndex) -> None:
        prompt = index.to_prompt_string()
        assert "ID | Chinese | English | Also known as" in prompt
        assert "美食" in prompt
        assert "Food" in prompt
        assert "cooking" in prompt

    def test_repr(self, index: CategoryIndex) -> None:
        assert "38" in repr(index)


class TestCategoryEntry:
    def test_frozen(self) -> None:
        entry = CategoryEntry(1, "舞蹈", "Dance", ["dancing"])
        with pytest.raises(AttributeError):
            entry.category_id = 2  # type: ignore[misc]

    def test_default_aliases(self) -> None:
        entry = CategoryEntry(1, "舞蹈", "Dance")
        assert entry.english_aliases == []
