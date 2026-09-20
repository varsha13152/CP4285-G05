"""Shared data loading and preprocessing for KuaiRec."""

from recsys.data.category_index import CategoryEntry, CategoryIndex
from recsys.data.loader import KuaiRecData, load_kuairec

__all__ = ["CategoryEntry", "CategoryIndex", "KuaiRecData", "load_kuairec"]
