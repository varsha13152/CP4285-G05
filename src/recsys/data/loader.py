"""Load all KuaiRec CSV files into a single data container."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class KuaiRecData:
    """All loaded KuaiRec dataframes."""

    small_matrix: pd.DataFrame
    big_matrix: pd.DataFrame | None
    user_features: pd.DataFrame
    user_features_raw: pd.DataFrame
    item_categories: pd.DataFrame
    item_daily_features: pd.DataFrame
    caption_categories: pd.DataFrame
    social_network: pd.DataFrame


def _safe_parse_list(val: str) -> list:
    """Parse a string representation of a list, e.g. '[1, 5, 12]'."""
    if isinstance(val, list):
        return val
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError):
        return []


def load_kuairec(data_dir: str = "data/raw", load_big: bool = False) -> KuaiRecData:
    """Load all KuaiRec CSVs from *data_dir*.

    Parameters
    ----------
    data_dir : str
        Directory containing the raw CSV files.
    load_big : bool
        Whether to load big_matrix.csv (slower, uses more memory).
        Defaults to False for faster development iteration.

    Returns
    -------
    KuaiRecData
        Container with all dataframes.
    """
    root = Path(data_dir)

    small_matrix = pd.read_csv(root / "small_matrix.csv")

    big_matrix = None
    if load_big:
        big_matrix = pd.read_csv(root / "big_matrix.csv")

    user_features = pd.read_csv(root / "user_features.csv")
    user_features_raw = pd.read_csv(root / "user_features_raw.csv")

    item_categories = pd.read_csv(root / "item_categories.csv")
    # Parse the 'feat' column from string to list
    if "feat" in item_categories.columns:
        item_categories["feat"] = item_categories["feat"].apply(_safe_parse_list)

    item_daily_features = pd.read_csv(root / "item_daily_features.csv")

    caption_path = root / "kuairec_caption_category.csv"
    if caption_path.exists():
        # Some rows have embedded commas in Chinese text; use Python engine
        caption_categories = pd.read_csv(caption_path, engine="python", on_bad_lines="skip")
    else:
        caption_categories = pd.DataFrame()

    social_network = pd.read_csv(root / "social_network.csv")
    if "friend_list" in social_network.columns:
        social_network["friend_list"] = social_network["friend_list"].apply(_safe_parse_list)

    return KuaiRecData(
        small_matrix=small_matrix,
        big_matrix=big_matrix,
        user_features=user_features,
        user_features_raw=user_features_raw,
        item_categories=item_categories,
        item_daily_features=item_daily_features,
        caption_categories=caption_categories,
        social_network=social_network,
    )
