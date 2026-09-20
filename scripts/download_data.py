#!/usr/bin/env python3
"""Download KuaiRec dataset files.

Usage:
    python scripts/download_data.py
    python scripts/download_data.py --dest data/raw

The KuaiRec dataset is hosted on Google Drive.  This script prints
instructions since automated Google Drive downloads require auth.
"""

import argparse
from pathlib import Path


EXPECTED_FILES = [
    "small_matrix.csv",
    "big_matrix.csv",
    "user_features.csv",
    "user_features_raw.csv",
    "item_categories.csv",
    "item_daily_features.csv",
    "kuairec_caption_category.csv",
    "social_network.csv",
]


def main():
    parser = argparse.ArgumentParser(description="Download KuaiRec data")
    parser.add_argument("--dest", default="data/raw", help="Destination directory")
    args = parser.parse_args()

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    print("KuaiRec Dataset Download")
    print("=" * 50)
    print()
    print("1. Go to: https://github.com/chongminggao/KuaiRec")
    print("2. Follow the download links in the README")
    print("3. Place the following CSV files in:", dest.resolve())
    print()
    for f in EXPECTED_FILES:
        path = dest / f
        status = "OK" if path.exists() else "MISSING"
        print(f"   [{status}] {f}")
    print()

    missing = [f for f in EXPECTED_FILES if not (dest / f).exists()]
    if missing:
        print(f"{len(missing)} file(s) missing. Download them to continue.")
    else:
        print("All files present!")


if __name__ == "__main__":
    main()
