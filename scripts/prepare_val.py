"""
Train/Val Split Script for Akkadian Translation Competition.

Splits train.csv into train_split.csv and val_split.csv at document level (8:2).
Each row in train.csv represents one document.

Usage:
    python scripts/prepare_val.py
    python scripts/prepare_val.py --data-dir path/to/data --seed 42 --val-ratio 0.2
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split train.csv into train/val sets at document level."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data",
        help="Directory containing train.csv (default: data/)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Fraction of data for validation (default: 0.2)",
    )
    return parser.parse_args()


def validate_dataframe(df: pd.DataFrame, filepath: Path) -> None:
    """Validate that the dataframe has expected columns."""
    required_cols = {"transliteration", "translation"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"{filepath} is missing required columns: {missing}. "
            f"Found columns: {list(df.columns)}"
        )


def print_split_stats(
    name: str, df: pd.DataFrame, total: int
) -> None:
    """Print statistics for a split."""
    avg_translit_len = df["transliteration"].astype(str).str.len().mean()
    avg_translation_len = df["translation"].astype(str).str.len().mean()
    empty_translations = df["translation"].isna().sum()

    print(f"\n  {name}:")
    print(f"    Documents:           {len(df)} ({len(df)/total*100:.1f}%)")
    print(f"    Avg transliteration: {avg_translit_len:.0f} chars")
    print(f"    Avg translation:     {avg_translation_len:.0f} chars")
    print(f"    Empty translations:  {empty_translations}")


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    train_csv = data_dir / "train.csv"

    if not train_csv.exists():
        print(f"ERROR: {train_csv} not found.")
        print(
            "Please download the competition data first:\n"
            "  kaggle competitions download "
            "-c deep-past-challenge-translate-akkadian-to-english "
            f"-p {data_dir}"
        )
        sys.exit(1)

    print(f"Loading {train_csv} ...")
    df = pd.read_csv(train_csv)
    validate_dataframe(df, train_csv)

    total = len(df)
    print(f"Total documents: {total}")
    print(f"Columns: {list(df.columns)}")

    train_df, val_df = train_test_split(
        df,
        test_size=args.val_ratio,
        random_state=args.seed,
    )

    train_out = data_dir / "train_split.csv"
    val_out = data_dir / "val_split.csv"

    train_df.to_csv(train_out, index=False)
    val_df.to_csv(val_out, index=False)

    print(f"\nSplit results (seed={args.seed}, val_ratio={args.val_ratio}):")
    print_split_stats("Train", train_df, total)
    print_split_stats("Val", val_df, total)

    print(f"\nSaved:")
    print(f"  {train_out}")
    print(f"  {val_out}")


if __name__ == "__main__":
    main()
