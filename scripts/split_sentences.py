"""
Sentence-Level Splitting Script for Akkadian Translation Evaluation.

Uses Sentences_Oare_FirstWord_LinNum.csv to split document-level data
into sentence-level data for fine-grained evaluation.

The script first inspects the sentence file structure, then performs the split.

Usage:
    python scripts/split_sentences.py
    python scripts/split_sentences.py --data-dir path/to/data
"""

import argparse
import io
import re
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split document-level data into sentence-level for evaluation."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data",
        help="Directory containing data files (default: data/)",
    )
    parser.add_argument(
        "--val-only",
        action="store_true",
        default=True,
        help="Only process val_split.csv (default: True)",
    )
    return parser.parse_args()


def find_sentence_file(data_dir: Path) -> Path:
    """Find the sentence boundary file with flexible name matching."""
    candidates = [
        "Sentences_Oare_FirstWord_LinNum.csv",
        "sentences_oare_firstword_linnum.csv",
        "Sentences_OARE_FirstWord_LinNum.csv",
    ]
    for name in candidates:
        path = data_dir / name
        if path.exists():
            return path

    csv_files = list(data_dir.glob("*entence*.*"))
    if csv_files:
        return csv_files[0]

    return data_dir / candidates[0]


def inspect_file(filepath: Path) -> pd.DataFrame:
    """Load and display file structure for inspection."""
    print(f"\nInspecting {filepath.name} ...")
    df = pd.read_csv(filepath)
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Dtypes:\n{df.dtypes.to_string()}")
    print(f"\n  First 5 rows:")
    print(df.head().to_string())
    return df


def split_transliteration_by_sentences(
    text: str, separator: str = "."
) -> list[str]:
    """Split a transliteration string into sentences.

    Akkadian transliterations typically use periods or line breaks
    as sentence boundaries.
    """
    if not isinstance(text, str) or not text.strip():
        return []

    sentences = re.split(r'(?<=[.!?])\s+|\n+', text)
    return [s.strip() for s in sentences if s.strip()]


def split_translation_by_sentences(text: str) -> list[str]:
    """Split an English translation into sentences."""
    if not isinstance(text, str) or not text.strip():
        return []

    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


def create_sentence_level_data(
    val_df: pd.DataFrame,
    sentence_df: pd.DataFrame | None,
) -> pd.DataFrame:
    """Create sentence-level evaluation data.

    Strategy:
    1. If sentence_df provides boundary info, use it to split documents.
    2. Otherwise, fall back to regex-based sentence splitting.
    """
    rows: list[dict] = []

    if sentence_df is not None:
        # Match val docs to sentences via oare_id <-> text_uuid
        sent_id_col = None
        val_id_col = None
        for s_col in ["text_uuid", "text_id", "id", "doc_id", "document_id"]:
            if s_col in sentence_df.columns:
                sent_id_col = s_col
                break
        for v_col in ["oare_id", "id", "doc_id"]:
            if v_col in val_df.columns:
                val_id_col = v_col
                break

        if sent_id_col and val_id_col:
            overlap = set(val_df[val_id_col]) & set(sentence_df[sent_id_col])
            print(f"  Matching {val_id_col} <-> {sent_id_col}: {len(overlap)} docs matched")

            matched = 0
            for _, doc_row in val_df.iterrows():
                doc_id = doc_row[val_id_col]
                doc_sentences = sentence_df[sentence_df[sent_id_col] == doc_id]

                if len(doc_sentences) > 0:
                    matched += 1
                    for sent_idx, (_, sent_row) in enumerate(
                        doc_sentences.iterrows()
                    ):
                        row = {
                            "doc_id": doc_id,
                            "sentence_idx": sent_idx,
                            "transliteration": str(sent_row.get("first_word_spelling", "")),
                            "translation": str(sent_row.get("translation", "")),
                        }
                        rows.append(row)
                else:
                    rows.extend(
                        _fallback_split_document(doc_row)
                    )
            print(f"  Sentence-file matched: {matched}, regex fallback: {len(val_df) - matched}")
        else:
            print(
                f"  No matching ID columns found (sent: {sent_id_col}, val: {val_id_col}). "
                "Using regex-based splitting."
            )
            for _, doc_row in val_df.iterrows():
                rows.extend(_fallback_split_document(doc_row))
    else:
        print("  No sentence file found. Using regex-based splitting.")
        for _, doc_row in val_df.iterrows():
            rows.extend(_fallback_split_document(doc_row))

    return pd.DataFrame(rows)


def _fallback_split_document(doc_row: pd.Series) -> list[dict]:
    """Split a single document into sentences using regex."""
    doc_id = doc_row.get("id", None)
    transliteration = str(doc_row.get("transliteration", ""))
    translation = str(doc_row.get("translation", ""))

    translit_sents = split_transliteration_by_sentences(transliteration)
    translation_sents = split_translation_by_sentences(translation)

    max_sents = max(len(translit_sents), len(translation_sents), 1)

    rows = []
    for i in range(max_sents):
        row = {
            "doc_id": doc_id,
            "sentence_idx": i,
            "transliteration": (
                translit_sents[i] if i < len(translit_sents) else ""
            ),
            "translation": (
                translation_sents[i]
                if i < len(translation_sents)
                else ""
            ),
        }
        rows.append(row)

    return rows


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()

    val_csv = data_dir / "val_split.csv"
    if not val_csv.exists():
        print(
            f"ERROR: {val_csv} not found. Run prepare_val.py first."
        )
        sys.exit(1)

    val_df = pd.read_csv(val_csv)
    print(f"Loaded val_split.csv: {len(val_df)} documents")

    sentence_file = find_sentence_file(data_dir)
    sentence_df = None
    if sentence_file.exists():
        sentence_df = inspect_file(sentence_file)
    else:
        print(
            f"\nWARNING: {sentence_file.name} not found in {data_dir}."
        )
        print("  Will use regex-based sentence splitting as fallback.")

    print("\nCreating sentence-level data ...")
    result_df = create_sentence_level_data(val_df, sentence_df)

    out_path = data_dir / "val_sentences.csv"
    result_df.to_csv(out_path, index=False)

    print(f"\nResults:")
    print(f"  Input documents:  {len(val_df)}")
    print(f"  Output sentences: {len(result_df)}")
    print(
        f"  Avg sentences/doc: {len(result_df)/max(len(val_df),1):.1f}"
    )

    if len(result_df) > 0:
        print(f"\n  Column list: {list(result_df.columns)}")
        print(f"\n  First 5 rows:")
        print(result_df.head().to_string())

    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
