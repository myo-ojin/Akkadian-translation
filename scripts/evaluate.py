"""Evaluation script for Akkadian translation competition.

Computes BLEU, chrF++, and their geometric mean (official Kaggle "Deep Past Challenge" metric).
Can be used as a CLI tool or imported as a library.

Usage:
    python evaluate.py --predictions pred.txt --references ref.txt
    python evaluate.py --predictions pred.csv --references ref.csv --pred-col translation --ref-col english
    python evaluate.py --predictions pred.txt --references ref.txt --json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Optional


def read_lines(filepath: Path) -> list[str]:
    """Read a text file, returning one stripped line per entry."""
    with open(filepath, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def read_csv_column(filepath: Path, column: str) -> list[str]:
    """Read a specific column from a CSV file."""
    rows: list[str] = []
    with open(filepath, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV file has no header: {filepath}")
        if column not in reader.fieldnames:
            raise ValueError(
                f"Column '{column}' not found in {filepath}. "
                f"Available columns: {', '.join(reader.fieldnames)}"
            )
        for row in reader:
            value = row[column].strip()
            rows.append(value)
    return rows


def load_texts(
    filepath: Path,
    column: Optional[str] = None,
) -> list[str]:
    """Load texts from a file (CSV with column name, or plain text)."""
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    if column is not None:
        return read_csv_column(filepath, column)

    suffix = filepath.suffix.lower()
    if suffix == ".csv":
        raise ValueError(
            f"File '{filepath}' appears to be CSV but no column name was specified. "
            f"Use --pred-col / --ref-col to specify the column."
        )

    return read_lines(filepath)


def evaluate(
    predictions: list[str],
    references: list[str],
) -> dict[str, float]:
    """Compute BLEU, chrF++, and geometric mean.

    Args:
        predictions: List of predicted translation strings.
        references: List of reference translation strings.

    Returns:
        Dictionary with keys 'bleu', 'chrf', and 'geo_mean'.

    Raises:
        ValueError: If inputs are empty or have mismatched lengths.
    """
    try:
        import sacrebleu
    except ImportError:
        raise ImportError(
            "sacrebleu is required. Install it with: pip install sacrebleu"
        )

    if not predictions:
        raise ValueError("Predictions list is empty")
    if not references:
        raise ValueError("References list is empty")
    if len(predictions) != len(references):
        raise ValueError(
            f"Length mismatch: {len(predictions)} predictions "
            f"vs {len(references)} references"
        )

    bleu_result = sacrebleu.corpus_bleu(predictions, [references])
    chrf_result = sacrebleu.corpus_chrf(predictions, [references], word_order=2)

    bleu_score = bleu_result.score
    chrf_score = chrf_result.score

    if bleu_score < 0 or chrf_score < 0:
        geo_mean = 0.0
    else:
        geo_mean = math.sqrt(bleu_score * chrf_score)

    return {
        "bleu": round(bleu_score, 4),
        "chrf": round(chrf_score, 4),
        "geo_mean": round(geo_mean, 4),
    }


def format_report(scores: dict[str, float]) -> str:
    """Format scores as a human-readable report."""
    lines = [
        "=" * 50,
        "  Akkadian Translation Evaluation Results",
        "=" * 50,
        f"  BLEU:            {scores['bleu']:>8.4f}",
        f"  chrF++:          {scores['chrf']:>8.4f}",
        "-" * 50,
        f"  Geometric Mean:  {scores['geo_mean']:>8.4f}",
        "=" * 50,
    ]
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Evaluate Akkadian translations using BLEU, chrF++, and geometric mean.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python evaluate.py --predictions pred.txt --references ref.txt\n"
            "  python evaluate.py --predictions pred.csv --references ref.csv "
            "--pred-col translation --ref-col english\n"
            "  python evaluate.py --predictions pred.txt --references ref.txt --json\n"
        ),
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="Path to predictions file (text or CSV)",
    )
    parser.add_argument(
        "--references",
        type=Path,
        required=True,
        help="Path to references file (text or CSV)",
    )
    parser.add_argument(
        "--pred-col",
        type=str,
        default=None,
        help="Column name for predictions in CSV file",
    )
    parser.add_argument(
        "--ref-col",
        type=str,
        default=None,
        help="Column name for references in CSV file",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="output_json",
        help="Output results as JSON",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry point.

    Args:
        argv: Command-line arguments (defaults to sys.argv[1:]).

    Returns:
        Exit code (0 for success, 1 for error).
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        predictions = load_texts(args.predictions, args.pred_col)
        references = load_texts(args.references, args.ref_col)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error loading files: {e}", file=sys.stderr)
        return 1

    print(
        f"Loaded {len(predictions)} predictions and {len(references)} references.",
        file=sys.stderr,
    )

    try:
        scores = evaluate(predictions, references)
    except (ValueError, ImportError) as e:
        print(f"Evaluation error: {e}", file=sys.stderr)
        return 1

    if args.output_json:
        print(json.dumps(scores, indent=2))
    else:
        print(format_report(scores))

    return 0


if __name__ == "__main__":
    sys.exit(main())
