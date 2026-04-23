#!/usr/bin/env python3
"""
Intrinsic tokenizer metrics for SentencePiece models: fertility, fragmentation, UNK coverage.
Appends one row to results/intrinsic/scores.csv.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import sentencepiece as spm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SCORES_CSV = PROJECT_ROOT / "results" / "intrinsic" / "scores.csv"

SCORE_FIELDS = (
    "condition",
    "lang",
    "script_type",
    "vocab_size",
    "fertility",
    "fragmentation_rate",
    "coverage",
)


def infer_script_type(test_file: Path) -> str:
    lowered = str(test_file.resolve()).lower()
    if "normalized" in lowered:
        return "normalized"
    return "native"


def load_words(test_file: Path) -> list[str]:
    words: list[str] = []
    with test_file.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            words.extend(line.split())
    return words


def compute_metrics(sp: spm.SentencePieceProcessor, words: list[str]) -> tuple[float, float, float]:
    """Return (fertility, fragmentation_rate, coverage)."""
    n = len(words)
    if n == 0:
        return 0.0, 0.0, 0.0

    unk_id = sp.unk_id()
    total_tok = 0
    frag_count = 0
    cov_count = 0

    for w in words:
        ids = sp.encode(w, out_type=int)
        t = len(ids)
        total_tok += t
        if t >= 3:
            frag_count += 1
        if unk_id not in ids:
            cov_count += 1

    fertility = total_tok / n
    frag_rate = frag_count / n
    coverage = cov_count / n
    return fertility, frag_rate, coverage


def append_score_row(csv_path: Path, row: dict[str, object]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(SCORE_FIELDS))
        if write_header:
            w.writeheader()
        w.writerow({k: row[k] for k in SCORE_FIELDS})


def main() -> None:
    p = argparse.ArgumentParser(description="SentencePiece intrinsic metrics (fertility, fragmentation, coverage).")
    p.add_argument("--model", type=Path, required=True, help="Path to .model file.")
    p.add_argument("--test_file", type=Path, required=True, help="Whitespace-tokenized test text (UTF-8, one or many lines).")
    p.add_argument("--lang", required=True, help="Language code, e.g. tt, kk, tr.")
    p.add_argument("--condition", required=True, help="Tokenizer condition, e.g. transfer_tr_kk.")
    p.add_argument(
        "--script_type",
        choices=("native", "normalized", "auto"),
        default="auto",
        help="Script bucket for CSV (default: infer from --test_file path).",
    )
    p.add_argument(
        "--scores_csv",
        type=Path,
        default=DEFAULT_SCORES_CSV,
        help=f"CSV to append (default: {DEFAULT_SCORES_CSV}).",
    )
    args = p.parse_args()

    model_path = args.model.expanduser().resolve()
    test_path = args.test_file.expanduser().resolve()
    if not model_path.is_file():
        raise SystemExit(f"Model not found: {model_path}")
    if not test_path.is_file():
        raise SystemExit(f"Test file not found: {test_path}")

    script_type = infer_script_type(test_path) if args.script_type == "auto" else args.script_type

    sp = spm.SentencePieceProcessor(model_file=str(model_path))
    vocab_size = sp.get_piece_size()

    words = load_words(test_path)
    fertility, fragmentation_rate, coverage = compute_metrics(sp, words)

    print(f"words={len(words)} vocab_size={vocab_size}")
    print(f"fertility={fertility:.6f}")
    print(f"fragmentation_rate={fragmentation_rate:.6f}")
    print(f"coverage={coverage:.6f}")

    scores_path = args.scores_csv.expanduser()
    if not scores_path.is_absolute():
        scores_path = (PROJECT_ROOT / scores_path).resolve()

    row = {
        "condition": args.condition,
        "lang": args.lang,
        "script_type": script_type,
        "vocab_size": vocab_size,
        "fertility": f"{fertility:.10g}",
        "fragmentation_rate": f"{fragmentation_rate:.10g}",
        "coverage": f"{coverage:.10g}",
    }
    append_score_row(scores_path, row)
    print(f"Appended row -> {scores_path}")


if __name__ == "__main__":
    main()
