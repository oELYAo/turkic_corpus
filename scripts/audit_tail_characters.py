#!/usr/bin/env python3
"""
List characters in the low-frequency tail beyond a cumulative coverage threshold.

Use before SentencePiece training to inspect noise (PUA, odd symbols, mojibake
substitutes) that lives past e.g. 99.99% mass — often more visible in tr than kk/tt.

Reads per-language split files (default: normalized_script splits, train only),
writes CSV (+ optional JSON) under results/ by default.
"""

from __future__ import annotations

import argparse
import csv
import json
import unicodedata
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SPLITS = PROJECT_ROOT / "data" / "processed" / "normalized_script" / "splits"
DEFAULT_OUT = PROJECT_ROOT / "results" / "tokenizer_corpus_metrics" / "tail_char_audit"
LANGS = ("tr", "kk", "tt")


def unicode_bucket(ch: str) -> str:
    cat = unicodedata.category(ch)
    if cat == "Cc":
        return "control"
    if cat == "Cf":
        return "format"
    if cat == "Cs":
        return "surrogate"
    if cat == "Co":
        return "private_use"
    if cat == "Cn":
        return "unassigned"
    if ch.isspace():
        return "space"
    if cat[0] == "L":
        return "letter"
    if cat[0] == "N":
        return "number"
    if cat[0] in ("P", "S"):
        return "punct_or_symbol"
    if cat[0] == "Z":
        return "separator"
    return "other"


def stream_char_freq(path: Path) -> Counter[str]:
    cf: Counter[str] = Counter()
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            for c in line.rstrip("\n\r"):
                cf[c] += 1
    return cf


def tail_after_coverage(char_freq: Counter[str], coverage: float) -> tuple[int, list[tuple[int, str, int]]]:
    """
    Return (cutoff_rank, tail_rows) where tail_rows are (rank, char, count) for all
    characters that appear strictly after the smallest prefix reaching `coverage`
    of total character mass (sorted by descending count, tie-break U+).
    """
    total = sum(char_freq.values())
    if total == 0:
        return 0, []
    ranked = sorted(char_freq.items(), key=lambda x: (-x[1], ord(x[0])))
    need = coverage * total
    cum = 0
    cutoff_rank = len(ranked)
    tail: list[tuple[int, str, int]] = []
    for rank, (ch, cnt) in enumerate(ranked, start=1):
        if cum >= need:
            tail.append((rank, ch, cnt))
        cum += cnt
        if cum >= need and cutoff_rank == len(ranked):
            cutoff_rank = rank
    return cutoff_rank, tail


def char_display(ch: str) -> str:
    if ch in ("\t", "\n", "\r"):
        return {"\t": "\\t", "\n": "\\n", "\r": "\\r"}[ch]
    if ord(ch) < 32 or ord(ch) == 0x7F:
        return f"U+{ord(ch):04X}"
    return ch


def main() -> None:
    p = argparse.ArgumentParser(description="Audit rare tail characters past a coverage threshold.")
    p.add_argument(
        "--splits_root",
        type=Path,
        default=DEFAULT_SPLITS,
        help="Directory with tr/, kk/, tt/ and train.txt (etc.).",
    )
    p.add_argument(
        "--split",
        default="train",
        help="Split file name stem (default train).",
    )
    p.add_argument("--coverage", type=float, default=0.9999, help="Cumulative mass threshold (default 0.9999).")
    p.add_argument("--out_dir", type=Path, default=DEFAULT_OUT, help="Directory for CSV/JSON outputs.")
    p.add_argument("--langs", nargs="+", default=list(LANGS), choices=list(LANGS), help="Languages to scan.")
    args = p.parse_args()

    splits_root = args.splits_root.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    if not splits_root.is_dir():
        raise SystemExit(f"Not a directory: {splits_root}")
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, object] = {
        "splits_root": str(splits_root.relative_to(PROJECT_ROOT)),
        "split": args.split,
        "coverage": args.coverage,
        "per_lang": {},
    }

    for lang in args.langs:
        path = splits_root / lang / f"{args.split}.txt"
        if not path.is_file():
            print(f"skip missing: {path}")
            continue
        print(f"scanning {path} ...")
        cf = stream_char_freq(path)
        total = sum(cf.values())
        cutoff_rank, tail = tail_after_coverage(cf, args.coverage)
        tail_mass = sum(c for _, _, c in tail)
        summary["per_lang"][lang] = {
            "path": str(path.relative_to(PROJECT_ROOT)),
            "total_chars": total,
            "unique_chars": len(cf),
            "cutoff_rank_at_threshold": cutoff_rank,
            "tail_unique_chars": len(tail),
            "tail_char_mass": tail_mass,
            "tail_mass_fraction": round(tail_mass / total, 8) if total else 0.0,
        }

        csv_path = out_dir / f"rare_tail_{lang}_{args.split}.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as wf:
            w = csv.writer(wf)
            w.writerow(
                [
                    "rank",
                    "codepoint_hex",
                    "char_display",
                    "count",
                    "fraction_of_total",
                    "unicode_major_category",
                    "unicode_bucket",
                    "unicode_name",
                ]
            )
            for rank, ch, cnt in tail:
                frac = cnt / total if total else 0.0
                w.writerow(
                    [
                        rank,
                        f"U+{ord(ch):04X}",
                        char_display(ch),
                        cnt,
                        f"{frac:.8e}",
                        unicodedata.category(ch),
                        unicode_bucket(ch),
                        unicodedata.name(ch, ""),
                    ]
                )
        print(f"  wrote {csv_path} ({len(tail)} code points in tail)")

    json_path = out_dir / "tail_audit_summary.json"
    with json_path.open("w", encoding="utf-8") as jf:
        json.dump(summary, jf, indent=2, ensure_ascii=False)
    print(f"wrote {json_path}")


if __name__ == "__main__":
    main()
