#!/usr/bin/env python3
"""
Streaming UTF-8 sanitizer for tokenizer training text (one segment per line).

Removes or normalizes common sources of long-tail noise:
- NFC normalization
- U+FFFD (replacement from invalid UTF-8 when using errors='replace')
- Unicode private-use (Co) and surrogate (Cs) code points
- Control characters (Cc): replaced with a single ASCII space (then collapsed)
- Optional: format characters (Cf) — zero-width spaces, BOM-as-text, joiners, etc.

Does not attempt language-specific orthography fixes; pair with clean_raw_text.py
and manual tail audits (see audit_tail_characters.py).
"""

from __future__ import annotations

import argparse
import re
import unicodedata
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LANGS = ("tr", "kk", "tt")
_SPACE_RE = re.compile(r" +")


def sanitize_line(
    line: str,
    *,
    strip_cf: bool,
    stats: Counter[str],
) -> str | None:
    """Return cleaned line or None if empty after cleaning."""
    s = unicodedata.normalize("NFC", line.rstrip("\n\r"))
    out: list[str] = []
    for ch in s:
        cat = unicodedata.category(ch)
        if ch == "\uFFFD":
            stats["dropped_replacement_char"] += 1
            continue
        if cat == "Co" or cat == "Cs":
            stats["dropped_private_or_surrogate"] += 1
            continue
        if cat == "Cc":
            stats["replaced_control"] += 1
            out.append(" ")
            continue
        if strip_cf and cat == "Cf":
            stats["dropped_format"] += 1
            continue
        out.append(ch)
    merged = _SPACE_RE.sub(" ", "".join(out)).strip()
    if not merged:
        stats["dropped_empty_line"] += 1
        return None
    return merged


def run_file(input_path: Path, output_path: Path, *, strip_cf: bool) -> dict[str, int]:
    stats: Counter[str] = Counter()
    kept = 0
    total_in = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open(encoding="utf-8", errors="replace") as inf, output_path.open(
        "w", encoding="utf-8"
    ) as outf:
        for line in inf:
            total_in += 1
            cleaned = sanitize_line(line, strip_cf=strip_cf, stats=stats)
            if cleaned is not None:
                outf.write(cleaned + "\n")
                kept += 1
    return {
        "input_lines": total_in,
        "output_lines": kept,
        **dict(stats),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Strip PUA/surrogates/controls (and optional Cf) from corpus lines.")
    p.add_argument("--input", type=Path, default=None, help="Input .txt (UTF-8, one segment per line).")
    p.add_argument("--output", type=Path, default=None, help="Output .txt path.")
    p.add_argument(
        "--strip-cf",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Remove format characters (Cf), e.g. ZWSP/ZWJ/BOM-in-text. Use --no-strip-cf to keep them.",
    )
    p.add_argument(
        "--splits-root",
        type=Path,
        default=None,
        help="If set with --mirror-splits, read lang/{train,dev,test}.txt under this directory.",
    )
    p.add_argument(
        "--mirror-splits",
        action="store_true",
        help=f"For each lang in {LANGS}, sanitize splits_root/lang/*.txt into out_root/lang/.",
    )
    p.add_argument(
        "--out-root",
        type=Path,
        default=None,
        help="Output root when using --mirror-splits (default: splits_root parent / splits_sanitized).",
    )
    args = p.parse_args()

    if args.mirror_splits:
        if args.splits_root is None:
            raise SystemExit("--mirror-splits requires --splits-root")
        splits_root = args.splits_root.expanduser().resolve()
        if args.out_root is None:
            out_root = splits_root.parent / "splits_sanitized"
        else:
            out_root = args.out_root.expanduser().resolve()
        if not splits_root.is_dir():
            raise SystemExit(f"Not a directory: {splits_root}")
        for lang in LANGS:
            lang_in = splits_root / lang
            if not lang_in.is_dir():
                print(f"skip missing lang dir: {lang_in}")
                continue
            lang_out = out_root / lang
            for name in ("train.txt", "dev.txt", "test.txt"):
                inp = lang_in / name
                if not inp.is_file():
                    print(f"skip missing: {inp}")
                    continue
                outp = lang_out / name
                rep = run_file(inp, outp, strip_cf=args.strip_cf)
                print(f"{inp.relative_to(PROJECT_ROOT)} -> {outp.relative_to(PROJECT_ROOT)}  {rep}")
        return

    if args.input is None or args.output is None:
        raise SystemExit("Provide --input and --output, or use --mirror-splits with --splits-root.")

    inp = args.input.expanduser().resolve()
    out = args.output.expanduser().resolve()
    if not inp.is_file():
        raise SystemExit(f"Input not found: {inp}")
    rep = run_file(inp, out, strip_cf=args.strip_cf)
    print(rep)


if __name__ == "__main__":
    main()
