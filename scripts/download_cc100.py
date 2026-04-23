#!/usr/bin/env python3
"""Stream CC-100 monolingual text from Hugging Face into per-language raw text files.

Uses ``load_dataset('statmt/cc100', lang=..., split='train', streaming=True)`` so rows
are not materialized in RAM.

Note: the upstream CC-100 release at https://data.statmt.org/cc-100/ does not ship
Tatar (``tt``). If ``tt`` is among ``--langs``, it is skipped with a warning; use
Wikipedia/OSCAR/etc. for Tatar.

Requires ``datasets<4`` (see ``requirements.txt``): 4.x no longer runs this dataset's
loading script.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

DEFAULT_LANGS = ("tr", "kk", "tt")
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ISO codes absent from https://data.statmt.org/cc-100/ (and HF builder configs).
CC100_MISSING_LANGS = {
    "tt": "Tatar (tt) is not included in CC-100; there is no tt.txt.xz on data.statmt.org.",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download CC-100 (statmt/cc100) monolingual splits for Turkic LRL study."
    )
    p.add_argument(
        "--langs",
        default=",".join(DEFAULT_LANGS),
        help=f"Comma-separated ISO 639-1 codes (default: {','.join(DEFAULT_LANGS)}).",
    )
    p.add_argument(
        "--max-lines",
        type=int,
        default=None,
        metavar="N",
        help="Stop after N lines per language (for debugging). Default: stream full split.",
    )
    return p.parse_args()


def normalize_line(text: str) -> str:
    """One CC-100 example -> one output line (collapse internal newlines/whitespace)."""
    return " ".join(text.split())


def download_lang(lang: str, max_lines: int | None) -> Path:
    raw_dir = PROJECT_ROOT / "data" / "raw" / lang
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_path = raw_dir / "cc100.txt"

    ds = load_dataset(
        "statmt/cc100",
        lang=lang,
        split="train",
        streaming=True,
        trust_remote_code=True,
    )

    n_written = 0
    with out_path.open("w", encoding="utf-8") as f:
        bar = tqdm(
            desc=f"cc100[{lang}]",
            unit="lines",
            total=max_lines,
            dynamic_ncols=True,
        )
        try:
            for row in ds:
                raw = row.get("text")
                if not raw:
                    continue
                line = normalize_line(raw)
                if not line:
                    continue
                f.write(line + "\n")
                n_written += 1
                bar.update(1)
                if max_lines is not None and n_written >= max_lines:
                    break
        finally:
            bar.close()

    return out_path


def main() -> None:
    args = parse_args()
    langs = [x.strip() for x in args.langs.split(",") if x.strip()]
    if not langs:
        print("No languages after parsing --langs.", file=sys.stderr)
        sys.exit(1)

    skipped = [lang for lang in langs if lang in CC100_MISSING_LANGS]
    langs = [lang for lang in langs if lang not in CC100_MISSING_LANGS]
    for lang in skipped:
        print(f"Skipping {lang}: {CC100_MISSING_LANGS[lang]}", file=sys.stderr)

    if not langs:
        print("No languages left to download after skipping unsupported codes.", file=sys.stderr)
        sys.exit(2)

    for lang in langs:
        path = download_lang(lang, args.max_lines)
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"{lang}: wrote {path} ({size_mb:.2f} MiB)")


if __name__ == "__main__":
    main()
