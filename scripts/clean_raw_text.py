#!/usr/bin/env python3
"""Clean monolingual raw .txt for tr/kk/tt: URLs, HTML entities, NFC, length/punct filters."""

from __future__ import annotations

import argparse
import html
import logging
import re
import unicodedata
from pathlib import Path

LOG = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LANGS = frozenset({"tr", "kk", "tt"})

# URLs: http(s) and common www. host patterns
_URL_RE = re.compile(
    r"(?:https?://[^\s<>\]\"{|}\\^`]+|www\.[^\s<>\]\"{|}\\^`]+)",
    flags=re.IGNORECASE,
)


def _punctuation_ratio(s: str) -> float:
    if not s:
        return 0.0
    punct = sum(1 for c in s if unicodedata.category(c).startswith("P"))
    return punct / len(s)


def clean_line(line: str, stats: dict[str, int]) -> str | None:
    """Return cleaned line or None if the line should be dropped. Updates stats."""
    s = _URL_RE.sub(" ", line)
    if not s.strip():
        stats["dropped_empty_after_urls"] += 1
        return None

    s = html.unescape(s)
    if not s.strip():
        stats["dropped_empty_after_html"] += 1
        return None

    s = unicodedata.normalize("NFC", s)
    if not s.strip():
        stats["dropped_empty_after_nfc"] += 1
        return None

    if len(s.split()) < 5:
        stats["dropped_too_few_tokens"] += 1
        return None

    s = re.sub(r"\s+", " ", s)
    if _punctuation_ratio(s) > 0.30:
        stats["dropped_high_punctuation"] += 1
        return None

    s = s.strip()
    if not s:
        stats["dropped_empty_after_final_strip"] += 1
        return None

    return s


def run(input_path: Path, output_path: Path, lang: str) -> None:
    stats = {
        "dropped_empty_after_urls": 0,
        "dropped_empty_after_html": 0,
        "dropped_empty_after_nfc": 0,
        "dropped_too_few_tokens": 0,
        "dropped_high_punctuation": 0,
        "dropped_empty_after_final_strip": 0,
    }
    kept = 0
    total_in = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open(encoding="utf-8", errors="replace") as inf, output_path.open(
        "w", encoding="utf-8"
    ) as outf:
        for line in inf:
            total_in += 1
            cleaned = clean_line(line.rstrip("\n\r"), stats)
            if cleaned is not None:
                outf.write(cleaned + "\n")
                kept += 1

    LOG.info("[%s] input lines: %d", lang, total_in)
    LOG.info("[%s] lines removed — after URLs (empty): %d", lang, stats["dropped_empty_after_urls"])
    LOG.info("[%s] lines removed — after HTML entities (empty): %d", lang, stats["dropped_empty_after_html"])
    LOG.info("[%s] lines removed — after NFC (empty): %d", lang, stats["dropped_empty_after_nfc"])
    LOG.info("[%s] lines removed — fewer than 5 tokens: %d", lang, stats["dropped_too_few_tokens"])
    LOG.info("[%s] lines removed — >30%% punctuation: %d", lang, stats["dropped_high_punctuation"])
    LOG.info(
        "[%s] lines removed — empty after final strip: %d",
        lang,
        stats["dropped_empty_after_final_strip"],
    )
    LOG.info("[%s] lines kept: %d -> %s", lang, kept, output_path)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    p = argparse.ArgumentParser(description="Clean raw text for Turkish, Kazakh, or Tatar NLP.")
    p.add_argument("--input", type=Path, required=True, help="Path to raw .txt (one segment per line).")
    p.add_argument(
        "--lang",
        required=True,
        choices=sorted(LANGS),
        help="Language code (tr, kk, tt).",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help=f"Output .txt (default: data/processed/native_script/{{lang}}.txt under project root).",
    )
    args = p.parse_args()

    inp = args.input.expanduser().resolve()
    if not inp.is_file():
        raise SystemExit(f"Input not found: {inp}")

    out = args.output
    if out is None:
        out = PROJECT_ROOT / "data" / "processed" / "native_script" / f"{args.lang}.txt"
    else:
        out = out.expanduser()
        if not out.is_absolute():
            out = (PROJECT_ROOT / out).resolve()

    run(inp, out, args.lang)


if __name__ == "__main__":
    main()
