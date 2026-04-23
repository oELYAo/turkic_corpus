#!/usr/bin/env python3
"""Parse WikiExtractor output (<doc>...</doc>), strip wikitext/HTML noise, emit cleaned lines."""

from __future__ import annotations

import argparse
import html
import logging
import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from clean_raw_text import clean_line  # noqa: E402

LOG = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LANGS = frozenset({"tr", "kk", "tt"})

# MediaWiki behavior lines like __NOTOC__, __NOEDITSECTION__, etc.
_MAGIC = re.compile(r"^__[A-Za-z0-9_]+__$")

_REF_RE = re.compile(r"<ref\b[^>]*>.*?</ref>|<ref\b[^>]*/>", re.DOTALL | re.IGNORECASE)


def _parse_doc_open(line: str) -> str | None:
    if "<doc id=" not in line or 'title="' not in line:
        return None
    rest = line.split('title="', 1)[1]
    if '">' not in rest:
        return None
    title, _, _ = rest.partition('">')
    return html.unescape(title)


def _strip_wikilinks(s: str) -> str:
    while True:
        m = re.search(r"\[\[([^\]]+)\]\]", s)
        if not m:
            break
        inner = m.group(1)
        if "|" in inner:
            inner = inner.split("|")[-1]
        s = s[: m.start()] + inner + s[m.end() :]
    return s


def _strip_templates(s: str) -> str:
    while True:
        start = s.find("{{")
        if start == -1:
            return s
        depth = 0
        j = start
        n = len(s)
        while j < n - 1:
            pair = s[j : j + 2]
            if pair == "{{":
                depth += 1
                j += 2
            elif pair == "}}":
                depth -= 1
                j += 2
                if depth == 0:
                    s = s[:start] + " " + s[j:]
                    break
            else:
                j += 1
        else:
            return s[:start] + s[start + 2 :]


def _preprocess_wiki_line(line: str) -> str | None:
    s = line.strip()
    if not s:
        return None
    if _MAGIC.match(s):
        return None
    s = _REF_RE.sub(" ", s)
    s = _strip_wikilinks(s)
    s = html.unescape(s)
    s = re.sub(r"<[^>]+>", " ", s)
    s = _strip_templates(s)
    s = re.sub(r"\s+", " ", s).strip()
    return s if s else None


def _iter_wiki_files(root: Path):
    for p in sorted(root.rglob("wiki_*")):
        if p.is_file():
            yield p


def clean_wiki_tree(input_dir: Path, output_path: Path, lang: str) -> None:
    stats_global = {
        "dropped_empty_after_urls": 0,
        "dropped_empty_after_html": 0,
        "dropped_empty_after_nfc": 0,
        "dropped_too_few_tokens": 0,
        "dropped_high_punctuation": 0,
        "dropped_empty_after_final_strip": 0,
    }
    kept = 0
    docs = 0
    pre_drop = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as outf:
        for wfile in _iter_wiki_files(input_dir):
            in_doc = False
            title = ""
            first_text = True
            with wfile.open(encoding="utf-8", errors="replace") as inf:
                for raw in inf:
                    if raw.startswith("<doc "):
                        in_doc = True
                        title = _parse_doc_open(raw) or ""
                        first_text = True
                        docs += 1
                        continue
                    if in_doc and raw.strip() == "</doc>":
                        in_doc = False
                        continue
                    if not in_doc:
                        continue

                    pre = _preprocess_wiki_line(raw.rstrip("\n\r"))
                    if pre is None:
                        pre_drop += 1
                        continue
                    if first_text and title and pre == title.strip():
                        first_text = False
                        continue
                    first_text = False

                    cleaned = clean_line(pre, stats_global)
                    if cleaned is not None:
                        outf.write(cleaned + "\n")
                        kept += 1

    LOG.info("[%s] docs seen (open tags): %d", lang, docs)
    LOG.info("[%s] lines dropped in wiki preprocess (empty/magic): %d", lang, pre_drop)
    LOG.info("[%s] lines kept: %d -> %s", lang, kept, output_path)
    for k, v in stats_global.items():
        if v:
            LOG.info("[%s] %s: %d", lang, k, v)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    p = argparse.ArgumentParser(description="Clean WikiExtractor dirs -> one .txt per language.")
    p.add_argument("--lang", required=True, choices=sorted(LANGS), help="tr, kk, or tt")
    p.add_argument("--input-dir", type=Path, required=True, help="e.g. data/raw/kk/wiki_extracted")
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output .txt (default: data/processed/wiki_cleaned/{lang}.txt)",
    )
    args = p.parse_args()

    root = args.input_dir.expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"Input directory not found: {root}")

    out = args.output
    if out is None:
        out = PROJECT_ROOT / "data" / "processed" / "wiki_cleaned" / f"{args.lang}.txt"
    else:
        out = out.expanduser().resolve()
        if not out.is_absolute():
            out = (PROJECT_ROOT / out).resolve()

    clean_wiki_tree(root, out, args.lang)


if __name__ == "__main__":
    main()
