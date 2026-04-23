#!/usr/bin/env python3
"""
Corpus metrics for SentencePiece / tokenizer training on normalized_script splits:
- Character coverage curves (cumulative mass vs rank; min #chars for target coverage)
- Split sizes (lines, characters, whitespace words)
- Line length distributions (chars per line)
- Unicode category mix (letter / number / punctuation / space / other)

Writes CSV/JSON summaries and PNG figures under results/tokenizer_corpus_metrics/.
"""

from __future__ import annotations

import argparse
import csv
import json
import unicodedata
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SPLITS = PROJECT_ROOT / "data" / "processed" / "normalized_script" / "splits"
DEFAULT_OUT = PROJECT_ROOT / "results" / "tokenizer_corpus_metrics"
LANGS = ("tr", "kk", "tt")
SPLITS = ("train", "dev", "test")
TARGET_COVERAGES = (0.9990, 0.9995, 0.9999)


def unicode_bucket(ch: str) -> str:
    if ch.isspace():
        return "space"
    cat = unicodedata.category(ch)[0]
    if cat == "L":
        return "letter"
    if cat == "N":
        return "number"
    if cat in ("P", "S"):
        return "punct_symbol"
    if cat == "Z":
        return "separator"
    return "other"


def stream_file_stats(path: Path) -> tuple[int, int, int, Counter[str], Counter[str], list[int]]:
    """Return lines, chars, words, char_freq, category_freq, line_lengths (chars per line)."""
    lines = chars = words = 0
    char_freq: Counter[str] = Counter()
    cat_freq: Counter[str] = Counter()
    line_lengths: list[int] = []
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            text = line.rstrip("\n\r")
            lines += 1
            line_lengths.append(len(text))
            chars += len(text)
            words += len(text.split())
            for c in text:
                char_freq[c] += 1
                cat_freq[unicode_bucket(c)] += 1
    return lines, chars, words, char_freq, cat_freq, line_lengths


def coverage_table(char_freq: Counter[str]) -> dict[str, float | int]:
    total = sum(char_freq.values())
    if total == 0:
        return {"total_chars": 0, "unique_chars": 0}
    ranked = sorted(char_freq.items(), key=lambda x: -x[1])
    cum = 0.0
    out: dict[str, float | int] = {
        "total_chars": total,
        "unique_chars": len(char_freq),
    }
    for i, (_, cnt) in enumerate(ranked, start=1):
        cum += cnt
        frac = cum / total
        for tc in TARGET_COVERAGES:
            key = f"min_rank_for_{tc:.4f}".replace(".", "_")
            if key not in out and frac >= tc:
                out[key] = i
        if frac >= max(TARGET_COVERAGES):
            break
    for tc in TARGET_COVERAGES:
        key = f"min_rank_for_{tc:.4f}".replace(".", "_")
        if key not in out:
            out[key] = len(ranked)
    return out


def cumulative_coverage_array(char_freq: Counter[str]) -> tuple[np.ndarray, np.ndarray]:
    """Returns (ranks 1..K, cumulative fraction of total mass)."""
    total = sum(char_freq.values())
    if total == 0:
        return np.array([]), np.array([])
    ranked = sorted(char_freq.values(), reverse=True)
    arr = np.cumsum(np.array(ranked, dtype=np.float64)) / total
    ranks = np.arange(1, len(arr) + 1)
    return ranks, arr


def plot_char_coverage(
    per_lang_train: dict[str, Counter[str]],
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = {"tr": "#2ecc71", "kk": "#3498db", "tt": "#e74c3c"}
    for lang in LANGS:
        if lang not in per_lang_train:
            continue
        ranks, cum = cumulative_coverage_array(per_lang_train[lang])
        if len(ranks) == 0:
            continue
        ax.plot(ranks, cum, label=f"{lang} (train)", color=colors.get(lang, None), linewidth=1.5)
    for tc in (0.9995, 0.999):
        ax.axhline(y=tc, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_xlabel("Character rank (by frequency, descending)")
    ax.set_ylabel("Cumulative fraction of all characters")
    ax.set_title("Character coverage curves — normalized_script train splits")
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_ylim(0.95, 1.0005)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_corpus_sizes(rows: list[dict], out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    langs = LANGS
    x = np.arange(len(langs))
    w = 0.25
    for si, split in enumerate(SPLITS):
        vals = [next((r["chars"] for r in rows if r["lang"] == L and r["split"] == split), 0) for L in langs]
        axes[0].bar(x + (si - 1) * w, vals, w, label=split)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(langs)
    axes[0].set_ylabel("Characters")
    axes[0].set_title("Characters by language and split")
    axes[0].legend()
    axes[0].set_yscale("log")

    for si, split in enumerate(SPLITS):
        vals = [next((r["lines"] for r in rows if r["lang"] == L and r["split"] == split), 0) for L in langs]
        axes[1].bar(x + (si - 1) * w, vals, w, label=split)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(langs)
    axes[1].set_ylabel("Lines")
    axes[1].set_title("Lines by language and split")
    axes[1].legend()
    axes[1].set_yscale("log")
    fig.suptitle("Normalized script splits — corpus size", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_line_lengths(
    per_key_lengths: dict[tuple[str, str], list[int]],
    out_path: Path,
    *,
    max_lines_sample: int = 500_000,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    for ax, lang in zip(axes, LANGS):
        lengths: list[int] = []
        for split in SPLITS:
            key = (lang, split)
            if key not in per_key_lengths:
                continue
            ls = per_key_lengths[key]
            if len(ls) > max_lines_sample:
                rng = np.random.default_rng(42)
                idx = rng.choice(len(ls), size=max_lines_sample, replace=False)
                lengths.extend(ls[i] for i in idx)
            else:
                lengths.extend(ls)
        if not lengths:
            continue
        ax.hist(lengths, bins=80, range=(0, min(2000, max(lengths))), color="#34495e", alpha=0.85, edgecolor="white", linewidth=0.3)
        ax.set_xlabel("Characters per line")
        ax.set_title(f"{lang} (all splits)")
        ax.set_yscale("log")
    axes[0].set_ylabel("Count (log scale)")
    fig.suptitle("Line length distribution (chars per line)", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_category_mix(rows: list[dict], out_path: Path) -> None:
    cats = ("letter", "number", "punct_symbol", "space", "separator", "other")
    langs = LANGS
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(langs))
    bottom = np.zeros(len(langs))
    colors_map = {
        "letter": "#27ae60",
        "number": "#f39c12",
        "punct_symbol": "#8e44ad",
        "space": "#95a5a6",
        "separator": "#bdc3c7",
        "other": "#7f8c8d",
    }
    train_rows = [r for r in rows if r["split"] == "train"]
    for cat in cats:
        heights = []
        for lang in langs:
            r = next((row for row in train_rows if row["lang"] == lang), None)
            if not r:
                heights.append(0.0)
                continue
            total = r["chars"] or 1
            h = r["cat_counts"].get(cat, 0) / total
            heights.append(h)
        ax.bar(x, heights, bottom=bottom, label=cat, color=colors_map[cat])
        bottom += np.array(heights)
    ax.set_xticks(x)
    ax.set_xticklabels(langs)
    ax.set_ylabel("Fraction of characters")
    ax.set_title("Unicode category mix — train split only")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Tokenizer training corpus metrics and plots.")
    p.add_argument(
        "--splits_root",
        type=Path,
        default=DEFAULT_SPLITS,
        help="Directory containing tr/, kk/, tt/ with train.txt, dev.txt, test.txt",
    )
    p.add_argument("--out_dir", type=Path, default=DEFAULT_OUT, help="Output directory for figures and tables.")
    args = p.parse_args()

    splits_root = args.splits_root.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    if not splits_root.is_dir():
        raise SystemExit(f"Not a directory: {splits_root}")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    per_lang_train: dict[str, Counter[str]] = {lang: Counter() for lang in LANGS}
    combined_train = Counter()
    per_key_lengths: dict[tuple[str, str], list[int]] = {}

    for lang in LANGS:
        for split in SPLITS:
            path = splits_root / lang / f"{split}.txt"
            if not path.is_file():
                print(f"skip missing: {path}")
                continue
            print(f"scanning {path} ...")
            lines, chars, words, cf, cat_f, line_lengths = stream_file_stats(path)
            per_key_lengths[(lang, split)] = line_lengths
            row = {
                "lang": lang,
                "split": split,
                "path": str(path.relative_to(PROJECT_ROOT)),
                "lines": lines,
                "chars": chars,
                "words": words,
                "chars_per_line_mean": round(chars / lines, 2) if lines else 0.0,
                "words_per_line_mean": round(words / lines, 2) if lines else 0.0,
            }
            row.update(coverage_table(cf))
            row["cat_counts"] = dict(cat_f)
            rows.append(row)
            if split == "train":
                per_lang_train[lang].update(cf)
                combined_train.update(cf)

    # Summary JSON (without huge nested dicts twice)
    summary = {
        "splits_root": str(splits_root.relative_to(PROJECT_ROOT)),
        "per_file": [],
        "train_character_coverage": {},
        "combined_train": coverage_table(combined_train),
    }
    for lang in LANGS:
        summary["train_character_coverage"][lang] = coverage_table(per_lang_train[lang])

    for r in rows:
        entry = {k: v for k, v in r.items() if k != "cat_counts"}
        entry["cat_fractions"] = {k: round(v / r["chars"], 6) for k, v in r["cat_counts"].items()} if r["chars"] else {}
        summary["per_file"].append(entry)

    json_path = out_dir / "metrics_summary.json"
    with json_path.open("w", encoding="utf-8") as jf:
        json.dump(summary, jf, indent=2, ensure_ascii=False)

    csv_path = out_dir / "split_stats.csv"

    flat_fields = [
        "lang",
        "split",
        "lines",
        "chars",
        "words",
        "unique_chars",
        "chars_per_line_mean",
        "words_per_line_mean",
    ] + [f"min_rank_for_{tc:.4f}".replace(".", "_") for tc in TARGET_COVERAGES]
    with csv_path.open("w", newline="", encoding="utf-8") as cf:
        w = csv.DictWriter(cf, fieldnames=flat_fields + ["letter_frac", "number_frac", "space_frac", "punct_symbol_frac"])
        w.writeheader()
        for r in rows:
            ct = r["chars"] or 1
            cc = r["cat_counts"]
            w.writerow(
                {
                    "lang": r["lang"],
                    "split": r["split"],
                    "lines": r["lines"],
                    "chars": r["chars"],
                    "words": r["words"],
                    "unique_chars": r.get("unique_chars", ""),
                    "chars_per_line_mean": r["chars_per_line_mean"],
                    "words_per_line_mean": r["words_per_line_mean"],
                    **{f"min_rank_for_{tc:.4f}".replace(".", "_"): r.get(f"min_rank_for_{tc:.4f}".replace(".", "_"), "") for tc in TARGET_COVERAGES},
                    "letter_frac": round(cc.get("letter", 0) / ct, 6),
                    "number_frac": round(cc.get("number", 0) / ct, 6),
                    "space_frac": round(cc.get("space", 0) / ct, 6),
                    "punct_symbol_frac": round((cc.get("punct_symbol", 0) + cc.get("separator", 0)) / ct, 6),
                }
            )

    plot_char_coverage(per_lang_train, out_dir / "char_coverage_curves.png")
    plot_corpus_sizes(rows, out_dir / "corpus_sizes.png")
    plot_line_lengths(per_key_lengths, out_dir / "line_length_histograms.png")
    plot_category_mix(rows, out_dir / "unicode_category_mix_train.png")

    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    print(f"Figures in {out_dir}")

    # Console summary for quick copy
    print("\n--- Train split: character coverage targets (min rank to reach mass) ---")
    for lang in LANGS:
        t = coverage_table(per_lang_train[lang])
        print(f"  {lang}: unique_chars={t.get('unique_chars')} total_chars={t.get('total_chars')}")
        for tc in TARGET_COVERAGES:
            k = f"min_rank_for_{tc:.4f}".replace(".", "_")
            print(f"    {tc}: rank<={t.get(k)}")
    comb = coverage_table(combined_train)
    print("  combined (tr+kk+tt train):")
    print(f"    unique_chars={comb.get('unique_chars')} total_chars={comb.get('total_chars')}")
    for tc in TARGET_COVERAGES:
        k = f"min_rank_for_{tc:.4f}".replace(".", "_")
        print(f"    {tc}: rank<={comb.get(k)}")


if __name__ == "__main__":
    main()
