#!/usr/bin/env python3
"""
Extra statistics for subword tokenizer design (normalized_script splits).

Focus: train splits by default — where you fit SentencePiece / BPE vocabulary.

Metrics
-------
- Word (whitespace) tokenization proxy: types, TTR, hapax rate, mass at top-k ranks
- Word-length distribution (characters per word); summary percentiles
- Zipf / rank–frequency (log–log) on train
- Character-set overlap: pairwise Jaccard, per-language exclusive chars, union size
- UTF-8 byte load: bytes per character, bytes per line (mean)
- Cross-language *type* overlap on a sampled vocabulary (optional): Jaccard, |∩|/min(|A|,|B|)

Outputs under results/tokenizer_design_stats/ (JSON, CSV, PNG).

For huge corpora, use --word-sample-rate < 1 to subsample *lines* for word-frequency
statistics only (word-length histograms always use all tokens on scanned lines).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SPLITS = PROJECT_ROOT / "data" / "processed" / "normalized_script" / "splits"
DEFAULT_OUT = PROJECT_ROOT / "results" / "tokenizer_design_stats"
LANGS = ("tr", "kk", "tt")
WORD_LEN_CAP = 64


def stream_train_word_and_char_stats(
    path: Path,
    *,
    word_sample_rate: float,
    rng: random.Random,
    max_rank_for_plot: int = 50_000,
) -> tuple[dict, set[str], Counter[str]]:
    """
    One pass: char set, utf8 bytes, word length histogram, optional sampled word Counter.

    word_sample_rate applies per *line*: if random() < rate, include that line's words
    in word_freq; word lengths always use every line's words.
    Returns (json_safe_summary, char_set, word_freq).
    """
    char_set: set[str] = set()
    utf8_bytes = 0
    n_lines = 0
    n_chars = 0
    word_freq: Counter[str] = Counter()
    len_hist = np.zeros(WORD_LEN_CAP + 1, dtype=np.int64)
    n_words_total = 0
    n_lines_in_word_counter = 0

    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            text = line.rstrip("\n\r")
            n_lines += 1
            b = text.encode("utf-8")
            utf8_bytes += len(b)
            n_chars += len(text)
            for ch in text:
                char_set.add(ch)

            words = text.split()
            n_words_total += len(words)
            for w in words:
                lw = len(w)
                if lw > WORD_LEN_CAP:
                    len_hist[WORD_LEN_CAP] += 1
                else:
                    len_hist[lw] += 1

            if word_sample_rate >= 1.0 or rng.random() < word_sample_rate:
                n_lines_in_word_counter += 1
                word_freq.update(words)

    # Word stats from word_freq (sampled if rate < 1)
    total_tok = sum(word_freq.values())
    types = len(word_freq)
    hapax = sum(1 for _w, c in word_freq.items() if c == 1)
    ttr = types / total_tok if total_tok else 0.0
    hapax_frac = hapax / types if types else 0.0

    ranked_freqs = sorted(word_freq.values(), reverse=True)
    def mass_at_top_ranks(k: int) -> float:
        if not ranked_freqs or total_tok == 0:
            return 0.0
        return min(1.0, sum(ranked_freqs[:k]) / total_tok)

    min_rank_for_mass = {}
    for target_mass in (0.5, 0.8, 0.9, 0.95):
        cum = 0
        for i, c in enumerate(ranked_freqs, start=1):
            cum += c
            if cum / total_tok >= target_mass:
                min_rank_for_mass[f"min_type_rank_for_{int(target_mass * 100)}pct_token_mass"] = i
                break
        else:
            min_rank_for_mass[f"min_type_rank_for_{int(target_mass * 100)}pct_token_mass"] = len(ranked_freqs)

    top_masses = {
        f"token_mass_top_{k}_types": mass_at_top_ranks(k)
        for k in (100, 500, 2000, 8000, 16000, 32000)
    }

    # Percentiles of word length (from histogram, over all words)
    percentiles = {}
    targets = (50, 90, 95, 99)
    for p in targets:
        threshold = math.ceil(n_words_total * (p / 100.0)) if n_words_total else 0
        cumulative = 0
        plen = None
        for L in range(WORD_LEN_CAP + 1):
            cumulative += int(len_hist[L])
            if cumulative >= threshold and plen is None:
                plen = L if L < WORD_LEN_CAP else f"{WORD_LEN_CAP}+"
                break
        percentiles[f"word_len_char_p{p}"] = plen

    mean_word_len = (
        float(sum(i * int(len_hist[i]) for i in range(WORD_LEN_CAP + 1)) / n_words_total)
        if n_words_total
        else 0.0
    )

    ranks_plot = []
    freqs_plot = []
    for r, c in enumerate(ranked_freqs[:max_rank_for_plot], start=1):
        ranks_plot.append(r)
        freqs_plot.append(c)

    summary = {
        "path": str(path.relative_to(PROJECT_ROOT)),
        "lines": n_lines,
        "lines_in_word_frequency_counter": n_lines_in_word_counter,
        "chars": n_chars,
        "utf8_bytes": utf8_bytes,
        "bytes_per_char": round(utf8_bytes / n_chars, 6) if n_chars else 0.0,
        "bytes_per_line_mean": round(utf8_bytes / n_lines, 2) if n_lines else 0.0,
        "char_set_size": len(char_set),
        "n_word_tokens_all_lines": n_words_total,
        "n_word_tokens_in_word_counter": total_tok,
        "word_sample_rate_effective": word_sample_rate,
        "unique_word_types_in_counter": types,
        "ttr_in_counter": round(ttr, 6),
        "hapax_types": hapax,
        "hapax_frac_of_types": round(hapax_frac, 6),
        "mean_word_len_chars": round(mean_word_len, 4),
        "word_len_percentiles": percentiles,
        **{k: round(v, 6) if isinstance(v, float) else v for k, v in min_rank_for_mass.items()},
        **{k: round(v, 6) for k, v in top_masses.items()},
        "word_len_histogram": len_hist.tolist(),
        "rank_freq_ranks": ranks_plot,
        "rank_freq_counts": freqs_plot,
    }
    return summary, char_set, word_freq


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    u = len(a | b)
    return len(a & b) / u if u else 0.0


def type_jaccard_from_counters(ca: Counter[str], cb: Counter[str]) -> float:
    sa, sb = set(ca), set(cb)
    return jaccard(sa, sb)


def overlap_coefficient_types(ca: Counter[str], cb: Counter[str]) -> float:
    """|A∩B| / min(|A|,|B|) — how much of the smaller vocab is shared."""
    sa, sb = set(ca), set(cb)
    inter = len(sa & sb)
    denom = min(len(sa), len(sb))
    return inter / denom if denom else 0.0


def plot_zipf(
    per_lang: dict[str, dict],
    out_path: Path,
    *,
    sampled: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"tr": "#2ecc71", "kk": "#3498db", "tt": "#e74c3c"}
    for lang in LANGS:
        if lang not in per_lang:
            continue
        r = per_lang[lang]["rank_freq_ranks"]
        f = per_lang[lang]["rank_freq_counts"]
        if not r:
            ax.plot([], [])
            continue
        ax.loglog(r, f, label=f"{lang}", color=colors.get(lang), alpha=0.85, linewidth=1.2)
    ax.set_xlabel("Rank r (by word frequency, descending)")
    ax.set_ylabel("Frequency (count in vocab counter)")
    title = "Word rank–frequency (train)"
    if sampled:
        title += " — note: word_counter sampled (--word-sample-rate < 1)"
    ax.set_title(title)
    ax.grid(True, alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_word_length_hists(per_lang: dict[str, dict], out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    colors = {"tr": "#2ecc71", "kk": "#3498db", "tt": "#e74c3c"}
    for ax, lang in zip(axes, LANGS):
        if lang not in per_lang:
            continue
        h = per_lang[lang]["word_len_histogram"]
        counts = np.array(h, dtype=float)
        xs = np.arange(WORD_LEN_CAP, dtype=float)
        ax.bar(
            xs,
            counts[:WORD_LEN_CAP],
            width=1.0,
            align="edge",
            color=colors.get(lang),
            alpha=0.85,
            edgecolor="white",
            linewidth=0.2,
        )
        last = counts[WORD_LEN_CAP]
        if last > 0:
            ax.bar([WORD_LEN_CAP], [last], width=1.0, color=colors.get(lang), alpha=0.5, label=f"{WORD_LEN_CAP}+")
        ax.set_xlabel("Word length (chars)")
        ax.set_title(f"{lang} (all words)")
        ax.set_xlim(0, min(40, WORD_LEN_CAP))
    axes[0].set_ylabel("Word count")
    fig.suptitle("Whitespace-token length (characters)", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_square_matrix_heatmap(
    matrix: np.ndarray,
    labels: list[str],
    out_path: Path,
    *,
    title: str,
    cbar_label: str,
) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(matrix, vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            v = matrix[i, j]
            ax.text(
                j,
                i,
                f"{v:.3f}",
                ha="center",
                va="center",
                color="w" if v < 0.5 else "k",
                fontsize=10,
            )
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=cbar_label)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Tokenizer design statistics (train-focused).")
    p.add_argument("--splits_root", type=Path, default=DEFAULT_SPLITS)
    p.add_argument("--out_dir", type=Path, default=DEFAULT_OUT)
    p.add_argument(
        "--word-sample-rate",
        type=float,
        default=1.0,
        help="Fraction of lines to include in word Counter / Zipf (0–1]. Length hists use all words.",
    )
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if not 0 < args.word_sample_rate <= 1:
        raise SystemExit("--word-sample-rate must be in (0, 1].")

    splits_root = args.splits_root.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    if not splits_root.is_dir():
        raise SystemExit(f"Not a directory: {splits_root}")
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    per_lang_summary: dict[str, dict] = {}
    char_sets: dict[str, set[str]] = {}
    word_counters: dict[str, Counter[str]] = {}

    for lang in LANGS:
        path = splits_root / lang / "train.txt"
        if not path.is_file():
            print(f"skip missing: {path}")
            continue
        print(f"scanning {path} ...")
        summary, cset, wf = stream_train_word_and_char_stats(
            path, word_sample_rate=args.word_sample_rate, rng=rng
        )
        per_lang_summary[lang] = summary
        char_sets[lang] = cset
        word_counters[lang] = wf

    labels = [L for L in LANGS if L in char_sets]
    n = len(labels)
    char_jac = np.eye(n)
    for i, a in enumerate(labels):
        for j, b in enumerate(labels):
            if i < j:
                v = jaccard(char_sets[a], char_sets[b])
                char_jac[i, j] = char_jac[j, i] = v

    union_chars: set[str] = set()
    for s in char_sets.values():
        union_chars |= s
    exclusive = {}
    exclusive_examples: dict[str, list[str]] = {}
    for lang in labels:
        others = set().union(*(char_sets[x] for x in labels if x != lang))
        only_here = char_sets[lang] - others
        exclusive[lang] = len(only_here)
        exclusive_examples[lang] = sorted(only_here)[:80]

    type_jac_mat = np.eye(n)
    overlap_coef_mat = np.eye(n)
    for i, a in enumerate(labels):
        for j, b in enumerate(labels):
            if i < j:
                ca, cb = word_counters[a], word_counters[b]
                type_jac_mat[i, j] = type_jac_mat[j, i] = type_jaccard_from_counters(ca, cb)
                overlap_coef_mat[i, j] = overlap_coef_mat[j, i] = overlap_coefficient_types(ca, cb)

    overlap_report = {
        "char_union_size": len(union_chars),
        "char_exclusive_type_count": exclusive,
        "char_exclusive_examples_up_to_80": exclusive_examples,
        "char_jaccard": {
            f"{labels[i]}_{labels[j]}": float(char_jac[i, j]) for i in range(n) for j in range(n) if i <= j
        },
        "word_type_jaccard_line_sampled": float(args.word_sample_rate < 1.0),
        "word_type_jaccard": {f"{labels[i]}_{labels[j]}": float(type_jac_mat[i, j]) for i in range(n) for j in range(n) if i <= j},
        "word_type_overlap_coefficient": {
            f"{labels[i]}_{labels[j]}": float(overlap_coef_mat[i, j]) for i in range(n) for j in range(n) if i <= j
        },
    }

    json_payload = {
        "meta": {
            "splits_root": str(splits_root.relative_to(PROJECT_ROOT)),
            "word_sample_rate": args.word_sample_rate,
            "seed": args.seed,
            "note_word_stats": "Zipf / TTR / hapax / token-mass-at-top-k use the word Counter, which subsamples lines when word_sample_rate < 1. Word-length histograms use all words.",
        },
        "per_language": {lang: {k: v for k, v in per_lang_summary[lang].items() if k not in ("word_len_histogram", "rank_freq_ranks", "rank_freq_counts")} for lang in per_lang_summary},
        "overlap": overlap_report,
    }

    json_path = out_dir / "tokenizer_design_stats.json"
    with json_path.open("w", encoding="utf-8") as jf:
        json.dump(json_payload, jf, indent=2, ensure_ascii=False)

    csv_path = out_dir / "word_stats_train.csv"
    flat_keys = [
        "lang",
        "lines",
        "chars",
        "bytes_per_char",
        "bytes_per_line_mean",
        "char_set_size",
        "n_word_tokens_all_lines",
        "n_word_tokens_in_word_counter",
        "unique_word_types_in_counter",
        "ttr_in_counter",
        "hapax_frac_of_types",
        "mean_word_len_chars",
        "min_type_rank_for_50pct_token_mass",
        "min_type_rank_for_80pct_token_mass",
        "min_type_rank_for_90pct_token_mass",
        "min_type_rank_for_95pct_token_mass",
        "token_mass_top_8000_types",
        "token_mass_top_16000_types",
        "token_mass_top_32000_types",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as cf:
        w = csv.DictWriter(cf, fieldnames=flat_keys, extrasaction="ignore")
        w.writeheader()
        for lang in LANGS:
            if lang not in per_lang_summary:
                continue
            row = {"lang": lang, **{k: per_lang_summary[lang].get(k, "") for k in flat_keys if k != "lang"}}
            w.writerow(row)

    plot_zipf(
        per_lang_summary,
        out_dir / "word_rank_frequency_train.png",
        sampled=args.word_sample_rate < 1.0,
    )
    plot_word_length_hists(per_lang_summary, out_dir / "word_length_histograms_train.png")
    plot_square_matrix_heatmap(
        char_jac,
        labels,
        out_dir / "char_set_jaccard_train.png",
        title="Character-set Jaccard (train)",
        cbar_label="Jaccard",
    )
    title_types = "Whitespace word-type Jaccard (train"
    title_types += "; line-sampled counter)" if args.word_sample_rate < 1.0 else ")"
    plot_square_matrix_heatmap(
        type_jac_mat,
        labels,
        out_dir / "word_type_jaccard_train.png",
        title=title_types,
        cbar_label="Jaccard",
    )
    plot_square_matrix_heatmap(
        overlap_coef_mat,
        labels,
        out_dir / "word_type_overlap_coefficient_train.png",
        title="Overlap coeff. |A∩B|/min(|A|,|B|) — word types (whitespace)",
        cbar_label="Overlap coef.",
    )

    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    print(f"Figures in {out_dir}")


if __name__ == "__main__":
    main()