#!/usr/bin/env python3
"""
End-to-end runner for the Turkic tokenizer transfer study (prompts.txt).

Trains SentencePiece models for matched vocabulary sizes and conditions:
  - tt_only
  - transfer_tr (Turkish-only training data)
  - transfer_tr_kk
  - transfer_tr_kk_tt
  - byte_fallback_bpe (BPE + byte_fallback on tr+kk+tt; reference byte-capable subword model)

For each *training script* (native vs normalized Latin), writes models under:
  outputs/experiments/<condition>/<train_script>/vocab_<N>/spm.model

Evaluates **Tatar** intrinsic metrics (fertility, fragmentation, word-level UNK-free coverage)
on **both** native and normalized Tatar *test* splits so you can read off script effects.

Example:
  cd turkic_corpus && .venv/bin/python scripts/run_transfer_experiment.py --train --eval --quick
  cd turkic_corpus && .venv/bin/python scripts/run_transfer_experiment.py --train --eval
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import sentencepiece as spm

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from intrinsic_tokenizer_metrics import compute_metrics, load_words  # noqa: E402

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

NATIVE_SPLIT = PROJECT_ROOT / "data" / "processed" / "native_script" / "splits"
NORM_SPLIT = PROJECT_ROOT / "data" / "processed" / "normalized_script" / "splits_sanitized"
EXPERIMENT_ROOT = PROJECT_ROOT / "outputs" / "experiments"
DEFAULT_RESULTS_JSON = PROJECT_ROOT / "results" / "intrinsic" / "transfer_experiment_summary.json"
DEFAULT_RESULTS_CSV = PROJECT_ROOT / "results" / "intrinsic" / "transfer_experiment_scores.csv"


@dataclass(frozen=True)
class TrainJob:
    condition: str
    train_langs: tuple[str, ...]
    model_type: str  # "unigram" | "bpe"
    byte_fallback: bool


TRAIN_MATRIX: tuple[TrainJob, ...] = (
    TrainJob("tt_only", ("tt",), "unigram", False),
    TrainJob("transfer_tr", ("tr",), "unigram", False),
    TrainJob("transfer_tr_kk", ("tr", "kk"), "unigram", False),
    TrainJob("transfer_tr_kk_tt", ("tr", "kk", "tt"), "unigram", False),
    TrainJob("byte_fallback_bpe", ("tr", "kk", "tt"), "bpe", True),
)


def split_train_path(train_script: str, lang: str) -> Path:
    root = NATIVE_SPLIT if train_script == "native" else NORM_SPLIT
    p = root / lang / "train.txt"
    if not p.is_file():
        raise FileNotFoundError(f"Missing training file: {p}")
    return p


def tatar_test_path(test_script: str) -> Path:
    root = NATIVE_SPLIT if test_script == "native" else NORM_SPLIT
    p = root / "tt" / "test.txt"
    if not p.is_file():
        raise FileNotFoundError(f"Missing Tatar test file: {p}")
    return p


def model_dir(condition: str, train_script: str, vocab_size: int) -> Path:
    return EXPERIMENT_ROOT / condition / train_script / f"vocab_{vocab_size}"


def model_path(condition: str, train_script: str, vocab_size: int) -> Path:
    return model_dir(condition, train_script, vocab_size) / "spm.model"


def train_sentencepiece(
    inputs: list[Path],
    out_dir: Path,
    vocab_size: int,
    model_type: str,
    *,
    byte_fallback: bool = False,
    character_coverage: float = 0.9995,
    input_sentence_size: int = 0,
    max_sentence_length: int = 8192,
    force: bool = False,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = out_dir / "spm"
    dst_model = Path(str(prefix) + ".model")
    if dst_model.is_file() and not force:
        log.info("skip train (exists): %s", dst_model)
        return dst_model

    for f in inputs:
        if not f.is_file():
            raise FileNotFoundError(f"Training input missing: {f}")

    kwargs: dict = {
        "input": ",".join(str(f.resolve()) for f in inputs),
        "model_prefix": str(prefix),
        "vocab_size": vocab_size,
        "character_coverage": character_coverage,
        "model_type": model_type,
        "num_threads": int(os.environ.get("SPM_NUM_THREADS", os.cpu_count() or 4)),
        "shuffle_input_sentence": True,
    }
    if input_sentence_size and input_sentence_size > 0:
        kwargs["input_sentence_size"] = input_sentence_size
    if max_sentence_length and max_sentence_length > 0:
        kwargs["max_sentence_length"] = max_sentence_length
    if model_type == "bpe" and byte_fallback:
        kwargs["byte_fallback"] = True

    log.info(
        "training SPM type=%s vocab=%s byte_fallback=%s -> %s",
        model_type,
        vocab_size,
        byte_fallback,
        dst_model,
    )
    spm.SentencePieceTrainer.train(**kwargs)
    if not dst_model.is_file():
        raise RuntimeError(f"Expected model not written: {dst_model}")
    return dst_model


def evaluate_one(
    model_file: Path,
    test_file: Path,
    *,
    condition: str,
    train_script: str,
    test_script: str,
    vocab_size: int,
) -> dict[str, object]:
    sp = spm.SentencePieceProcessor(model_file=str(model_file))
    words = load_words(test_file)
    fertility, fragmentation_rate, coverage = compute_metrics(sp, words)
    return {
        "condition": condition,
        "train_script": train_script,
        "test_script": test_script,
        "vocab_size_requested": vocab_size,
        "vocab_size_model": sp.get_piece_size(),
        "lang": "tt",
        "n_words": len(words),
        "fertility": fertility,
        "fragmentation_rate": fragmentation_rate,
        "coverage": coverage,
        "model_path": str(model_file),
        "test_path": str(test_file),
    }


def append_csv_rows(path: Path, rows: Iterable[dict[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    new_file = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if new_file:
            w.writeheader()
        for r in rows:
            w.writerow({k: "" if r.get(k) is None else r[k] for k in fieldnames})


def run_train(
    *,
    train_scripts: tuple[str, ...],
    vocab_sizes: list[int],
    input_sentence_size: int,
    max_sentence_length: int,
    force: bool,
) -> list[dict[str, object]]:
    manifest: list[dict[str, object]] = []
    for train_script in train_scripts:
        for job in TRAIN_MATRIX:
            inputs = [split_train_path(train_script, lg) for lg in job.train_langs]
            for vs in vocab_sizes:
                out = model_dir(job.condition, train_script, vs)
                m = train_sentencepiece(
                    inputs,
                    out,
                    vs,
                    job.model_type,
                    byte_fallback=job.byte_fallback,
                    input_sentence_size=input_sentence_size,
                    max_sentence_length=max_sentence_length,
                    force=force,
                )
                manifest.append(
                    {
                        **asdict(job),
                        "train_script": train_script,
                        "vocab_size": vs,
                        "inputs": [str(p) for p in inputs],
                        "model_path": str(m),
                    }
                )
    man_path = EXPERIMENT_ROOT / "manifest.json"
    man_path.parent.mkdir(parents=True, exist_ok=True)
    man_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    log.info("wrote manifest: %s (%d models)", man_path, len(manifest))
    return manifest


def run_eval(
    *,
    train_scripts: tuple[str, ...],
    vocab_sizes: list[int],
    results_json: Path,
    results_csv: Path,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for train_script in train_scripts:
        for job in TRAIN_MATRIX:
            for vs in vocab_sizes:
                mp = model_path(job.condition, train_script, vs)
                if not mp.is_file():
                    log.warning("missing model, skip eval: %s", mp)
                    continue
                for test_script in ("native", "normalized"):
                    tt_test = tatar_test_path(test_script)
                    r = evaluate_one(
                        mp,
                        tt_test,
                        condition=job.condition,
                        train_script=train_script,
                        test_script=test_script,
                        vocab_size=vs,
                    )
                    rows.append(r)

    results_json.parent.mkdir(parents=True, exist_ok=True)
    results_json.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    log.info("wrote JSON: %s (%d rows)", results_json, len(rows))

    # Flatten floats for CSV readability
    csv_rows = []
    for r in rows:
        csv_rows.append(
            {
                **r,
                "fertility": f"{r['fertility']:.10g}",
                "fragmentation_rate": f"{r['fragmentation_rate']:.10g}",
                "coverage": f"{r['coverage']:.10g}",
            }
        )
    append_csv_rows(results_csv, csv_rows)
    log.info("appended CSV: %s", results_csv)
    return rows


def parse_vocab_sizes(s: str) -> list[int]:
    out = [int(x.strip()) for x in s.split(",") if x.strip()]
    if not out:
        raise argparse.ArgumentTypeError("vocab_sizes empty")
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")

    p = argparse.ArgumentParser(description="Train + eval Turkic transfer tokenizers (Tatar-centric).")
    p.add_argument("--train", action="store_true", help="Train all SentencePiece conditions.")
    p.add_argument("--eval", action="store_true", help="Evaluate trained models on Tatar test (native+normalized).")
    p.add_argument(
        "--train-scripts",
        default="native,normalized",
        help="Comma list: native,normalized (default both).",
    )
    p.add_argument(
        "--vocab-sizes",
        type=parse_vocab_sizes,
        default=parse_vocab_sizes("8000,16000,32000"),
        help="Comma-separated vocab sizes (default 8000,16000,32000).",
    )
    p.add_argument(
        "--input-sentence-size",
        type=int,
        default=1_000_000,
        help="SPM training cap on shuffled sentences per run (0 = no cap). Default 1e6 for speed.",
    )
    p.add_argument(
        "--max-sentence-length",
        type=int,
        default=8192,
        help="SPM max line length (default 8192; SPM default 4192 skips long wiki lines).",
    )
    p.add_argument("--force", action="store_true", help="Retrain even if spm.model exists.")
    p.add_argument(
        "--quick",
        action="store_true",
        help="Small smoke run: vocab_sizes=4000 and input_sentence_size=50_000.",
    )
    p.add_argument("--results-json", type=Path, default=DEFAULT_RESULTS_JSON)
    p.add_argument("--results-csv", type=Path, default=DEFAULT_RESULTS_CSV)
    args = p.parse_args()

    if not args.train and not args.eval:
        p.error("pass at least one of --train / --eval")

    train_scripts = tuple(x.strip() for x in args.train_scripts.split(",") if x.strip())
    for s in train_scripts:
        if s not in ("native", "normalized"):
            p.error(f"unknown train_script: {s}")

    vocab_sizes = [4000] if args.quick else list(args.vocab_sizes)
    input_sentence_size = 50_000 if args.quick else args.input_sentence_size

    if args.train:
        run_train(
            train_scripts=train_scripts,
            vocab_sizes=vocab_sizes,
            input_sentence_size=input_sentence_size,
            max_sentence_length=args.max_sentence_length,
            force=args.force,
        )
    if args.eval:
        run_eval(
            train_scripts=train_scripts,
            vocab_sizes=vocab_sizes,
            results_json=args.results_json.resolve(),
            results_csv=args.results_csv.resolve(),
        )


if __name__ == "__main__":
    main()
