#!/usr/bin/env python3
"""Shuffle and split a line-segmented corpus into train (90%), dev (5%), test (5%)."""

from __future__ import annotations

import argparse
import hashlib
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LANGS = ("tr", "kk", "tt")
SEED = 42


def load_lines(path: Path) -> list[str]:
    with path.open(encoding="utf-8", errors="replace") as f:
        return [line.rstrip("\n\r") for line in f]


def split_lines(lines: list[str], seed: int = SEED) -> tuple[list[str], list[str], list[str]]:
    """90% / 5% / 5% via integer partitioning: floor(n*5/100) for dev and test."""
    rng = random.Random(seed)
    bucket = list(lines)
    rng.shuffle(bucket)
    n = len(bucket)
    n_dev = n * 5 // 100
    n_test = n * 5 // 100
    n_train = n - n_dev - n_test
    train = bucket[:n_train]
    dev = bucket[n_train : n_train + n_dev]
    test = bucket[n_train + n_dev :]
    return train, dev, test


def write_split(output_dir: Path, train: list[str], dev: list[str], test: list[str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "train.txt").write_text("\n".join(train) + ("\n" if train else ""), encoding="utf-8")
    (output_dir / "dev.txt").write_text("\n".join(dev) + ("\n" if dev else ""), encoding="utf-8")
    (output_dir / "test.txt").write_text("\n".join(test) + ("\n" if test else ""), encoding="utf-8")


def _stream_bucket(seed: int, line_index: int) -> int:
    """Stable pseudo-random bucket in 0..99 from (seed, line index)."""
    h = hashlib.blake2b(digest_size=16)
    h.update(seed.to_bytes(8, "big", signed=True))
    h.update(line_index.to_bytes(8, "big", signed=False))
    return int.from_bytes(h.digest()[:8], "big") % 100


def run_split_streaming(input_path: Path, output_dir: Path, seed: int = SEED, label: str = "") -> None:
    """One-pass split: ~90/5/5 via hashed bucket per line (O(1) RAM)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.txt"
    dev_path = output_dir / "dev.txt"
    test_path = output_dir / "test.txt"
    n_train = n_dev = n_test = 0
    idx = 0
    with input_path.open(encoding="utf-8", errors="replace") as inf, train_path.open(
        "w", encoding="utf-8"
    ) as train_f, dev_path.open("w", encoding="utf-8") as dev_f, test_path.open(
        "w", encoding="utf-8"
    ) as test_f:
        for line in inf:
            text = line.rstrip("\n\r")
            b = _stream_bucket(seed, idx)
            idx += 1
            if b < 90:
                train_f.write(text + "\n")
                n_train += 1
            elif b < 95:
                dev_f.write(text + "\n")
                n_dev += 1
            else:
                test_f.write(text + "\n")
                n_test += 1
    prefix = f"[{label}] " if label else ""
    total = n_train + n_dev + n_test
    print(
        f"{prefix}{input_path.name} -> {output_dir}: "
        f"train={n_train} dev={n_dev} test={n_test} (total={total}, streaming ~90/5/5)"
    )


def run_split(
    input_path: Path,
    output_dir: Path,
    seed: int = SEED,
    label: str = "",
    *,
    shuffle_in_memory: bool = False,
) -> None:
    if shuffle_in_memory:
        lines = load_lines(input_path)
        train, dev, test = split_lines(lines, seed=seed)
        write_split(output_dir, train, dev, test)
        prefix = f"[{label}] " if label else ""
        print(
            f"{prefix}{input_path.name} -> {output_dir}: "
            f"train={len(train)} dev={len(dev)} test={len(test)} (total={len(lines)}, in-memory shuffle)"
        )
    else:
        run_split_streaming(input_path, output_dir, seed=seed, label=label)


def run_all_processed(root: Path, seed: int = SEED, *, shuffle_in_memory: bool = False) -> None:
    """Split native_script/{lang}.txt and normalized_script/{lang}_norm.txt for tr, kk, tt."""
    proc = root / "data" / "processed"
    native_base = proc / "native_script"
    norm_base = proc / "normalized_script"

    for lang in LANGS:
        inp = native_base / f"{lang}.txt"
        out = native_base / "splits" / lang
        if inp.is_file():
            run_split(
                inp, out, seed=seed, label=f"native_script/{lang}", shuffle_in_memory=shuffle_in_memory
            )
        else:
            print(f"[native_script/{lang}] skip — missing {inp}")

    for lang in LANGS:
        inp = norm_base / f"{lang}_norm.txt"
        out = norm_base / "splits" / lang
        if inp.is_file():
            run_split(
                inp, out, seed=seed, label=f"normalized_script/{lang}", shuffle_in_memory=shuffle_in_memory
            )
        else:
            print(f"[normalized_script/{lang}] skip — missing {inp}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Shuffle (seed=42) and split lines into train.txt, dev.txt, test.txt (90/5/5)."
    )
    p.add_argument("--input", type=Path, default=None, help="Processed text file (one segment per line).")
    p.add_argument("--output_dir", type=Path, default=None, help="Directory for train.txt, dev.txt, test.txt.")
    p.add_argument(
        "--run-all-processed",
        action="store_true",
        help=f"Run splits for native_script and normalized_script for {', '.join(LANGS)} under project data/processed/.",
    )
    p.add_argument("--seed", type=int, default=SEED, help=f"RNG seed (default {SEED}).")
    p.add_argument(
        "--shuffle-in-memory",
        action="store_true",
        help="Load all lines, shuffle, then split (exact 90/5/5 counts; high RAM on large files). "
        "Default is a one-pass hashed split (~90/5/5, constant memory).",
    )
    args = p.parse_args()

    if args.run_all_processed:
        run_all_processed(PROJECT_ROOT, seed=args.seed, shuffle_in_memory=args.shuffle_in_memory)
        return

    if args.input is None or args.output_dir is None:
        raise SystemExit("Provide --input and --output_dir, or use --run-all-processed.")

    inp = args.input.expanduser()
    out = args.output_dir.expanduser()
    if not inp.is_absolute():
        inp = (PROJECT_ROOT / inp).resolve()
    if not out.is_absolute():
        out = (PROJECT_ROOT / out).resolve()

    if not inp.is_file():
        raise SystemExit(f"Input not found: {inp}")

    run_split(inp, out, seed=args.seed, shuffle_in_memory=args.shuffle_in_memory)


if __name__ == "__main__":
    main()
