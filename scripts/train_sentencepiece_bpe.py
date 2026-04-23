#!/usr/bin/env python3
"""Train a SentencePiece BPE model on one or more line-delimited text files."""

from __future__ import annotations

import argparse
from pathlib import Path

import sentencepiece as spm

TATAR_TEST_WORDS = [
    "эшчәнлек",
    "укытучылар",
    "мәктәптә",
    "татар",
    "балалар",
    "казан",
    "якынлык",
    "укыту",
    "чагыштыру",
    "хакимият",
]


def count_training_lines(paths: list[Path]) -> int:
    n = 0
    for p in paths:
        with p.open(encoding="utf-8", errors="replace") as f:
            n += sum(1 for _ in f)
    return n


def main() -> None:
    p = argparse.ArgumentParser(description="Train SentencePiece BPE tokenizer.")
    p.add_argument(
        "--input",
        nargs="+",
        type=Path,
        required=True,
        help="One or more training text files (one sentence/segment per line, UTF-8).",
    )
    p.add_argument(
        "--output_prefix",
        type=Path,
        required=True,
        help="Path prefix for outputs: <prefix>.model and <prefix>.vocab",
    )
    p.add_argument("--vocab_size", type=int, default=16000)
    p.add_argument("--character_coverage", type=float, default=0.9995)
    p.add_argument("--num_threads", type=int, default=4)
    args = p.parse_args()

    input_paths = [x.expanduser().resolve() for x in args.input]
    for ip in input_paths:
        if not ip.is_file():
            raise SystemExit(f"Input not found: {ip}")

    prefix = args.output_prefix.expanduser().resolve()
    prefix.parent.mkdir(parents=True, exist_ok=True)
    prefix_str = str(prefix)

    n_sents = count_training_lines(input_paths)
    input_csv = ",".join(str(x) for x in input_paths)

    spm.SentencePieceTrainer.train(
        input=input_csv,
        model_prefix=prefix_str,
        vocab_size=args.vocab_size,
        character_coverage=args.character_coverage,
        model_type="bpe",
        num_threads=args.num_threads,
    )

    model_path = Path(prefix_str + ".model")
    if not model_path.is_file():
        raise SystemExit(f"Expected model missing: {model_path}")

    sp = spm.SentencePieceProcessor(model_file=str(model_path))
    vocab_size = sp.get_piece_size()

    print(f"vocab_size: {vocab_size}")
    print(f"training_sentences (input lines): {n_sents}")
    print("example BPE tokenizations (Tatar test words):")
    for w in TATAR_TEST_WORDS:
        pieces = sp.encode_as_pieces(w)
        print(f"  {w!r} -> {pieces}")


if __name__ == "__main__":
    main()
