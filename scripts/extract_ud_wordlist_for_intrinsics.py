#!/usr/bin/env python3
"""
Build a whitespace-tokenized word stream from CoNLL-U for UD-aligned intrinsics.

Concatenates FORM (or LEMMA) fields in treebank order: file order, then sentence
order, then token order — the same surface units the parser sees. Feed the
output to intrinsic_tokenizer_metrics.py as --test_file.

Example (after downloading full tt_nmctt splits into data/treebanks/tt/):

  python3 scripts/extract_ud_wordlist_for_intrinsics.py \\
    --out data/processed/ud_intrinsic/tt_nmctt_forms_train_dev_test.txt \\
    data/treebanks/tt/tt-ud-train.conllu \\
    data/treebanks/tt/tt-ud-dev.conllu \\
    data/treebanks/tt/tt-ud-test.conllu

  python3 scripts/intrinsic_tokenizer_metrics.py \\
    --model outputs/experiments/transfer_tr_kk/native/vocab_16000/spm.model \\
    --test_file data/processed/ud_intrinsic/tt_nmctt_forms_train_dev_test.txt \\
    --lang tt --condition transfer_tr_kk_ud_forms
"""

from __future__ import annotations

import argparse
from pathlib import Path

from conllu import parse_incr

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def collect_sentence_lines(paths: list[Path], *, field: str) -> list[str]:
    lines: list[str] = []
    for path in paths:
        with path.open(encoding="utf-8") as f:
            for sent in parse_incr(f):
                toks: list[str] = []
                for t in sent:
                    if not isinstance(t["id"], int):
                        continue
                    if field == "form":
                        toks.append(t["form"])
                    else:
                        lem = t["lemma"]
                        if lem is not None and lem != "_":
                            toks.append(lem)
                if toks:
                    lines.append(" ".join(toks))
    return lines


def main() -> None:
    p = argparse.ArgumentParser(
        description="Extract FORM/LEMMA stream from CoNLL-U for tokenizer intrinsics."
    )
    p.add_argument(
        "conllu",
        type=Path,
        nargs="+",
        help="One or more .conllu files (e.g. train dev test).",
    )
    p.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output .txt: one UD sentence per line, space-joined tokens.",
    )
    p.add_argument(
        "--field",
        choices=("form", "lemma"),
        default="form",
        help="CoNLL-U column to emit (default: form).",
    )
    p.add_argument(
        "--dedupe",
        action="store_true",
        help="Emit each distinct token once (sorted), single output line — type-level, not token frequency.",
    )
    args = p.parse_args()

    paths = [p.expanduser().resolve() for p in args.conllu]
    for path in paths:
        if not path.is_file():
            raise SystemExit(f"Not found: {path}")

    lines = collect_sentence_lines(paths, field=args.field)

    out_path = args.out
    if not out_path.is_absolute():
        out_path = (PROJECT_ROOT / out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.dedupe:
        uniq = sorted({tok for line in lines for tok in line.split()})
        with out_path.open("w", encoding="utf-8", newline="\n") as f:
            f.write(" ".join(uniq) + "\n")
        n_lines, n_tokens = 1, len(uniq)
    else:
        with out_path.open("w", encoding="utf-8", newline="\n") as f:
            for line in lines:
                f.write(line + "\n")
        n_lines, n_tokens = len(lines), sum(len(line.split()) for line in lines)

    print(f"wrote {out_path}")
    print(f"  files={len(paths)} field={args.field} dedupe={args.dedupe}")
    print(f"  ud_sentences={n_lines} tokens={n_tokens}")


if __name__ == "__main__":
    main()
