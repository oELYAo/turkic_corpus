#!/usr/bin/env bash
# Minimal reproducible path for tokenizer + downstream numbers in prompts.txt.
# Heavy corpus steps (Wikipedia, CC-100, preprocess) stay manual — see scripts/.
set -euo pipefail
cd "$(dirname "$0")"

PY="${PY:-}"
if [ -z "$PY" ]; then
  if [ -x .venv/bin/python ]; then
    PY=".venv/bin/python"
  else
    echo ">>> Creating .venv (install turkic_corpus/requirements.txt here)"
    python3 -m venv .venv
    PY=".venv/bin/python"
  fi
fi

echo ">>> Dependencies ($PY)"
"$PY" -m pip install -q -r requirements.txt

echo ">>> UD: treebank curls + Tatar test CoNLL-U (required for downstream)"
bash scripts/download_ud_treebanks.sh

echo ">>> Train + intrinsic eval (SentencePiece transfer matrix)"
"$PY" scripts/run_transfer_experiment.py --train --eval

echo ">>> Downstream: UD Tatar k-fold depparse (native + Latin-normalized)"
"$PY" scripts/run_downstream_depparse.py --script both --vocab-size 16000

echo ">>> Merge downstream summary tables"
"$PY" scripts/merge_downstream_summaries.py --vocab-size 16000

echo "Done."
echo "  Intrinsic: results/intrinsic/transfer_experiment_summary.json"
echo "  Downstream: results/downstream/ud_depparse_tatar_merged_v16000.json"
