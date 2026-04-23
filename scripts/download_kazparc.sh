#!/usr/bin/env bash
# Clone IS2AI/KazParC from GitHub, fetch Kazakh–Turkish parallel CSVs from Hugging Face
# (issai/kazparc; the Git repo does not ship the corpus files), and build aligned line files.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

DEST="${ROOT}/data/parallel/kazparc"
REPO_URL="https://github.com/IS2AI/KazParC.git"
HF_DATASET="issai/kazparc"
HF_LOCAL="${DEST}/hf_dataset"

# Order: KazParC train / valid / test, then SynC train / valid (HF paths; see dataset card).
KK_TR_CSVS=(
  "kazparc/04_kazparc_train_kk_tr.csv"
  "kazparc/10_kazparc_valid_kk_tr.csv"
  "kazparc/16_kazparc_test_kk_tr.csv"
  "sync/23_sync_train_kk_tr.csv"
  "sync/29_sync_valid_kk_tr.csv"
)

mkdir -p "$(dirname "${DEST}")"

if [[ -d "${DEST}/.git" ]]; then
  echo ">>> Updating Git clone: ${DEST}"
  git -C "${DEST}" pull --ff-only
else
  if [[ -e "${DEST}" ]] && [[ -n "$(ls -A "${DEST}" 2>/dev/null)" ]]; then
    echo "error: ${DEST} exists, is not a clone of ${REPO_URL}, and is not empty." >&2
    exit 1
  fi
  echo ">>> Cloning ${REPO_URL} -> ${DEST}"
  git clone "${REPO_URL}" "${DEST}"
fi

if ! command -v huggingface-cli >/dev/null 2>&1; then
  echo "error: huggingface-cli not found (pip install huggingface_hub)" >&2
  exit 1
fi

mkdir -p "${HF_LOCAL}"
echo ">>> Downloading KK–TR CSVs from Hugging Face (${HF_DATASET}) into ${HF_LOCAL}"
echo "    (Gated dataset: use 'huggingface-cli login' and accept access on the dataset page.)"
for rel in "${KK_TR_CSVS[@]}"; do
  huggingface-cli download "${HF_DATASET}" "${rel}" \
    --repo-type dataset \
    --local-dir "${HF_LOCAL}"
done

OUT_KK="${DEST}/kk_parallel.txt"
OUT_TR="${DEST}/tr_parallel.txt"

{
  for r in "${KK_TR_CSVS[@]}"; do
    printf '%s\n' "${HF_LOCAL}/${r}"
  done
} > "${DEST}/._kazparc_kk_tr_manifest.txt"

echo ">>> Building ${OUT_KK} and ${OUT_TR}"
PAIR_COUNT="$(
  DEST="${DEST}" OUT_KK="${OUT_KK}" OUT_TR="${OUT_TR}" python3 - <<'PY'
import csv
import os
from pathlib import Path

manifest = Path(os.environ["DEST"]) / "._kazparc_kk_tr_manifest.txt"
out_kk = Path(os.environ["OUT_KK"])
out_tr = Path(os.environ["OUT_TR"])

paths = [ln.strip() for ln in manifest.read_text(encoding="utf-8").splitlines() if ln.strip()]
n = 0
with out_kk.open("w", encoding="utf-8") as fkk, out_tr.open("w", encoding="utf-8") as ftr:
    for path_str in paths:
        path = Path(path_str)
        if not path.is_file():
            raise SystemExit(f"missing file: {path}")
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                kk = " ".join((row.get("kk") or "").split())
                tr = " ".join((row.get("tr") or "").split())
                fkk.write(kk + "\n")
                ftr.write(tr + "\n")
                n += 1
print(n)
PY
)"

rm -f "${DEST}/._kazparc_kk_tr_manifest.txt"

KK_LINES="$(wc -l < "${OUT_KK}" | tr -d '[:space:]')"
TR_LINES="$(wc -l < "${OUT_TR}" | tr -d '[:space:]')"
if [[ "${KK_LINES}" != "${TR_LINES}" ]]; then
  echo "error: line count mismatch: kk=${KK_LINES} tr=${TR_LINES}" >&2
  exit 1
fi

if [[ "${PAIR_COUNT}" != "${KK_LINES}" ]]; then
  echo "error: Python wrote ${PAIR_COUNT} pairs but wc -l reports ${KK_LINES}" >&2
  exit 1
fi

echo ">>> Sentence pairs (aligned lines per file): ${KK_LINES}"
