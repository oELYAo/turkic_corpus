#!/usr/bin/env bash
# Download latest pages-articles XML dumps from dumps.wikimedia.org and plain-text
# extract with wikiextractor into data/raw/{tr,kk,tt}/.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

if ! command -v wget >/dev/null 2>&1; then
  echo "error: wget is required (e.g. brew install wget)" >&2
  exit 1
fi

if ! command -v wikiextractor >/dev/null 2>&1; then
  echo "error: wikiextractor not on PATH (pip install wikiextractor)" >&2
  exit 1
fi

# WikiExtractor defaults: (1) a first pass loads *all* template definitions from the dump
# into RAM — often tens of GB on large wikis; (2) many worker processes multiply memory
# and enlarge the reducer's out-of-order buffer. Use --no-templates for one-pass streaming
# (raw wikitext markers may remain; fine for plain LM/tokenizer corpora). Lower parallel
# jobs on small-RAM machines, e.g. WIKIEXTRACTOR_PROCESSES=1.
: "${WIKIEXTRACTOR_PROCESSES:=2}"

download_and_extract() {
  local wiki="$1" lang="$2"
  local url="https://dumps.wikimedia.org/${wiki}/latest/${wiki}-latest-pages-articles.xml.bz2"
  local dest="data/raw/${lang}/wiki_dump.xml.bz2"
  local extract_dir="data/raw/${lang}/wiki_extracted"

  mkdir -p "$(dirname "${dest}")"

  if [[ -f "${dest}" ]]; then
    echo "[${lang}] Dump already present, skipping download: ${dest}"
  else
    echo "[${lang}] Downloading ${url}"
    wget --continue --output-document="${dest}" "${url}"
  fi

  echo "[${lang}] WikiExtractor -> ${extract_dir} (--no-templates, --processes=${WIKIEXTRACTOR_PROCESSES})"
  rm -rf "${extract_dir}"
  mkdir -p "${extract_dir}"
  wikiextractor \
    --no-templates \
    --processes "${WIKIEXTRACTOR_PROCESSES}" \
    -q \
    --output "${extract_dir}" \
    "${dest}"
}

download_and_extract trwiki tr
download_and_extract kkwiki kk
download_and_extract ttwiki tt

echo "Done. Dumps under data/raw/{tr,kk,tt}/wiki_dump.xml.bz2; text under wiki_extracted/."
