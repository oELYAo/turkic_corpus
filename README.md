# Turkic corpus — tokenizer transfer study

Code and configs for comparing **SentencePiece** tokenizers for **Tatar** (`tt`) when trained on Tatar only, on **Turkish** (`tr`), on **Turkish + Kazakh** (`kk`), and with a **byte-fallback BPE** reference — in both **native** and **Latin-normalized** scripts. Intrinsic metrics (fertility, fragmentation, coverage) and a lightweight **UD Tatar** dependency-parsing experiment quantify transfer effects. 
## Requirements

- Python 3.10+ (3.11 used in development)
- A virtual environment under `turkic_corpus/.venv` is recommended (see below)

Install dependencies:

```bash
cd turkic_corpus
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

`requirements.txt` pins `datasets` below 4.x so `statmt/cc100` and similar script-based loaders keep working.

## Quick reproducible path

From `turkic_corpus/`, after you have **processed train/test splits** (see [Corpus preparation](#corpus-preparation)):

```bash
./run_pipeline.sh
```

This script:

1. Ensures `.venv` exists and installs `requirements.txt`
2. Runs `scripts/download_ud_treebanks.sh` (UD CoNLL-U under `data/treebanks/`, plus `data/ud/tt_nmctt-ud-test.conllu` for downstream)
3. Trains and evaluates transfer tokenizers: `scripts/run_transfer_experiment.py --train --eval`
4. Runs Tatar UD k-fold dependency parsing: `scripts/run_downstream_depparse.py --script both --vocab-size 16000`
5. Merges summaries: `scripts/merge_downstream_summaries.py --vocab-size 16000`

**Outputs**

- Intrinsic: `results/intrinsic/transfer_experiment_summary.json` (and `transfer_experiment_scores.csv`)
- Downstream: `results/downstream/ud_depparse_tatar_merged_v16000.json`

Override the interpreter with `PY=/path/to/python ./run_pipeline.sh` if needed.

## Repository layout

| Path | Role |
|------|------|
| `config/project_paths.yaml` | Canonical relative paths for raw/processed data, tokenizer outputs, results |
| `config/evaluation.yaml` | Evaluation anchors (some paths overlap with `project_paths`; align when changing layout) |
| `config/sentencepiece_training.yaml` | SentencePiece training defaults |
| `data/raw/` | Language-prefixed raw dumps (ignored in git; see `.gitignore`) |
| `data/processed/` | Cleaned text, splits (`native_script/…`, `normalized_script/…`) |
| `data/ud/` | Tatar test CoNLL-U used by the depparse script |
| `data/treebanks/` | UD treebanks fetched by `download_ud_treebanks.sh` |
| `outputs/` | Trained tokenizers and experiment checkpoints (ignored in git) |
| `results/` | Metrics tables and JSON summaries (tracked selectively) |
| `scripts/` | Download, cleaning, normalization, training, and evaluation |

## Corpus preparation

`run_transfer_experiment.py` expects **per-language** `train.txt` / `test.txt` (and related splits) under:

- `data/processed/native_script/splits/<tr|kk|tt>/`
- `data/processed/normalized_script/splits_sanitized/<tr|kk|tt>/`

Building those steps is **not** fully automated in `run_pipeline.sh` (Wikipedia dumps, CC-100, wikiextractor, etc. are heavy). Relevant helpers include:

- `scripts/download_cc100.py` — CC-100 via Hugging Face
- `scripts/clean_wiki_extracted.py`, `scripts/clean_raw_text.py` — text cleanup
- `scripts/sanitize_corpus_unicode.py` — Unicode normalization for sanitized splits
- `scripts/normalize_script.py`, `scripts/preprocess.py` — script / Latin pipeline
- `scripts/split_corpus.py` — train/dev/test splits
- `scripts/audit_tail_characters.py` — corpus QA
- `scripts/download_kazparc.sh` — optional parallel Kazakh–Turkish data
- `scripts/train_sentencepiece_bpe.py`, `scripts/train_tokenizers.py` — standalone tokenizer training utilities

Use `prompts.txt` for the recommended Wikipedia + CC-100 + UD stack.

## Main scripts (experiments)

| Script | Purpose |
|--------|---------|
| `run_transfer_experiment.py` | Train all study conditions + intrinsic metrics on Tatar test (native and normalized) |
| `intrinsic_tokenizer_metrics.py` | Fertility, fragmentation, word-level coverage (imported by the runner) |
| `run_downstream_depparse.py` | BiLSTM depparse on UD Tatar with frozen SentencePiece; `--script native\|normalized\|both` |
| `merge_downstream_summaries.py` | Combine per-script downstream CSV/JSON summaries |
| `extract_ud_wordlist_for_intrinsics.py` | Build word lists from full UD CoNLL-U when available locally |
| `analyze_tokenizer_training_corpus.py`, `analyze_tokenizer_design_stats.py` | Corpus / tokenizer statistics |

**Examples**

```bash
# Fast smoke run (smaller vocab / quicker train where supported — see `--help`)
.venv/bin/python scripts/run_transfer_experiment.py --train --eval --quick

# Downstream only (after models exist)
.venv/bin/python scripts/run_downstream_depparse.py --script both --vocab-size 16000
```

## Conditions

Training conditions (see `TRAIN_MATRIX` in `run_transfer_experiment.py`): `tt_only`, `transfer_tr`, `transfer_tr_kk`, `transfer_tr_kk_tt`, and `byte_fallback_bpe` (BPE with byte fallback on `tr+kk+tt`). Models are written under `outputs/experiments/<condition>/<train_script>/vocab_<N>/`.
