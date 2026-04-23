import json, logging, statistics
from pathlib import Path
from collections import Counter
from tokenizers import Tokenizer
import sentencepiece as spm

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)

BASE   = Path(__file__).parent.parent / "data"
TOK    = BASE / "tokenizers"
TEST_N = 10_000

def load_lines(path, n):
    with open(path, encoding="utf-8") as f:
        all_lines = [l.strip() for l in f if l.strip()]
    return all_lines[-n:]

def eval_hf(name, lines):
    f = TOK / name / "tokenizer.json"
    if not f.exists(): return None
    tok = Tokenizer.from_file(str(f))
    ferts, conts, oovs, total = [], [], 0, 0
    word_types = Counter()
    vocab = set(tok.get_vocab().keys())
    for line in lines:
        words = line.split()
        if not words: continue
        enc  = tok.encode(line)
        toks = enc.tokens
        if not toks: continue
        ferts.append(len(toks) / len(words))
        total += len(words)
        conts.append(sum(1 for t in toks if t.startswith("Ġ") or t == "[UNK]") / len(toks))
        for w in words:
            word_types[w] += 1
            if tok.encode(w).tokens == ["[UNK]"]: oovs += 1
    coverage = sum(1 for w in word_types if w in vocab) / len(word_types) if word_types else 0
    return dict(
        fertility_mean    = round(statistics.mean(ferts), 3),
        fertility_stdev   = round(statistics.stdev(ferts) if len(ferts) > 1 else 0, 3),
        continuation_rate = round(statistics.mean(conts), 3),
        oov_rate          = round(oovs / total, 4) if total else None,
        vocab_coverage    = round(coverage, 4),
    )

def eval_spm(name, lines):
    f = TOK / name / "spm.model"
    if not f.exists(): return None
    sp = spm.SentencePieceProcessor(model_file=str(f))
    ferts, conts, oovs, total = [], [], 0, 0
    word_types = Counter()
    for line in lines:
        words = line.split()
        if not words: continue
        pieces = sp.encode(line, out_type=str)
        if not pieces: continue
        ferts.append(len(pieces) / len(words))
        total += len(words)
        conts.append(sum(1 for p in pieces if not p.startswith("▁")) / len(pieces))
        for w in words:
            word_types[w] += 1
            if sp.encode(w, out_type=str) == ["<unk>"]: oovs += 1
    coverage = sum(1 for w in word_types
                   if len(sp.encode(w, out_type=str)) == 1) / len(word_types) \
               if word_types else 0
    return dict(
        fertility_mean    = round(statistics.mean(ferts), 3),
        fertility_stdev   = round(statistics.stdev(ferts) if len(ferts) > 1 else 0, 3),
        continuation_rate = round(statistics.mean(conts), 3),
        oov_rate          = round(oovs / total, 4) if total else None,
        vocab_coverage    = round(coverage, 4),
    )

VARIANTS = [
    ("A1_tatar_only_bpe",          eval_hf,  "native"),
    ("A2_tatar_only_unigram",      eval_spm, "native"),
    ("A3_tatar_byte_bpe",          eval_hf,  "native"),
    ("B1_turkish_bpe",             eval_hf,  "native"),
    ("B2_tur_kaz_bpe",             eval_hf,  "native"),
    ("B3_tur_kaz_tat_bpe",         eval_hf,  "native"),
    ("B4_tur_kaz_tat_unigram",     eval_spm, "native"),
    ("C1_latin_tur_kaz_tat_bpe",   eval_hf,  "latin"),
]

def run():
    results = {}
    native = load_lines(BASE / "processed/tt_combined.txt", TEST_N)
    latin  = load_lines(BASE / "normalized/tt_latin.txt",   TEST_N)

    print(f"\n{'Variant':<32} {'Fertility':>10} {'±':>6} "
          f"{'ContRate':>10} {'OOV%':>8} {'Coverage':>10}")
    print("─" * 82)
    for name, fn, script in VARIANTS:
        lines = latin if script == "latin" else native
        m = fn(name, lines)
        if not m:
            print(f"  {name:<30}  [not trained]"); continue
        results[name] = m
        print(f"  {name:<30} "
              f"{m['fertility_mean']:>9.3f} "
              f"{m['fertility_stdev']:>6.3f} "
              f"{m['continuation_rate']:>10.3f} "
              f"{m['oov_rate']*100:>7.2f}% "
              f"{m['vocab_coverage']*100:>9.1f}%")

    out = BASE / "results.json"
    json.dump(results, open(out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    log.info(f"Saved -> {out}")

if __name__ == "__main__":
    run()