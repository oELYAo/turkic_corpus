import os, logging
from pathlib import Path
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders
import sentencepiece as spm

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)

BASE  = Path(__file__).parent.parent / "data"
PROC  = BASE / "processed"
NORM  = BASE / "normalized"
TOK   = BASE / "tokenizers"
TOK.mkdir(parents=True, exist_ok=True)

VOCAB = 32_000
SPE   = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]

def p(lang, script="native"):
    return (NORM / f"{lang}_latin.txt") if script == "latin" \
           else (PROC / f"{lang}_combined.txt")

def existing(files):
    return [str(f) for f in files if Path(f).exists()]

def bpe(name, files, byte_level=False):
    out = TOK / name / "tokenizer.json"
    (TOK / name).mkdir(exist_ok=True)
    if out.exists(): log.info(f"[{name}] skip"); return
    srcs = existing(files)
    if not srcs: log.error(f"[{name}] no inputs"); return
    log.info(f"[{name}] Training BPE on {[Path(s).name for s in srcs]}...")
    tok = Tokenizer(models.BPE(unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder       = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB, special_tokens=SPE,
        min_frequency=2, show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet() if byte_level else [],
    )
    tok.train(srcs, trainer)
    tok.save(str(out))
    log.info(f"  -> {out}")

def unigram(name, files):
    prefix = str(TOK / name / "spm")
    (TOK / name).mkdir(exist_ok=True)
    if Path(prefix + ".model").exists(): log.info(f"[{name}] skip"); return
    srcs = existing(files)
    if not srcs: log.error(f"[{name}] no inputs"); return
    combined = TOK / name / "_input.txt"
    with open(combined, "w", encoding="utf-8") as fw:
        for s in srcs:
            with open(s, encoding="utf-8") as fr:
                for line in fr: fw.write(line)
    log.info(f"[{name}] Training Unigram SPM...")
    spm.SentencePieceTrainer.train(
        input=str(combined), model_prefix=prefix,
        vocab_size=VOCAB, model_type="unigram",
        character_coverage=0.9995,
        pad_id=3, unk_id=0, bos_id=1, eos_id=2,
        pad_piece="[PAD]", unk_piece="[UNK]",
        bos_piece="[CLS]", eos_piece="[SEP]",
        num_threads=os.cpu_count() or 4,
        input_sentence_size=5_000_000,
        shuffle_input_sentence=True,
    )
    combined.unlink()
    log.info(f"  -> {prefix}.model")

if __name__ == "__main__":
    tt, kk, tr = p("tt"), p("kk"), p("tr")
    tl, kl, rl = p("tt","latin"), p("kk","latin"), p("tr","latin")

    bpe("A1_tatar_only_bpe",         [tt])
    unigram("A2_tatar_only_unigram", [tt])
    bpe("A3_tatar_byte_bpe",         [tt], byte_level=True)

    bpe("B1_turkish_bpe",            [tr])
    bpe("B2_tur_kaz_bpe",            [tr, kk])
    bpe("B3_tur_kaz_tat_bpe",        [tr, kk, tt])
    unigram("B4_tur_kaz_tat_unigram",[tr, kk, tt])

    bpe("C1_latin_tur_kaz_tat_bpe",  [rl, kl, tl])

    print("\nAll tokenizers:")
    for d in sorted(TOK.iterdir()):
        print(f"  {d.name}/  ({len(list(d.iterdir()))} file(s))")