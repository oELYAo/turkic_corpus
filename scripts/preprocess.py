import re, logging
from pathlib import Path
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)

BASE  = Path(__file__).parent.parent / "data"
PROC  = BASE / "processed"
NORM  = BASE / "normalized"
NORM.mkdir(parents=True, exist_ok=True)

SOURCES = {
    "tt": [PROC / "tt_wiki.txt", PROC / "tt_cc100.txt"],
    "kk": [PROC / "kk_wiki.txt", PROC / "kk_cc100.txt"],
    "tr": [PROC / "tr_wiki.txt", PROC / "tr_cc100.txt"],
}

TATAR = str.maketrans({
    "а":"a","б":"b","в":"v","г":"g","д":"d","е":"ye","ё":"yo","ж":"zh",
    "з":"z","и":"i","й":"y","к":"k","л":"l","м":"m","н":"n","о":"o",
    "п":"p","р":"r","с":"s","т":"t","у":"u","ф":"f","х":"kh","ц":"ts",
    "ч":"ch","ш":"sh","щ":"shch","ъ":"","ы":"y","ь":"","э":"e","ю":"yu",
    "я":"ya","ә":"a","ң":"ng","ө":"o","ү":"u","һ":"h","ғ":"g","қ":"q",
    "А":"A","Б":"B","В":"V","Г":"G","Д":"D","Е":"Ye","Ё":"Yo","Ж":"Zh",
    "З":"Z","И":"I","Й":"Y","К":"K","Л":"L","М":"M","Н":"N","О":"O",
    "П":"P","Р":"R","С":"S","Т":"T","У":"U","Ф":"F","Х":"Kh","Ц":"Ts",
    "Ч":"Ch","Ш":"Sh","Щ":"Shch","Ъ":"","Ы":"Y","Ь":"","Э":"E","Ю":"Yu",
    "Я":"Ya","Ә":"A","Ң":"Ng","Ө":"O","Ү":"U","Һ":"H","Ғ":"G","Қ":"Q",
})

KAZAKH = str.maketrans({
    "а":"a","ә":"a","б":"b","в":"v","г":"g","ғ":"gh","д":"d","е":"e",
    "ё":"yo","ж":"zh","з":"z","и":"i","й":"y","к":"k","қ":"q","л":"l",
    "м":"m","н":"n","ң":"ng","о":"o","ө":"o","п":"p","р":"r","с":"s",
    "т":"t","у":"u","ұ":"u","ү":"u","ф":"f","х":"kh","ц":"ts","ч":"ch",
    "ш":"sh","щ":"shch","ъ":"","ы":"y","і":"i","ь":"","э":"e","ю":"yu",
    "я":"ya","һ":"h",
    "А":"A","Ә":"A","Б":"B","В":"V","Г":"G","Ғ":"Gh","Д":"D","Е":"E",
    "Ё":"Yo","Ж":"Zh","З":"Z","И":"I","Й":"Y","К":"K","Қ":"Q","Л":"L",
    "М":"M","Н":"N","Ң":"Ng","О":"O","Ө":"O","П":"P","Р":"R","С":"S",
    "Т":"T","У":"U","Ұ":"U","Ү":"U","Ф":"F","Х":"Kh","Ц":"Ts","Ч":"Ch",
    "Ш":"Sh","Щ":"Shch","Ъ":"","Ы":"Y","І":"I","Ь":"","Э":"E","Ю":"Yu",
    "Я":"Ya","Һ":"H",
})

MAPS = {"tt": TATAR, "kk": KAZAKH, "tr": None}  

_URL = re.compile(r"https?://\S+")
_TAG = re.compile(r"<[^>]+>")
_WSP = re.compile(r"\s+")

def clean(line: str) -> str:
    line = _URL.sub("", line)
    line = _TAG.sub("", line)
    return _WSP.sub(" ", line).strip()

def to_latin(text: str, lang: str) -> str:
    m = MAPS[lang]
    return text.translate(m) if m else text

def process(lang: str) -> None:
    native_out = PROC / f"{lang}_combined.txt"
    latin_out  = NORM / f"{lang}_latin.txt"
    if native_out.exists() and latin_out.exists():
        log.info(f"[{lang}] Already processed."); return

    total = 0
    with open(native_out, "w", encoding="utf-8") as fn, \
         open(latin_out,  "w", encoding="utf-8") as fl:
        for src in SOURCES[lang]:
            if not src.exists():
                log.warning(f"  Missing: {src}"); continue
            log.info(f"  {src.name}...")
            with open(src, encoding="utf-8") as f:
                for raw in tqdm(f, desc=f"    {lang}/{src.stem}", unit=" lines"):
                    line = clean(raw)
                    if len(line) < 15: continue
                    fn.write(line + "\n")
                    fl.write(to_latin(line, lang) + "\n")
                    total += 1
    log.info(f"  {total:,} lines -> {native_out.name} + {latin_out.name}")

if __name__ == "__main__":
    for lang in SOURCES:
        process(lang)
    print("\nPreprocessing done.")
    for lang in SOURCES:
        for folder, suffix in [(PROC, "combined"), (NORM, "latin")]:
            p = folder / f"{lang}_{suffix}.txt"
            if p.exists():
                n = sum(1 for _ in open(p, encoding="utf-8"))
                print(f"  {lang} {suffix}: {n:,} lines  ({p.stat().st_size/1e6:.1f} MB)")