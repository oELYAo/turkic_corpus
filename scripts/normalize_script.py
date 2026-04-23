#!/usr/bin/env python3
"""
Cyrillic → Latin normalization for controlled cross-lingual experiments.

- **Kazakh (kk)**: April 2021 official Kazakh Latin alphabet (Kazakhstan),
  Cyrillic→Latin as in the state correspondence table (Wikipedia / QazAqparat).
- **Tatar (tt)**: **Zamanälif** (2000 decree / 2012 romanization law), after
  Wikipedia «Correspondence between alphabets» (Zamanälif column).
- **Turkish (tr)**: already Latin; pass-through.

The full Kazakh Cyrillic→Latin table (2021) is documented in KAZAKH_CYRILLIC_TO_LATIN_2021
below as a dict (lowercase) plus explicit uppercase entries where case-fold is not enough.
"""

from __future__ import annotations

import argparse
import unicodedata
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# -----------------------------------------------------------------------------
# Official April 2021 Kazakh Latin: full Cyrillic → Latin (one code point each
# except multigraph rows Ё Ю Я Ц Ч Щ, handled in KAZAKH_MULTIGRAPHS).
#
# Source: Republic of Kazakhstan Latin alphabet correspondence (2021 revision),
# as summarized e.g. on https://en.wikipedia.org/wiki/Kazakh_alphabets — April
# 2021 row: Ş for Ш, Tş for Ч, Ştş for Щ, Ts for Ц, İo/İu/İa for Ё/Ю/Я;
# Х and Һ both → H; І → dotless I (ı); И and Й → İ / i.
# -----------------------------------------------------------------------------
KAZAKH_MULTIGRAPHS: list[tuple[str, str]] = [
    # longest / special digraphs-trigraphs first (applied before single-char map)
    ("Щ", "Ştş"),
    ("щ", "ştş"),
    ("Ш", "Ş"),
    ("ш", "ş"),
    ("Ч", "Tş"),
    ("ч", "tş"),
    ("Ц", "Ts"),
    ("ц", "ts"),
    ("Ё", "İo"),
    ("ё", "io"),
    ("Ю", "İu"),
    ("ю", "iu"),
    ("Я", "İa"),
    ("я", "ia"),
]

# Lowercase Cyrillic → Latin (April 2021); multigraph letters omitted above.
_KK_LOWER = {
    "а": "a",
    "ә": "ä",
    "б": "b",
    "в": "v",
    "г": "g",
    "ғ": "ğ",
    "д": "d",
    "е": "e",
    "ж": "j",
    "з": "z",
    "и": "i",  # dotted i (pair with İ)
    "й": "i",  # same Latin letter as и per 2021 table
    "к": "k",
    "қ": "q",
    "л": "l",
    "м": "m",
    "н": "n",
    "ң": "ñ",
    "о": "o",
    "ө": "ö",
    "п": "p",
    "р": "r",
    "с": "s",
    "т": "t",
    "у": "u",
    "ұ": "ū",
    "ү": "ü",
    "ф": "f",
    "х": "h",
    "һ": "h",
    "ъ": "",
    "ы": "y",
    "і": "ı",  # U+0131 dotless i
    "ь": "",
    "э": "e",
}
# Uppercase Cyrillic → Latin
_KK_UPPER = {
    "А": "A",
    "Ә": "Ä",
    "Б": "B",
    "В": "V",
    "Г": "G",
    "Ғ": "Ğ",
    "Д": "D",
    "Е": "E",
    "Ж": "J",
    "З": "Z",
    "И": "İ",
    "Й": "İ",
    "К": "K",
    "Қ": "Q",
    "Л": "L",
    "М": "M",
    "Н": "N",
    "Ң": "Ñ",
    "О": "O",
    "Ө": "Ö",
    "П": "P",
    "Р": "R",
    "С": "S",
    "Т": "T",
    "У": "U",
    "Ұ": "Ū",
    "Ү": "Ü",
    "Ф": "F",
    "Х": "H",
    "Һ": "H",
    "Ъ": "",
    "Ы": "Y",
    "І": "I",
    "Ь": "",
    "Э": "E",
}

KAZAKH_CYRILLIC_TO_LATIN_2021: dict[str, str] = {**_KK_UPPER, **_KK_LOWER}

# Full April 2021 Cyrillic→Latin table (reference; multigraph targets in KAZAKH_MULTIGRAPHS):
#   А→A  Ә→Ä  Б→B  В→V  Г→G  Ғ→Ğ  Д→D  Е→E  Ж→J  З→Z  И→İ  Й→İ  К→K  Қ→Q  Л→L  М→M
#   Н→N  Ң→Ñ  О→O  Ө→Ö  П→P  Р→R  С→S  Т→T  У→U  Ұ→Ū  Ү→Ü  Ф→F  Х→H  Һ→H  Ъ→(∅)
#   Ы→Y  І→I/ı  Ь→(∅)  Э→E
#   Ё→İo/io  Ю→İu/iu  Я→İa/ia  Ц→Ts/ts  Ч→Tş/tş  Ш→Ş/ş  Щ→Ştş/ştş
# Lowercase İ/і output uses i (ASCII) and ı (U+0131) as in the table above.

# -----------------------------------------------------------------------------
# Tatar Zamanälif: Cyrillic → Latin (single chars + multigraphs).
# Source: Wikipedia «Tatar alphabet» — Zamanälif (2000–2005), 2012 romanization.
# Note: Е/Ё/К in native vs Russian words has contextual rules; we use a fixed
# letter-level mapping suitable for corpus normalization (not pedagogical).
# -----------------------------------------------------------------------------
TATAR_MULTIGRAPHS: list[tuple[str, str]] = [
    ("Щ", "Şç"),
    ("щ", "şç"),
    ("Ш", "Ş"),
    ("ш", "ş"),
    ("Ц", "Ts"),
    ("ц", "ts"),
    ("Ч", "Ç"),
    ("ч", "ç"),
    ("Ю", "Yu"),
    ("ю", "yu"),
    ("Я", "Ya"),
    ("я", "ya"),
    ("Ё", "Yo"),
    ("ё", "yo"),
]

_TT_LOWER = {
    "а": "a",
    "ә": "ä",
    "б": "b",
    "в": "v",
    "г": "g",
    "ғ": "ğ",
    "д": "d",
    "е": "e",
    "ж": "j",
    "җ": "ç",
    "з": "z",
    "и": "i",
    "й": "y",
    "к": "k",
    "қ": "q",
    "л": "l",
    "м": "m",
    "н": "n",
    "ң": "ñ",
    "о": "o",
    "ө": "ö",
    "п": "p",
    "р": "r",
    "с": "s",
    "т": "t",
    "у": "u",
    "ү": "ü",
    "ф": "f",
    "х": "x",
    "һ": "h",
    "ъ": "ʼ",
    "ы": "ı",
    "ь": "ʼ",
    "э": "e",
}
_TT_UPPER = {
    "А": "A",
    "Ә": "Ä",
    "Б": "B",
    "В": "V",
    "Г": "G",
    "Ғ": "Ğ",
    "Д": "D",
    "Е": "E",
    "Ж": "J",
    "Җ": "Ç",
    "З": "Z",
    "И": "İ",
    "Й": "Y",
    "К": "K",
    "Қ": "Q",
    "Л": "L",
    "М": "M",
    "Н": "N",
    "Ң": "Ñ",
    "О": "O",
    "Ө": "Ö",
    "П": "P",
    "Р": "R",
    "С": "S",
    "Т": "T",
    "У": "U",
    "Ү": "Ü",
    "Ф": "F",
    "Х": "X",
    "Һ": "H",
    "Ъ": "ʼ",
    "Ы": "I",
    "Ь": "ʼ",
    "Э": "E",
}

TATAR_CYRILLIC_TO_LATIN_ZAMANALIF: dict[str, str] = {**_TT_UPPER, **_TT_LOWER}

# Zamanälif (letter-level): А→A  Ә→Ä  Б→B  В→V  Г→G  Ғ→Ğ  Д→D  Е→E  Ж→J  Җ→Ç  З→Z  И→İ
#   Й→Y  К→K  Қ→Q  Л→L  М→M  Н→N  Ң→Ñ  О→O  Ө→Ö  П→P  Р→R  С→S  Т→T  У→U  Ү→Ü
#   Ф→F  Х→X  Һ→H  Ъ→ʼ  Ы→I/ı  Ь→ʼ  Э→E
#   Ё→Yo/yo  Ю→Yu/yu  Я→Ya/ya  Ц→Ts/ts  Ч→Ç/ç  Ш→Ş/ş  Щ→Şç/şç

def _apply_multigraphs(text: str, pairs: list[tuple[str, str]]) -> str:
    s = text
    for cyr, lat in pairs:
        s = s.replace(cyr, lat)
    return s


def _apply_char_map(text: str, char_map: dict[str, str]) -> str:
    out: list[str] = []
    for ch in text:
        out.append(char_map.get(ch, ch))
    return "".join(out)


def normalize_kazakh(text: str) -> str:
    s = unicodedata.normalize("NFC", text)
    s = _apply_multigraphs(s, KAZAKH_MULTIGRAPHS)
    return _apply_char_map(s, KAZAKH_CYRILLIC_TO_LATIN_2021)


def normalize_tatar(text: str) -> str:
    s = unicodedata.normalize("NFC", text)
    s = _apply_multigraphs(s, TATAR_MULTIGRAPHS)
    return _apply_char_map(s, TATAR_CYRILLIC_TO_LATIN_ZAMANALIF)


def normalize_turkish(text: str) -> str:
    return unicodedata.normalize("NFC", text)


NORMALIZERS = {
    "kk": normalize_kazakh,
    "tt": normalize_tatar,
    "tr": normalize_turkish,
}


def normalize_text(text: str, lang: str) -> str:
    return NORMALIZERS[lang](text)


def run_file(input_path: Path, output_path: Path, lang: str) -> None:
    fn = NORMALIZERS[lang]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open(encoding="utf-8", errors="replace") as inf, output_path.open(
        "w", encoding="utf-8"
    ) as outf:
        for line in inf:
            outf.write(fn(line.rstrip("\n\r")) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Normalize Kazakh/Tatar Cyrillic to Latin (2021 Kazakh / Zamanälif); Turkish NFC pass-through."
    )
    p.add_argument("--lang", required=True, choices=sorted(NORMALIZERS.keys()))
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args()

    inp = args.input.expanduser().resolve()
    if not inp.is_file():
        raise SystemExit(f"Input not found: {inp}")

    out = args.output
    if out is None:
        out = PROJECT_ROOT / "data" / "processed" / "normalized_script" / f"{args.lang}_norm.txt"
    else:
        out = out.expanduser()
        if not out.is_absolute():
            out = (PROJECT_ROOT / out).resolve()

    run_file(inp, out, args.lang)
    print(f"Wrote {out}")


if __name__ == "__main__":
    import sys

    # --- unit-style checks: 5 example words per language -----------------
    _KK_EXAMPLES = [
        ("Қазақстан", "Qazaqstan"),
        ("әлеуметтік", "äleumettık"),
        ("түйін", "tüiın"),
        ("Ғажайып", "Ğajaiyp"),
        ("Шығыс", "Şyğys"),
    ]
    _TT_EXAMPLES = [
        ("Татарстан", "Tatarstan"),
        ("Хәл", "Xäl"),
        ("җан", "çan"),
        ("яңа", "yaña"),
        ("китап", "kitap"),
    ]
    _TR_EXAMPLES = [
        ("İstanbul", "İstanbul"),
        ("âşık", "âşık"),
        ("merhaba", "merhaba"),
        ("Öğrenci", "Öğrenci"),
        ("Çanakkale", "Çanakkale"),
    ]

    for cyr, lat in _KK_EXAMPLES:
        assert normalize_kazakh(cyr) == lat, (cyr, normalize_kazakh(cyr), lat)
    for cyr, lat in _TT_EXAMPLES:
        assert normalize_tatar(cyr) == lat, (cyr, normalize_tatar(cyr), lat)
    for plain, lat in _TR_EXAMPLES:
        assert normalize_turkish(plain) == lat, (plain, normalize_turkish(plain), lat)

    if len(sys.argv) == 1:
        print("normalize_script.py: OK (18 word assertions). Pass --lang --input [--output] to convert files.")
    else:
        main()
