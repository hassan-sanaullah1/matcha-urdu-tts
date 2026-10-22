"""Urdu number verbalisation. Hand-written, because no library does this.

num2words has no 'ur' and no 'hi' locale -- it raises. Its 'ar' converter emits
Arabic ("ألف و مئتان و أربعة و ثلاثون"), not Urdu, and 'fa' diverges from Urdu
above twenty and on the scale words. So this table exists.

Two properties of Urdu numerals force the table to be explicit:
  - 1-99 are suppletive. They are not composed from tens + units the way English
    "forty-two" is; اکتالیس (41) and بیالیس (42) have to be listed.
  - The scale is South Asian: hundred / thousand / lakh (10^5) / crore (10^7),
    not thousand / million / billion. 2,500,000 is پچیس لاکھ, not "2.5 million".

Why verbalise at all rather than emit digit characters: ASCII 0-9 are not in the
350-symbol vocab and are silently deleted today, and of the two Arabic-Indic digit
blocks that ARE in the vocab, only U+06F0-06F9 was trained (symbol_audit.py:
U+0660-0669 scores KS p=0.94 against the untrained reference). Even emitting the
trained block would leave the model to guess whether "۲۰۲۴" is read as a year, a
quantity or four separate digits. Words remove the ambiguity.

The word forms below follow standard Pakistani Urdu usage and should be checked by
a native speaker before this ships -- a handful of them have common variants
(چھ/چھے, نوے/نوّے) and the choice is a pronunciation decision, not a code one.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import paths  # noqa: E402  -- kiosk-tts/src/paths.py: every location + the import path

import re

ONES = [
    "صفر", "ایک", "دو", "تین", "چار", "پانچ", "چھ", "سات", "آٹھ", "نو",
    "دس", "گیارہ", "بارہ", "تیرہ", "چودہ", "پندرہ", "سولہ", "سترہ", "اٹھارہ", "انیس",
    "بیس", "اکیس", "بائیس", "تیئیس", "چوبیس", "پچیس", "چھبیس", "ستائیس", "اٹھائیس", "انتیس",
    "تیس", "اکتیس", "بتیس", "تینتیس", "چونتیس", "پینتیس", "چھتیس", "سینتیس", "اڑتیس", "انتالیس",
    "چالیس", "اکتالیس", "بیالیس", "تینتالیس", "چوالیس", "پینتالیس", "چھیالیس", "سینتالیس", "اڑتالیس", "انچاس",
    "پچاس", "اکاون", "باون", "تریپن", "چوون", "پچپن", "چھپن", "ستاون", "اٹھاون", "انسٹھ",
    "ساٹھ", "اکسٹھ", "باسٹھ", "تریسٹھ", "چونسٹھ", "پینسٹھ", "چھیاسٹھ", "سڑسٹھ", "اڑسٹھ", "انہتر",
    "ستر", "اکہتر", "بہتر", "تہتر", "چوہتر", "پچہتر", "چھہتر", "ستتر", "اٹھہتر", "اناسی",
    "اسی", "اکیاسی", "بیاسی", "تراسی", "چوراسی", "پچاسی", "چھیاسی", "ستاسی", "اٹھاسی", "نواسی",
    "نوے", "اکیانوے", "بانوے", "ترانوے", "چورانوے", "پچانوے", "چھیانوے", "ستانوے", "اٹھانوے", "ننانوے",
]

# The same numerals as phone strings (the phones.py inventory), which is what we now emit.
#
# Why: Urdu does not write its short vowels, so the script form gives the model no cue for them and it
# runs the consonants together -- the user heard پینسٹھ (65, "painsaṭh") as "penst" (1 Oct). espeak-ng's
# Urdu voice has the identical bug (پینسٹھ -> pˈeːnsʈʰ, اسی -> ˈɪsiː instead of ˈəssiː), so the phones
# cannot be derived from it and are written out here. This is the mechanism text_norm.PK_PHONES already
# uses for names whose spelling hides the vowels (سجاول -> "Sajawal", فوجی -> "Fuji").
#
# Gemination is a doubled consonant, as phones._map_word renders it. Nasals before a consonant are a
# plain "n" rather than the NASAL symbol, which this checkpoint only ever saw inside English spans.
ONES_PH = [
    "sˈɪfər", "ˈeːk", "dˈoː", "tˈiːn", "ʧˈaːr", "pˈaːnʧ", "ʧʰˈeː", "sˈaːt", "ˈaːʈʰ", "nˈɔ",
    "dˈəs", "ɡɪjˈaːrə", "bˈaːrə", "tˈeːrə", "ʧˈɔdə", "pˈəndrə", "sˈoːlə", "sˈətrə", "əʈʰˈaːrə", "ʊnnˈiːs",
    "bˈiːs", "ɪkkˈiːs", "bˈaːiːs", "tˈeːiːs", "ʧɔbˈiːs", "pəʧʧˈiːs", "ʧʰəbbˈiːs", "səttˈaːiːs", "əʈʰˈaːiːs", "ʊntˈiːs",
    "tˈiːs", "ɪkəttˈiːs", "bəttˈiːs", "tɛntˈiːs", "ʧɔntˈiːs", "pɛntˈiːs", "ʧʰəttˈiːs", "sɛntˈiːs", "əɽtˈiːs", "ʊntaːlˈiːs",
    "ʧaːlˈiːs", "ɪkətaːlˈiːs", "bəjaːlˈiːs", "tɛntaːlˈiːs", "ʧəʋaːlˈiːs", "pɛntaːlˈiːs", "ʧʰɪjaːlˈiːs", "sɛntaːlˈiːs",
    "əɽtaːlˈiːs", "ʊnʧˈaːs",
    "pəʧˈaːs", "ɪkjˈaːʋən", "bˈaːʋən", "tɪrˈeːpən", "ʧəʋʋˈən", "pəʧpˈən", "ʧʰəppˈən", "səttˈaːʋən", "əʈʰˈaːʋən",
    "ʊnsˈəʈʰ",
    "sˈaːʈʰ", "ɪksˈəʈʰ", "bˈaːsəʈʰ", "tɪrsˈəʈʰ", "ʧɔnsˈəʈʰ", "pɛnsˈəʈʰ", "ʧʰɪjaːsˈəʈʰ", "səɽsˈəʈʰ", "əɽsˈəʈʰ",
    "ʊnhˈəttər",
    "sˈəttər", "ɪkhˈəttər", "bəhˈəttər", "tɪhˈəttər", "ʧɔhˈəttər", "pəʧhˈəttər", "ʧʰɪhˈəttər", "sətˈəttər",
    "əʈʰˈəttər", "ʊnˈaːsiː",
    "ˈəssiː", "ɪkjˈaːsiː", "bəjˈaːsiː", "tɪrˈaːsiː", "ʧɔrˈaːsiː", "pəʧˈaːsiː", "ʧʰɪjˈaːsiː", "sətˈaːsiː",
    "əʈʰˈaːsiː", "nəʋˈaːsiː",
    "nˈəʋʋeː", "ɪkjˈaːnʋeː", "bˈaːnʋeː", "tɪrˈaːnʋeː", "ʧɔrˈaːnʋeː", "pəʧˈaːnʋeː", "ʧʰɪjˈaːnʋeː", "sətˈaːnʋeː",
    "əʈʰˈaːnʋeː", "nɪnjˈaːnʋeː",
]

# South Asian scale, largest first.
SCALES = [(10 ** 9, "ارب"), (10 ** 7, "کروڑ"), (10 ** 5, "لاکھ"),
          (10 ** 3, "ہزار"), (10 ** 2, "سو")]
SCALES_PH = {"ارب": "ˈərəb", "کروڑ": "kərˈoːɽ", "لاکھ": "lˈaːkʰ", "ہزار": "həzˈaːr", "سو": "sˈɔ",
             "کھرب": "kʰˈərəb"}

POINT = "اعشاریہ"
NEG = "منفی"
PERCENT = "فیصد"
# اعشاریہ came out as "acharia" and منفی as "munfi" with the wrong vowels, so both are spelled out in phones.
# اعشاریہ was then read as the English word "point" instead -- a stopgap, and a bad one: pˈɔɪnʈ is heard as
# "baint", because ɔɪ is the diphthong that also made "oil" come out as "ail". Spelled as ɛʃˈaːrɪjaː it is
# heard as "ɛʃaːriːjaː", i.e. the real word. Measured with src/say_check.py, in a decimal, 3 seeds:
#   sˈəttər pˈɔɪnʈ tˈiːn      -> "səʈərbaːiːɴʈiː"        (the separator lost)
#   sˈəttər ɛʃˈaːrɪjaː tˈiːn  -> "soːʈərɛʃaːriːjaːʈiː"   (the separator clear)
POINT_PH = "ɛʃˈaːrɪjaː"
NEG_PH = "mˈənfiː"

# Currency and units seen often enough in LLM output to be worth expanding.
# "+" is deliberately absent: it was read "جمع", which is right for arithmetic and wrong for
# every "+92" in the knowledge base (user's call, 19 Oct). It is not in the vocab, so with no
# mapping tts.clean drops it silently -- which is the wanted "skip it".
SYMBOLS = {"%": PERCENT, "₨": "روپے", "$": "ڈالر", "€": "یورو", "£": "پاؤنڈ"}


def int_to_urdu(n, ph=False):
    """Non-negative int -> Urdu, as phone strings (ph=True, what the model reads) or script (ph=False)."""
    n = int(n)
    ones, neg = (ONES_PH, NEG_PH) if ph else (ONES, NEG)
    if n < 0:
        return neg + " " + int_to_urdu(-n, ph)
    if n < 100:
        return ones[n]
    for value, name in SCALES:
        if n >= value:
            head, rest = divmod(n, value)
            out = int_to_urdu(head, ph) + " " + (SCALES_PH[name] if ph else name)
            return out if rest == 0 else out + " " + int_to_urdu(rest, ph)
    return ones[n]


def _digits(s, ph=False):
    """Read a digit string one digit at a time -- for years-as-digits, IDs, phone
    numbers, and anything where the grouped reading would be wrong."""
    ones = ONES_PH if ph else ONES
    return " ".join(ones[int(c)] for c in s if c.isdigit())


def number_to_urdu(tok, ph=False):
    """One numeric token (already ASCII digits) -> Urdu words.

    Handles decimals and thousands separators. A token with a leading zero, or one
    of 10+ digits, is read digit-by-digit: those are phone numbers or identifiers
    (Pakistani mobile numbers are 11 digits starting 03), where the grouped reading
    is actively wrong. Everything shorter is a quantity or a year and is grouped --
    1947 is "ایک ہزار نو سو سینتالیس", which is how Urdu says a year.
    """
    tok = re.sub(r"[^0-9.,،+-]", "", str(tok))
    tok = tok.replace(",", "").replace("،", "")
    neg = tok.startswith("-")
    tok = tok.lstrip("+-")
    if not tok:
        return ""
    if "." in tok:
        whole, _, frac = tok.partition(".")
        whole = whole or "0"
        out = int_to_urdu(int(whole), ph) + " " + (POINT_PH if ph else POINT) + " " + _digits(frac, ph)
    elif len(tok) >= 10 or (len(tok) > 1 and tok[0] == "0"):
        out = _digits(tok, ph)
    else:
        out = int_to_urdu(int(tok), ph)
    return ((NEG_PH if ph else NEG) + " " + out) if neg else out


# Script word -> phones, for the swap frontend.prepare does on marked spans. Derived from the tables above,
# so the two cannot drift. Scale words an LLM writes in Urdu are included even though no numeral emits them.
WORD_PH = {**{w: p for w, p in zip(ONES, ONES_PH)}, **SCALES_PH, POINT: POINT_PH, NEG: NEG_PH,
           "ملین": "mˈɪliːən", "بلین": "ˈərəb", "ٹریلین": "ʈrˈɪliːən"}

# Every number this module expands is wrapped in this marker, so frontend.prepare can tell a numeral's "سو"
# from the ordinary word and swap only the former for phones. It is not in the vocab and no rule matches it:
# the text keeps flowing through every existing step as Urdu script, exactly as before. frontend.prepare is
# the only consumer; on the other front ends (tts.prepare, urdu_text_norm.normalize_for_tts) it collapses to
# nothing in urdu_norm, so those keep emitting script numerals and their output is unchanged.
MARK = "\x01"


def mark(words):
    return MARK + words + MARK


# Latin digits, Arabic-Indic (U+0660-0669) and extended Arabic-Indic (U+06F0-06F9)
# all normalise to ASCII before parsing, so the caller does not have to care which
# convention the upstream text used.
_ASCII = str.maketrans(
    "".join(chr(0x0660 + i) for i in range(10)) + "".join(chr(0x06F0 + i) for i in range(10)),
    "0123456789" * 2,
)
_NUM = re.compile(r"[+-]?\d[\d,،]*(?:\.\d+)?%?")


def expand_numbers(text):
    """Replace every numeric token in `text` with its Urdu words.

    Also expands the currency and percent symbols directly adjacent to a number,
    since "%" alone is not in the vocab and would otherwise vanish.
    """
    text = str(text).translate(_ASCII)

    def sub(m):
        tok = m.group(0)
        # A trailing % binds to the number and is consumed here, so the sweep
        # below cannot expand it a second time: "20%" -> "بیس فیصد".
        words = mark(number_to_urdu(tok))
        return words + " " + PERCENT if tok.endswith("%") else words

    out = _NUM.sub(sub, text)
    for sym, word in SYMBOLS.items():
        if sym != "%":
            out = out.replace(sym, " " + word + " ")
    return re.sub(r"\s+", " ", out).strip()


def check_phones():
    """Every phone string must be 100 long, in the vocab, and made only of phones.py units.

    A typo here is silent -- an unknown symbol is deleted by tts.clean and the numeral just loses a sound --
    so this is asserted rather than hoped for. Run: python src/common/text/urdu_numbers.py --test
    """
    import json

    import phones
    v = json.load(open(paths.BASE_VOCAB, encoding="utf-8"))
    units = sorted(phones.PHONES, key=len, reverse=True)
    bad = []
    if len(ONES_PH) != len(ONES) != 100:
        bad.append(f"ONES_PH has {len(ONES_PH)} entries, ONES has {len(ONES)}")
    for name, s in [(ONES[i], ONES_PH[i]) for i in range(min(len(ONES), len(ONES_PH)))] \
            + list(SCALES_PH.items()) + [(POINT, POINT_PH), (NEG, NEG_PH)]:
        missing = [c for c in s if c not in v]
        rest = s.replace(phones.STRESS, "")
        while rest:
            for u in units:
                if rest.startswith(u):
                    rest = rest[len(u):]
                    break
            else:
                bad.append(f"{name} {s!r}: not a phone at {rest!r}")
                break
        if missing:
            bad.append(f"{name} {s!r}: not in vocab: {missing}")
        if s.count(phones.STRESS) != 1:
            bad.append(f"{name} {s!r}: {s.count(phones.STRESS)} stress marks, want 1")
    return bad


if __name__ == "__main__":
    if "--test" in sys.argv:
        problems = check_phones()
        print("\n".join(problems) if problems else f"OK: {len(ONES_PH)} numerals + "
              f"{len(SCALES_PH)} scales are valid phones in the vocab")
        sys.exit(1 if problems else 0)
    for probe in ["0", "3", "11", "26", "41", "49", "67", "79", "99", "100", "101",
                  "250", "1000", "2024", "100000", "2500000", "12345678",
                  "3.5", "20%", "-7", "07", "1,250"]:
        print(f"{probe:>10s}  {number_to_urdu(probe)}")
    print()
    for s in ["میری meeting 3 بجے ہے۔", "قیمت 2,500 روپے ہے", "تقریبا 20% اضافہ",
              "سال ۲۰۲۴ میں", "فون 03001234567 پر"]:
        print(f"  {s}\n  -> {expand_numbers(s)}\n")
