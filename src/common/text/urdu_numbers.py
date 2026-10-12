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

# South Asian scale, largest first.
SCALES = [(10 ** 9, "ارب"), (10 ** 7, "کروڑ"), (10 ** 5, "لاکھ"),
          (10 ** 3, "ہزار"), (10 ** 2, "سو")]

POINT = "اعشاریہ"
NEG = "منفی"
PERCENT = "فیصد"

# Currency and units seen often enough in LLM output to be worth expanding.
SYMBOLS = {"%": PERCENT, "₨": "روپے", "$": "ڈالر", "€": "یورو", "£": "پاؤنڈ", "+": "جمع"}


def int_to_urdu(n):
    """Non-negative int -> Urdu words."""
    n = int(n)
    if n < 0:
        return NEG + " " + int_to_urdu(-n)
    if n < 100:
        return ONES[n]
    for value, name in SCALES:
        if n >= value:
            head, rest = divmod(n, value)
            out = int_to_urdu(head) + " " + name
            return out if rest == 0 else out + " " + int_to_urdu(rest)
    return ONES[n]


def _digits(s):
    """Read a digit string one digit at a time -- for years-as-digits, IDs, phone
    numbers, and anything where the grouped reading would be wrong."""
    return " ".join(ONES[int(c)] for c in s if c.isdigit())


def number_to_urdu(tok):
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
        out = int_to_urdu(int(whole)) + " " + POINT + " " + _digits(frac)
    elif len(tok) >= 10 or (len(tok) > 1 and tok[0] == "0"):
        out = _digits(tok)
    else:
        out = int_to_urdu(int(tok))
    return (NEG + " " + out) if neg else out


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
        words = number_to_urdu(tok)
        return words + " " + PERCENT if tok.endswith("%") else words

    out = _NUM.sub(sub, text)
    for sym, word in SYMBOLS.items():
        if sym != "%":
            out = out.replace(sym, " " + word + " ")
    return re.sub(r"\s+", " ", out).strip()


if __name__ == "__main__":
    for probe in ["0", "3", "11", "26", "41", "49", "67", "79", "99", "100", "101",
                  "250", "1000", "2024", "100000", "2500000", "12345678",
                  "3.5", "20%", "-7", "07", "1,250"]:
        print(f"{probe:>10s}  {number_to_urdu(probe)}")
    print()
    for s in ["میری meeting 3 بجے ہے۔", "قیمت 2,500 روپے ہے", "تقریبا 20% اضافہ",
              "سال ۲۰۲۴ میں", "فون 03001234567 پر"]:
        print(f"  {s}\n  -> {expand_numbers(s)}\n")
