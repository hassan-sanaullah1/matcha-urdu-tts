"""Round 4: general text normalisation for whatever a chatbot LLM says, before the urdu-phones front end.

The front end (frontend.py, from urdu-phones) already spells capitals, mixed-case abbreviations (PhD,
iOS), letter-digit codes (FY2024-25) and English-context numbers (NeMo). This pass handles the rest, by shape and
pattern rather than word lists, so it works on text no one has seen:

  markup      markdown, bullets, links, emoji -> plain text
  web         emails and URLs -> "info at mari energies dot com"
  money       PKR / Rs / Rs. / USD / US$ with an amount -> amount + "rupees"/"dollars" (روپے/ڈالر in Urdu)
  units       km, kg, MW, sq km, cm, mm, ft, °C after a number -> words
  identifiers long digit strings and hyphenated codes (phone, NTN, registration) -> read digit by digit
  times       10:30 in an Urdu sentence -> دس بج کر تیس منٹ (NeMo handles English sentences)
  acronyms    any acronym becomes Urdu letter names (ETF -> ای ٹی ایف): that is how Pakistanis say them, and the
              base model knows these words from 228k Urdu steps. Measured on the Round 3 pilot: 9/10 heard right
              as Urdu letter names vs 6/10 as English sound symbols. Here: plurals and possessives (CEOs -> سی ای
              اوز), & forms (E&P -> ای اینڈ پی), dotted capitals (A.F.), capitals of 7-8 letters; the front end
              then spells the rest in the same letters because prepare_pk calls it with letters="ur".
  greeting    every spelling of the greeting (السلام علیکم, Assalam-o-Alaikum, AOA...) -> اَسّلامُ علیکم, the
              spelling you picked by ear; the reply وعلیکم السلام likewise
  camel case  MariEnergies -> Mari Energies, so glued names reach the names table
  word list   the only list: acronyms SAID AS WORDS (NASA, NADRA, UNESCO...). SECP, HEC and KSE are taken off it
              and out of the phone overrides, so they are spelled like any other acronym.

normalize(text) -> text for pk_gazetteer.prepare_pk.   Usage: python src/text_norm.py --test | --render-check
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import paths  # noqa: E402  -- kiosk-tts/src/paths.py: every location + the import path

import argparse
import functools
import re
import threading
import unicodedata

_URDU = re.compile(r"[؀-ۿݐ-ݿ]")
SPELLED_NOT_WORDS = {"SECP", "HEC", "KSE"}                 # were on the front end's "say as a word" list
EXTRA_WORD_ACRONYMS = {                                    # said as words; everything else is spelled
    "ASEAN", "SAARC", "UEFA", "IATA", "ICAO", "NASDAQ", "AIDS", "SARS", "MERS", "CAPTCHA", "SONAR", "PEMRA",
    "NESCOM", "PIDE", "EBITDA", "IPSEC", "ANSI", "SOAS",
}
# Exact sounds for names whose Urdu-script spelling hides the vowels (user's marks, 24 Sep): Urdu writes u/o/au with
# one letter and leaves short vowels out, so سجاول came out "Sajawal", فوجی "Fuji", بنوں "Banno", کلفٹن "Kalftan".
# Also Huawei with its H, and General the way it is said in Pakistan ("jenral": the model dropped the unstressed end).
PK_PHONES = {
    "fauji": "fˈɔʤiː", "salim": "səlˈiːm", "saleem": "səlˈiːm", "sumair": "sʊmˈeːr", "bannu": "bənnˈuː",
    "clifton": "klˈɪfʈən", "khel": "xˈeːl", "daud": "dˈaːʊd", "spinwam": "spˈiːnʋaːm", "sujawal": "sʊʤˈaːʋəl",
    "bhitai": "bʰɪʈˈaːiː", "huawei": "hˈuːaːʋeː", "general": "ʤˈɛnrəl",
    # "-cy"/"-ship" on chairman: not dictionary words, so espeak would guess at them.
    "chairmancy": "ʧˈeːrmənsiː", "chairmanship": "ʧˈeːrmənʃɪp",
}
# The front end's mixed-case pattern (PhD, MSc) also matched ANY 2-4 capitals, so NASA, FIFA, NATO, OPEC and
# headings like ONLY were spelled before the word list was consulted. Mixed case now needs a lowercase letter;
# all-capital tokens go through the normal spelled / word-list / dictionary checks.
_MIXED_CASE_ABBR = re.compile(r"^(?=[A-Za-z]*[a-z])(?=[A-Za-z]*[A-Z][A-Za-z]*[A-Z])[A-Za-z]{2,4}$")


@functools.lru_cache(maxsize=1)
def frontend():
    """The front end (frontend.py) with the acronym word list corrected (SECP/HEC/KSE are spelled, not said)."""
    import frontend as fe  # only its module-level sets are adjusted
    fe.ACRONYM_WORDS.difference_update(SPELLED_NOT_WORDS)
    fe.ACRONYM_WORDS.update(EXTRA_WORD_ACRONYMS)
    fe._MIXED_ABBR = _MIXED_CASE_ABBR
    fe.PHONE_OVERRIDE.update(PK_PHONES)
    # the Roman-Urdu guard runs first and would turn these names back into vowel-less Urdu script
    table, _ = fe._roman_urdu()
    keep = {k: v for k, v in table.items() if k not in PK_PHONES}
    rx = re.compile(r"(?<![A-Za-z])(" + "|".join(re.escape(k) for k in sorted(keep, key=len, reverse=True))
                    + r")(?![A-Za-z])", re.I)
    fe.roman_urdu_guard = lambda text: rx.sub(lambda m: keep[m.group(0).lower()], text)
    for k in ("secp", "hec", "kse"):
        fe.PHONE_OVERRIDE.pop(k, None)
    return fe


_ctx = threading.local()


def sentence_is_urdu(t):
    """Urdu-ness of the sentence as it ARRIVED, not as it looks now.

    Several passes insert Urdu script themselves -- ranks_and_codes spells an acronym in Urdu letter
    names, the gazetteer rewrites Pakistani names -- so a later pass re-deriving this from the text
    would call an English sentence Urdu and read "the Companies Act, 2017" as "دو ہزار سترہ".
    normalize() fixes the verdict once; called on its own (the tests do), this falls back to the text.
    """
    v = getattr(_ctx, "urdu", None)
    return is_urdu(t) if v is None else v


def is_urdu(text):
    """Urdu reply if the sentence has Urdu script at all: its numbers and currencies are then read in Urdu."""
    return bool(_URDU.search(text))


def _spell(letters):
    fe = frontend()
    return " ".join(fe.LETTER_UR[c] for c in letters.upper() if c in fe.LETTER_UR)


def _word_acronym(tok):
    return tok.upper() in frontend().ACRONYM_WORDS


# ---------------------------------------------------------------------------
# markup and web
# ---------------------------------------------------------------------------
_EMOJI = re.compile("[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U0001F900-\U0001F9FF️‍]")


def strip_markup(t):
    t = re.sub(r"\[([^\]]+)\]\((?:[^)]+)\)", r"\1", t)            # [text](url) -> text
    t = re.sub(r"(\*\*|__|\*|`+|~~)", "", t)
    t = re.sub(r"(?m)^\s{0,3}(?:#{1,6}\s+|>\s+|[-*•]\s+|\d+[.)]\s+)", "", t)
    t = re.sub(r"(?m)^\s*\d+(?:\.\d+)+\s+", "", t)          # "11.2.2 Liquidity" heading numbers
    t = _EMOJI.sub("", t)
    return t.replace("−", "-").replace("–", "-").replace("—", ", ")


def _say_host(s):
    parts = [p for p in re.split(r"[.\-_/]", s) if p]
    return " dot ".join(parts)


def web(t):
    t = re.sub(r"\b([\w.+\-]+)@([\w\-]+(?:\.[\w\-]+)+)\b",
               lambda m: f"{_say_host(m.group(1))} at {_say_host(m.group(2))}", t)
    t = re.sub(r"\bhttps?://", "", t)
    t = re.sub(r"\bwww\.", "w w w dot ", t)
    return re.sub(r"\b([a-z0-9\-]+)\.(com|org|net|pk|gov|edu|io|ai)((?:\.[a-z]{2,3})?)(/[^\s]*)?",
                  lambda m: _say_host(m.group(1) + "." + m.group(2) + m.group(3)), t)


# ---------------------------------------------------------------------------
# money, units, identifiers, times
# ---------------------------------------------------------------------------
_AMOUNT = r"(\d[\d,]*(?:\.\d+)?)(\s*(?:million|billion|trillion|thousand|lakh|crore|mn|bn))?"


_UR_SCALE = {"million": "ملین", "mn": "ملین", "billion": "ارب", "bn": "ارب", "بلین": "ارب",
             "trillion": "کھرب", "thousand": "ہزار", "lakh": "لاکھ", "crore": "کروڑ"}
_UR_CURRENCY = r"(?:روپے|روپیہ|روپئے|ڈالرز|ڈالر)"


def _ur_amount(num, scale):
    """An amount in an Urdu sentence, said in Urdu (user's call, 1 Oct): 65bn -> پینسٹھ ارب, 5.2 -> پانچ اعشاریہ دو.
    Marked, so frontend.prepare swaps the numerals for phones (urdu_numbers.WORD_PH)."""
    from urdu_numbers import ONES, POINT, int_to_urdu, mark
    whole, _, frac = num.replace(",", "").partition(".")
    words = int_to_urdu(int(whole or "0"))
    if frac:
        words += f" {POINT} " + " ".join(ONES[int(d)] for d in frac)
    scale = (scale or "").strip()
    return mark(words + (" " + _UR_SCALE.get(scale.lower(), scale) if scale else ""))


def money(t):
    ur = sentence_is_urdu(t)
    if ur:
        say = lambda m, cur: f"{_ur_amount(m.group(1), m.group(2))} {cur}"
        t = re.sub(r"\b(?:PKR|Rs\.?|Rs)\s?" + _AMOUNT, lambda m: say(m, "روپے"), t)
        t = re.sub(_AMOUNT + r"\s?(?:PKR|rupees?)\b", lambda m: say(m, "روپے"), t)
        t = re.sub(r"(?:\bUSD\s?|\bUS\$\s?|\$\s?)" + _AMOUNT, lambda m: say(m, "ڈالر"), t)
        t = re.sub(_AMOUNT + r"\s?(?:USD|dollars?)\b", lambda m: say(m, "ڈالر"), t)
        # an amount already followed by an Urdu currency word (45.2 ارب روپے) or a scale word alone (150 billion)
        t = re.sub(r"(?<![\d.,])" + _AMOUNT + r"(?=\s?" + _UR_CURRENCY + ")", lambda m: _ur_amount(m.group(1), m.group(2)), t)
        return re.sub(r"(?<![\d.,])(\d[\d,]*(?:\.\d+)?)(\s*(?:million|billion|trillion|mn|bn|ملین|بلین|ٹریلین|ارب|کھرب|کروڑ))"
                      r"(?![A-Za-z؀-ۿ])", lambda m: _ur_amount(m.group(1), m.group(2)), t)
    t = re.sub(r"\b(?:PKR|Rs\.?|Rs)\s?" + _AMOUNT, lambda m: f"{m.group(1)}{m.group(2) or ''} rupees", t)
    t = re.sub(_AMOUNT + r"\s?(?:PKR|rupees?)\b", lambda m: f"{m.group(1)}{m.group(2) or ''} rupees", t)
    t = re.sub(r"(?:\bUSD\s?|\bUS\$\s?)" + _AMOUNT, lambda m: f"{m.group(1)}{m.group(2) or ''} dollars", t)
    return t                                                        # English sentences keep $ for NeMo


def urdu_year(n):
    """1954 -> انیس سو چوون (how a year is said, not ایک ہزار نو سو چوون); 2025 -> دو ہزار پچیس.
    Marked for the phone swap, like every other numeral -- see urdu_numbers.WORD_PH."""
    from urdu_numbers import ONES, int_to_urdu, mark
    if n >= 2000:
        return mark(int_to_urdu(n))
    return mark(ONES[n // 100] + " سو" + (" " + ONES[n % 100] if n % 100 else ""))


def urdu_years(t):
    """In a sentence with Urdu script every year is said in Urdu (user's call, 1 Oct), also when English words are
    mixed in. A four-digit number from 1800 to 2099 with no comma, decimal, % or unit after it counts as a year;
    2024-25 is both years. Dates with an English month name were already spelled out by dates()."""
    from urdu_numbers import ONES, mark
    if not sentence_is_urdu(t):
        return t
    yr = r"(?:18|19|20)\d\d"
    t = re.sub(rf"(?<![\w.,/:\-])({yr})-(\d\d|{yr})(?![\w.,%\-])",
               lambda m: urdu_year(int(m.group(1))) + " "
               + (urdu_year(int(m.group(2))) if len(m.group(2)) == 4 else mark(ONES[int(m.group(2))])), t)
    return re.sub(rf"(?<![\w.,/:\-+])({yr})(?![\w%\-]|[.,]\d|\s*(?:%|{_URDU_NUM_WORD}|{_UR_CURRENCY}|فیصد))",
                  lambda m: urdu_year(int(m.group(1))), t)


UNITS = {
    "sq km": "square kilometres", "km2": "square kilometres", "km²": "square kilometres", "km": "kilometres",
    "kg": "kilograms", "mw": "megawatts", "kw": "kilowatts", "gw": "gigawatts", "kwh": "kilowatt hours",
    "cm": "centimetres", "mm": "millimetres", "ft": "feet", "°c": "degrees Celsius", "°f": "degrees Fahrenheit",
    "mb": "megabytes", "gb": "gigabytes", "tb": "terabytes", "mbps": "megabits per second",
}
_UNIT = re.compile(r"(\d)\s?(" + "|".join(re.escape(u) for u in sorted(UNITS, key=len, reverse=True)) + r")(?![A-Za-z])",
                   re.I)


def units(t):
    # 2.81x -> "2.81 گنا" in an Urdu sentence; the unit names below stay English, which is how
    # MMSCFD, BOPD and kilowatts are said here.
    times = " گنا" if sentence_is_urdu(t) else " times"
    t = re.sub(r"(\d(?:[\d.,]*\d)?)x\b", lambda m: m.group(1) + times, t)
    t = re.sub(r"\b\d+(?:\.\d+){2,}\b",                                       # 2.1.3 -> 2 point 1 point 3
               lambda m: " point ".join(m.group(0).split(".")), t)
    return _UNIT.sub(lambda m: f"{m.group(1)} {UNITS[m.group(2).lower()]}", t)


# A postcode is anchored to a city or an explicit label: a bare five-digit run is more often a
# standard number or a quantity (44000 barrels), so it is not read digit by digit on its own.
_PC_CITY = (r"(?:اسلام آباد|کراچی|لاہور|کوئٹہ|پشاور|راولپنڈی|ملتان|فیصل آباد|حیدرآباد|سکھر|بنوں"
            r"|Islamabad|Karachi|Lahore|Quetta|Peshawar|Rawalpindi|Multan|Faisalabad|Hyderabad|Sukkur|Bannu)")
_POSTCODE = re.compile(r"((?:" + _PC_CITY + r")\s*[،,\u2013\u2014-]\s*"
                       r"|(?:postal|post|zip)\s*code\s*:?\s*|(?:پوسٹل|ڈاک)\s*کوڈ\s*:?\s*)(\d{5})\b", re.I)
# A box or extension number is an identifier too, not a count of boxes.
_BOX = re.compile(r"\b(P\.?\s?O\.?\s*Box|Ext(?:ension)?\.?|باکس|ایکسٹینشن)\s*:?\s*(\d{2,6})\b", re.I)


def identifiers(t):
    """Phone numbers, NTN, registration, postcode, box and extension numbers are read digit by digit,
    never as one big number. The leading "+" of a dialling code is dropped rather than spoken (it was
    said "جمع", and "plus" before that); every digit after it is still read."""
    def spaced(m):
        digits = re.sub(r"\D", "", m.group(0))
        return " " + " ".join(digits) + " "
    t = re.sub(r"(?<![\d.,])\+?\d{1,4}(?:[\- ]\d{2,}){2,}(?![\d.])(?!,\d)", spaced, t)   # +92 51 111 410 410
    year_range = re.compile(r"^(?:19|20)\d\d-(?:\d\d|(?:19|20)\d\d)$")          # 2024-25, 2020-2021
    t = re.sub(r"(?<![\d.,])\d+(?:-\d+)+(?![\d.%])(?!,\d)(?!\s*(?:to|سے)\b)",            # 1414673-8, 07-01-2710
               lambda m: spaced(m) if sum(c.isdigit() for c in m.group(0)) >= 6
               and not year_range.match(m.group(0)) else m.group(0), t)
    # 00012471, 111410410, and the 7-8 digit local numbers that share a dialling code earlier in the
    # line ("Tel (+92) 81 2821052, 2864085, 2839790"). A run that long is an identifier unless a scale,
    # currency or unit word follows it, which is what tells 2821052 from a quantity.
    t = re.sub(r"(?<![\d.,])(?:0\d{5,}|\d{7,})(?![\d.])(?!,\d)"
               r"(?!\s*(?:" + _URDU_NUM_WORD + r"|" + _UR_CURRENCY + r"|million|billion|trillion|thousand"
               r"|mn|bn|MT|BOPD|BOEPD|MMSCFD|MMBOE|KBOEPD))", spaced, t)
    t = _POSTCODE.sub(lambda m: m.group(1) + " " + " ".join(m.group(2)) + " ", t)
    t = _BOX.sub(lambda m: m.group(1) + " " + " ".join(m.group(2)) + " ", t)
    return re.sub(r"\s{2,}", " ", t)


# Urdu ordinals 1-10; above that the cardinal plus واں.
_URDU_ORDINAL = {1: "پہلا", 2: "دوسرا", 3: "تیسرا", 4: "چوتھا", 5: "پانچواں",
                 6: "چھٹا", 7: "ساتواں", 8: "آٹھواں", 9: "نواں", 10: "دسواں"}


def fiscal_year_label(t):
    """FY2024-25 -> "مالی سال 2024-25". Without this the token is a letter+number code, so its digits were
    read in English while the tail year was read in Urdu ("ایف وائی twenty twenty four پچیس"). Leaving the
    digits here lets ranges() juxtapose them and urdu_years() read them as years. Runs before ranges()."""
    if not sentence_is_urdu(t):
        return t                           # an English sentence keeps "FY" as the code it is
    t = re.sub(r"\bFY\s?(\d{2,4})\s*[-\u2013]\s*(\d{2,4})\b", r"مالی سال \1 \2", t)
    return re.sub(r"\bFY\s?(?=\d)", "مالی سال ", t)


def ranges(t):
    """A hyphen or "to" BETWEEN TWO NUMBERS is a range -- not a minus sign, and not the English word "two".

    urdu_numbers reads a leading "-" as منفی, so "gas 30-36 MMSCFD" was read "thirty MINUS thirty six"
    (57 hyphen ranges in the knowledge base), and "100 to 115" was read with the English "to", which comes
    out as ʈˈuː and is heard as "two" -- "one hundred two one hundred fifteen". Both now read سے.

    A fiscal or calendar year range is juxtaposed instead, which is how it is said: FY24-25 -> "چوبیس پچیس".
    Runs after identifiers(), so NTN/registration/GST numbers are already spaced out digit by digit and
    their hyphens are gone; letter-hyphen-number codes (G-10, Block-28, Bhitai-6) have a letter before the
    hyphen and are left to ranks_and_codes().
    """
    if not sentence_is_urdu(t):
        return t                           # an English sentence keeps "10-15" and "2020-21"
    t = re.sub(r"\b(FY\s?)(\d{2,4})\s*[-\u2013]\s*(\d{2,4})\b", r"\1\2 \3", t)
    t = re.sub(r"(?<![\d.,\-])((?:19|20)\d{2})\s*[-\u2013]\s*(\d{2}|(?:19|20)\d{2})(?![\d.,\-])",
               r"\1 \2", t)
    t = re.sub(r"(?<![\d.,\-])(\d[\d,]*(?:\.\d+)?)\s*[-\u2013]\s*(\d[\d,]*(?:\.\d+)?)(?![\d.,\-])",
               r"\1 سے \2", t)
    return re.sub(r"(?<=\d)\s+to\s+(?=\d)", " سے ", t)


def ordinals(t):
    """2nd -> دوسرا. They were read as the English "second": dates() only rewrites an ordinal that sits in
    front of a month name, so a standalone one ("2nd quarter") fell through to the English span."""
    if not sentence_is_urdu(t):
        return t                           # pk_gazetteer reads "3rd March" as "third March"

    def one(m):
        n = int(m.group(1))
        return _URDU_ORDINAL.get(n) or (_words(n) + "واں")
    return re.sub(r"\b(\d{1,3})(?:st|nd|rd|th)\b", one, t)

def urdu_times(t):
    if not sentence_is_urdu(t):
        return t
    frontend()                                  # loads frontend.py (and through it urdu_numbers)
    from urdu_numbers import mark, number_to_urdu
    def one(m):
        h, mnt = int(m.group(1)), int(m.group(2))
        return mark(number_to_urdu(str(h))) + " بجے" if mnt == 0 else \
            f"{mark(number_to_urdu(str(h)))} بج کر {mark(number_to_urdu(str(mnt)))} منٹ"
    return re.sub(r"\b(\d{1,2}):(\d{2})\b", one, t)


# ---------------------------------------------------------------------------
# acronyms the front end does not cover
# ---------------------------------------------------------------------------
def acronyms(t):
    # & inside an acronym: E&P, HR&R, D&PL, M&A
    t = re.sub(r"\b([A-Z]{1,4})&([A-Z]{1,4})\b", lambda m: f"{_spell(m.group(1))} اینڈ {_spell(m.group(2))}", t)
    # dotted capitals: A.F., U.S.A. (not e.g./i.e., which the front end expands)
    t = re.sub(r"\b((?:[A-Z]\.){2,})(?=\s|$|[,;:)])", lambda m: _spell(m.group(1).replace(".", "")) + " ", t)
    # plural or possessive: CEOs, SDGs, PSX's -> letters with ز on the last one
    t = re.sub(r"\b([A-Z]{2,8})['’]?s\b",
               lambda m: m.group(0) if _word_acronym(m.group(1)) else _spell(m.group(1)) + "ز", t)
    # 7-8 capitals, not a word: spelled as well (the front end stops at 6)
    fe = frontend()
    t = re.sub(r"\b[A-Z]{7,8}\b",
               lambda m: m.group(0) if (_word_acronym(m.group(0)) or m.group(0).lower() in fe._dictionary())
               else _spell(m.group(0)), t)
    # bare "&" between words
    return re.sub(r"\s&\s", " اور " if is_urdu(t) else " and ", t)


# The greeting, however it is written, becomes one spelling: shadda on the doubled s, silent laam and the
# short vowels marked. Chosen by ear (greeting_D); the plain spellings made the model say "al-salaam" or a short s.
GREETING = "اَسّلامُ علیکم"
GREETING_REPLY = "وَعلیکُم اَسّلام"
# optional marks throughout, so the chosen spelling itself matches and the replacement is idempotent
_GREET_UR = re.compile(r"(?:ال|اَ?)?س{1,2}ّ?لامُ?(?:\s*[وُ]\s*|\s+)علیکم")
_GREET_LAT = re.compile(r"\b(?:as+ala+m+(?:u|o)?[\s\-']*(?:o[\s\-']*|u[\s\-']*|wa[\s\-']*)?a?l[ae]i?kum|AOA)\b", re.I)
_REPLY_UR = re.compile(r"وَ?علیکُ?م\s*(?:ال|اَ?)?س{1,2}ّ?لام")
_REPLY_LAT = re.compile(r"\bwa[\s\-']*a?l[ae]i?kum[\s\-']*(?:a?s+ala+m+|us+ala+m+)\b", re.I)


def greeting(t):
    t = _REPLY_LAT.sub(GREETING_REPLY, t)
    t = _REPLY_UR.sub(GREETING_REPLY, t)
    t = _GREET_LAT.sub(GREETING, t)
    return _GREET_UR.sub(GREETING, t)


def split_camel(t):
    """MariEnergies -> Mari Energies, BankIslami -> Bank Islami, so each part reaches the names table and the
    English rules as a word. Only joins of two 3+ letter parts, so LinkedIn, iPhone and McDonald stay whole."""
    return re.sub(r"\b([A-Z][a-z]{2,})(?=[A-Z][a-z]{2,})", r"\1 ", t)


def abbreviations_without_period(t):
    t = re.sub(r"(?<![A-Za-z])w\.\s?e\.\s?f\.?(?![A-Za-z])", "with effect from", t, flags=re.I)   # w.e.f. 24/6/2025
    return re.sub(r"\b(Pvt|Ltd|Co|Inc|Corp)\b(?!\.)",
                  lambda m: {"pvt": "Private", "ltd": "Limited", "co": "Company", "inc": "Incorporated",
                             "corp": "Corporation"}[m.group(1).lower()], t)


_RANKS = {"brig": "Brigadier", "retd": "Retired", "col": "Colonel", "maj": "Major", "capt": "Captain"}


_MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"
_MONTH_LIST = _MONTHS.split("|")


# Numbers are read in Urdu (user's call, 19 Oct). The three reasons they used to be English are
# documented below -- mixed_numbers (English for within-sentence consistency), decimals ("اعشاریہ"
# came out as "acharia") and standards (ISO codes). The "acharia" problem no longer applies:
# urdu_numbers spells all 100 numerals, اعشاریہ and منفی as phones, so the short vowels are heard.
# Set URDU_NUMBERS = False to get the old English reading back.
URDU_NUMBERS = True


def _words(n, **kw):
    """The one place text_norm turns a number into words. to='ordinal'/'year' are ignored in Urdu: Urdu dates
    use the cardinal ("سولہ اکتوبر", not an ordinal) and a year is read as a plain number."""
    if URDU_NUMBERS:
        frontend()                                  # loads urdu_numbers through frontend.py
        from urdu_numbers import mark, number_to_urdu
        return mark(number_to_urdu(str(int(n))))
    return _en_words(n, **kw)


def _en_words(n, **kw):
    """Always English, whatever URDU_NUMBERS says: the digits inside a letter+number code are read in
    English ("Bhitai six", "Block four", "ISO nine thousand and one"), which is how they are said."""
    from num2words import num2words
    return num2words(n, **kw).replace("-", " ").replace(",", "")


def dates(t):
    """23 March 2025 / March 23, 2025 -> "twenty third March twenty twenty five": said without "the ... of"."""
    # Per sentence, not per number: an English sentence keeps the English date, and mixing the two
    # readings inside one sentence is exactly what mixed_numbers() was written to stop.
    w = _words if (URDU_NUMBERS and sentence_is_urdu(t)) else _en_words

    def dm(m):
        return f"{w(int(m.group(1)), to='ordinal')} {m.group(3)}" + (f" {w(int(m.group(4)), to='year')}" if m.group(4) else "")

    def md(m):
        return f"{w(int(m.group(2)), to='ordinal')} {m.group(1)}" + (f" {w(int(m.group(4)), to='year')}" if m.group(4) else "")
    # numeric dates first, day first as written in Pakistan: 05/04/2024, 24/6/2025, 2025-03-23 -> "5 April 2024"
    def slash(m):
        d, mo, y = int(m.group(1)), int(m.group(2)), m.group(3)
        return f"{d} {_MONTH_LIST[mo - 1]} {y}" if 1 <= d <= 31 and 1 <= mo <= 12 else m.group(0)
    t = re.sub(r"(?<![\d/])(\d{1,2})/(\d{1,2})/((?:19|20)\d{2})(?![\d/])", slash, t)
    t = re.sub(r"(?<![\d-])((?:19|20)\d{2})-(\d{2})-(\d{2})(?![\d-])",
               lambda m: slash(re.match(r"(\d+)/(\d+)/(\d+)", f"{m.group(3)}/{m.group(2)}/{m.group(1)}")), t)
    t = re.sub(rf"\b(\d{{1,2}})(st|nd|rd|th)?\s+({_MONTHS}),?(?:\s+(\d{{4}}))?\b", dm, t)
    return re.sub(rf"\b({_MONTHS})\s+(\d{{1,2}})(st|nd|rd|th)?\b(?:,?\s+(\d{{4}}))?", md, t)


def decimals(t):
    """55.12 -> "fifty five point one two" in English: the Urdu reading's "اعشاریہ" came out as "acharia"."""
    if URDU_NUMBERS and sentence_is_urdu(t):
        # urdu_numbers.expand_numbers reads the decimal in Urdu, اعشاریہ spelled as phones.
        return t
    def w(m):
        whole, frac = m.group(1).replace(",", ""), m.group(2)
        return f"{_en_words(int(whole))} point {' '.join(_en_words(int(d)) for d in frac)}"
    return re.sub(r"(?<![\d.])(\d{1,3}(?:,\d{3})*|\d+)\.(\d+)(?![\d.])", w, t)


def ranks_and_codes(t):
    """Military ranks (Brig, Col., Maj., Capt, Retd), Islamabad sector codes (G-10/4 -> G ten four, all English, as
    it is said) and a capital letter with a short number (A1, B2 -> the letter name, then the number)."""
    # two-part ranks with or without dots (Lt Gen, Lt. Gen., Maj Gen), before the one-part ones
    t = re.sub(r"\b(Lt|Maj)\.?\s+Gen\b\.?", lambda m: ("Lieutenant" if m.group(1) == "Lt" else "Major") + " General", t)
    t = re.sub(r"\b(Brig|Retd|Capt)\b\.?|\b(Col|Maj)\.", lambda m: _RANKS[(m.group(1) or m.group(2)).lower()], t)
    from num2words import num2words
    t = re.sub(r"\b([A-HJ-Z])-(\d{1,2})/(\d{1,2})\b",
               lambda m: f"{m.group(1)} {num2words(int(m.group(2)))} {num2words(int(m.group(3)))}", t)
    t = re.sub(r"(?<![\w-])([A-Z])(\d{1,2})(?![\w-])", lambda m: f"{_spell(m.group(1))} {num2words(int(m.group(2)))}", t)
    # a name joined to its number (Bhitai-6, Spinwam-1, Block-28, KSE-100): the hyphen is not a minus sign, which
    # the Urdu number reader otherwise says as "منفی"
    # "as a service" acronyms are spelled: IaaS, PaaS, DRaaS, GPUaaS (DRaaS was read as the word "draas")
    t = re.sub(r"\b([A-Z]{1,5})aaS\b", lambda m: _spell(m.group(1) + "AAS"), t)
    # a name joined to its number (Bhitai-6, Spinwam-1, Block-28, KSE-100) is said in English, and the hyphen is not a
    # minus sign; same for Block / Well / Phase / Tier numbers
    t = re.sub(r"(?<=[A-Za-z\u0600-\u06FF])-(\d{1,3})\b", lambda m: " " + _en_words(int(m.group(1))), t)
    return re.sub(r"\b(Block|Well|Phase|Tier)\s+(\d{1,3})\b", lambda m: f"{m.group(1)} {_en_words(int(m.group(2)))}", t)


def standards(t):
    """ISO 9001:2015, ISO 45001:2018, IEC 62443 -> the code's number and year in English words ("ISO nine thousand
    and one twenty fifteen"); the colon was read as a pause and the numbers in Urdu inside Urdu sentences."""
    def one(m):
        out = f"{m.group(1)} {_en_words(int(m.group(2)))}"
        return out + (f" {_en_words(int(m.group(3)), to='year')}" if m.group(3) else "")
    return re.sub(r"\b(ISO|IEC|OHSAS|IATF|EN)\s?(\d{3,5})(?::((?:19|20)\d{2}))?\b", one, t)


_URDU_NUM_WORD = r"(?:لاکھ|کروڑ|ارب|کھرب|ہزار|سو|بجے|بج|منٹ|تاریخ)"


def mixed_numbers(t):
    """In a sentence that mixes Urdu and English, every number is said in English (years as years): the reader used
    to pick Urdu or English per number from its neighbours, so one sentence said "sixty percent" and "تیس فیصد".
    Pure Urdu sentences keep Urdu numbers, and so does a number followed by an Urdu scale or time word (2 لاکھ)."""
    if URDU_NUMBERS or not (sentence_is_urdu(t) and re.search(r"[A-Za-z]{2}", t)):
        return t                                # the digits reach urdu_numbers.expand_numbers instead

    def one(m):
        num, suffix = m.group(1), m.group(2) or ""
        n = int(num.replace(",", ""))
        if suffix == "" and "," not in num and 1900 <= n <= 2099:
            words = _words(n, to="year")
        else:
            words = _words(n)
        return words + (" percent" if suffix == "%" else " times" if suffix.lower() == "x" else "")
    return re.sub(r"(?<![\w.,/:\-])(\d{1,3}(?:,\d{3})+|\d+)(%|[xX](?![A-Za-z]))?(?![\d.,]*\d)(?![A-Za-z])"
                  r"(?!\s*" + _URDU_NUM_WORD + ")", one, t)


def phone_parens(t):
    """ "(+92) 51-111 ..." -> "+92 51-111 ..." so the phone rule (identifiers) reads the plus and every digit."""
    return re.sub(r"\(\s*\+(\d{1,3})\s*\)", r"+\1", t)


def normalize(text):
    t = unicodedata.normalize("NFC", str(text))
    _ctx.urdu = is_urdu(t)                          # judged once, before any pass can rewrite it
    try:
        return _normalize(t)
    finally:
        _ctx.urdu = None


def _normalize(t):
    for step in (strip_markup, greeting, web, dates, standards, money, units, decimals, phone_parens, identifiers,
                 fiscal_year_label, ranges, ordinals, urdu_times, ranks_and_codes, urdu_years, mixed_numbers, acronyms,
                 split_camel, abbreviations_without_period):
        t = step(t)
    return re.sub(r"[ \t]{2,}", " ", t).strip()


# (input, must appear in normalize() output, must NOT appear)
TESTS = [
    # Urdu dates take the cardinal, not an ordinal: "سولہ اکتوبر", never "سولہواں".
    ("Seema Adil 24/06/2022 کو join ہوئیں", "چوبیس June دو ہزار بائیس", "/"),
    ("اور w.e.f. 24/6/2025 دوبارہ منتخب ہوئیں۔", "with effect from چوبیس June", "w.e.f"),
    ("Mari Ghazij سے 16/10/2025 کو oil ملا", "سولہ October دو ہزار پچیس", "16/10"),
    ("ISO 9001:2015 اور ISO 45001:2018 کے مطابق", "ISO nine thousand and one twenty fifteen", ":"),
    # The digits and the % survive normalize(); urdu_numbers reads them ("تیس فیصد") in prepare.
    ("Sky47 میں 60% stake ہے، جبکہ Paramount کے پاس 30% ہے۔", "30%", "thirty"),
    ("GHG emissions 1,680,500 MT رہیں (2024: 1,608,492 MT)", "دو ہزار چوبیس", "twenty twenty four"),
    ("مجھے مہینے کے 2 لاکھ کمانے ہیں، honestly", "2 لاکھ", None),
    ("اس سال 30 فیصد اضافہ ہوا۔", "30 فیصد", "thirty"),
    ("The meeting is on 3rd March.", "third March", "3rd"),
    ("Company Secretary Brig Sumair Ashraf Sheikh (Retd)", "Brigadier", "Brig "),
    ("Col. Ahmed and Maj. Ali", "Colonel Ahmed and Major Ali", "Col."),
    ("Head office in G-10/4, Islamabad", "G ten four", "10/4"),
    ("short term rating A1", "one", "A1"),
    ("Bhitai-6 well", "Bhitai six", "-6"),
    ("production began on 23 March 2025", "twenty third March twenty twenty five", "the"),
    ("As at June 30, 2025", "thirtieth June twenty twenty five", "of"),
    ("EBITDA margin 55.12% رہا", "55.12%", "fifty five"),
    ("Sky47 Cloud offers DRaaS", "ڈی آر اے اے ایس", "DRaaS"),
    ("Block-28 and Block 4", "Block twenty eight and Block four", "Block 4"),
    ("Check your **ETF** balance.", "ETF", "**"),
    ("Visit [our site](https://mari.com.pk) today", "our site", "https"),
    ("Email info@marienergies.com.pk for details", "info at marienergies dot com dot pk", "@"),
    ("See www.psx.com.pk", "w w w dot psx dot com dot pk", "www."),
    ("The share price is PKR 651.65 today.", "six hundred and fifty one point six five rupees", "PKR"),
    ("Profit was Rs. 5 billion", "5 billion rupees", "Rs"),
    ("آج شیئر کی قیمت PKR 651.65 ہے", "چھ سو اکاون اعشاریہ چھ پانچ روپے", "PKR"),
    ("Revenue of USD 2.5 million", "two point five million dollars", "USD"),
    ("اس کی قیمت $5 ہے", "پانچ ڈالر", "$"),
    ("The block covers 155,756 sq km", "square kilometres", "sq km"),
    ("a 50MW solar plant", "50 megawatts", "MW"),
    # The "+" is dropped, not spoken: it was "جمع", and "plus" before that.
    ("Call +92 51 111 410 410 now", "9 2 5 1 1 1 1 4 1 0 4 1 0", "plus"),
    ("NTN: 1414673-8", "1 4 1 4 6 7 3 8", "1414673-8"),
    ("Registration Number: 00012471", "0 0 0 1 2 4 7 1", "00012471"),
    ("growth of 10-15 percent", "10-15", "1 0"),
    ("میٹنگ 10:30 پر ہے", "دس بج کر تیس منٹ", "10:30"),
    ("The E&P sector", "ای اینڈ پی", "E&P"),
    ("Our HR&R team", "ایچ آر اینڈ آر", "HR&R"),
    ("Audited by A.F. Ferguson", "اے ایف", "A.F."),
    ("Three CEOs spoke", "سی ای اوز", "CEOs"),
    ("the UN SDGs", "ایس ڈی جیز", "SDGs"),
    ("The PSX's index", "پی ایس ایکسز", "PSX's"),
    ("NASA's mission", "NASA", "این اے"),
    ("Corplink (Pvt) Limited", "Private", "Pvt"),
    ("Ferguson & Co.", "and", "&"),
    ("🚀 Great news!", "Great news", "🚀"),
    ("MariEnergies is listed", "Mari Energies", "MariEnergies"),
    ("السلام علیکم! میں مریم ہوں", "اَسّلامُ علیکم", "السلام"),
    ("اسلام و علیکم۔ خوش آمدید", "اَسّلامُ علیکم", "اسلام و"),
    ("Assalam-o-Alaikum, welcome", "اَسّلامُ علیکم", "اَاَ"),
    ("Assalamualaikum everyone", "اَسّلامُ علیکم", "اَاَ"),
    ("AOA! How can I help?", "اَسّلامُ علیکم", "اَاَ"),
    ("وعلیکم السلام", "وَعلیکُم اَسّلام", "وعلیکم السلام"),
    ("Walaikum Assalam", "وَعلیکُم اَسّلام", "Walaikum"),
    ("اسلام آباد میں دفتر ہے", "اسلام آباد", "اَسّلام"),
    ("اَسّلامُ علیکم", "اَسّلامُ علیکم", "اَاَ"),
    ("Assalam-o-Alaikum", "اَسّلامُ علیکم", "Assalam"),
    ("FY2024-25 EPS: PKR 54.25", "FY2024-25", "2 0 2 4"),
    ("from 260 BSCF (2020-21)", "2020-21", "2 0 2 0"),
    ("- **11.2.2** Liquidity: Current ratio 2.81x", "two point eight one times", "11.2.2"),
    ("update to version 2.1.3", "2 point 1 point 3", "2.1.3"),
    ("Visit LinkedIn", "LinkedIn", "Linked In"),
]

# (input, must appear in the full prepare_pk output) -- acronyms end up as Urdu letter names
PREPARED_TESTS = [
    ("Check the SECP website", "ایس ای سی پی"),
    ("Apply to HEC", "ایچ ای سی"),
    ("The KSE index", "کے ایس ای"),
    ("Your ETF and NAV", "ای ٹی ایف"),
    ("Your ETF and NAV", "این اے وی"),
    ("Mari produces 36 MMSCFD of gas", "ایم ایم ایس سی ایف ڈی"),
    ("He has a PhD", "پی ایچ ڈی"),
    ("OGDCL and SNGPL", "او جی ڈی سی ایل"),
    ("NADRA office", "نادرا"),
    ("an upgrade to 5G", "جی"),
    ("MariEnergies and BankIslami", "ماڑی انرجیز"),
    ("Eid Mubarak from all of us", "عید"),
]


def run_tests():
    import pk_gazetteer
    bad = 0
    from urdu_numbers import MARK
    for src, must, must_not in TESTS:
        out = normalize(src).replace(MARK, "")       # the marker is plumbing; it never reaches the model
        ok = must in out and (not must_not or must_not not in out)
        bad += not ok
        print(f"{'PASS' if ok else 'FAIL'} {src!r} -> {out!r}")
    for src, must in PREPARED_TESTS:
        out = pk_gazetteer.prepare_pk(src)
        ok = must in out
        bad += not ok
        print(f"{'PASS' if ok else 'FAIL'} [prepared] {src!r} -> {out!r}")
    n = len(TESTS) + len(PREPARED_TESTS)
    print(f"{n - bad}/{n} passed")
    return bad


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    a = ap.parse_args()
    sys.exit(1 if run_tests() else 0)
