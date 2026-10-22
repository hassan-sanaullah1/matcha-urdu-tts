"""Deployment front end: chatbot text (Urdu script + Latin English) -> the model's input.

tts.prepare("phones") handles plain English words (espeak en-us -> Pakistani-English
phones). This module adds the cases no amount of training fixes, all measured on the
user's own sentences and mari_energies_knowledge_base.md:

  1. parenthetical glosses  "ایم ایم بی او ای (MMBOE)" was read twice
  2. alphanumeric codes     2P, FY2024-25, COP29, X-1, Sky47: digits become ENGLISH
                            number words (they were read in Urdu, mid-English-code)
  3. letter-by-letter acronyms, two arms for an A/B by ear:
       letters="ur"  Urdu-script letter names (A -> اے, W -> ڈبلیو, Z -> زیڈ), read
                     through the Urdu letters the base model learned from hours of audio
       letters="ph"  the same names as phone strings
     the pk-voice host's audio had 0 spoken B H J K Q W X Z, so neither arm relies on
     learning letters from speech
  4. which all-caps tokens are words: ACRONYM_WORDS (NADRA, LUMS...) plus a dictionary
     rule for headings (SMART, ONLY are words; OGDCL is not)
  5. PHONE_OVERRIDE: names espeak misreads, as literal phone strings
  6. Roman-Urdu names (pk-voice's guard + the knowledge base's people and places) go to
     Urdu script, so they are not read as English

Usage:
  python src/frontend.py --demo
  python src/frontend.py "Huawei کے CEO نے 2P reserves کی بات کی" --letters ur
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import paths  # noqa: E402  -- kiosk-tts/src/paths.py: every location + the import path

import argparse
import functools
import importlib.util
import re
import unicodedata

import phones
import tts

# ---------------------------------------------------------------------------
# letter names, as said in Pakistan (British-derived: Z is "zed")
# ---------------------------------------------------------------------------
LETTER_UR = {
    "A": "اے", "B": "بی", "C": "سی", "D": "ڈی", "E": "ای", "F": "ایف", "G": "جی", "H": "ایچ",
    "I": "آئی", "J": "جے", "K": "کے", "L": "ایل", "M": "ایم", "N": "این", "O": "او", "P": "پی",
    "Q": "کیو", "R": "آر", "S": "ایس", "T": "ٹی", "U": "یو", "V": "وی", "W": "ڈبلیو",
    "X": "ایکس", "Y": "وائی", "Z": "زیڈ",
}
LETTER_PH = {
    "A": "ˈeː", "B": "ˈbiː", "C": "ˈsiː", "D": "ˈɖiː", "E": "ˈiː", "F": "ˈɛf", "G": "ˈʤiː",
    "H": "ˈeːʧ", "I": "ˈaːɪ", "J": "ˈʤeː", "K": "ˈkeː", "L": "ˈɛl", "M": "ˈɛm", "N": "ˈɛn",
    "O": "ˈoː", "P": "ˈpiː", "Q": "ˈkjuː", "R": "ˈaːr", "S": "ˈɛs", "T": "ˈʈiː", "U": "ˈjuː",
    "V": "ˈʋiː", "W": "ˈɖəbəljuː", "X": "ˈɛks", "Y": "ˈʋaːɪ", "Z": "ˈzɛɖ",
}
# reverse table for gloss detection; the extra spellings are common variants
_UR_TO_LETTER = {**{u: l for l, u in LETTER_UR.items()},
                 "ای": "E", "وائے": "Y", "زی": "Z", "ڈبلیو": "W", "اؤ": "O"}

# all-caps tokens said as one word, not spelled
ACRONYM_WORDS = {
    "NASA", "COVID", "UNESCO", "UNICEF", "NATO", "OPEC", "SIM", "PIN", "LAN", "RAM", "ROM",
    "JPEG", "GIF", "SCUBA", "LASER", "RADAR", "WAPDA", "NADRA", "FIFA", "ASAP",
    # knowledge base / Pakistani energy and finance
    "LUMS", "PARCO", "OGRA", "NEPRA", "SECP", "ICAP", "COP", "FAST", "NUST", "MARI",
    "SUPARCO", "HEC", "KSE",
}
# said as words although spelled out by the rule above would be wrong; lowercase keys
PHONE_OVERRIDE = {
    "huawei": "ʋˈaːʋeː",
    "schlumberger": "ʃlˈʊmbərʒeː",
    "exxonmobil": "ˈɛksɔn mˈoːbɪl",
    "nvidia": "ɛnʋˈɪɖiːaː",
    "secp": "ˈɛs ˈiː ˈsiː ˈpiː",       # said letter by letter despite being pronounceable
    "icap": "ˈaːɪkɛp",
    "kse": "ˈkeː ˈɛs ˈiː",
    "hec": "ˈeːʧ ˈiː ˈsiː",
}
# knowledge-base people and places typed in Latin (Urdu script, so they are read as Urdu)
ROMAN_URDU_EXTRA = {
    "faheem": "فہیم", "haider": "حیدر", "waziristan": "وزیرستان", "sujawal": "سجاول",
    "ghazij": "غازج", "shewa": "شیوا", "spinwam": "سپنوام", "kot": "کوٹ", "hamed": "حامد",
    "yaqoob": "یعقوب", "zafar": "ظفر", "abbas": "عباس", "niaz": "نیاز", "majid": "ماجد",
    "syed": "سید", "sheikh": "شیخ", "hayat": "حیات", "fauji": "فوجی", "careem": "کریم",
    "daraz": "دراز", "meezan": "میزان", "mashal": "مشعل", "khan": "خان", "jinnah": "جناح",
    "quaid": "قائد", "azam": "اعظم",
    # Pakistani institutions said as Urdu words, not English: espeak read NADRA as nˈɛɖrə
    "nadra": "نادرا", "wapda": "واپڈا", "nepra": "نیپرا", "ogra": "اوگرا", "parco": "پارکو",
    "suparco": "سپارکو", "lums": "لمز", "nust": "نسٹ",
}


@functools.lru_cache(maxsize=1)
def _roman_urdu():
    """text_ipa.py's Roman-Urdu table (loaded by file path; from pk-voice) + ROMAN_URDU_EXTRA."""
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "text_ipa.py")
    spec = importlib.util.spec_from_file_location("pk_text_ipa", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    table = {**mod.ROMAN_URDU, **ROMAN_URDU_EXTRA}
    rx = re.compile(r"(?<![A-Za-z])(" + "|".join(
        re.escape(k) for k in sorted(table, key=len, reverse=True)) + r")(?![A-Za-z])", re.I)
    return table, rx


def roman_urdu_guard(text):
    table, rx = _roman_urdu()
    return rx.sub(lambda m: table[m.group(0).lower()], text)


@functools.lru_cache(maxsize=1)
def _dictionary():
    try:
        return {w.strip().lower() for w in open(paths.DICT_WORDS, encoding="utf-8")
                if w.strip().isalpha()}
    except OSError:
        return set()


# ---------------------------------------------------------------------------
# 1. parenthetical glosses
# ---------------------------------------------------------------------------
_PAREN = re.compile(r"\s*\(\s*([A-Za-z][A-Za-z0-9\-]*)\s*\)")


def drop_parenthetical_gloss(text):
    """Drop "(MMBOE)" when the words just before it already spell M M B O E in Urdu
    letter names, or are the same Latin token. Other parentheses are left alone."""
    out, last = [], 0
    for m in _PAREN.finditer(text):
        gloss = m.group(1)
        letters = [c for c in gloss.upper() if c.isalpha()]
        before = text[last:m.start()]
        words = before.split()
        drop = False
        if words and words[-1].strip("،,") == gloss:
            drop = True
        elif letters and len(words) >= len(letters) and gloss.replace("-", "").isalpha():
            tail = [w.strip("،,") for w in words[-len(letters):]]
            drop = [_UR_TO_LETTER.get(w) for w in tail] == letters
        out.append(before)
        if not drop:
            out.append(m.group(0))
        last = m.end()
    out.append(text[last:])
    return "".join(out)


# ---------------------------------------------------------------------------
# 2. alphanumeric codes
# ---------------------------------------------------------------------------
_CODE = re.compile(r"(?<![A-Za-z0-9])(?=[A-Za-z0-9\-]*[A-Za-z])(?=[A-Za-z0-9\-]*[0-9])"
                   r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*(?![A-Za-z0-9])")


def _en_number(digits):
    from num2words import num2words
    n = int(digits)
    if len(digits) == 4 and 1900 <= n <= 2099:
        return num2words(n, to="year")
    return num2words(n)


def expand_codes(text):
    """FY2024-25 -> FY twenty twenty-four twenty-five; 2P -> two P; X-1 -> X one.
    Only tokens mixing letters and digits; plain numbers stay for Urdu expand_numbers."""
    def one(m):
        parts = re.findall(r"[A-Za-z]+|[0-9]+", m.group(0))
        return " ".join(_en_number(p) if p.isdigit() else p for p in parts)
    return _CODE.sub(one, text)


# ---------------------------------------------------------------------------
# 3-5. a Latin span -> pieces
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# period abbreviations: the standard TTS table, not domain words
# ---------------------------------------------------------------------------
ABBREVIATIONS = {
    "mr": "mister", "mrs": "missus", "ms": "miss", "dr": "doctor", "prof": "professor",
    "st": "saint", "jr": "junior", "sr": "senior", "pvt": "private", "ltd": "limited",
    "inc": "incorporated", "co": "company", "corp": "corporation", "govt": "government",
    "dept": "department", "etc": "et cetera", "vs": "versus", "approx": "approximately",
    "no": "number", "nos": "numbers", "e.g": "for example", "i.e": "that is", "rs": "rupees",
    "est": "established", "engr": "engineer", "capt": "captain", "col": "colonel",
    "gen": "general", "lt": "lieutenant", "maj": "major", "brig": "brigadier", "retd": "retired", "min": "minimum", "max": "maximum", "avg": "average",
}
_ABBR = re.compile(r"(?<![A-Za-z])(" + "|".join(
    re.escape(k) for k in sorted(ABBREVIATIONS, key=len, reverse=True)) + r")\.(?=\s|$|[،,;:)])", re.I)


def expand_abbreviations(text):
    """Dr. -> doctor, Pvt. Ltd. -> private limited, e.g. -> for example. Only with the
    period, so "no" and "co" as words are untouched; "No." before a digit stays for NeMo."""
    def one(m):
        if m.group(1).lower() == "no" and not re.match(r"\s*[0-9]", m.string[m.end():]):
            return m.group(0)
        return ABBREVIATIONS[m.group(1).lower()]
    return _ABBR.sub(one, text)


# ---------------------------------------------------------------------------
# numbers in ENGLISH context: NeMo's English text normaliser (money, %, dates, times,
# decimals, measures, ordinals). Numbers in Urdu context stay with urdu_numbers.
# ---------------------------------------------------------------------------
_ARABIC = re.compile(r"[؀-ۿݐ-ݿ]")


@functools.lru_cache(maxsize=1)
def _nemo():
    from nemo_text_processing.text_normalization.normalize import Normalizer
    return Normalizer(input_case="cased", lang="en",
                      cache_dir=paths.NEMO_CACHE)


def english_numbers(text):
    """A run of whitespace-separated tokens with no Urdu script is English context if it
    has at least one Latin word; its digits and $ % symbols are verbalised in English
    ("revenue 45% بڑھا" -> forty five percent). A number between Urdu words is not touched."""
    parts = re.split(r"(\s+)", text)
    out, group = [], []

    def close():
        if not group:
            return
        g = "".join(group)
        if re.search(r"[A-Za-z]{2,}", g) and re.search(r"[0-9$£€%]", g):
            lead = g[:len(g) - len(g.lstrip())]
            trail = g[len(g.rstrip()):]
            g = lead + _nemo().normalize(g.strip(), verbose=False, punct_post_process=True) + trail
        out.append(g)
        group.clear()

    for tok in parts:
        if _ARABIC.search(tok):
            close()
            out.append(tok)
        else:
            group.append(tok)
    close()
    return "".join(out)


# ---------------------------------------------------------------------------
# 3-5. a Latin span -> pieces
# ---------------------------------------------------------------------------
_SPAN = re.compile(r"[A-Za-z]+(?:[ '’\-]+[A-Za-z]+)*")
_PLURAL_ACRONYM = re.compile(r"^([A-Z]{2,6})['’]?s$")
_MIXED_ABBR = re.compile(r"^(?=[A-Za-z]*[A-Z][A-Za-z]*[A-Z])[A-Za-z]{2,4}$")
_ROMAN = re.compile(r"^(X{0,3})(IX|IV|V?I{0,3})$")


def _roman_value(tok):
    m = _ROMAN.match(tok)
    if not m or len(tok) < 2:
        return None
    vals = {"I": 1, "V": 5, "X": 10}
    n = 0
    for i, c in enumerate(tok):
        v = vals[c]
        n += -v if i + 1 < len(tok) and vals[tok[i + 1]] > v else v
    return n


_ONSETS = {"bl", "br", "ch", "cl", "cr", "dr", "fl", "fr", "gl", "gr", "kh", "gh", "kn", "pl",
           "pr", "sc", "sh", "sk", "sl", "sm", "sn", "sp", "st", "sw", "th", "tr", "tw", "wh",
           "wr", "ph", "bh", "dh", "jh"}
_CODAS = {"ch", "ck", "ct", "ff", "ft", "ld", "lf", "lk", "ll", "lm", "lp", "lt", "mp", "nd",
          "ng", "nk", "nt", "pt", "rb", "rd", "rf", "rg", "rk", "rl", "rm", "rn", "rp", "rt",
          "sh", "sk", "sp", "ss", "st", "th", "tt", "zz", "gh", "rs", "ns", "ms", "ls", "ts"}
_SYLLABLE = re.compile(r"^([^aeiouy]{0,2})[aeiouy]+([^aeiouy]{0,2}?)s?$")


def _one_syllable(w):
    m = _SYLLABLE.match(w)
    if not m:
        return False
    onset, coda = m.group(1), m.group(2)
    return (len(onset) < 2 or onset in _ONSETS) and (len(coda) < 2 or coda in _CODAS)


def _looks_initialism(tok):
    """A lowercase/typed initialism the ASR or a user wrote as a word: no vowel at all
    (gpt, dnb, psxdps), or a short non-dictionary token with one vowel that is not a
    single syllable shape (ogdc, amcs). Dictionary words and names never get here
    spelled: those checks come first."""
    import names
    low = tok.lower()
    if not low.isalpha() or len(low) > 6:
        return False
    if not re.search(r"[aeiouy]", low):
        return len(low) >= 2
    if len(low) > 5 or low in names.english_words() or low in names.lexicon():
        return False
    return len(re.findall(r"[aeiouy]", low)) <= 1 and not _one_syllable(low)


def _is_spelled(tok, nxt):
    if not (tok.isalpha() and tok.isupper() and len(tok) <= 6):
        return False
    if tok in ACRONYM_WORDS:
        return False
    if tok == "I":
        return False                                    # the pronoun
    if tok == "A" and nxt[:1].islower():
        return False                                    # the article: "A company"
    if len(tok) >= 4 and tok.lower() in _dictionary():
        return False                                    # headings: SMART, ONLY, SEED
    return True


def _letters(tok, letters):
    tok = tok.upper()
    if letters == "ur":
        return ("ur", " ".join(LETTER_UR[c] for c in tok))
    return ("ph", " ".join(LETTER_PH[c] for c in tok))


def _span_pieces(span, letters):
    import names
    toks = [t for t in re.split(r"[ \-]+", span) if t]
    pieces, run = [], []

    def flush():
        if run:
            pieces.append(("en", " ".join(run)))
            run.clear()

    for i, tok in enumerate(toks):
        prev = toks[i - 1] if i else ""
        nxt = toks[i + 1] if i + 1 < len(toks) else ""
        key = tok.lower()
        roman = _roman_value(tok)
        plural = _PLURAL_ACRONYM.match(tok)
        if key in PHONE_OVERRIDE:
            flush()
            pieces.append(("ph", PHONE_OVERRIDE[key]))
        elif roman and prev[:1].isupper() and prev[1:].islower():
            run.append(__import__("num2words").num2words(roman))     # World War II
        elif _is_spelled(tok, nxt):
            flush()
            pieces.append(_letters(tok, letters))
        elif plural and plural.group(1) not in ACRONYM_WORDS:
            flush()
            pieces.append(_letters(plural.group(1), letters))        # CEOs, APIs
            pieces.append(("ph", "z"))
        elif _MIXED_ABBR.match(tok) or _looks_initialism(tok):
            flush()
            pieces.append(_letters(tok, letters))                    # PhD, MSc; no vowel: kWh, km, Hz
        elif tok.isupper() and len(tok) > 1:
            run.append(key)                                          # NADRA-like words, MANAGEMENT
        else:
            # contractions (don't, it's) are English; everything else is routed
            kind, val = ("en", tok) if re.search(r"['’]", tok) else names.route(tok)
            if kind == "en":
                run.append(tok)
            else:
                flush()
                pieces.append((kind, val))                           # Roman-Urdu names, lexicon names
    flush()
    return pieces


def prepare(text, letters="ph", epenthesis=False, urdu=None):
    """-> the exact string the model tokenises (same contract as tts.prepare).

    `urdu`: whether to read numbers in Urdu. Pass the verdict for the ORIGINAL sentence, as
    pk_gazetteer.prepare_pk does -- by the time text reaches here the gazetteer has rewritten every
    Pakistani name into Urdu script ("Mari Energies" -> "ماڑی انرجیز"), so judging it from this text
    would call an English sentence Urdu and read "Fauji Foundation (40%)" as "چالیس فیصد".
    None means judge from the text, which is right only when nothing has rewritten it yet.
    """
    import text_norm
    import urdu_numbers as un
    from urdu_numbers import expand_numbers
    from urdu_text_norm import normalize_for_tts
    if letters not in ("ur", "ph"):
        raise ValueError(letters)
    v = tts.vocab()
    text = unicodedata.normalize("NFC", str(text))
    text = drop_parenthetical_gloss(text)
    text = expand_abbreviations(text)
    text = roman_urdu_guard(text)
    # expand_codes FIRST, so the digits inside a letter+number code (FY2024-25, Sky47, 2P) are claimed
    # in English before the Urdu pass can reach them -- that is what keeps a code read as a code.
    text = expand_codes(text)
    if text_norm.is_urdu(text) if urdu is None else urdu:
        # Then Urdu, BEFORE english_numbers: NeMo verbalises any digit that sits near a Latin word, and
        # with Mari's replies almost always carrying one ("revenue", "MMSCFD", "PKR") it used to claim
        # every number in the sentence. After expand_numbers no digit is left for it to take.
        text = expand_numbers(text)
        text = english_numbers(text)
    else:
        text = english_numbers(text)                # a Latin-only sentence keeps the English reading
        text = expand_numbers(text)

    pieces = []
    # Numerals arrive wrapped in urdu_numbers.MARK as Urdu script, so every rule above saw ordinary text.
    # Here they become phones: the script does not write Urdu's short vowels, so the model ran the
    # consonants together (پینسٹھ heard as "penst"). Marked spans skip the Latin-span router below, whose
    # espeak path would otherwise spell the phones' ASCII letters out ("ns" -> "این ایس").
    for i, part in enumerate(str(text).split(un.MARK)):
        if i % 2:
            pieces.append(("ph", " ".join(un.WORD_PH.get(w, w) for w in part.split())))
            continue
        last = 0
        for m in _SPAN.finditer(part):
            pieces.append(("ur", part[last:m.start()]))
            pieces.extend(_span_pieces(m.group(0), letters))
            last = m.end()
        pieces.append(("ur", part[last:]))

    out = []
    for kind, seg in pieces:
        if kind == "en":
            out.append(phones.english_span(seg, epenthesis=epenthesis))
        elif kind == "ph":
            out.append(seg)
        else:
            out.append(normalize_for_tts(seg, v, latin="drop", expand_nums=False))
    return tts.clean(" ".join(p for p in out if p))


DEMO = [
    "جی ہاں، 30 جون 2025 تک کمپنی کے ذخائر اور وسائل مجموعی طور پر 952 ایم ایم بی او ای (MMBOE) ہیں، "
    "جن میں تقریباً 20 سال کا ریزرو ٹو پروڈکشن ریشو موجود ہے۔ اس عرصے کے دوران پرووڈ اور پروبیبل ریزروز (2P) "
    "میں تقریباً 39.7 ایم ایم بی او ای کی کمی، جبکہ کنٹینجنٹ ریسورسز (2C) میں تقریباً 108.7 ایم ایم بی او ای "
    "کا اضافہ ریکارڈ کیا گیا ہے۔",
    "Management نے کہا کہ ہمیں thoroughly investigate کرنا ہوگا کہ previous implementation میں exactly "
    "کیا غلط ہوا اور future میں ایسی صورتحال کو کیسے prevent کیا جا سکتا ہے۔",
    "The preliminary analysis کے مطابق infrastructure optimization، operational efficiency، اور resource "
    "utilization میں considerable improvement کی گنجائش موجود ہے۔",
    "board members نے Huawei Ascend chips کے بارے میں پوچھا، CEO اور CFO بھی موجود تھے۔",
    "FY2024-25 میں OGDCL، SNGPL اور PSX کے ساتھ Sky47 کا COP29 سے پہلے MMSCFD کا ڈیٹا شیئر ہوا۔",
    "Faheem Haider نے Waziristan اور Daharki میں NADRA اور SECP کے ساتھ SMART meeting کی۔",
    "A company کا X-1 well اور Plan B دونوں I think اچھے ہیں۔",
    "Dr. Naveed اور Engr. Kamran نے کہا کہ revenue 45% بڑھ کر $5.2 billion ہو گئی، اور World War II کے بعد سے CEOs کی salary میں 12 kWh کا PhD والا فرق ہے۔",
    "Mashal Khan نے Quetta، Hyderabad اور Sialkot میں Zafar اور Ayesha سے 30 جون کو ملاقات کی۔",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("text", nargs="?")
    ap.add_argument("--letters", default="ph", choices=["ph", "ur"])
    ap.add_argument("--demo", action="store_true")
    a = ap.parse_args()
    items = DEMO if a.demo or not a.text else [a.text]
    for t in items:
        print("IN :", t)
        for arm in (["ph", "ur"] if a.demo else [a.letters]):
            print(f"{arm} :", prepare(t, arm))
        print()


if __name__ == "__main__":
    main()
