"""Mixed Urdu + English text -> the exact string the model tokenises.

ONE function builds the training manifests AND runs at inference, so the two cannot
drift (the failure reference_urdu_tokenizer_parity warns about).

Modes:
  letters  -- finetune's normalize_for_tts(latin="keep"): English stays as letters.
              This is Stage 2/3's representation, kept as the pilot's control arm.
  ipa-gb   -- every Latin span becomes espeak en-gb IPA; Urdu untouched.
  ipa-us   -- same with en-us.
  drop     -- base-model behaviour (Latin removed).

Order matters for ipa modes:
  1. numbers are verbalised first (in Urdu, same as the other modes);
  2. Latin spans are phonemised from the RAW text, so espeak still sees "I'm" and
     "don't" (normalize_for_tts deletes the apostrophe, and "Im" / "dont" read wrong);
  3. normalize_for_tts(latin="keep") runs last on the result for the usual symbol
     hygiene. IPA symbols are all in the vocab and contain no characters that
     hygiene rewrites, which __main__ asserts.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import paths  # noqa: E402  -- kiosk-tts/src/paths.py: every location + the import path

import re
import unicodedata

import en_g2p
from urdu_text_norm import normalize_for_tts

MODES = ("letters", "ipa-gb", "ipa-us", "drop")

# A Latin span: English words joined by spaces / apostrophes / hyphens. Punctuation
# (ASCII or Urdu) ends a span, so clause prosody marks stay outside the IPA.
_SPAN = re.compile(r"[A-Za-z]+(?:[ '’\-]+[A-Za-z]+)*")

_VOCAB = None


def vocab():
    global _VOCAB
    if _VOCAB is None:
        import json
        _VOCAB = json.load(open(paths.BASE_VOCAB, encoding="utf-8"))
    return _VOCAB


# A digit run touching a Latin letter ("Sky47", "5G") is verbalised in Urdu by
# expand_numbers and would otherwise weld onto the word (skˈaɪسینتالیس). Split first.
_LETTER_DIGIT = re.compile(r"(?<=[A-Za-z])(?=[0-9])|(?<=[0-9])(?=[A-Za-z])")

# Roman-Urdu guard (pk-voice). Urdu names, greetings and places typed in Latin letters
# are NOT English, and espeak reads them as English ("Maryam" -> mˈɛɹɪæm, "Mari" ->
# mˈæɹi). They are rewritten into Urdu script before any English handling, so the model
# reads them with the Urdu it learned from the speaker. Whole-word, case-insensitive.
ROMAN_URDU = {
    "assalamualaikum": "السلام علیکم", "assalamu alaikum": "السلام علیکم",
    "assalam-o-alaikum": "السلام علیکم", "asalamualaikum": "السلام علیکم",
    "salam": "سلام", "walaikumassalam": "وعلیکم السلام", "khuda hafiz": "خدا حافظ",
    "allah hafiz": "اللہ حافظ", "inshallah": "ان شاء اللہ", "insha allah": "ان شاء اللہ",
    "mashallah": "ماشاءاللہ", "alhamdulillah": "الحمدللہ", "subhanallah": "سبحان اللہ",
    "jazakallah": "جزاک اللہ", "shukriya": "شکریہ", "allah": "اللہ",
    "maryam": "مریم", "mari": "ماری", "muhammad": "محمد", "mohammad": "محمد",
    "ahmed": "احمد", "ahmad": "احمد", "ali": "علی", "fatima": "فاطمہ", "ayesha": "عائشہ",
    "hassan": "حسن", "hasan": "حسن", "hussain": "حسین", "imran": "عمران", "usman": "عثمان",
    "bilal": "بلال", "zainab": "زینب", "hamza": "حمزہ", "abdullah": "عبداللہ",
    "sana": "ثنا", "sara": "سارہ", "muzamil": "مزمل",
    "pakistan": "پاکستان", "karachi": "کراچی", "lahore": "لاہور", "islamabad": "اسلام آباد",
    "rawalpindi": "راولپنڈی", "peshawar": "پشاور", "quetta": "کوئٹہ", "multan": "ملتان",
    "faisalabad": "فیصل آباد", "sindh": "سندھ", "punjab": "پنجاب", "balochistan": "بلوچستان",
    "sui": "سوئی", "daharki": "ڈہرکی",
}
_ROMAN_URDU = re.compile(r"(?<![A-Za-z])(" + "|".join(
    re.escape(k) for k in sorted(ROMAN_URDU, key=len, reverse=True)) + r")(?![A-Za-z])", re.I)


def roman_urdu_guard(text):
    return _ROMAN_URDU.sub(lambda m: ROMAN_URDU[m.group(0).lower()], text)


def prepare(text, mode, stats=None):
    """Return the model-ready string. `stats` (dict) collects dropped IPA symbols."""
    v = vocab()
    text = roman_urdu_guard(_LETTER_DIGIT.sub(" ", str(text)))
    if mode == "letters":
        return normalize_for_tts(text, v, latin="keep")
    if mode == "drop":
        return normalize_for_tts(text, v, latin="drop")
    voice = {"ipa-gb": "en-gb", "ipa-us": "en-us"}[mode]

    text = unicodedata.normalize("NFC", str(text))
    from urdu_numbers import expand_numbers
    text = expand_numbers(text)

    def sub(m):
        p, dropped = en_g2p.to_vocab(en_g2p.ipa(m.group(0), voice), v)
        if stats is not None and dropped:
            for ch in dropped:
                stats[ch] = stats.get(ch, 0) + 1
        return p

    text = _SPAN.sub(sub, text)
    # expand_nums=False: already done above, and a digit cannot come out of espeak IPA
    return normalize_for_tts(text, v, latin="keep", expand_nums=False)


def encode(text, mode):
    """-> interspersed id list, identical to matcha_torch.encode's layout."""
    v = vocab()
    s = prepare(text, mode)
    space = v.get(" ", 0)
    ids = [v.get(ch, space) for ch in s]
    inter = [0] * (len(ids) * 2 + 1)
    for i, x in enumerate(ids):
        inter[i * 2 + 1] = x
    return inter


if __name__ == "__main__":
    samples = [
        "میں کل office میں meeting کے لیے جاؤں گا۔",
        "آپ WhatsApp پر مجھے PDF بھیج دیں، میں 5 منٹ میں check کر لوں گا۔",
        "I'm going to the office, then a meeting.",
        "یہ لفظ yacht ہے۔",
        "مثلاً عید کی باقاعدگی، سالگرہ کی خاص محافل، یا گھر کی خوبصورت سجاوٹیں۔",
    ]
    for s in samples:
        for m in MODES:
            print(f"{m:8s} {prepare(s, m)}")
        print()
