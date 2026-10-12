"""General routing for Latin words that are not ordinary English: names, places, brands.

A fixed name list cannot cover "any name" (the 26k mined Wikipedia pairs hit 17/62 of a
Pakistani name/place/brand test list). So every Latin token is routed by rules and data:

  1. common English word (lowercase entry in the system dictionary, or a WordNet lemma
     with at least one non-proper-noun sense)  -> English (espeak)
  2. exact entry in the mined English->Urdu transliteration lexicon
     (assets/translit_pairs.json, from respelling/data; 26k, mostly proper nouns)  -> Urdu script
  3. otherwise a character n-gram classifier decides "Roman Urdu name" vs "English-like":
       Roman Urdu  -> rule-based reading into the shared phone inventory (roman_phones)
       English     -> English (espeak), e.g. brands and English surnames
     Classes: the lexicon's Latin side minus English words (South Asian / Arabic /
     Persian proper nouns) vs English dictionary words.

Usage:
  python src/names.py --eval
  python src/names.py Faheem Waziristan Huawei Schlumberger blockchain
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import paths  # noqa: E402  -- kiosk-tts/src/paths.py: every location + the import path

import argparse
import collections
import functools
import json
import math
import random
import re

import phones

TRANSLIT = paths.TRANSLIT


@functools.lru_cache(maxsize=1)
def english_words():
    """Common English words only. Measured 2026-09-16: the code-switch frequency list
    (us_cs_en_freq.json) and en_lexicon.MODERN both contain Roman Urdu (zafar, sindh,
    yaar, bhai, careem), and WordNet lemmas include proper nouns (hyderabad), so those
    sources would send Pakistani names to espeak English. WordNet words are kept only
    if at least one sense is not a named instance."""
    words = set()
    try:
        words |= {w.strip() for w in open(paths.DICT_WORDS, encoding="utf-8")
                  if w.strip().isalpha() and w.strip().islower()}
    except OSError:
        pass
    try:
        import nltk
        if paths.NLTK_DATA not in nltk.data.path:
            nltk.data.path.insert(0, paths.NLTK_DATA)          # assets/nltk_data (WordNet)
        from nltk.corpus import wordnet as wn
        for w in wn.all_lemma_names():
            if w.isalpha() and w.islower() and w not in words:
                if any(not s.instance_hypernyms() for s in wn.synsets(w)):
                    words.add(w)
    except Exception:
        pass
    return words


@functools.lru_cache(maxsize=1)
def lexicon():
    table = collections.defaultdict(collections.Counter)
    for en, ur in json.load(open(TRANSLIT, encoding="utf-8")):
        if en.isalpha():
            table[en.lower()][ur] += 1
    return {k: c.most_common(1)[0][0] for k, c in table.items()}


# ---------------------------------------------------------------------------
# character n-gram naive Bayes: Roman-Urdu name vs English word
# ---------------------------------------------------------------------------
N = 3


def _grams(w):
    s = f"^^{w.lower()}$"
    return [s[i:i + N] for i in range(len(s) - N + 1)]


class NameClassifier:
    def __init__(self, ur_words, en_words, alpha=0.5):
        self.counts = {}
        self.totals = {}
        vocab = set()
        for label, ws in (("ur", ur_words), ("en", en_words)):
            c = collections.Counter(g for w in ws for g in _grams(w))
            self.counts[label], self.totals[label] = c, sum(c.values())
            vocab |= set(c)
        self.V, self.alpha = len(vocab), alpha

    def logodds(self, w):
        """> 0 means Roman Urdu is more likely. Equal priors on purpose: the question
        is which spelling system the word looks like, not how common names are."""
        s = 0.0
        for g in _grams(w):
            pu = (self.counts["ur"][g] + self.alpha) / (self.totals["ur"] + self.alpha * self.V)
            pe = (self.counts["en"][g] + self.alpha) / (self.totals["en"] + self.alpha * self.V)
            s += math.log(pu / pe)
        return s / max(len(_grams(w)), 1)


def _training_sets(seed=0, holdout=0.1):
    en = english_words()
    ur = sorted(w for w in lexicon() if w not in en and len(w) >= 3)
    en_list = sorted(w for w in en if len(w) >= 3)
    rnd = random.Random(seed)
    rnd.shuffle(ur)
    rnd.shuffle(en_list)
    ku, ke = int(len(ur) * holdout), int(len(en_list) * holdout)
    return (ur[ku:], en_list[ke:]), (ur[:ku], en_list[:ke])


@functools.lru_cache(maxsize=1)
def classifier():
    (ur, en), _ = _training_sets()
    return NameClassifier(ur, en)


# ---------------------------------------------------------------------------
# Roman Urdu -> phones (the conventions Pakistanis use when typing names)
# ---------------------------------------------------------------------------
_DIGRAPHS = [
    ("sch", ["ʃ"]),
    ("aa", ["aː"]), ("ee", ["iː"]), ("ii", ["iː"]), ("oo", ["uː"]), ("ou", ["oː"]),
    ("ai", ["ɛ"]), ("ay", ["eː"]), ("ey", ["eː"]), ("ei", ["eː"]), ("au", ["ɔ"]), ("aw", ["ɔ"]),
    ("kh", ["x"]), ("gh", ["ɣ"]), ("sh", ["ʃ"]), ("ch", ["ʧ"]), ("zh", ["ʒ"]),
    ("th", ["tʰ"]), ("dh", ["dʰ"]), ("bh", ["bʰ"]), ("ph", ["pʰ"]), ("jh", ["ʤʰ"]),
    ("ck", ["k"]), ("qu", ["k", "ʋ"]),
]
# a vowel digraph whose second letter is a glide is NOT a digraph before another vowel:
# Hayat = ha-yat, Sujawal = su-ja-wal, Naveed stays nə-veed
_GLIDE_DIGRAPHS = {"ay", "ey", "aw", "ai", "au"}
_SINGLE = {"a": "ə", "b": "b", "c": "k", "d": "d", "e": "eː", "f": "f", "g": "ɡ", "h": "h",
           "i": "ɪ", "j": "ʤ", "k": "k", "l": "l", "m": "m", "n": "n", "o": "oː", "p": "p",
           "q": "q", "r": "r", "s": "s", "t": "t", "u": "ʊ", "v": "ʋ", "w": "ʋ", "x": "ks",
           "y": "j", "z": "z"}
_FINAL = {"a": "aː", "i": "iː", "u": "uː", "y": "iː", "e": "eː"}
_LONG = {"aː", "iː", "uː", "eː", "oː"}


def roman_phones(word):
    w = word.lower()
    out, i = [], 0
    while i < len(w):
        for dg, ph in _DIGRAPHS:
            if w.startswith(dg, i) and not (dg in _GLIDE_DIGRAPHS and w[i + 2:i + 3] in "aeiou"
                                            and i + 2 < len(w)):
                out.extend(ph)
                i += len(dg)
                break
        else:
            ch = w[i]
            if i == len(w) - 1 and ch in _FINAL:
                out.append(_FINAL[ch])
            elif ch == "h" and i == len(w) - 1 and out and out[-1] in phones.VOWELS:
                pass                                        # Shah, Allah: final h after a vowel is weak
            elif ch == "y" and i + 1 < len(w) and w[i + 1] not in "aeiou" and i > 0:
                out.append("ɛ")                             # Hyder, Syed: y between consonants
            elif ch in _SINGLE:
                out.extend(_SINGLE[ch].split() if " " in _SINGLE[ch] else
                           (["k", "s"] if ch == "x" else [_SINGLE[ch]]))
            i += 1
    vi = next((k for k, p in enumerate(out) if p in _LONG), None)
    if vi is None:
        vi = next((k for k, p in enumerate(out) if p in phones.VOWELS), None)
    if vi is not None:
        out.insert(vi, phones.STRESS)
    return phones.render(out)


_SUFFIXES = ("ies", "es", "s", "ers", "er", "ing", "ed", "ly")


def _english_inflection(w, en):
    """gainers, bros, listed: an English stem plus a regular suffix is English."""
    for suf in _SUFFIXES:
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            stem = w[:-len(suf)]
            if stem in en or (suf == "ies" and stem + "y" in en) or (stem + "e") in en:
                return True
    return False


def _english_compound(w, en):
    """jazzcash, paypal, facebook: two common English words glued together read as English."""
    return any(w[:i] in en and w[i:] in en for i in range(3, len(w) - 2))


# Measured on held-out lexicon words: thr 0.0 -> name recall 0.830 / English 0.845;
# -0.2 -> 0.902 / 0.757. Dictionary English never reaches the classifier (step 1), so
# the English side's errors fall only on non-dictionary brands; names get the benefit.
THRESHOLD = -0.2


def route(token, threshold=THRESHOLD):
    """-> ("en", token) | ("ur", urdu_script) | ("ph", phone_string)."""
    low = token.lower()
    en = english_words()
    if low in en or _english_compound(low, en) or _english_inflection(low, en):
        return "en", token
    lex = lexicon()
    if low in lex:
        return "ur", lex[low]
    if classifier().logodds(low) > threshold:
        return "ph", roman_phones(low)
    return "en", token


def evaluate():
    _, (ur_te, en_te) = _training_sets()
    c = classifier()
    acc_u = sum(c.logodds(w) > THRESHOLD for w in ur_te) / len(ur_te)
    acc_e = sum(c.logodds(w) <= THRESHOLD for w in en_te) / len(en_te)
    print(f"held-out: Roman-Urdu/proper-noun recall {acc_u:.3f} (n={len(ur_te)}), "
          f"English recall {acc_e:.3f} (n={len(en_te)})")
    pk = ("faheem waziristan sujawal daharki mashal muzamil quetta hyderabad ghazij zafar abbas niaz "
          "syed hayat ayesha usman naveed kamran asif sialkot gujranwala mardan swat hunza gilgit "
          "skardu thar sindh balochistan fauji careem daraz jazzcash easypaisa meezan habib alfalah "
          "hascol engro attock").split()
    brands = "huawei schlumberger nvidia exxonmobil siemens toyota samsung microsoft unilever".split()
    new_en = "fintech crypto blockchain upskilling cybersecurity dashboard".split()
    for name, ws in (("Pakistani names/places/brands", pk), ("international brands", brands),
                     ("modern English", new_en)):
        rs = [(w, *route(w)) for w in ws]
        print(f"\n{name}:")
        for w, k, s in rs:
            print(f"  {w:14s} {k}  {s}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("words", nargs="*")
    ap.add_argument("--eval", action="store_true")
    a = ap.parse_args()
    if a.eval or not a.words:
        evaluate()
    for w in a.words:
        print(w, route(w), f"logodds={classifier().logodds(w):+.2f}")


if __name__ == "__main__":
    main()
