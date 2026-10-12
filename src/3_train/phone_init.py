"""Where each English phone symbol's embedding row starts before training (used by train.py).

The base model only ever read Urdu letters, so the rows for phone symbols (ʈ, ə, iː, ...) are untrained.
Each one starts from the trained row of the Urdu letter that makes the same sound, so training adapts a
meaningful start instead of learning the symbol from noise. train.py's SEED_V2 then overrides the
vowels so that contrasting vowels (cooled / cold) do not start identical.

Copied unchanged from urdu-phones/src/train.py (INIT_FROM).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import paths  # noqa: E402,F401  -- kiosk-tts/src/paths.py: every location + the import path

import phones

# phone symbol character -> Urdu grapheme whose trained row it starts from
INIT_FROM = {
    "p": "پ", "b": "ب", "t": "ت", "d": "د", "ʈ": "ٹ", "ɖ": "ڈ", "k": "ک", "ɡ": "گ", "q": "ق",
    "ʔ": "ء", "f": "ف", "s": "س", "z": "ز", "ʃ": "ش", "ʒ": "ژ", "x": "خ", "ɣ": "غ", "h": "ہ",
    "m": "م", "n": "ن", "l": "ل", "r": "ر", "ɽ": "ڑ", "j": "ی", "ʋ": "و", "ʧ": "چ", "ʤ": "ج",
    "ʰ": "ھ",
    "ə": "َ", "ɪ": "ِ", "ʊ": "ُ",          # zabar, zer, pesh
    "a": "ا", "i": "ی", "u": "و", "e": "ے", "o": "و", "ɛ": "ے", "ɔ": "و",
    "ː": "ا",
    phones.NASAL: "ں",
    phones.STRESS: "_",                                  # starts as the (trained) blank
}
