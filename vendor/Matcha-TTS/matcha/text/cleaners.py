import re as _re
""" from https://github.com/keithito/tacotron

Cleaners are transformations that run over the input text at both training and eval time.

Cleaners can be selected by passing a comma-delimited list of cleaner names as the "cleaners"
hyperparameter. Some cleaners are English-specific. You'll typically want to use:
  1. "english_cleaners" for English text
  2. "transliteration_cleaners" for non-English text that can be transliterated to ASCII using
     the Unidecode library (https://pypi.python.org/pypi/Unidecode)
  3. "basic_cleaners" if you do not want to transliterate (in this case, you should also update
     the symbols in symbols.py to match your data).
"""

import logging
import re

try:  # English-only; unused by urdu_cleaners_graphemes
    import phonemizer
    _HAS_PHONEMIZER = True
except ImportError:  # pragma: no cover
    phonemizer = None
    _HAS_PHONEMIZER = False
try:
    from unidecode import unidecode
except ImportError:  # pragma: no cover
    def unidecode(x):
        return x


# To avoid excessive logging we set the log level of the phonemizer package to Critical
critical_logger = logging.getLogger("phonemizer")
critical_logger.setLevel(logging.CRITICAL)

# Intializing the phonemizer globally significantly reduces the speed
# now the phonemizer is not initialising at every call
# Might be less flexible, but it is much-much faster
global_phonemizer = None if not _HAS_PHONEMIZER else phonemizer.backend.EspeakBackend(
    language="en-us",
    preserve_punctuation=True,
    with_stress=True,
    language_switch="remove-flags",
    logger=critical_logger,
)


# Regular expression matching whitespace:
_whitespace_re = re.compile(r"\s+")

# Remove brackets
_brackets_re = re.compile(r"[\[\]\(\)\{\}]")

# List of (regular expression, replacement) pairs for abbreviations:
_abbreviations = [
    (re.compile(f"\\b{x[0]}\\.", re.IGNORECASE), x[1])
    for x in [
        ("mrs", "misess"),
        ("mr", "mister"),
        ("dr", "doctor"),
        ("st", "saint"),
        ("co", "company"),
        ("jr", "junior"),
        ("maj", "major"),
        ("gen", "general"),
        ("drs", "doctors"),
        ("rev", "reverend"),
        ("lt", "lieutenant"),
        ("hon", "honorable"),
        ("sgt", "sergeant"),
        ("capt", "captain"),
        ("esq", "esquire"),
        ("ltd", "limited"),
        ("col", "colonel"),
        ("ft", "fort"),
    ]
]


def expand_abbreviations(text):
    for regex, replacement in _abbreviations:
        text = re.sub(regex, replacement, text)
    return text


def lowercase(text):
    return text.lower()


def remove_brackets(text):
    return re.sub(_brackets_re, "", text)


def collapse_whitespace(text):
    return re.sub(_whitespace_re, " ", text)


def convert_to_ascii(text):
    return unidecode(text)


def basic_cleaners(text):
    """Basic pipeline that lowercases and collapses whitespace without transliteration."""
    text = lowercase(text)
    text = collapse_whitespace(text)
    return text


def transliteration_cleaners(text):
    """Pipeline for non-English text that transliterates to ASCII."""
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = collapse_whitespace(text)
    return text


def english_cleaners2(text):
    """Pipeline for English text, including abbreviation expansion. + punctuation + stress"""
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = expand_abbreviations(text)
    phonemes = global_phonemizer.phonemize([text], strip=True, njobs=1)[0]
    # Added in some cases espeak is not removing brackets
    phonemes = remove_brackets(phonemes)
    phonemes = collapse_whitespace(phonemes)
    return phonemes


def ipa_simplifier(text):
    replacements = [
        ("ɐ", "ə"),
        ("ˈə", "ə"),
        ("ʤ", "dʒ"),
        ("ʧ", "tʃ"),
        ("ᵻ", "ɪ"),
    ]
    for replacement in replacements:
        text = text.replace(replacement[0], replacement[1])
    phonemes = collapse_whitespace(text)
    return phonemes


# I am removing this due to incompatibility with several version of python
# However, if you want to use it, you can uncomment it
# and install piper-phonemize with the following command:
# pip install piper-phonemize

# import piper_phonemize
# def english_cleaners_piper(text):
#     """Pipeline for English text, including abbreviation expansion. + punctuation + stress"""
#     text = convert_to_ascii(text)
#     text = lowercase(text)
#     text = expand_abbreviations(text)
#     phonemes = "".join(piper_phonemize.phonemize_espeak(text=text, voice="en-US")[0])
#     phonemes = collapse_whitespace(phonemes)
#     return phonemes


def urdu_cleaners_graphemes(text):
    """Grapheme-level Urdu cleaner (no phonemizer), matching matcha_tts.py inference.

    Characters outside the 350-symbol vocab become spaces rather than raising, which
    is what the ONNX inference path already does. Whitespace is collapsed so a
    stray newline cannot become several blank frames.
    """
    import unicodedata as _ud

    from matcha.text.symbols import symbols as _syms

    _ok = set(_syms)
    text = _ud.normalize("NFC", str(text))
    text = "".join(c if c in _ok else " " for c in text)
    return _re.sub(r"\s+", " ", text).strip()
