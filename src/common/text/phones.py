"""One phone inventory for Urdu and Pakistani English, written in the model's own vocab.

Idea (see README): English words are never learned from English audio. They are spelled
in the SAME phone symbols that Urdu words get during training, so every symbol the model
reads in an English word has been heard thousands of times in the base voice's Urdu.

  Urdu word    --espeak-ng -v ur-->  espeak IPA  --UR_MAP-->  canonical phones
  English span --en_g2p.shape + espeak-ng -v en-us-->  IPA  --EN_MAP (Pakistani English)-->  phones

Every canonical phone is a short string of characters that are in the 350-symbol vocab
and survive NFC (the training cleaner NFC-normalises; a combining tilde would fuse with
its vowel into a character the vocab does not have). Nasalisation is therefore the
spacing symbol NASAL after the vowel, not U+0303.

The Pakistani-English choices in EN_MAP follow Mahboob & Ahmar, "Pakistani English:
Phonology": retroflex t/d, no flap, rhotic r, /θ ð/ -> dental stops, /v w/ -> one
labiodental approximant, /eɪ oʊ/ -> monophthongs, /æ/ -> /ɛ/.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import paths  # noqa: E402  -- kiosk-tts/src/paths.py: every location + the import path

import re
import subprocess
import threading

STRESS = "ˈ"
NASAL = "ɴ"          # IPA uvular nasal; unused otherwise, so free to mean "nasalised vowel"

VOWELS = ["ə", "ɪ", "ʊ", "aː", "iː", "uː", "eː", "oː", "ɛ", "ɔ"]
CONSONANTS = ["p", "b", "t", "d", "ʈ", "ɖ", "k", "ɡ", "q", "ʔ", "f", "s", "z", "ʃ", "ʒ", "x",
              "ɣ", "h", "m", "n", "l", "r", "ɽ", "j", "ʋ", "ʧ", "ʤ",
              "pʰ", "bʰ", "tʰ", "dʰ", "ʈʰ", "ɖʰ", "kʰ", "ɡʰ", "ʧʰ", "ʤʰ", "ɽʰ", "lʰ", "mʰ", "nʰ"]
PHONES = VOWELS + CONSONANTS + [NASAL]

# ---------------------------------------------------------------------------
# espeak-ng -v ur token -> canonical phones. Measured on the anchor's 3,468 word types.
# ---------------------------------------------------------------------------
UR_MAP = {
    # espeak spells the short a three ways (a, ʌ, ə) with no consistent contrast in its
    # Urdu output (آخرکار x|ə|r, الخلاء x|ʌ|l, ابھرتا h|a|r): one short vowel, zabar
    "a": ["ə"], "ʌ": ["ə"], "ə": ["ə"],
    "aː": ["aː"], "i": ["iː"], "iː": ["iː"], "ɪ": ["ɪ"], "e": ["eː"], "eː": ["eː"],
    "ɛ": ["ɛ"], "ɛː": ["ɛ"], "ʊ": ["ʊ"], "u": ["uː"], "uː": ["uː"], "o": ["oː"], "oː": ["oː"],
    "ɔ": ["ɔ"], "ɔː": ["ɔ"], "aɪ": ["aː", "ɪ"], "aʊ": ["aː", "ʊ"],
    # espeak's Urdu uses palatal stops for ج چ; the English side uses the affricates
    "ɟ": ["ʤ"], "ɟʰ": ["ʤʰ"], "c": ["ʧ"], "cʰ": ["ʧʰ"],
    "r.": ["ɽ"],
    # ص ظ ذ come out as retroflex/dental fricatives; in Urdu they are plain s z z
    "ʂ": ["s"], "ʐ": ["z"], "ð": ["z"],
    "w": ["ʋ"], "g": ["ɡ"],
}

# ---------------------------------------------------------------------------
# espeak-ng -v en-us token -> Pakistani-English phones.
# ---------------------------------------------------------------------------
EN_MAP = {
    # stops: retroflex t/d; the American flap and glottal stop are a plain t
    "t": ["ʈ"], "d": ["ɖ"], "ɾ": ["ʈ"], "ʔ": ["ʈ"],
    # dental fricatives -> Urdu dental stops (تھ, د); v/w -> و
    "θ": ["tʰ"], "ð": ["d"], "v": ["ʋ"], "w": ["ʋ"],
    "tʃ": ["ʧ"], "dʒ": ["ʤ"], "ɹ": ["r"], "r": ["r"], "g": ["ɡ"],
    # vowels
    "iː": ["iː"], "i": ["iː"], "ɪ": ["ɪ"], "ᵻ": ["ɪ"],
    "eɪ": ["eː"], "e": ["eː"], "ɛ": ["ɛ"], "æ": ["ɛ"],
    "ɑː": ["aː"], "ɑ": ["aː"], "a": ["aː"],
    "ʌ": ["ə"], "ə": ["ə"], "ɐ": ["ə"],
    "ɔː": ["ɔ"], "ɒ": ["ɔ"], "ɔ": ["ɔ"],
    "oʊ": ["oː"], "o": ["oː"], "oː": ["oː"], "ɵ": ["oː"],
    "uː": ["uː"], "u": ["uː"], "ʊ": ["ʊ"],
    "aɪ": ["aː", "ɪ"], "aʊ": ["aː", "ʊ"], "ɔɪ": ["ɔ", "ɪ"],
    # rhotic: Pakistani English pronounces the r
    "ɚ": ["ə", "r"], "ɜː": ["ə", "r"], "ɜ": ["ə", "r"],
    "ɪə": ["iː", "ə"], "ɛə": ["eː", "r"], "ʊə": ["ʊ", "ə"],
    # en-us hiatus tokens. Missing until 2026-09-16, when they were silently DROPPED:
    # previous -> prˈiːʋs, compliance -> kəmplˈaːns (scan of 4,416 Mashal + KB word types)
    "iə": ["iː", "ə"], "aɪə": ["aː", "ɪ", "ə"],
    "oːɹ": ["oː", "r"], "ɔːɹ": ["oː", "r"], "ɑːɹ": ["aː", "r"], "ɛɹ": ["eː", "r"],
    "ɪɹ": ["iː", "r"], "ʊɹ": ["uː", "r"], "aɪɚ": ["aː", "ɪ", "ə", "r"], "aʊɚ": ["aː", "ʊ", "ə", "r"],
    "əl": ["ə", "l"], "n̩": ["ə", "n"], "l̩": ["ə", "l"], "m̩": ["ə", "m"],
}

_VOWEL_SET = set(VOWELS)
_ESPEAK_LOCK = threading.Lock()
_CACHE = {}


def espeak(lines, voice):
    """IPA with '|' between phonemes, one output line per input line, cached."""
    todo = [l for l in dict.fromkeys(lines) if (voice, l) not in _CACHE]
    if todo:
        with _ESPEAK_LOCK:
            p = subprocess.run(["espeak-ng", "-v", voice, "-q", "--ipa", "--sep=|"],
                               input="\n".join(todo), capture_output=True, text=True, check=True)
        outs = [o.strip() for o in p.stdout.split("\n")]
        if len(outs) < len(todo) or any(not o for o in outs[:len(todo)]):
            # misaligned (a line espeak read as silence): redo one at a time
            outs = []
            for l in todo:
                q = subprocess.run(["espeak-ng", "-v", voice, "-q", "--ipa", "--sep=|"],
                                   input=l, capture_output=True, text=True, check=True)
                outs.append(" ".join(q.stdout.split()))
        for l, o in zip(todo, outs):
            _CACHE[(voice, l)] = o
    return [_CACHE[(voice, l)] for l in lines]


def _map_word(ipa_word, table, unknown):
    """One espeak word ('m|ˈiː|ɾ|ɪ|ŋ') -> list of canonical phones, stress kept."""
    toks = [t for t in ipa_word.split("|") if t]
    out = []
    for i, t in enumerate(toks):
        stressed = STRESS in t
        t = t.replace("ˈ", "").replace("ˌ", "")
        nasal = "̃" in t
        t = t.replace("̃", "")
        if not t:
            continue
        if t in table:
            ph = list(table[t])
        elif t in PHONES:
            ph = [t]
        elif t == "ŋ":
            nxt = toks[i + 1].replace("ˈ", "").replace("ˌ", "") if i + 1 < len(toks) else ""
            ph = ["n"] if nxt[:1] in ("k", "ɡ", "g") else ["n", "ɡ"]
        elif len(t) == 2 and t[1] == "ː" and t[0] in PHONES:      # gemination: bː -> b b
            ph = [t[0], t[0]]
        else:
            unknown[t] = unknown.get(t, 0) + 1
            continue
        if stressed:
            vi = next((k for k, p in enumerate(ph) if p in _VOWEL_SET), 0)
            ph.insert(vi, STRESS)
        if nasal:
            ph.append(NASAL)
        out.extend(ph)
    # 'battery' b|æ|ɾ|ɚ|ɹ|i -> ... ə r r ... : one r
    dedup = []
    for p in out:
        if p == "r" and dedup and dedup[-1] == "r":
            continue
        dedup.append(p)
    return dedup


def render(phones):
    return "".join(phones)


_URDU_WORD = re.compile(r"^[؀-ۿݐ-ݿ]+$")


def urdu_words(words, unknown=None):
    """Urdu script words -> phone strings (None where espeak leaves Urdu, e.g. '(en)')."""
    unknown = {} if unknown is None else unknown
    outs = espeak(words, "ur")
    res = []
    for w, o in zip(words, outs):
        if not o or "(" in o or " " in o.strip() or not _URDU_WORD.match(w):
            res.append(None)
            continue
        ph = _map_word(o, UR_MAP, unknown)
        res.append(render(ph) if ph else None)
    return res


def english_span(span, epenthesis=False, unknown=None):
    """English (Latin) span -> Pakistani-English phone string, words space-separated."""
    import en_g2p                                    # common/text/en_g2p.py
    unknown = {} if unknown is None else unknown
    shaped = en_g2p.shape(span).strip()
    if not shaped:
        return ""
    o = espeak([shaped], "en-us")[0]
    o = re.sub(r"\([a-z\-]+\)", "", o)
    words = []
    for w in o.split():
        ph = _map_word(w, EN_MAP, unknown)
        if epenthesis:
            core = [p for p in ph if p != STRESS]
            # s + consonant onset: Pakistani speakers prepend a short vowel (school -> iskool)
            if len(core) > 1 and core[0] == "s" and core[1] not in _VOWEL_SET:
                ph = ["ɪ"] + ph
        if ph:
            words.append(render(ph))
    return " ".join(words)


def check_vocab():
    import json
    v = json.load(open(paths.BASE_VOCAB, encoding="utf-8"))
    chars = set("".join(PHONES + [STRESS]))
    missing = sorted(c for c in chars if c not in v)
    import unicodedata
    unstable = [p for p in PHONES if unicodedata.normalize("NFC", p) != p]
    return v, missing, unstable


if __name__ == "__main__":
    v, missing, unstable = check_vocab()
    print("symbols missing from vocab:", missing, "| NFC-unstable phones:", unstable)
    print("ids used:", sorted({v[c] for c in set("".join(PHONES + [STRESS]))}))
    unk = {}
    for w, p in zip(["آج", "آنکھوں", "پڑتی", "خاموشی", "اظہار", "ہنس"],
                    urdu_words(["آج", "آنکھوں", "پڑتی", "خاموشی", "اظہار", "ہنس"], unk)):
        print(w, p)
    for s in ["meeting", "office laptop", "WhatsApp", "school", "three thousand", "battery",
              "presentation", "hydrocarbon", "throughput", "GPU", "Kubernetes", "the weather"]:
        print(f"{s:14s} {english_span(s, unknown=unk)}   | epenth: {english_span(s, True)}")
    print("unknown tokens:", unk)
