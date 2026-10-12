"""English text -> IPA phoneme string, for the Latin spans of Urdu+English text.

Why phonemes. The base checkpoint's Latin rows (ids 24-75) AND its IPA rows (ids
76-183, left over from upstream English Matcha) are both at random init. Stage 2/3
taught the Latin rows, which makes the model learn English spelling->sound from ~2 h
of audio; unseen-word hit rate topped out at 0.20. Feeding espeak IPA instead makes
the model learn ~45 sounds, and spelling is espeak's problem.

Implementation: libespeak-ng in-process via ctypes (espeak_TextToPhonemes), not a
subprocess per word -- this sits in a chatbot's TTS path, so latency matters. Results
are cached per span.

Two measured details:
  * espeak_SetVoiceByName wants the voice NAME ("English (Great Britain)"), not the
    language code; "en-gb" returns EE_NOT_FOUND and TextToPhonemes then segfaults.
  * en-us flaps /t/ (meeting -> mˈiːɾɪŋ); ref06's Pakistani English does not. Both
    voices are kept so the pilot can decide.

Input shaping (before espeak), all generic rules rather than a dictionary:
  * camelCase / brand casing is split at lower->Upper boundaries (YouTube -> You Tube,
    iPhone -> i Phone, WhatsApp -> Whats App), which is how these are spoken.
  * all-caps tokens of 2-5 letters are spelled letter by letter (API -> A P I),
    except a short list that is said as a word (NASA, COVID...).
  * OVERRIDE maps a handful of tokens espeak demonstrably misreads to an English
    respelling espeak reads correctly. It changes what espeak READS, never the IPA.
"""
import ctypes
import functools
import re
import threading

VOICES = {"en-gb": b"English (Great Britain)", "en-us": b"English (America)"}

# said as words, not spelled
ACRONYM_WORDS = {"NASA", "COVID", "UNESCO", "UNICEF", "NATO", "OPEC", "SIM", "PIN",
                 "LAN", "RAM", "ROM", "JPEG", "GIF", "SCUBA", "LASER", "RADAR", "WAPDA",
                 "NADRA", "FIFA", "ASAP"}

# token (case-sensitive after camel split is NOT applied) -> English respelling
OVERRIDE = {
    "ChatGPT": "chat G P T",
    "WiFi": "why fye", "Wi-Fi": "why fye", "wifi": "why fye",
    "email": "ee mail", "e-mail": "ee mail", "Email": "ee mail",
    "OK": "okay", "Ok": "okay", "ok": "okay",
    "vs": "versus",
    "LinkedIn": "linked in",
    "TikTok": "tick tock",
    "PayPal": "pay pal",
    "iOS": "eye O S",
    "JavaScript": "java script",
    "GitHub": "git hub",
    "Gmail": "gee mail",
    "Huawei": "wah way",          # espeak: hjˈuːɐweɪ
}

_LIB = None
_LOCK = threading.Lock()
_CUR_VOICE = [None]


def _lib():
    global _LIB
    if _LIB is None:
        lib = ctypes.cdll.LoadLibrary("libespeak-ng.so.1")
        lib.espeak_Initialize.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
        lib.espeak_SetVoiceByName.argtypes = [ctypes.c_char_p]
        lib.espeak_TextToPhonemes.restype = ctypes.c_char_p
        lib.espeak_TextToPhonemes.argtypes = [ctypes.POINTER(ctypes.c_char_p), ctypes.c_int, ctypes.c_int]
        if lib.espeak_Initialize(2, 0, None, 0) <= 0:       # AUDIO_OUTPUT_SYNCHRONOUS
            raise RuntimeError("espeak_Initialize failed")
        _LIB = lib
    return _LIB


def _set_voice(voice):
    if _CUR_VOICE[0] != voice:
        rc = _lib().espeak_SetVoiceByName(VOICES[voice])
        if rc != 0:
            raise RuntimeError(f"espeak voice {voice!r} not found (rc={rc})")
        _CUR_VOICE[0] = voice


_CAMEL = re.compile(r"(?<=[a-z])(?=[A-Z])")
_TOKEN = re.compile(r"[A-Za-z]+(?:['’\-][A-Za-z]+)*")


def shape(span):
    """Rewrite a Latin span into what espeak should read. Pure text, deterministic."""
    def one(m):
        t = m.group(0)
        if t in OVERRIDE:
            return OVERRIDE[t]
        t = t.replace("’", "'")
        if t.isupper() and 2 <= len(t) <= 5 and t not in ACRONYM_WORDS:
            # a lone "A" is read as the article (API -> ɐ pˈiː aɪ), and "ay" is read as
            # "aye" (AI -> aɪ aɪ, found on a real chatbot sentence). "eigh" -> eɪ in both
            # voices, measured.
            return " ".join("eigh" if c == "A" else c for c in t)
        if "-" in t:
            return " ".join(one(re.match(_TOKEN, p)) if re.match(_TOKEN, p) else p
                            for p in t.split("-"))
        if not t.isupper() and not t.islower() and not t.istitle():
            t = _CAMEL.sub(" ", t)
        return t
    return _TOKEN.sub(one, span)


@functools.lru_cache(maxsize=200_000)
def ipa(span, voice="en-gb"):
    """IPA for an English span (words separated by spaces). Word spaces preserved."""
    text = shape(span).strip()
    if not text:
        return ""
    with _LOCK:
        _set_voice(voice)
        buf = ctypes.create_string_buffer(text.encode("utf-8"))
        ptr = ctypes.c_char_p(ctypes.addressof(buf))
        out = []
        while ptr.value:
            r = _lib().espeak_TextToPhonemes(ctypes.pointer(ptr), 1, 0x02)   # UTF8, IPA
            if r:
                out.append(r.decode("utf-8"))
    s = " ".join(o.strip() for o in out if o.strip())
    # espeak marks a language switch as "(fr)" etc. inside the IPA -- drop the tags
    s = re.sub(r"\([a-z\-]+\)", "", s)
    return " ".join(s.split())


def to_vocab(s, vocab):
    """Map IPA onto the 350-symbol vocab. Unknown symbols are DELETED, not spaced:
    a space would insert a word boundary mid-word. Returns (string, dropped_chars)."""
    out, dropped = [], []
    for ch in s:
        if ch in vocab:
            out.append(ch)
        else:
            dropped.append(ch)
    return "".join(out), dropped


if __name__ == "__main__":
    import subprocess
    import sys
    tests = sys.argv[1:] or ["office meeting laptop", "I'm going to the office then a meeting",
                             "yacht island choir debris", "WhatsApp YouTube iPhone ChatGPT",
                             "API PDF NASA", "Wi-Fi e-mail follow-up"]
    for v in VOICES:
        for t in tests:
            cli = subprocess.run(["espeak-ng", "-q", "--ipa", "-v", v, shape(t)],
                                 capture_output=True, text=True).stdout.strip()
            mine = ipa(t, v)
            flag = "" if " ".join(cli.split()) == mine else "   <-- differs from CLI: " + cli
            print(f"[{v}] {t!r} -> {mine}{flag}")
