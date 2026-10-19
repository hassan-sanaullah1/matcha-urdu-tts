"""The kaani male+female voice, loaded once and kept warm: text -> 22050 Hz mono float32 audio, sentence by
sentence. One checkpoint carries two speaker rows (Round 29 kb_male_v2_asis, step 3999):

    speaker 0 = the MALE voice   (UpliftAI prime-time-anchor dataset)
    speaker 1 = the FEMALE voice (the Lullaby narrator, carried into this checkpoint)

Rendering is exactly kiosk-tts/src/6_speak/speak.py (and byte-identical to pk-r29 src/infer.py for the same
text + seed): split into sentences -> pk_gazetteer.prepare_pk -> tts.clean -> the chosen speaker row, 32 steps,
temperature 0.35, seed = seed*100 + sentence index -> Vocos -> RMS/peak normalise -> sentences joined with
0.3 s of silence.

Latency:
  - the checkpoint, Vocos, NeMo grammars, espeak and the CUDA kernels are loaded/warmed once, in __init__;
  - both speaker rows share one loaded model (spk_cond holds a row per speaker), so switching voice is free;
  - stream() yields each sentence as soon as it is rendered, so playback can start after the first one;
  - a sentence's reading and its audio are cached per (voice, text, speed, seed), so repeated kiosk lines cost ~0 ms.

    from engine import KaaniTTS
    tts = KaaniTTS()                                 # ~10 s once: load + warm up both rows
    for part in tts.stream("پہلا جملہ۔ دوسرا جملہ۔", voice="male"):
        play(part["audio"])                          # float32 numpy at tts.SR
"""
import os
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                                 # kiosk-tts
sys.path.insert(0, os.path.join(ROOT, "src"))
import paths  # noqa: E402  -- kiosk-tts/src/paths.py: every location + the import path
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import collections
import io
import logging
import re
import threading
import time

import numpy as np
import soundfile as sf
import torch

# The male+female checkpoint shipped with this server (see checkpoints/kaani_male_female/).
CKPT = os.path.join(ROOT, "checkpoints", "kaani_male_female", "step=003999.ckpt")

# Speaker rows in the checkpoint. "male"/"female" are the names the server and Mari speak in.
VOICES = {"male": 0, "female": 1}
DEFAULT_VOICE = os.environ.get("KIOSK_TTS_DEFAULT_VOICE", "female").strip().lower()

SR = 22050
GAP_S = 0.3                                                  # silence between sentences (as on the demo page)
SPLIT = re.compile(r"(?<=[۔.!?؟])\s+|\n+")                  # the kiosk's sentence cut
# ...except after an abbreviation or an initial: "Lt. Gen. Anwar", "Dr. Ayesha", "A.F. Ferguson" stay one sentence
ABBR_END = re.compile(r"(?:\b(?:Lt|Gen|Dr|Mr|Mrs|Ms|Col|Maj|Capt|Brig|Prof|Engr|Retd|No|Co|Inc|Ltd|Pvt|St|Jr|Sr|vs|Rs|"
                      r"approx|e\.g|i\.e)|(?<![A-Za-z])[A-Z])\.$")
MAX_CHARS = 3000
STEPS, TEMPERATURE = 32, 0.35
WARM_UP = [
    "warm up.",
    "السلام علیکم، میں آپ کی کیسے مدد کر سکتا ہوں؟",
    "MARI Petroleum نے 2023 میں PKR 65 بلین کا منافع کمایا۔",
    "Dr. Ayesha Khan joined the company in 1998 as a senior geologist.",
]


def resolve_voice(voice):
    """Map a voice name (male/female, m/f, 0/1, or None) to a speaker row. Unknown -> the server default."""
    if voice is None:
        voice = DEFAULT_VOICE
    v = str(voice).strip().lower()
    aliases = {"m": "male", "male": "male", "0": "male", "hamza": "male",
               "f": "female", "female": "female", "1": "female", "maryam": "female"}
    name = aliases.get(v, DEFAULT_VOICE if DEFAULT_VOICE in VOICES else "female")
    return name, VOICES[name]


def split(text):
    """Text -> sentences, the way the kiosk cuts them (no cut after "Dr.", "Lt. Gen.", initials)."""
    out = []
    for s in SPLIT.split(str(text)[:MAX_CHARS]):
        s = s.strip(" -*")
        if not s:
            continue
        if out and ABBR_END.search(out[-1]):
            out[-1] += " " + s
        else:
            out.append(s)
    return out


def wav_bytes(audio, sr=SR):
    """float32 audio -> a 16-bit PCM WAV file in memory."""
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def pcm16(audio):
    """float32 audio -> raw little-endian 16-bit PCM bytes (no header)."""
    buf = io.BytesIO()
    sf.write(buf, audio, SR, format="RAW", subtype="PCM_16", endian="LITTLE")
    return buf.getvalue()


class KaaniTTS:
    SR = SR
    VOICES = VOICES

    def __init__(self, ckpt=CKPT, device=None, vocoder_device="cpu", cache_size=256, warm=True):
        logging.disable(logging.WARNING)
        import matcha_torch
        import pk_gazetteer
        import spk_cond
        import tts
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.vocoder_device = vocoder_device
        self.ckpt = ckpt
        self._tts, self._prep = tts, pk_gazetteer.prepare_pk
        if not spk_cond.has_spk_cond(ckpt):
            raise RuntimeError(f"{ckpt} has no speaker rows; this server needs the male+female checkpoint")
        # Load the model once. load_for_inference attaches every speaker row; we pass spks explicitly per call,
        # so the default row does not matter. matcha_torch caches the model, so _load returns the same object.
        spk_cond.load_for_inference(ckpt, self.device, spk=0)
        c = matcha_torch._load(ckpt, self.device)
        self.model, self.vocos = c["model"], c["vocos"]
        n_rows = self.model.spk_cond.emb_t.weight.shape[0]
        if n_rows < 2:
            raise RuntimeError(f"checkpoint has {n_rows} speaker row(s); need at least 2 (male=0, female=1)")
        if self.vocoder_device != "cpu":                               # the wrapper hides backbone+head in a closure
            for cell in type(self.vocos).decode.__closure__:
                cell.cell_contents.to(self.vocoder_device)
        self.gap = np.zeros(int(GAP_S * SR), np.float32)
        self.lock = threading.Lock()                                   # one render at a time on the GPU
        self._reads = collections.OrderedDict()                        # sentence -> what the model reads
        self._audio = collections.OrderedDict()                        # (read, spk, k, seed, speed) -> audio
        self.cache_size = cache_size
        self.warm_up_s = 0.0
        if warm:
            t0 = time.time()
            import frontend
            import names
            for load in (frontend._roman_urdu, frontend._dictionary, frontend._nemo, names.english_words,
                         names.lexicon, names.classifier):         # lazy one-time tables (~0.6 s on first use)
                load()
            for spk in sorted(set(VOICES.values())):                  # warm every speaker row's CUDA path
                for k, s in enumerate(WARM_UP):
                    self._render(s, spk, k, 1, 1.0, use_cache=False)
            self.warm_up_s = round(time.time() - t0, 2)

    # ---- text -------------------------------------------------------------------------------------------
    def read(self, sentence):
        """One sentence -> the exact string the model reads (numbers, acronyms, English, names all expanded)."""
        with self.lock:
            return self._read(sentence)

    def _read(self, sentence):
        r = self._reads.get(sentence)
        if r is None:
            r = self._tts.clean(self._prep(sentence))
            self._put(self._reads, sentence, r)
        else:
            self._reads.move_to_end(sentence)
        return r

    def _put(self, cache, key, value):
        cache[key] = value
        if len(cache) > self.cache_size:
            cache.popitem(last=False)

    # ---- audio ------------------------------------------------------------------------------------------
    def _render(self, sentence, spk, k, seed, speed, use_cache=True):
        """-> (audio or None, read, ms_text, ms_model, cached). Caller holds no lock; this takes it."""
        with self.lock:
            t0 = time.perf_counter()
            read = self._read(sentence)
            t1 = time.perf_counter()
            if not read:
                return None, read, (t1 - t0) * 1000, 0.0, False
            key = (read, spk, k, seed, speed)
            if use_cache and key in self._audio:
                self._audio.move_to_end(key)
                return self._audio[key], read, (t1 - t0) * 1000, 0.0, True
            inter = self._tts.encode(read)
            torch.manual_seed(seed * 100 + k)
            with torch.inference_mode():
                o = self.model.synthesise(torch.tensor([inter], device=self.device),
                                          torch.tensor([len(inter)], device=self.device),
                                          n_timesteps=STEPS, temperature=TEMPERATURE, length_scale=1.0 / speed,
                                          spks=torch.full((1,), spk, dtype=torch.long, device=self.device))
                w = self.vocos.decode(o["mel"].to(self.vocoder_device)).cpu().numpy().squeeze().astype("float32")
            audio = self._tts.normalize(w)
            if use_cache:
                self._put(self._audio, key, audio)
            return audio, read, (t1 - t0) * 1000, (time.perf_counter() - t1) * 1000, False

    def stream(self, text, voice=None, speed=1.0, seed=1):
        """Yield one dict per sentence as soon as it is rendered:
        {index, text, read, audio (float32 or None), ms_text, ms_model, cached}."""
        _, spk = resolve_voice(voice)
        speed = min(1.5, max(0.6, float(speed)))
        for k, s in enumerate(split(text)):
            audio, read, ms_text, ms_model, cached = self._render(s, spk, k, int(seed), speed)
            yield dict(index=k, text=s, read=read, audio=audio, ms_text=round(ms_text, 1),
                       ms_model=round(ms_model, 1), cached=cached)

    def synth(self, text, voice=None, speed=1.0, seed=1):
        """All sentences joined with 0.3 s of silence -> (float32 audio, list of per-sentence dicts)."""
        parts = list(self.stream(text, voice, speed, seed))
        audio = [p["audio"] for p in parts if p["audio"] is not None]
        joined = [x for a in audio for x in (a, self.gap)][:-1]
        return (np.concatenate(joined) if joined else np.zeros(1, np.float32)), parts
