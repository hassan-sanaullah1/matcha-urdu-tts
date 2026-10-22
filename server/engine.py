"""The kaani male+female voice, loaded once and kept warm: text -> 22050 Hz mono float32 audio, sentence by
sentence. Urdu only (English TTS is served separately, by mari-tts-server).

TWO entirely separate checkpoints, one per gender -- NOT one shared dual-speaker checkpoint. The previous setup
(one checkpoint, speaker 0 = male, speaker 1 = female, Round 29 kb_male_v2_asis step 3999) shared a backbone
between both speaker rows; the female row could audibly pick up male-ish qualities from that shared backbone
(the two voices are "close" by design, per the training note in checkpoints/kaani_male_female/args.json), which
is the bleed this split removes:

    MALE   = checkpoints/kaani_male_female/step=003999.ckpt, speaker row 0 (UpliftAI prime-time-anchor dataset).
             Unchanged from before -- this checkpoint still exists and still carries a female row (id 1), but
             this server now never asks it for that row, so the female voice can never come from here.
    FEMALE = checkpoints/voice_demuffled/model.ckpt, speaker row 0 (her own training recipe's speaker 1 is
             Rasa, a studio Urdu support voice -- not male, and this server never asks for that row either).
             There is no male data anywhere in this checkpoint's lineage, so there is nothing for the male
             voice to bleed from on this side. This is also the best Lullaby checkpoint on its own merits
             (see docs/training_report.md Round 16): r5_p2 recipe, her own duration predictor timing,
             de-muffled for clarity -- picked over the plain 8k and over a 6k variant.

Rendering is exactly kiosk-tts/src/6_speak/speak.py for the same text + seed: split into sentences ->
pk_gazetteer.prepare_pk -> tts.clean -> the voice's own model (+ its speaker row, for the male checkpoint
only), 32 steps, temperature 0.35, seed = seed*100 + sentence index -> that voice's own Vocos -> RMS/peak
normalise -> sentences joined with 0.3 s of silence.

Latency:
  - both checkpoints, their Vocos, NeMo grammars, espeak and the CUDA kernels are loaded/warmed once, in __init__;
  - switching voice now means switching which loaded model is used (still ~free: both stay resident);
  - stream() yields each sentence as soon as it is rendered, so playback can start after the first one;
  - a sentence's reading and its audio are cached per (voice, text, speed, seed), so repeated kiosk lines cost ~0 ms.

    from engine import KaaniTTS
    tts = KaaniTTS()                                 # ~15 s once: load + warm up both checkpoints
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

# One checkpoint per gender -- see the module docstring for why this is two files, not one shared checkpoint.
VOICE_CKPTS = {
    "male": os.path.join(ROOT, "checkpoints", "kaani_male_female", "step=003999.ckpt"),
    "female": os.path.join(ROOT, "checkpoints", "voice_demuffled", "model.ckpt"),
}
# Speaker row to request from that voice's own checkpoint (her row 1 is Rasa support; his row 1 is her Lullaby
# support -- this server never asks either checkpoint for its "other" row, so there is nothing to bleed from).
VOICE_SPK = {"male": 0, "female": 0}

VOICES = {"male": "male", "female": "female"}          # name the server and Mari speak in -> itself (kept as a
                                                         # dict so `sorted(engine.VOICES)` still works unchanged)
DEFAULT_VOICE = os.environ.get("KIOSK_TTS_DEFAULT_VOICE", "female").strip().lower()

SR = 22050
GAP_S = 0.3                                                  # silence between sentences (as on the demo page)
SPLIT = re.compile(r"(?<=[۔.!?؟])\s+|\n+")                  # the kiosk's sentence cut
# ...except after an abbreviation or an initial: "Lt. Gen. Anwar", "Dr. Ayesha", "A.F. Ferguson" stay one sentence
ABBR_END = re.compile(r"(?:\b(?:Lt|Gen|Dr|Mr|Mrs|Ms|Col|Maj|Capt|Brig|Prof|Engr|Retd|No|Co|Inc|Ltd|Pvt|St|Jr|Sr|vs|Rs|"
                      r"approx|e\.g|i\.e)|(?<![A-Za-z])[A-Z])\.$")
MAX_CHARS = 3000

# A sentence-final nasal is swallowed: "وعلیکم السلام" came out "walaikum asalaa", in both voices. The
# model only closes a word-final consonant when more speech follows it -- the same م is clean mid-sentence
# (measured off the model's own alignment: energy over the م's frames is ~0.00 of the utterance's speech
# level when it ends the input, 0.36-0.49 when a clause follows). Spelling cannot fix it: shadda, sukun, a
# doubled م, a split-off م, and trailing ! ۔ , were all still ~0.0-0.16. So render with a throwaway clause
# appended, then cut the audio back at the frame the alignment says that clause starts. The kiosk sends one
# sentence per request (its own split_sentences), so the greeting always arrives as a whole utterance and
# there is no following text to borrow.
PAD_TEXT = " آپ کیسے ہیں"
PAD_AFTER = re.compile(r"[من]\s*[!.۔?؟,،\"'\)]*$")      # last word ends in م or ن
PAD_FADE_MS = 12                                                   # no click where the cut lands
STEPS, TEMPERATURE = 32, 0.35
WARM_UP = [
    "warm up.",
    "السلام علیکم، میں آپ کی کیسے مدد کر سکتا ہوں؟",
    "MARI Petroleum نے 2023 میں PKR 65 بلین کا منافع کمایا۔",
    "Dr. Ayesha Khan joined the company in 1998 as a senior geologist.",
]


def resolve_voice(voice):
    """Map a voice name (male/female, m/f, 0/1, or None) to its canonical name. Unknown -> the server default."""
    if voice is None:
        voice = DEFAULT_VOICE
    v = str(voice).strip().lower()
    aliases = {"m": "male", "male": "male", "0": "male", "hamza": "male",
               "f": "female", "female": "female", "1": "female", "maryam": "female"}
    name = aliases.get(v, DEFAULT_VOICE if DEFAULT_VOICE in VOICES else "female")
    return name, name


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

    def __init__(self, ckpts=None, device=None, vocoder_device="cpu", cache_size=256, warm=True):
        logging.disable(logging.WARNING)
        import matcha_torch
        import pk_gazetteer
        import spk_cond
        import tts
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.vocoder_device = vocoder_device
        self.ckpts = dict(ckpts) if ckpts else dict(VOICE_CKPTS)
        self._tts, self._prep = tts, pk_gazetteer.prepare_pk
        self._models = {}                                               # name -> (model, vocos)
        for name, ckpt in self.ckpts.items():
            spk = VOICE_SPK[name]
            has_spk = spk_cond.has_spk_cond(ckpt)
            if spk is not None and not has_spk:
                raise RuntimeError(f"{name}: {ckpt} has no speaker rows; expected speaker {spk}")
            if spk is None and has_spk:
                raise RuntimeError(f"{name}: {ckpt} unexpectedly has speaker rows; expected a single-speaker "
                                    f"checkpoint so this voice cannot bleed from another speaker's row")
            if spk is not None:
                spk_cond.load_for_inference(ckpt, self.device, spk=spk)
            c = matcha_torch._load(ckpt, self.device)
            model, vocos = c["model"], c["vocos"]
            if spk is not None:
                n_rows = model.spk_cond.emb_t.weight.shape[0]
                if n_rows <= spk:
                    raise RuntimeError(f"{name}: checkpoint has {n_rows} speaker row(s); need row {spk}")
            if self.vocoder_device != "cpu":                            # the wrapper hides backbone+head in a closure
                for cell in type(vocos).decode.__closure__:
                    cell.cell_contents.to(self.vocoder_device)
            self._models[name] = (model, vocos)
        self.gap = np.zeros(int(GAP_S * SR), np.float32)
        self.lock = threading.Lock()                                   # one render at a time on the GPU
        self._reads = collections.OrderedDict()                        # sentence -> what the model reads
        self._audio = collections.OrderedDict()                        # (read, voice, k, seed, speed) -> audio
        self.cache_size = cache_size
        self.warm_up_s = 0.0
        if warm:
            t0 = time.time()
            import frontend
            import names
            for load in (frontend._roman_urdu, frontend._dictionary, frontend._nemo, names.english_words,
                         names.lexicon, names.classifier):         # lazy one-time tables (~0.6 s on first use)
                load()
            for name in VOICES:                                      # warm every voice's own checkpoint/CUDA path
                for k, s in enumerate(WARM_UP):
                    self._render(s, name, k, 1, 1.0, use_cache=False)
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
    @staticmethod
    def _drop_pad(w, attn, n_read, hop=256):
        """Cut PAD_TEXT's audio off the end, at the frame the model's own alignment puts it.

        encode() interleaves blanks (inter = blank, c0, blank, c1, ... ), so read character i is token
        2i+1. Keeping the blank that follows the last real character keeps that consonant's release.
        """
        a = attn[0, 0].cpu().numpy()
        keep = int(a[:(n_read - 1) * 2 + 3].sum()) * hop                # frames -> samples
        if not 0 < keep < len(w):
            return w
        w = w[:keep].copy()
        fade = min(int(PAD_FADE_MS / 1000 * SR), len(w))
        w[-fade:] *= np.linspace(1.0, 0.0, fade, dtype="float32")
        return w

    def _render(self, sentence, voice_name, k, seed, speed, use_cache=True):
        """-> (audio or None, read, ms_text, ms_model, cached). Caller holds no lock; this takes it."""
        with self.lock:
            t0 = time.perf_counter()
            read = self._read(sentence)
            t1 = time.perf_counter()
            if not read:
                return None, read, (t1 - t0) * 1000, 0.0, False
            key = (read, voice_name, k, seed, speed)
            if use_cache and key in self._audio:
                self._audio.move_to_end(key)
                return self._audio[key], read, (t1 - t0) * 1000, 0.0, True
            model, vocos = self._models[voice_name]
            spk = VOICE_SPK[voice_name]
            pad = PAD_TEXT if PAD_AFTER.search(read) else ""          # see PAD_TEXT: saves a final م / ن
            inter = self._tts.encode(read + pad)
            torch.manual_seed(seed * 100 + k)
            kwargs = dict(n_timesteps=STEPS, temperature=TEMPERATURE, length_scale=1.0 / speed)
            if spk is not None:                                       # only the multi-speaker (male) checkpoint
                kwargs["spks"] = torch.full((1,), spk, dtype=torch.long, device=self.device)
            with torch.inference_mode():
                o = model.synthesise(torch.tensor([inter], device=self.device),
                                     torch.tensor([len(inter)], device=self.device), **kwargs)
                w = vocos.decode(o["mel"].to(self.vocoder_device)).cpu().numpy().squeeze().astype("float32")
            if pad:
                w = self._drop_pad(w, o["attn"], len(read))
            audio = self._tts.normalize(w)
            if use_cache:
                self._put(self._audio, key, audio)
            return audio, read, (t1 - t0) * 1000, (time.perf_counter() - t1) * 1000, False

    def stream(self, text, voice=None, speed=1.0, seed=1):
        """Yield one dict per sentence as soon as it is rendered:
        {index, text, read, audio (float32 or None), ms_text, ms_model, cached}."""
        name, _ = resolve_voice(voice)
        speed = min(1.5, max(0.6, float(speed)))
        for k, s in enumerate(split(text)):
            audio, read, ms_text, ms_model, cached = self._render(s, name, k, int(seed), speed)
            yield dict(index=k, text=s, read=read, audio=audio, ms_text=round(ms_text, 1),
                       ms_model=round(ms_model, 1), cached=cached)

    def synth(self, text, voice=None, speed=1.0, seed=1):
        """All sentences joined with 0.3 s of silence -> (float32 audio, list of per-sentence dicts)."""
        parts = list(self.stream(text, voice, speed, seed))
        audio = [p["audio"] for p in parts if p["audio"] is not None]
        joined = [x for a in audio for x in (a, self.gap)][:-1]
        return (np.concatenate(joined) if joined else np.zeros(1, np.float32)), parts
