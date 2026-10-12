"""Text -> audio for the models compared in the pilot.

Front ends (what the model reads):
  phones   Urdu stays in script; every Latin span becomes Pakistani-English phones
           (phones.english_span). For urdu-phones checkpoints.
  letters  English kept as Latin letters (finetune Stage 2 checkpoint).
  drop     English deleted (base checkpoint, Urdu-only text).
  script   text already fully in Urdu script (e.g. respelling's output) on the base model.

In kiosk-tts the speaking path is: pk_gazetteer.prepare_pk(text) -> clean/encode (here) -> synth (here), after
spk_cond.load_for_inference has loaded the checkpoint. vocab() reads models/base/vocab.json.
(The `phones` front end is the one the kiosk uses; the others are kept for reference only.)
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import paths  # noqa: E402  -- kiosk-tts/src/paths.py: every location + the import path

import argparse
import json
import re
import unicodedata

import numpy as np
import soundfile as sf
import torch

import phones

OUT_SR = 22050
_SPAN = re.compile(r"[A-Za-z]+(?:[ '’\-]+[A-Za-z]+)*")
_LETTER_DIGIT = re.compile(r"(?<=[A-Za-z])(?=[0-9])|(?<=[0-9])(?=[A-Za-z])")
_VOCAB = None


def vocab():
    global _VOCAB
    if _VOCAB is None:
        _VOCAB = json.load(open(paths.BASE_VOCAB, encoding="utf-8"))
    return _VOCAB


def prepare(text, mode, epenthesis=False):
    """-> the exact string the model tokenises."""
    from urdu_text_norm import normalize_for_tts
    v = vocab()
    if mode == "letters":
        return normalize_for_tts(text, v, latin="keep")
    if mode in ("drop", "script"):
        return normalize_for_tts(text, v, latin="drop")

    text = unicodedata.normalize("NFC", _LETTER_DIGIT.sub(" ", str(text)))
    from urdu_numbers import expand_numbers
    text = expand_numbers(text)
    out = []
    last = 0
    for m in _SPAN.finditer(text):
        out.append(("ur", text[last:m.start()]))
        out.append(("en", m.group(0)))
        last = m.end()
    out.append(("ur", text[last:]))

    pieces = []
    for kind, seg in out:
        if kind == "en":
            pieces.append(phones.english_span(seg, epenthesis=epenthesis))
            continue
        pieces.append(normalize_for_tts(seg, v, latin="drop", expand_nums=False))
    return clean(" ".join(p for p in pieces if p))


def clean(s):
    """What the training cleaner (urdu_cleaners_graphemes) does: NFC, out-of-vocab -> space."""
    v = vocab()
    s = "".join(c if c in v else " " for c in unicodedata.normalize("NFC", s))
    return re.sub(r"\s+", " ", s).strip()


def encode(s):
    v = vocab()
    space = v.get(" ", 0)
    ids = [v.get(ch, space) for ch in s]
    inter = [0] * (len(ids) * 2 + 1)
    for i, x in enumerate(ids):
        inter[i * 2 + 1] = x
    return inter


def normalize(w, target_rms=0.06, peak=0.97):
    r = float(np.sqrt((w ** 2).mean())) or 1.0
    w = w * (target_rms / r)
    m = float(np.abs(w).max())
    return w * (peak / m) if m > peak else w


def synth(text, ckpt=None, mode="phones", seed=1, steps=10, temp=0.667, device="cuda",
          epenthesis=False, prepared=None):
    import matcha_torch
    matcha_torch._prepare_imports()
    c = matcha_torch._load(ckpt, device)
    s = prepared if prepared is not None else prepare(text, mode, epenthesis)
    inter = encode(s)
    x = torch.tensor([inter], dtype=torch.long, device=device)
    xl = torch.tensor([len(inter)], dtype=torch.long, device=device)
    torch.manual_seed(seed)
    with torch.inference_mode():
        out = c["model"].synthesise(x, xl, n_timesteps=steps, temperature=temp, length_scale=1.0)
        audio = c["vocos"].decode(out["mel"].cpu())
    return normalize(audio.numpy().squeeze().astype("float32")).astype("float32"), s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("text")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--mode", default="phones", choices=["phones", "letters", "drop", "script"])
    ap.add_argument("--epenthesis", action="store_true")
    ap.add_argument("--out", default=os.path.join(paths.OUTPUT, "tts.wav"))
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    w, s = synth(a.text, a.ckpt, a.mode, a.seed, device=a.device, epenthesis=a.epenthesis)
    print("model input:", s)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    sf.write(a.out, w, OUT_SR)
    print(f"{len(w) / OUT_SR:.2f}s -> {a.out}")


if __name__ == "__main__":
    main()
