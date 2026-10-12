"""Small helpers shared by the list-building step (src/2_build_lists).

Each function is copied unchanged from the file named in its comment, so that those older, larger scripts
(which built other voices' datasets) do not have to come along.
"""
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import paths  # noqa: E402,F401  -- kiosk-tts/src/paths.py: every location + the import path

import numpy as np
import soundfile as sf
import torch

# ---- from urdu-phones/src/build_data.py ------------------------------------------------------------------
# One Urdu word with any punctuation around it: (before)(Urdu letters)(after).
_TOKEN = re.compile(r"^([^؀-ۿݐ-ݿ]*)([؀-ۿݐ-ݿ]+)([^؀-ۿݐ-ݿ]*)$")


def rows_from(text, table, p, rnd):
    """Rewrite each Urdu word as its phones with probability p ("row B" of a clip). The shipped recipe trains
    with --no-row-b, so these rows are written to the lists but only every other row (row A) is used."""
    out = []
    for tok in text.split():
        m = _TOKEN.match(tok)
        core = m.group(2) if m else None
        ph = table.get(core) if core else None
        # an Arabic-script punctuation mark (، ۔ ؟) is inside the Unicode block the
        # regex treats as letters; phones.urdu_words returns None for it, so it stays
        if ph and rnd.random() < p:
            out.append(m.group(1) + ph + m.group(3))
        else:
            out.append(tok)
    return " ".join(out)


# ---- from urdu-phones/src/build_mashal.py ----------------------------------------------------------------
def mel_mean(wavs, gain):
    """Mean log-mel of wavs at a gain. Copied from pk-voice/src/build_final.py."""
    import matcha_torch
    matcha_torch._prepare_imports()
    from matcha.utils.audio import mel_spectrogram
    vals = []
    for p in wavs:
        w, _ = sf.read(p, dtype="float32")
        y = torch.from_numpy(np.clip(w * gain, -1, 1))[None]
        m = mel_spectrogram(y, 1024, 80, 22050, 256, 1024, 0, 8000, center=False)
        vals.append(m.mean().item())
    return float(np.mean(vals))


# ---- from pk-bilingual/src/dataset/build_target.py -------------------------------------------------------
def soft_limit(y, knee=0.85, ceil=0.98):
    """Gain-matched lecture audio peaks above 1.0; hard clipping distorts, so compress above the knee."""
    a = np.abs(y)
    over = a > knee
    out = y.copy()
    out[over] = np.sign(y[over]) * (knee + (ceil - knee) * np.tanh((a[over] - knee) / (ceil - knee)))
    return out
