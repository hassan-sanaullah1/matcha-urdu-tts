"""STEP 4 engine -- edit a trained checkpoint's weights only (no retraining, no altered audio).

This is the shared weight-editor. The de-muffle step calls it through demuffle.py for the --demuffle edit; it
also knows --swap / --blend / --pace (kept from an earlier project that used them to swap in a studio support
voice), which demuffle.py does not use. The result is a drop-in checkpoint that speak.py, test_page.py,
pron_eval.py and demo/server.py load as usual.

Edits (each optional):
  --demuffle s  THE ONE demuffle.py USES. A per-band bias on the decoder's last layer
                (decoder.estimator.final_proj), fitted in CLOSED LOOP: render the reference sentences, measure the
                band-shape gap (mean log-mel over the louder half of frames) to the real recordings, move the bias
                part of the way, repeat 5 times. s = 1.0 removes the whole gap. Open loop ("just add the gap")
                overshoots, because the network reacts to the bias at every solver step. Level-neutral, smoothed
                over 5 bands, capped at +-6 dB.
  --swap        swap speaker rows 0 <-> 1 (spk_cond.emb_t / emb_mu). Self-check: the new speaker 0 must equal the
                old speaker 1 bit for bit. (Not used by demuffle.py; there speaker 0 is already the target voice.)
  --blend p     speaker 0 becomes (1-p)*speaker0 + p*speaker1: a voice between the two. (Not used.)
  --pace f      durations x f (log f added to the duration predictor's output bias). (Not used.)

Reference clips: demuffle.py supplies them -- it overrides refs() and split() with the target voice's own clips
(some of its training clips fit the curve, its held-out recording tests it). Run standalone, refs() reads
RASA_REF; for de-muffling a voice always call demuffle.py instead.

Input:  a trained checkpoint (default checkpoints/voice/step=007999.ckpt) and the reference clips
Output: checkpoints/<name>/model.ckpt, edit.json (the edit, bias curve, fit error per round, test gap before/after),
        val_ms.txt (the test half, as speaker 0)

Usage: do not call directly to de-muffle a voice -- use:  python src/4_demuffle/demuffle.py
Copied from an earlier kaani fine-tune round's voice_edit.py; only file locations changed.
"""
import os
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import paths  # noqa: E402  -- kiosk-tts/src/paths.py: every location + the import path
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import argparse
import json
import math
import re

import numpy as np
import torch

RASA_REF = os.path.join(paths.EVAL, "rasa_voice_ref.txt")
CAP = 6 / (20 / math.log(10))                       # 6 dB in natural-log mel units
ITERS, STEP = 5, 0.5                                 # closed-loop de-muffle fit
LATIN = re.compile(r"[A-Za-zəɪʊɛɔæʌɑɒθðʃʒŋɡʈɖɽɳʋɦʔːˈˌ]")


def refs():
    """[(real wav, model-ready text)] for Rasa, one row per clip, sorted by wav path.

    The list was made once, from the Rasa clips in data/3_training_lists/rasa_female/train.txt that no training
    run used, keeping for each clip the row with the fewest phone symbols (row A)."""
    out = []
    for l in open(RASA_REF, encoding="utf-8"):
        wav, text = l.rstrip("\n").split("|", 1)
        out.append((os.path.join(paths.ROOT, wav), text))
    return out


def split(rs):
    return rs[0::2], rs[1::2]


def model_of(path, dev):
    import tts  # noqa: F401
    import matcha_torch
    import spk_cond
    spk_cond.load_for_inference(path, dev, spk=0)
    return matcha_torch._load(path, dev)


def mel_of_wav(path):
    import soundfile as sf
    import matcha_torch
    matcha_torch._prepare_imports()
    from matcha.utils.audio import mel_spectrogram
    w, sr = sf.read(path, dtype="float32")
    assert sr == 22050, path
    return mel_spectrogram(torch.from_numpy(w)[None], 1024, 80, 22050, 256, 1024, 0, 8000, center=False)[0].numpy()


def render_mel(c, text, seed, dev):
    import tts
    inter = tts.encode(tts.clean(text))
    torch.manual_seed(seed)
    with torch.inference_mode():
        o = c["model"].synthesise(torch.tensor([inter], device=dev), torch.tensor([len(inter)], device=dev),
                                  n_timesteps=32, temperature=0.35, length_scale=1.0)
    return o["mel"][0].cpu().numpy()


def loud_mean(m):
    e = m.mean(0)
    return m[:, e > np.median(e)].mean(1)


def spectra(c, rs, dev):
    real = np.mean([loud_mean(mel_of_wav(w)) for w, _ in rs], 0)
    model = np.mean([loud_mean(render_mel(c, t, 1234 + i, dev)) for i, (_, t) in enumerate(rs)], 0)
    return real, model


def curve_from(real, model):
    d = real - model
    d = d - d.mean()                                             # shape only: output is loudness-normalised anyway
    d = np.convolve(np.pad(d, 2, mode="edge"), np.ones(5) / 5, mode="valid")
    return np.clip(d, -CAP, CAP)


def db(x):
    return [round(float(v) * 20 / math.log(10), 2) for v in x]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="output folder name under checkpoints/")
    ap.add_argument("--base", default=paths.VOICE_8K, help="the trained two-speaker checkpoint to edit")
    ap.add_argument("--swap", action="store_true")
    ap.add_argument("--blend", type=float, default=0.0)
    ap.add_argument("--demuffle", type=float, default=0.0)
    ap.add_argument("--pace", type=float, default=1.0)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    base = os.path.abspath(a.base)
    out_dir = os.path.join(paths.CHECKPOINTS, a.name)
    os.makedirs(out_dir, exist_ok=True)
    ck = torch.load(base, map_location="cpu", weights_only=False)
    sd = ck["state_dict"]
    std = float(sd["mel_std"])
    edit = dict(base=os.path.relpath(base, paths.ROOT), swap=a.swap, blend=a.blend, demuffle=a.demuffle, pace=a.pace,
                ref="rasa")

    for k in ("spk_cond.emb_t.weight", "spk_cond.emb_mu.weight"):
        w = sd[k].clone()
        if a.swap:
            w[[0, 1]] = w[[1, 0]]
        if a.blend:
            w[0] = (1 - a.blend) * w[0] + a.blend * w[1]
        sd[k] = w
    if a.pace != 1.0:
        sd["encoder.proj_w.proj.bias"] = sd["encoder.proj_w.proj.bias"] + math.log(a.pace)
    path = os.path.join(out_dir, "model.ckpt")
    torch.save(ck, path)
    base_edit = os.path.join(out_dir, "pre_demuffle.ckpt")
    if a.demuffle:
        torch.save(ck, base_edit)                  # the same edits without the de-muffle: the "before" on the test half

    rs = refs()
    fit, test = split(rs)
    with open(os.path.join(out_dir, "val_ms.txt"), "w", encoding="utf-8") as f:      # her real reference, as tag 0
        f.write("".join(f"{os.path.relpath(w, paths.ROOT)}|0|{t}\n" for w, t in test))
    print(f"{a.name}: ref rasa, {len(fit)} clips fit / {len(test)} test", flush=True)

    if a.swap:                                     # self-check: new tag 0 == old tag 1, bit for bit
        c0 = model_of(base, dev)
        c0["model"]._default_spk = 1
        m_old = render_mel(c0, test[0][1], 7, dev)
        c0["model"]._default_spk = 0
        m_new = render_mel(model_of(path, dev), test[0][1], 7, dev)
        edit["swap_check_max_abs"] = float(np.abs(m_old - m_new).max())
        print("swap self-check max |diff|", edit["swap_check_max_abs"], flush=True)

    if a.demuffle:
        # Closed loop: render the fit half, measure the remaining band-shape gap, move the bias by part of the error,
        # repeat. Target: remove a fraction `demuffle` of the original gap. The test half is scored only at the end.
        b0 = sd["decoder.estimator.final_proj.bias"].clone()
        gap = lambda r, m: curve_from(r, m)                                   # noqa: E731  level-neutral, smoothed, capped
        r0, m0 = spectra(model_of_fresh(path, dev), fit, dev)
        g0 = gap(r0, m0)
        target = (1 - a.demuffle) * g0
        bias, hist = np.zeros(80), []
        for it in range(ITERS):
            ck = torch.load(path, map_location="cpu", weights_only=False)
            r, m = (r0, m0) if it == 0 else spectra(model_of_fresh(path, dev), fit, dev)
            err = gap(r, m) - target
            hist.append(round(float(np.abs(err).mean()) * 20 / math.log(10), 3))
            bias = np.clip(bias + STEP * err, -CAP, CAP)
            ck["state_dict"]["decoder.estimator.final_proj.bias"] = b0 + torch.tensor(bias / std, dtype=torch.float32)
            torch.save(ck, path)
        rt0, mt0 = spectra(model_of_fresh(base_edit, dev), test, dev)
        rt, mt = spectra(model_of_fresh(path, dev), test, dev)
        edit.update(bias_db=db(bias), fit_err_db_by_iter=hist,
                    gap_db_test_before=db(gap(rt0, mt0)), gap_db_test_after=db(gap(rt, mt)),
                    test_mean_abs_gap_db=[round(float(np.abs(gap(rt0, mt0)).mean()) * 20 / math.log(10), 3),
                                          round(float(np.abs(gap(rt, mt)).mean()) * 20 / math.log(10), 3)])
        os.remove(base_edit)                       # only needed for the "before" measurement above
        print("fit error by iteration (mean |dB|):", hist, flush=True)
        print("TEST half mean |band-shape gap| dB, before -> after:", edit["test_mean_abs_gap_db"], flush=True)
    ck = torch.load(path, map_location="cpu", weights_only=False)
    ck["voice_edit"] = edit
    torch.save(ck, path)
    json.dump(edit, open(os.path.join(out_dir, "edit.json"), "w"), indent=1)
    print("->", path, flush=True)


def model_of_fresh(path, dev):
    """Load a checkpoint that was rewritten on disk (matcha_torch caches by path)."""
    import tts  # noqa: F401
    import matcha_torch
    matcha_torch._cache["key"] = None               # one-slot cache keyed by (path, device): force a reload
    return model_of(path, dev)


if __name__ == "__main__":
    main()
