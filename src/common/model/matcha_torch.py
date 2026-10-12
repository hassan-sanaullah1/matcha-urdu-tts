"""Loads a Matcha-TTS checkpoint (the base model or a fine-tune) in PyTorch, plus the vocoder and vocabulary.

Used by training (to build the model) and by speaking/evaluation (spk_cond.load_for_inference calls it).
Everything is read from local files: the vocabulary from models/base/vocab.json, the vocoder from
models/vocoder/. Copied from finetune/src/inference/matcha_torch.py; only the file
locations changed (and the old --batch demo mode was removed).

Why PyTorch and not the ONNX file AhsanTalal ships: the ONNX export is baked at n_timesteps=10, the
coarsest setting; the PyTorch checkpoint lets the number of solver steps be chosen, and it is also the
checkpoint that is fine-tuned, so loading it the same way in both places keeps training and speaking honest.

Usage (base model, Urdu only):
    python src/common/model/matcha_torch.py "متن" out.wav [--steps 32] [--temp 0.667] [--rate 1.0]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import paths  # noqa: E402  -- kiosk-tts/src/paths.py: every location + the import path

import argparse
import json
import os
import sys
import types

import numpy as np
import soundfile as sf
import torch

MATCHA_SRC = paths.MATCHA_SRC
OUT_SR = 22050
_cache = {}


def _prepare_imports():
    """Make `matcha` importable without pip-installing it.

    matcha.utils.utils pulls in gdown (only for downloading the authors' English
    checkpoints) which drags bs4/requests behind it. We load our own checkpoint,
    so a stub is enough and keeps the pinned env untouched.
    """
    # diffusers 0.25.0 (the version Matcha's decoder/transformer API targets) imports
    # huggingface_hub.cached_download, which newer hub releases dropped. It sits on the
    # remote-code-download path we never touch, so aliasing it is enough -- the
    # alternative is a newer diffusers that has moved LoRACompatibleLinear/AdaLayerNorm
    # out from under matcha.models.components.transformer.
    import huggingface_hub
    if not hasattr(huggingface_hub, "cached_download"):
        huggingface_hub.cached_download = huggingface_hub.hf_hub_download

    if "gdown" not in sys.modules:
        stub = types.ModuleType("gdown")
        stub.download = lambda *a, **k: (_ for _ in ()).throw(
            RuntimeError("gdown is stubbed out; matcha_torch loads local checkpoints only")
        )
        sys.modules["gdown"] = stub
    if MATCHA_SRC not in sys.path:
        sys.path.insert(0, MATCHA_SRC)


def load_model(ckpt=None, device="cpu"):
    """Build MatchaTTS and load weights.

    Two things the stock load_from_checkpoint gets wrong for this checkpoint:
      - hyper_parameters says n_vocab=178, but encoder.emb.weight is [350, 192].
        350 is the truth (vocab.json has 350 entries); 178 is a stale config value.
      - mel_mean/mel_std are buffers that load_from_checkpoint does not restore,
        and without them denormalisation silently produces garbage mel.
    """
    _prepare_imports()
    from matcha.models.matcha_tts import MatchaTTS

    if ckpt is None:
        ckpt = paths.BASE_CKPT

    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    hp = dict(ck["hyper_parameters"])
    hp["n_vocab"] = ck["state_dict"]["encoder.emb.weight"].shape[0]
    hp.pop("optimizer", None)
    hp.pop("scheduler", None)

    model = MatchaTTS(**hp)
    missing, unexpected = model.load_state_dict(ck["state_dict"], strict=False)
    if missing or unexpected:
        print(f"[matcha_torch] missing={missing} unexpected={unexpected}", file=sys.stderr)

    ds = ck["hyper_parameters"]["data_statistics"]
    model.mel_mean = torch.tensor(float(ds["mel_mean"]))
    model.mel_std = torch.tensor(float(ds["mel_std"]))
    return model.to(device).eval()


def _load(ckpt=None, device="cpu"):
    key = (ckpt, device)
    if _cache.get("key") != key:
        from vocoder import _load_vocos_22k

        _cache["stoi"] = json.load(open(paths.BASE_VOCAB, encoding="utf-8"))
        _cache["model"] = load_model(ckpt, device)
        _cache["vocos"] = _load_vocos_22k()
        _cache["key"] = key
    return _cache


def encode(text, stoi, hygiene=True, latin="drop"):
    """Graphemes -> ids, with a blank (0) interspersed between every character.

    Normalisation is delegated to urdu_text_norm so this path and the training
    cleaner cannot diverge; they previously disagreed on whitespace, which changed
    the sequence LENGTH on real corpus text.

    hygiene=True additionally runs normalize_for_tts, which expands numbers and
    rewrites characters whose embedding rows pretraining never trained -- ASCII
    punctuation, zero-width joiners, and the U+0660 digit block, all of which are
    IN the vocab and so survive urdu_norm untouched while meaning nothing to the
    model. hygiene=False reproduces the pre-fix behaviour, for A/B measurement.

    latin="keep" passes English letters through. Only correct for a checkpoint
    that has been trained to read them; on the base checkpoint ids 24-75 are at
    random initialisation.
    """
    space = stoi.get(" ", 0)
    if hygiene:
        from urdu_text_norm import normalize_for_tts
        text = normalize_for_tts(text, stoi, latin=latin)
    else:
        from urdu_text_norm import urdu_norm
        text = urdu_norm(text, stoi)
    ids = [stoi.get(ch, space) for ch in text]
    inter = [0] * (len(ids) * 2 + 1)
    for i, v in enumerate(ids):
        inter[i * 2 + 1] = v
    return inter


def synth(text, steps=32, temp=0.667, rate=1.0, ckpt=None, device="cpu",
          hygiene=True, latin="drop", seed=None):
    """steps = ODE solver steps (ONNX was hard-locked to 10). temp = sampling noise.

    seed fixes the CFM prior (flow_matching.py:51 draws torch.randn_like(mu) *
    temperature), which makes synthesis reproducible and, more importantly, makes
    an A/B comparison PAIRED -- both sides get the same noise vector, so the only
    difference measured is the change under test.

    This is not a nicety. Measured over four identical runs of the 20 Urdu eval
    sentences, set-level CER ranged 0.043-0.091 purely from resampling, and the
    entire spread came from one sentence that degenerates about half the time.
    An unpaired comparison at that noise level cannot resolve anything smaller
    than the effect it is trying to detect.
    """
    c = _load(ckpt, device)
    inter = encode(text, c["stoi"], hygiene=hygiene, latin=latin)
    x = torch.tensor([inter], dtype=torch.long, device=device)
    x_len = torch.tensor([len(inter)], dtype=torch.long, device=device)
    if seed is not None:
        torch.manual_seed(seed)
    with torch.inference_mode():
        out = c["model"].synthesise(x, x_len, n_timesteps=steps,
                                    temperature=temp, length_scale=rate)
        audio = c["vocos"].decode(out["mel"].cpu())
    return audio.numpy().squeeze().astype("float32")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("text", nargs="?")
    ap.add_argument("out", nargs="?", default=os.path.join(paths.OUTPUT, "matcha_torch.wav"))
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--temp", type=float, default=0.667)
    ap.add_argument("--rate", type=float, default=1.0)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    w = synth(a.text, a.steps, a.temp, a.rate, a.ckpt, a.device)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    sf.write(a.out, w, OUT_SR)
    print(f"{len(w)/OUT_SR:.2f}s -> {a.out}")


if __name__ == "__main__":
    main()
