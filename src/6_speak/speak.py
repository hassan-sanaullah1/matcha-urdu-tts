"""STEP 6 -- speak: one text -> one wav file, in the target voice (speaker 0).

Text may be Urdu script, English, or mixed (LLM chatbot output). What happens:
  1. pk_gazetteer.prepare_pk(text)   the text front end (src/common/text/): numbers, dates, acronyms, Pakistani
                                     names -> Urdu script, English words -> Pakistani-English sound symbols
  2. tts.clean                       keep only symbols in the model's vocabulary
  3. Matcha (32 solver steps, temperature 0.35) -> mel spectrogram -> Vocos vocoder -> 22,050 Hz audio

Input:  the text; by default checkpoints/voice_demuffled/model.ckpt (the voice in use). Point --ckpt at any
        checkpoint to hear a run you trained, e.g. checkpoints/voice/step=007999.ckpt (before the de-muffle).
Output: the wav file given with -o; also prints the exact symbols the model read.

Usage (from the kiosk-tts folder):
  python src/6_speak/speak.py "آج کی meeting میں ہم Mari Energies کے results discuss کریں گے۔" -o out.wav
  python src/6_speak/speak.py --ckpt checkpoints/voice/step=007999.ckpt "..." -o out.wav   # the 8k, un-demuffled model
  python src/6_speak/speak.py --length-scale 1.05 "..." -o out.wav                            # >1 = slower
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import paths  # noqa: E402  -- kiosk-tts/src/paths.py: every location + the import path

import argparse

import soundfile as sf
import torch

import pk_gazetteer
import spk_cond
import tts

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("text")
    ap.add_argument("-o", "--out", default="out.wav")
    ap.add_argument("--ckpt", default=None, help="checkpoint to speak with (default: checkpoints/voice_demuffled)")
    ap.add_argument("--seed", type=int, default=100)
    # Measured on the Round 3 pilot: 10 steps at 0.667 gave UTMOS 3.17, 32 steps at 0.2-0.4 gave 3.86-4.04.
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--temp", type=float, default=0.35)
    ap.add_argument("--length-scale", type=float, default=1.0, help=">1 slower, <1 faster")
    a = ap.parse_args()
    ckpt = a.ckpt or paths.VOICE
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    # speaker 0 = the target voice (the kiosk voice); speaker 1 = the Rasa support voice used only in training.
    c = spk_cond.load_for_inference(ckpt, dev, spk=0) if spk_cond.has_spk_cond(ckpt) else None
    prepared = tts.clean(pk_gazetteer.prepare_pk(a.text))
    if a.length_scale == 1.0:
        w, _ = tts.synth(a.text, ckpt, seed=a.seed, steps=a.steps, temp=a.temp, device=dev, prepared=prepared)
    else:
        import matcha_torch
        c = matcha_torch._load(ckpt, dev) if c is None else c
        inter = tts.encode(prepared)
        x = torch.tensor([inter], dtype=torch.long, device=dev)
        xl = torch.tensor([len(inter)], dtype=torch.long, device=dev)
        torch.manual_seed(a.seed)
        with torch.inference_mode():
            mel = c["model"].synthesise(x, xl, n_timesteps=a.steps, temperature=a.temp,
                                        length_scale=a.length_scale)["mel"]
            w = tts.normalize(c["vocos"].decode(mel.cpu()).numpy().squeeze().astype("float32"))
    sf.write(a.out, w, tts.OUT_SR)
    print("model input:", prepared)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
