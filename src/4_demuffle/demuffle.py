"""STEP 4 -- de-muffle a trained model through its weights only (no retraining, no audio altered).

A fine-tune comes out duller than the speaker's real recordings (for the shipped voice: 2-4 kHz clarity 0.93 vs 2.07).
This runs the closed-loop de-muffle in voice_edit.py (the shared engine next to it), matched to the voice's OWN
recordings, which it takes from that voice's training lists:
  fit   --fit-clips of its train.txt clips (row A text; random.Random(0))
  test  the held-out recording (val + test lists, never trained on), scored once at the end
A per-band bias on the decoder's last layer is fitted in 5 render-measure-adjust rounds so the model's band shape
moves `--amount` of the way to the speaker's. There is no speaker swap: speaker 0 is already the target voice.

Works for any voice: point --base at its checkpoint and --lists at its training lists.

Input:  a trained checkpoint (default checkpoints/voice/step=007999.ckpt) and the voice's lists (--lists)
Output: checkpoints/<name>/model.ckpt, edit.json, val_ms.txt.
        `--name voice_demuffled` is the shipped model; the default name is a new folder, never overwritten.

Usage (from the kiosk-tts folder; GPU, ~2 min):
  python src/4_demuffle/demuffle.py --name voice_demuffled
  python src/4_demuffle/demuffle.py --name myvoice_demuffled \\
      --base checkpoints/myvoice/step=007999.ckpt --lists data/3_training_lists/myvoice
Copied from an earlier kaani fine-tune round's demuffle.py; only file locations changed.
"""
import os
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import paths  # noqa: E402  -- kiosk-tts/src/paths.py: every location + the import path

import argparse
import json
import random


def rows(path):
    """(wav, text) per clip, first row only (row A: the text the model reads; build_voice writes A then B)."""
    out, seen = [], set()
    for l in open(path, encoding="utf-8"):
        w, t = l.rstrip("\n").split("|", 1)
        if w not in seen:
            seen.add(w)
            out.append((os.path.join(paths.ROOT, w), t))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="voice_demuffled_rebuild", help="output folder name under checkpoints/")
    ap.add_argument("--base", default=os.path.join(paths.CHECKPOINTS, "voice", "step=007999.ckpt"))
    ap.add_argument("--amount", type=float, default=1.0, help="share of the band-shape gap to remove")
    ap.add_argument("--lists", default=paths.VOICE_LISTS,
                    help="the voice's training lists (train/val/test.txt): its real clips are the reference")
    ap.add_argument("--fit-clips", type=int, default=60, help="how many training clips to fit the curve on")
    a = ap.parse_args()
    import voice_edit as V
    tr = rows(os.path.join(a.lists, "train.txt"))
    fit = random.Random(0).sample(tr, min(a.fit_clips, len(tr)))
    test = []
    for f in ("val.txt", "test.txt"):                      # the held-out recording: never trained on, scored at the end
        p = os.path.join(a.lists, f)
        if os.path.exists(p):
            test += rows(p)
    if not test:                                           # no held-out list: score on the fit clips and say so
        test = fit
    V.refs = lambda: fit + test
    V.split = lambda rs: (fit, test)
    sys.argv = ["voice_edit.py", "--name", a.name, "--base", os.path.abspath(a.base), "--demuffle", str(a.amount)]
    V.main()
    p = os.path.join(paths.CHECKPOINTS, a.name, "edit.json")
    e = json.load(open(p))
    e["ref"] = (f"{os.path.relpath(a.lists, paths.ROOT)}: fit {len(fit)} training clips, "
                f"test {'the held-out recording' if test is not fit else 'the same fit clips (no held-out list)'}")
    json.dump(e, open(p, "w"), indent=1)


if __name__ == "__main__":
    main()
