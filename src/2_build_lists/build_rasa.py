"""STEP 2b -- build the SUPPORT voice's training list: AI4Bharat Rasa Urdu, FEMALE, neutral styles
(WIKI/CONV/BOOK/NEWS), studio quality, accurate text.

Why: it keeps the model hearing correct Urdu sounds (غ ق خ, aspirates) that are rare or mistranscribed in the
target voice's YouTube audio. It is trained as a SEPARATE speaker (id 1), 30% of the rows; the kiosk speaks as
the target voice (id 0). Same text pipeline as build_voice.py (pk front end, two rows per clip, tokenizer check),
own loudness match. Every English sound the model learns comes from the target voice's clips; Rasa only steadies
the Urdu. This is optional for a new voice -- see README: fine-tune on your own dataset.

Input:  data/1_source/rasa/train-0000{1..4}-of-00056.parquet, data/1_source/anchor (loudness reference)
Output: <--out>/{wavs/, train.txt, summary.json}   (893 clips, 1.5 h with shards 1-4)

The list the shipped model trained on is data/3_training_lists/rasa_female; a rebuild reproduces its text exactly
(the default output is a new folder, rasa_female_rebuild, so that copy is never overwritten).

Usage (from the kiosk-tts folder):  python src/2_build_lists/build_rasa.py --shards 1,2,3,4
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import paths  # noqa: E402  -- kiosk-tts/src/paths.py: every location + the import path

import argparse
import glob
import io
import json
import random

import numpy as np
import soundfile as sf

import phones
import pk_gazetteer
from dataset_utils import _TOKEN, mel_mean, rows_from, soft_limit

OUT = os.path.join(os.path.relpath(paths.TRAINING_LISTS, paths.ROOT), "rasa_female_rebuild")
STYLES = {"WIKI", "CONV", "BOOK", "NEWS"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", default="1,2,3,4")
    ap.add_argument("--max-sec", type=float, default=12.0)
    ap.add_argument("--out", default=OUT, help="output folder")
    a = ap.parse_args()
    globals()["OUT"] = a.out
    import librosa
    import pyarrow.parquet as pq
    wav_dir = os.path.join(OUT, "wavs")
    os.makedirs(wav_dir, exist_ok=True)
    rows = []
    for sh in a.shards.split(","):
        # a copy of https://huggingface.co/datasets/ai4bharat/Rasa , Urdu/train-0000N-of-00056.parquet
        p = os.path.join(paths.RASA_SOURCE, f"train-{int(sh):05d}-of-00056.parquet")
        t = pq.read_table(p)
        meta = t.select([c for c in t.column_names if c != "audio"]).to_pylist()
        audio = t.column("audio")
        for i, m in enumerate(meta):
            if m["gender"] != "Female" or m["style"] not in STYLES:
                continue
            if float(m["duration"]) > a.max_sec or float(m["duration"]) < 1.5:
                continue
            dst = os.path.join(wav_dir, m["filename"] + "_raw.wav")
            if not os.path.exists(dst):
                w, sr = sf.read(io.BytesIO(audio[i].as_py()["bytes"]), dtype="float32")
                if w.ndim > 1:
                    w = w.mean(1)
                w = librosa.resample(w, orig_sr=sr, target_sr=22050)
                sf.write(dst, w, 22050, subtype="PCM_16")
            rows.append(dict(id=m["filename"], raw=dst, text=m["text"], dur=float(m["duration"])))
        print(f"shard {sh}: total {len(rows)} clips, {sum(r['dur'] for r in rows) / 3600:.2f} h", flush=True)

    # Loudness target: the same 50 base-voice clips build_voice uses (data/1_source/anchor).
    anchor = sorted(glob.glob(os.path.join(paths.ANCHOR, "*.wav")))
    target = mel_mean(anchor, 1.0)
    sample = [r["raw"] for r in rows[::max(1, len(rows) // 150)]]
    gain = 1.0
    for _ in range(6):
        gain *= float(np.exp(target - mel_mean(sample, gain)))
    print(f"gain {gain:.3f}")

    prepared = {r["id"]: pk_gazetteer.prepare_pk(r["text"]) for r in rows}
    words = sorted({m.group(2) for t in prepared.values() for tok in t.split() for m in [_TOKEN.match(tok)] if m})
    table = {w: ph for w, ph in zip(words, phones.urdu_words(words, {})) if ph}

    import matcha_torch
    matcha_torch._prepare_imports()
    from matcha.text import text_to_sequence
    import tts
    v = tts.vocab()
    lines = []
    for r in rows:
        dst = os.path.join(wav_dir, r["id"] + ".wav")
        if not os.path.exists(dst):
            w, sr = sf.read(r["raw"], dtype="float32")
            sf.write(dst, soft_limit(w * gain), sr, subtype="PCM_16")
        ta = prepared[r["id"]]
        tb = rows_from(ta, table, 0.5, random.Random(r["id"]))
        ok = all(text_to_sequence(t, ["urdu_cleaners_graphemes"])[1] == t for t in (ta, tb))
        if ok:
            lines += [f"{dst}|{ta}", f"{dst}|{tb}"]
    random.Random(1).shuffle(lines)
    open(os.path.join(OUT, "train.txt"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    json.dump(dict(clips=len(rows), hours=round(sum(r["dur"] for r in rows) / 3600, 2), rows=len(lines), gain=gain),
              open(os.path.join(OUT, "summary.json"), "w"), indent=1)
    print("RASA_DONE", len(rows), "clips", len(lines), "rows")


if __name__ == "__main__":
    main()
