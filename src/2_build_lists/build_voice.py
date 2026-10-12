"""STEP 2a -- build a target speaker's training lists: the clean clips (step 1) -> Matcha filelists (speaker 0).

Works for ANY voice: its defaults are the shipped voice, but --clean/--drop/--held/--out point this at a voice
you gated yourself (gate.py for the shipped dataset's own format, import_voice.py for any other dataset).

What it does to every clip:
  gain      match the base voice's anchor log-mel mean, so loudness matches what the checkpoint was trained on.
            Each clip's gain is capped so its peak stays under the limiter's knee: the limiter never touches a
            training clip (it used to waveshape the loudest frames), and the gain is solved with that same cap.
  rows      row A = Urdu script + English phones (pk_gazetteer.prepare_pk), row B = some Urdu words replaced
            by espeak-ur phones (p=0.5). Both rows must tokenise identically at train and inference time.
  split     one whole recording ("video") is held out, so no recording appears in both train and val/test.

Each list line is "<wav path>|<text the model reads>". The wav paths are relative to the kiosk-tts folder,
so run from there.

Input:  a clean-clip folder (manifest.jsonl + wav/ + drop.txt), data/1_source/anchor (loudness reference)
Output: <--out>/{wavs/, train.txt, val.txt, test.txt, test_raw.json, summary.json}
        (wavs/ also gets *_raw.wav: each clip before the gain; only needed while building)

The lists the shipped model trained on are data/3_training_lists/voice (760 training clips, 1.53 h). The
default output is a NEW folder (voice_rebuild), so that copy is never overwritten.

Usage (from the kiosk-tts folder):
  python src/2_build_lists/build_voice.py       # the shipped voice; held-out video auto-filled
  python src/2_build_lists/build_voice.py --clean data/2_clean/myvoice --drop data/2_clean/myvoice/drop.txt \\
      --held myvoice_heldout --out data/3_training_lists/myvoice        # any new voice (import_voice.py prints this)
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import paths  # noqa: E402  -- kiosk-tts/src/paths.py: every location + the import path

import argparse
import collections
import glob
import json
import random
import re

import numpy as np
import soundfile as sf

import phones
import pk_gazetteer
from dataset_utils import _TOKEN, rows_from, soft_limit

PEAK = 0.84          # just under soft_limit's 0.85 knee


def clip_gain(w, gain):
    peak = float(np.abs(w).max()) or 1.0
    return min(gain, PEAK / peak)


def mel_mean(wavs, gain, cap=True):
    """Mean log-mel of wavs at a gain, with the same per-clip peak cap the written files get (cap=False for the
    anchor, which is the reference level and is measured as it is)."""
    import torch
    import matcha_torch
    matcha_torch._prepare_imports()
    from matcha.utils.audio import mel_spectrogram
    vals = []
    for p in wavs:
        w, _ = sf.read(p, dtype="float32")
        y = torch.from_numpy(w * (clip_gain(w, gain) if cap else gain))[None]
        vals.append(mel_spectrogram(y, 1024, 80, 22050, 256, 1024, 0, 8000, center=False).mean().item())
    return float(np.mean(vals))


CLEAN = os.path.relpath(paths.VOICE_CLEAN, paths.ROOT)                     # data/2_clean/voice_clean
OUT = os.path.join(os.path.relpath(paths.TRAINING_LISTS, paths.ROOT), "voice_rebuild")
DROP = os.path.join(CLEAN, "drop.txt")                                       # the gate's dropped-clip list (step 1)
HELD = "voice_-gI_ALIlkR8"                                                 # the dataset's own validation video
_URDU = re.compile(r"[؀-ۿ]")
_LATIN = re.compile(r"[A-Za-z]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--alter", default="none", help="none only (voice alteration was tried and not used; its code is "
                    "not included)")
    ap.add_argument("--p-mix", type=float, default=0.5)
    ap.add_argument("--clean", default=CLEAN, help="main clean-clip folder (manifest.jsonl + wav/)")
    ap.add_argument("--extra", default="", help="more clean-clip folders (manifest.jsonl + wav/), comma-separated")
    ap.add_argument("--drop", default=DROP, help="file of clip names to leave out (the gate's drop.txt; '' = none)")
    ap.add_argument("--held", default="",
                    help="held-out recording id. '' = auto: the recording closest to 8%% of clips "
                         f"(for the shipped voice, that is {HELD!r})")
    a = ap.parse_args()
    if a.alter != "none":
        raise SystemExit("--alter: voice alteration is not part of kiosk-tts (the voice is used unaltered)")
    strength = None
    if not a.held and a.clean == CLEAN:
        a.held = HELD                                      # reproduce the shipped voice's split exactly

    rows = []
    for d in [a.clean] + [x for x in a.extra.split(",") if x]:
        for l in open(os.path.join(d, "manifest.jsonl"), encoding="utf-8"):
            r = json.loads(l)
            r["src"] = os.path.join(d, "wav", r["clip"])
            rows.append(r)
    if a.drop:
        bad = {l.strip() for l in open(a.drop, encoding="utf-8") if l.strip()}
        rows = [r for r in rows if r["clip"] not in bad]
        print(f"dropped {len(bad)} clips listed in {a.drop}")
    by_video = collections.Counter(r["video"] for r in rows)
    # Hold out one whole recording, the one closest to 8% of the clips, so val/test share no recording with train.
    held_id = a.held or min(by_video, key=lambda v: abs(by_video[v] - 0.08 * len(rows)))
    train = [r for r in rows if r["video"] != held_id]
    held = [r for r in rows if r["video"] == held_id]
    rnd = random.Random(7)
    test = rnd.sample(held, min(12, len(held)))
    tset = {t["clip"] for t in test}
    val = [h for h in held if h["clip"] not in tset][:80]
    print(f"{len(rows)} clips; held-out video {held_id} ({len(held)} clips) -> {len(val)} val, {len(test)} test")

    wav_dir = os.path.join(a.out, "wavs")
    os.makedirs(wav_dir, exist_ok=True)
    raw = {}
    for r in train + val + test:
        src = r["src"]
        dst = os.path.join(wav_dir, r["clip"].replace(".wav", "_raw.wav"))
        if not os.path.exists(dst):
            w, sr = sf.read(src, dtype="float32")
            sf.write(dst, w, sr, subtype="PCM_16")
        raw[r["clip"]] = dst

    # Loudness target: 50 clips of the base model's own training voice (every 35th of its 1,733 anchor clips,
    # already selected into data/1_source/anchor), so her clips match the level the base model knows.
    anchor = sorted(glob.glob(os.path.join(paths.ANCHOR, "*.wav")))
    target_mean = mel_mean(anchor, 1.0, cap=False)
    sample = [raw[r["clip"]] for r in train[:: max(1, len(train) // 200)]]
    gain = 1.0
    for _ in range(6):
        gain *= float(np.exp(target_mean - mel_mean(sample, gain)))
    print(f"gain {gain:.3f} (anchor log-mel mean {target_mean:.3f}, residual {mel_mean(sample, gain) - target_mean:+.3f})")

    prepared, failed = {}, 0
    for r in train + val + test:
        try:
            prepared[r["clip"]] = pk_gazetteer.prepare_pk(r["text"])
        except Exception:  # noqa: BLE001
            failed += 1
            prepared[r["clip"]] = None
    print("front-end failures:", failed)
    words = sorted({m.group(2) for t in prepared.values() if t for tok in t.split() for m in [_TOKEN.match(tok)] if m})
    table = {w: ph for w, ph in zip(words, phones.urdu_words(words, {})) if ph}
    print(f"Urdu word types {len(words)}, phonemised {len(table)}")

    import matcha_torch
    matcha_torch._prepare_imports()
    from matcha.text import text_to_sequence
    import tts
    v = tts.vocab()

    cover, cls_count, lists = collections.Counter(), collections.Counter(), {}
    dropped = collections.Counter()
    capped = 0
    phone_rx = re.compile("|".join(sorted(map(re.escape, phones.PHONES), key=len, reverse=True)))
    for name, rs in (("train", train), ("val", val), ("test", test)):
        lines = []
        for r in rs:
            ta = prepared.get(r["clip"])
            if not ta:
                dropped["frontend"] += 1
                continue
            dst = os.path.join(wav_dir, r["clip"])
            if not os.path.exists(dst):
                w, sr = sf.read(raw[r["clip"]], dtype="float32")
                g = clip_gain(w, gain)
                capped += g < gain
                sf.write(dst, soft_limit(w * g), sr, subtype="PCM_16")      # soft_limit is now only a guard
            tb = rows_from(ta, table, a.p_mix, random.Random(r["clip"]))
            ok = True
            for t in (ta, tb):       # the cleaner must be a no-op and the ids must match inference tokenisation
                seq, clean = text_to_sequence(t, ["urdu_cleaners_graphemes"])
                if not (clean == t and seq == [v.get(ch, v[" "]) for ch in t]):
                    ok = False
            if not ok:
                dropped["tokenizer"] += 1
                continue
            lines += [f"{dst}|{ta}", f"{dst}|{tb}"]
            if name == "train":
                cover.update(ph for tok in tb.split() if not _TOKEN.match(tok) for ph in phone_rx.findall(tok))
                n_ur = len([t for t in r["text"].split() if _URDU.search(t)])
                n_en = len([t for t in r["text"].split() if _LATIN.search(t)])
                cls_count["pure_ur" if n_en == 0 else "pure_en" if n_ur == 0 else "mixed"] += r["duration"]
        # Row A then row B per clip, unshuffled: --no-row-b in train.py takes every other line, and the
        # dataloader shuffles anyway.
        lists[name] = lines
        open(os.path.join(a.out, f"{name}.txt"), "w", encoding="utf-8").write("\n".join(lines) + "\n")

    json.dump([dict(clip=r["clip"], wav=os.path.join(wav_dir, r["clip"]), text=r["text"]) for r in test],
              open(os.path.join(a.out, "test_raw.json"), "w", encoding="utf-8"), ensure_ascii=False)
    tot = sum(cls_count.values())
    summary = dict(alter=a.alter, gain=round(gain, 3), clips_with_capped_gain=int(capped), held_out_video=held_id, clips=len(rows),
                   train_clips=len(lists["train"]) // 2,
                   train_hours=round(sum(r["duration"] for r in train) / 3600, 2),
                   val_rows=len(lists["val"]), test_rows=len(lists["test"]), dropped=dict(dropped),
                   class_share={c: round(s / tot, 3) for c, s in cls_count.items()} if tot else {},
                   by_queue=dict(collections.Counter(r["queue"] for r in train)),
                   rare_phones={p: cover[p] for p in phones.PHONES if cover[p] < 100})
    json.dump(summary, open(os.path.join(a.out, "summary.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
