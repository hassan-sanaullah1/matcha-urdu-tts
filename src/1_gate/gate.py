"""STEP 1 -- gate the shipped voice's clips: keep only clips that are clearly its speaker with clean Urdu text.

The dataset (data/1_source/voice, 864 clips, 1.74 h) was built by ~/Desktop/urdu-dataset-builder: de-musicked with
MossFormer2, cut in silences, gate dropouts repaired. Its audio is used exactly as delivered. This step only decides
which clips to keep, with the rule used for every new voice since Round 13:

  reasr_C     the transcriber's Whisper re-decode agreed < 0.70 with the transcript (grade C)
  promoted    auto-released from the review pool (the dataset README's own "strict" option)
  loop        one word is > 40% of a 6+ word text
  voice       TitaNet similarity to her centroid < 0.70 (centroid = 300 random clips outside the held-out video)
  urdu_cer    an independent recogniser (Omnilingual ASR) disagrees with the Urdu more than a known-clean line
              (ANAM_URDU_CER_LINE below: the p90 of the Anam clips the r5_p2 recipe was first proven on -- a fixed
              number baked in here; no Anam audio is read)

For the shipped dataset this kept 782 of 864 clips (1.57 h; 1.53 h for training).

Input:  data/1_source/voice/{train.txt, val.txt, metadata.csv, wav/}
Output: <--out>/manifest.jsonl (one line per clip, the build step's input), wav -> the source clips (a link),
        urdu_check.jsonl (Omnilingual transcript + CER per clip; resumable), drop.txt (clips to leave out),
        keep_summary.json, titanet_centroid.npy
        The run used for the trained model is data/2_clean/voice_clean. The default output is a NEW folder
        (voice_clean_rebuild), so that copy is never overwritten.

Models (Hugging Face cache, not copied): nvidia/speakerverification_en_titanet_large, EmreAkgul/omniASR-CTC-300M-ONNX.
Usage (from the kiosk-tts folder):  python src/1_gate/gate.py
Copied from an earlier kaani fine-tune round's prep.py; only file locations changed.
"""
import os
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import paths  # noqa: E402  -- kiosk-tts/src/paths.py: every location + the import path
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import argparse
import collections
import csv
import json
import re

import numpy as np

SRC = paths.VOICE_SOURCE
HELD = "voice_-gI_ALIlkR8"                    # the dataset's own validation video: never trained on
LATIN = re.compile(r"[A-Za-z]")
VOICE_MIN = 0.70
# Omnilingual Urdu CER of Anam's 391 real clips (the clips r5_p2 was trained on), computed in Round 13
# (pk-r13-mahwash-r5p2-recipe/src/urdu_check.py anam_line, from the Round 9 meter's cache). A clip is dropped when its
# score is above Anam's p90: "no less clean than the data that made r5_p2".
ANAM_URDU_CER_LINE = {"n": 391, "p50": 0.07462686567164178, "p90": 0.375, "p95": 0.9}


def is_loop(text):
    w = text.split()
    return len(w) >= 6 and collections.Counter(w).most_common(1)[0][1] / len(w) > 0.4


def manifest():
    meta = {r["clip_file"]: r for r in csv.DictReader(open(os.path.join(SRC, "metadata.csv"), encoding="utf-8"))}
    rows = []
    for fl in ("train.txt", "val.txt"):
        for l in open(os.path.join(SRC, fl), encoding="utf-8"):
            path, text = l.rstrip("\n").split("|", 1)
            m = meta[path]
            ws = text.split()
            rows.append(dict(clip=os.path.basename(path), video="voice_" + m["video_id"], seg=int(m["seg_index"]),
                             text=text, queue="accepted", flags=[], duration=float(m["duration"]),
                             words=len(ws), english_words=sum(bool(LATIN.search(w)) for w in ws),
                             reasr=float(m["reasr_similarity"]), ctc=float(m["ctc_similarity"]),
                             promoted=bool(m["promoted"])))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=paths.VOICE_CLEAN + "_rebuild", help="output folder")
    a = ap.parse_args()
    import librosa
    import soundfile as sf
    import torch
    import gate_utils as G

    clean = a.out
    os.makedirs(clean, exist_ok=True)
    link = os.path.join(clean, "wav")
    if not os.path.islink(link):
        os.symlink(os.path.relpath(os.path.join(SRC, "wav"), clean), link)     # relative: the folder stays movable
    rows = manifest()
    wav = lambda c: os.path.join(link, c)                                       # noqa: E731
    load16 = lambda c: librosa.resample(sf.read(wav(c), dtype="float32")[0], orig_sr=22050, target_sr=16000)  # noqa: E731

    # voice: TitaNet to her centroid (300 random clips outside the held-out video)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    emb = G.Embedder(dev)
    rng = np.random.default_rng(0)
    pool = [r["clip"] for r in rows if r["video"] != HELD]
    pick = [pool[i] for i in rng.permutation(len(pool))[:300]]
    cen = np.concatenate([emb.embed([load16(c)]) for c in pick]).mean(0)
    cen /= np.linalg.norm(cen)
    np.save(os.path.join(clean, "titanet_centroid.npy"), cen)
    for r in rows:
        r["spk_sim"] = round(float(emb.embed([load16(r["clip"])])[0] @ cen), 3)
    del emb
    torch.cuda.empty_cache()
    with open(os.path.join(clean, "manifest.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"manifest: {len(rows)} clips, {sum(r['duration'] for r in rows) / 3600:.2f} h", flush=True)

    # urdu: Omnilingual CER per clip (resumable)
    line = ANAM_URDU_CER_LINE
    out_p = os.path.join(clean, "urdu_check.jsonl")
    done = {r["clip"]: r for r in map(json.loads, open(out_p, encoding="utf-8"))} if os.path.exists(out_p) else {}
    omni = G.Omni(6)
    with open(out_p, "a", encoding="utf-8") as f:
        for n, r in enumerate(rows):
            if r["clip"] in done:
                continue
            txt = omni(G.load16(wav(r["clip"])))
            c = G.urdu_cers([r], [dict(clip=r["clip"], omni=txt)]).get(r["clip"])
            done[r["clip"]] = dict(clip=r["clip"], omni=txt, cer=c)
            f.write(json.dumps(done[r["clip"]], ensure_ascii=False) + "\n")
            if (n + 1) % 100 == 0:
                f.flush()
                print(f"  urdu {n + 1}/{len(rows)}", flush=True)

    # keep rule
    why = collections.defaultdict(list)
    for r in rows:
        c = r["clip"]
        if r["reasr"] < 0.70:
            why[c].append("reasr_C")
        if r["promoted"]:
            why[c].append("promoted")
        if is_loop(r["text"]):
            why[c].append("loop")
        if r["spk_sim"] < VOICE_MIN:
            why[c].append("voice")
        cer = done.get(c, {}).get("cer")
        if cer is not None and cer > line["p90"]:
            why[c].append("urdu_cer")
    kept = [r for r in rows if r["clip"] not in why]
    h = lambda rs: round(sum(r["duration"] for r in rs) / 3600, 2)             # noqa: E731
    cers = np.array([d["cer"] for d in done.values() if d["cer"] is not None])
    sims = np.array([r["spk_sim"] for r in rows])
    summ = dict(clips=len(rows), hours=h(rows), kept_clips=len(kept), kept_hours=h(kept),
                kept_train_hours=h([r for r in kept if r["video"] != HELD]),
                dropped_clips=len(why), dropped_hours=h([r for r in rows if r["clip"] in why]),
                drop_reasons=dict(collections.Counter(x for v in why.values() for x in v)),
                drop_reason_alone=dict(collections.Counter(v[0] for v in why.values() if len(v) == 1)),
                anam_urdu_cer_line=line, urdu_cer=dict(n_scored=len(cers), p50=float(np.median(cers)),
                                                       p90=float(np.percentile(cers, 90))),
                spk_sim=dict(p5=float(np.percentile(sims, 5)), p50=float(np.median(sims)), min=float(sims.min())),
                kept_english_share=round(sum(r["english_words"] for r in kept) / max(1, sum(r["words"] for r in kept)), 3))
    open(os.path.join(clean, "drop.txt"), "w").write("\n".join(sorted(why)) + "\n")
    json.dump(summ, open(os.path.join(clean, "keep_summary.json"), "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
