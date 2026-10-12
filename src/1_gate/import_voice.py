"""STEP 1 (any dataset) -- turn a folder of audio + transcripts into the clean-clip folder step 2 reads.

Use this for a NEW voice. (gate.py is the same step for the shipped dataset specifically: it understands that
dataset's own metadata.csv and its re-ASR scores. This script asks only for audio + text.)

What it does to every clip:
  audio     decoded, mixed to mono and resampled to 22,050 Hz, written as 16-bit PCM. The whole pipeline assumes
            22,050 Hz mono (the base model's rate); a dataset at 44.1 kHz trains wrong, so this is not optional.
  length    clips shorter than --min-sec or longer than --max-sec are dropped (the decoder trains on 4 s windows)
  group     clips are grouped into "videos" (recordings). Step 2 holds one whole group out, so val/test share no
            recording with train. With a single group it would hold out everything, so this script then carves a
            held-out group of --heldout-share of the clips and prints the --held value to pass to step 2.
  gate      with --gate: keep only clips that are clearly one voice (TitaNet) with text that an independent
            recogniser agrees with (Omnilingual, Urdu stretches only). Optional but recommended for scraped audio.

Input:  --audio <dir of audio files>  --text <metadata file>, either
          * lines of "<audio file name>|<transcript>"  (also accepts TAB), or
          * a .csv with a header; the file-name and text columns are found by name.
Output: <--out>/{wav/, manifest.jsonl, drop.txt, summary.json} (+ urdu_check.jsonl, titanet_centroid.npy with --gate)
        manifest.jsonl is what step 2 reads: one {clip, video, text, duration, queue} per line.

Usage (from the kiosk-tts folder):
  python src/1_gate/import_voice.py --name myvoice --audio ~/data/myvoice/wav --text ~/data/myvoice/metadata.csv
  python src/1_gate/import_voice.py --name myvoice --audio ... --text ... --gate     # add the quality checks (GPU)
Then build the lists with step 2 (the command it prints at the end).
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

SR = 22050                                      # the base model's rate: every later step assumes it
AUDIO_EXT = (".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus", ".aac", ".wma", ".aiff", ".aif")
CLIP_COLS = ("clip_file", "clip", "filename", "file_name", "file", "audio", "audio_file", "audio_path",
             "wav", "wav_path", "path", "name", "id", "utt", "utt_id", "utterance")
TEXT_COLS = ("text", "transcript", "transcription", "sentence", "sentences", "raw_text", "normalized_text",
             "normalised_text", "words", "label")
URDU = re.compile(r"[؀-ۿ]")
LATIN = re.compile(r"[A-Za-z]")


def read_pairs(path):
    """[(audio file name as written, text)] from a .csv with a header, or from "<name>|<text>" / "<name>\\ttext" lines."""
    if path.lower().endswith(".csv"):
        with open(path, newline="", encoding="utf-8") as f:
            rd = csv.DictReader(f)
            cols = {c.strip().lower(): c for c in (rd.fieldnames or [])}
            cc = next((cols[c] for c in CLIP_COLS if c in cols), None)
            tc = next((cols[c] for c in TEXT_COLS if c in cols), None)
            if not cc or not tc:
                raise SystemExit(f"--text {path}: could not find a file-name and a text column in {rd.fieldnames}.\n"
                                 f"Rename them to e.g. clip_file,text or pass a '<name>|<text>' .txt instead.")
            print(f"csv columns: file='{cc}' text='{tc}'")
            return [(r[cc], (r[tc] or "").strip()) for r in rd if r.get(cc) and (r.get(tc) or "").strip()]
    out = []
    for ln in open(path, encoding="utf-8"):
        ln = ln.rstrip("\n")
        if not ln.strip() or ln.lstrip().startswith("#"):
            continue
        parts = ln.split("|", 1) if "|" in ln else ln.split("\t", 1)
        if len(parts) == 2 and parts[1].strip():
            out.append((parts[0].strip(), parts[1].strip()))
    return out


def index_audio(audio_dir):
    """{lower-case stem: full path} and {lower-case file name: full path} for every audio file, searched recursively."""
    by_stem, by_name = {}, {}
    for root, _, files in os.walk(audio_dir):
        for fn in files:
            if fn.lower().endswith(AUDIO_EXT):
                p = os.path.join(root, fn)
                by_name.setdefault(fn.lower(), p)
                by_stem.setdefault(os.path.splitext(fn)[0].lower(), p)
    return by_stem, by_name


def resolve(ref, by_stem, by_name):
    """Find the audio file a metadata row points at: exact name, then bare name, then stem (extension ignored)."""
    base = os.path.basename(ref.strip().replace("\\", "/"))
    return (by_name.get(base.lower()) or by_stem.get(os.path.splitext(base)[0].lower()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="voice name; used for the output folder and the group names")
    ap.add_argument("--audio", required=True, help="folder with the audio files (searched recursively)")
    ap.add_argument("--text", required=True, help="metadata: a .csv with a header, or '<file name>|<transcript>' lines")
    ap.add_argument("--out", default="", help="output folder (default data/2_clean/<name>)")
    ap.add_argument("--min-sec", type=float, default=1.0)
    ap.add_argument("--max-sec", type=float, default=12.5)
    ap.add_argument("--group-by", choices=("prefix", "none"), default="prefix",
                    help="prefix: group clips by the file name before --group-sep (one group per recording); "
                         "none: one group for everything")
    ap.add_argument("--group-sep", default="__", help="separator that ends the recording id in a file name")
    ap.add_argument("--heldout-share", type=float, default=0.08,
                    help="share of clips carved into a held-out group when there is only one group")
    ap.add_argument("--gate", action="store_true", help="also run the voice (TitaNet) and Urdu (Omnilingual) checks")
    ap.add_argument("--voice-min", type=float, default=0.70, help="--gate: drop clips below this TitaNet similarity")
    ap.add_argument("--urdu-cer-max", type=float, default=0.375,
                    help="--gate: drop clips whose independent Urdu error is above this (gate.py's line)")
    a = ap.parse_args()
    import librosa
    import soundfile as sf

    out = a.out or os.path.join(paths.DATA, "2_clean", a.name)
    wav_dir = os.path.join(out, "wav")
    os.makedirs(wav_dir, exist_ok=True)

    pairs = read_pairs(a.text)
    if not pairs:
        raise SystemExit(f"--text {a.text}: no usable rows found")
    by_stem, by_name = index_audio(a.audio)
    if not by_name:
        raise SystemExit(f"--audio {a.audio}: no audio files found (looked for {', '.join(AUDIO_EXT)})")
    print(f"{len(pairs)} metadata rows, {len(by_name)} audio files in {a.audio}")

    # ---- decode -> mono 22,050 Hz -> wav/, and measure each clip -------------------------------------------
    rows, drop, missing = [], {}, 0
    for ref, text in pairs:
        src = resolve(ref, by_stem, by_name)
        clip = os.path.splitext(os.path.basename(ref.strip().replace("\\", "/")))[0] + ".wav"
        if src is None:
            missing += 1
            drop[clip] = ["no_audio"]
            continue
        dst = os.path.join(wav_dir, clip)
        if os.path.exists(dst):
            w, _ = sf.read(dst, dtype="float32")
        else:
            try:
                w, _ = librosa.load(src, sr=SR, mono=True)
            except Exception as e:  # noqa: BLE001  -- a file we cannot decode is a dropped clip, not a crash
                drop[clip] = [f"decode_error:{type(e).__name__}"]
                continue
            peak = float(np.abs(w).max()) if w.size else 0.0
            if peak > 1.0:
                w = w / peak                                  # keep it inside the 16-bit range; step 2 sets the level
            sf.write(dst, w, SR, subtype="PCM_16")
        dur = len(w) / SR
        if not (a.min_sec <= dur <= a.max_sec):
            drop[clip] = [f"length:{dur:.2f}s"]
            continue
        ws = text.split()
        rows.append(dict(clip=clip, video="", seg=0, text=text, queue="accepted", flags=[], duration=round(dur, 3),
                         words=len(ws), english_words=sum(bool(LATIN.search(x)) for x in ws)))
    if missing:
        print(f"WARNING {missing} metadata rows had no matching audio file (listed in drop.txt as no_audio)")
    if not rows:
        raise SystemExit("no clips survived; check --audio / --text and the length bounds")

    # ---- group into recordings, so step 2 can hold one whole group out ------------------------------------
    for r in rows:
        stem = r["clip"][:-4]
        g = stem.split(a.group_sep)[0] if (a.group_by == "prefix" and a.group_sep in stem) else a.name
        r["video"] = f"{a.name}_{g}" if g != a.name else a.name
    groups = collections.Counter(r["video"] for r in rows)
    if len(groups) == 1:
        # One recording: holding out a whole group would hold out everything. Carve a deterministic slice instead.
        k = max(1, min(len(rows) - 1, round(a.heldout_share * len(rows))))
        held_name = f"{a.name}_heldout"
        for r in sorted(rows, key=lambda r: r["clip"])[-k:]:
            r["video"] = held_name
        groups = collections.Counter(r["video"] for r in rows)
        print(f"one recording only -> carved {k} clips into the group '{held_name}'")
    held = min(groups, key=lambda g: abs(groups[g] - a.heldout_share * len(rows)))
    print(f"{len(groups)} group(s); step 2 should hold out '{held}' ({groups[held]} clips)")

    # ---- optional quality gate ----------------------------------------------------------------------------
    if a.gate:
        import torch
        import gate_utils as G
        wav_of = lambda c: os.path.join(wav_dir, c)                                     # noqa: E731
        load16 = lambda c: librosa.resample(sf.read(wav_of(c), dtype="float32")[0], orig_sr=SR, target_sr=16000)  # noqa: E731
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        emb = G.Embedder(dev)
        rng = np.random.default_rng(0)
        pool = [r["clip"] for r in rows if r["video"] != held] or [r["clip"] for r in rows]
        pick = [pool[i] for i in rng.permutation(len(pool))[:300]]
        cen = np.concatenate([emb.embed([load16(c)]) for c in pick]).mean(0)
        cen /= np.linalg.norm(cen)
        np.save(os.path.join(out, "titanet_centroid.npy"), cen)
        for r in rows:
            r["spk_sim"] = round(float(emb.embed([load16(r["clip"])])[0] @ cen), 3)
        del emb
        torch.cuda.empty_cache()
        print(f"voice: median similarity {np.median([r['spk_sim'] for r in rows]):.3f}")

        out_p = os.path.join(out, "urdu_check.jsonl")
        done = {d["clip"]: d for d in map(json.loads, open(out_p, encoding="utf-8"))} if os.path.exists(out_p) else {}
        if any(URDU.search(r["text"]) for r in rows):
            omni = G.Omni(6)
            with open(out_p, "a", encoding="utf-8") as f:
                for n, r in enumerate(rows):
                    if r["clip"] in done:
                        continue
                    txt = omni(G.load16(wav_of(r["clip"])))
                    c = G.urdu_cers([r], [dict(clip=r["clip"], omni=txt)]).get(r["clip"])
                    done[r["clip"]] = dict(clip=r["clip"], omni=txt, cer=c)
                    f.write(json.dumps(done[r["clip"]], ensure_ascii=False) + "\n")
                    if (n + 1) % 100 == 0:
                        f.flush()
                        print(f"  urdu {n + 1}/{len(rows)}", flush=True)
        else:
            print("no Urdu text found: skipping the Urdu check")
        for r in rows:
            if r.get("spk_sim") is not None and r["spk_sim"] < a.voice_min:
                drop.setdefault(r["clip"], []).append("voice")
            cer = done.get(r["clip"], {}).get("cer")
            if cer is not None and cer > a.urdu_cer_max:
                drop.setdefault(r["clip"], []).append("urdu_cer")

    # ---- write the manifest step 2 reads -----------------------------------------------------------------
    with open(os.path.join(out, "manifest.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    open(os.path.join(out, "drop.txt"), "w", encoding="utf-8").write("".join(f"{c}\n" for c in sorted(drop)))
    kept = [r for r in rows if r["clip"] not in drop]
    h = lambda rs: round(sum(r["duration"] for r in rs) / 3600, 2)                      # noqa: E731
    summary = dict(name=a.name, audio=os.path.abspath(a.audio), text=os.path.abspath(a.text), sample_rate=SR,
                   metadata_rows=len(pairs), clips=len(rows), hours=h(rows),
                   kept_clips=len(kept), kept_hours=h(kept),
                   kept_train_hours=h([r for r in kept if r["video"] != held]),
                   dropped=len(drop), drop_reasons=dict(collections.Counter(x for v in drop.values() for x in v)),
                   groups=len(collections.Counter(r["video"] for r in rows)), held_out_group=held,
                   held_out_clips=sum(r["video"] == held for r in rows),
                   english_word_share=round(sum(r["english_words"] for r in kept) / max(1, sum(r["words"] for r in kept)), 3),
                   gated=bool(a.gate))
    json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=1)
    print(json.dumps(summary, indent=1))

    rel = os.path.relpath(out, paths.ROOT)
    print("\nNEXT -- build the training lists (step 2):\n"
          f"  python src/2_build_lists/build_voice.py --clean {rel} --drop {rel}/drop.txt \\\n"
          f"      --held {held} --out data/3_training_lists/{a.name}")


if __name__ == "__main__":
    main()
