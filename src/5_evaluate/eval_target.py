"""STEP 5b -- Urdu error rate, English word check and voice similarity, plus a listening page.

Renders the frozen eval suite (data/4_eval/eval_suite.json) with one or more checkpoints and scores them.

Renders (seed 100): every eval_suite item through the pk front end (gazetteer + urdu-phones frontend).
The BASE AhsanTalal checkpoint renders only the Urdu categories (it cannot read phones) as the parity reference.

Scores (faster-whisper large-v3 int8, greedy):
  urdu CER   urdu_parity (60) and urdu_rare (12), language=ur (Urdu character error rate; base model ~0.010)
  en hit     share of reference English words recovered, language=en, on pure_en / en_names / en_business.
             Collapse detector only: it rewards American English and cannot judge the Pakistani accent.
  voice      WavLM similarity to a fixed reference voice and to the Rasa support voice. NOTE: the reference voice
             is Dr. Sarah Shahed (data/4_eval/shahed_ref), a legacy distance anchor kept from when the recipe was
             first scored; "sim_shahed" is NOT "is this the target voice" -- compare it between checkpoints only.
             (For "is this her voice", use the TitaNet check in docs/training_report.md.)

Models used (Hugging Face cache, not copied): faster-whisper large-v3, microsoft/wavlm-base-plus-sv.
Output: results/eval_<name>/index.html, scores.json

Usage (from the kiosk-tts folder):
  python src/5_evaluate/eval_target.py --name voice_demuffled --ckpt checkpoints/voice_demuffled/model.ckpt
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import paths  # noqa: E402  -- kiosk-tts/src/paths.py: every location + the import path

import argparse
import gc
import html
import json
import re
import shutil
import subprocess

import numpy as np
import soundfile as sf
import torch

import tts
import pk_gazetteer
import spk_cond

EN_CATS = ("pure_en", "en_names", "en_business")
# Speaking settings, as in src/speak.py: 32 steps at temperature 0.35 measured +0.7-0.9 UTMOS over 10 at 0.667.
STEPS, TEMP = 32, 0.35
UR_CATS = ("urdu_parity", "urdu_rare")
ORDER = ["journey", "chatbot", "dense", "ur_word", "ur_phrase", "en_led", "pure_en", "en_names", "en_business",
         "numbers", "urdu_rare", "urdu_parity"]


def normalize_urdu(s):
    """Copied from finetune/src/eval/score_wavs.py: drop punctuation and diacritics, spell out digits."""
    drop = "۔،؟!.,?؛:\"'()[]{}«»…-–—‌‍"
    diacritics = "".join(chr(c) for c in range(0x064B, 0x0653))
    for ch in drop + diacritics:
        s = s.replace(ch, " ")
    try:
        from urdu_numbers import expand_numbers     # Whisper writes spoken numbers as digits
        s = expand_numbers(s)
    except Exception:
        pass
    return " ".join(s.split())


def ckpt_name(p):
    if p == "base":
        return "base"
    return os.path.basename(os.path.dirname(p)) + "@" + re.sub(r"\D", "", os.path.basename(p)).lstrip("0")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--ckpt", action="append", required=True)
    ap.add_argument("--no-base", action="store_true")
    ap.add_argument("--parity-n", type=int, default=60)
    a = ap.parse_args()

    suite = json.load(open(os.path.join(paths.EVAL, "eval_suite.json"), encoding="utf-8"))
    out = os.path.join(paths.RESULTS, f"eval_{a.name}")
    os.makedirs(os.path.join(out, "audio"), exist_ok=True)
    systems = ([] if a.no_base else ["base"]) + a.ckpt

    import matcha_torch
    renders = {}
    for sysn in systems:
        tag = ckpt_name(sysn)
        renders[tag] = {}
        for cat in ORDER:
            if sysn == "base" and cat not in UR_CATS:
                continue
            items = suite[cat]["items"][: a.parity_n if cat == "urdu_parity" else None]
            for it in items:
                p = os.path.join(out, "audio", f"{tag.replace('@', '_')}__{it['id']}.wav")
                if not os.path.exists(p):
                    if sysn == "base":
                        w, _ = tts.synth(it["text"], None, "script", seed=100, steps=STEPS, temp=TEMP)
                    else:
                        if spk_cond.has_spk_cond(sysn):
                            spk_cond.load_for_inference(sysn, "cuda", spk=0)
                        prepared = pk_gazetteer.prepare_pk(it["text"])
                        w, _ = tts.synth(it["text"], sysn, seed=100, steps=STEPS, temp=TEMP, prepared=tts.clean(prepared))
                    sf.write(p, w, tts.OUT_SR)
                renders[tag][it["id"]] = p
        print("rendered", tag, len(renders[tag]), flush=True)
        matcha_torch._cache.clear()
        gc.collect()
        torch.cuda.empty_cache()

    import jiwer
    from faster_whisper import WhisperModel
    asr = WhisperModel("large-v3", device="cuda", compute_type="int8")

    def dec(p, lang):
        segs, _ = asr.transcribe(p, language=lang, beam_size=1, temperature=0.0, vad_filter=False,
                                 condition_on_previous_text=False)
        return " ".join(x.text.strip() for x in segs)

    scores = {}
    for tag, rs in renders.items():
        sc = {}
        for cat in UR_CATS:
            items = suite[cat]["items"][: a.parity_n if cat == "urdu_parity" else None]
            cers = [min(jiwer.cer(normalize_urdu(it["text"]), normalize_urdu(dec(rs[it["id"]], "ur")) or "-"), 1.0)
                    for it in items if it["id"] in rs]
            sc[f"cer_{cat}"] = round(float(np.mean(cers)), 4) if cers else None
        for cat in EN_CATS:
            hits = []
            for it in suite[cat]["items"]:
                if it["id"] not in rs:
                    continue
                ref = re.findall(r"[a-z]+", it["text"].lower())
                hyp = set(re.findall(r"[a-z]+", dec(rs[it["id"]], "en").lower()))
                hits.append(np.mean([w in hyp for w in ref]) if ref else 1.0)
            if hits:
                sc[f"en_hit_{cat}"] = round(float(np.mean(hits)), 3)
        scores[tag] = sc
        print(tag, sc, flush=True)
    # voice identity: WavLM-sv cosine to real Shahed clips minus cosine to Rasa support voice (speaker margin)
    del asr
    gc.collect()
    torch.cuda.empty_cache()
    import glob as _g
    import librosa
    from transformers import AutoFeatureExtractor, WavLMForXVector
    fe = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")
    sv = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv").to("cuda").eval()

    def emb(paths):
        E = []
        for p in paths:
            w, sr = sf.read(p, dtype="float32")
            w = librosa.resample(w, orig_sr=sr, target_sr=16000)
            with torch.inference_mode():
                e = sv(**fe(w, sampling_rate=16000, return_tensors="pt").to("cuda")).embeddings[0].float().cpu().numpy()
            E.append(e / np.linalg.norm(e))
        return np.array(E)

    def centroid(E):
        c = E.mean(0)
        return c / np.linalg.norm(c)
    # the 40 reference clips were pre-selected (every 2nd row of Shahed's val list, first 40) into shahed_ref/
    val_wavs = [os.path.join(paths.ROOT, l.split("|")[0])
                for l in open(os.path.join(paths.EVAL, "shahed_ref", "val.txt"), encoding="utf-8") if "|" in l][:40]
    c_tgt = centroid(emb(val_wavs))
    rasa = sorted(_g.glob(os.path.join(paths.RASA_LISTS, "wavs", "*[0-9].wav")))[:40]
    c_sup = centroid(emb(rasa)) if rasa else None
    probe_cats = ("chatbot", "pure_en", "ur_word", "urdu_rare")
    for tag, rs in renders.items():
        ids = [it["id"] for c in probe_cats for it in suite[c]["items"] if it["id"] in rs]
        if not ids:
            continue
        E = emb([rs[i] for i in ids])
        sim_t = float(np.mean(E @ c_tgt))
        scores[tag]["sim_shahed"] = round(sim_t, 3)
        if c_sup is not None:
            scores[tag]["sim_rasa"] = round(float(np.mean(E @ c_sup)), 3)
            scores[tag]["voice_margin"] = round(sim_t - scores[tag]["sim_rasa"], 3)
        print(tag, "voice", {k: scores[tag].get(k) for k in ("sim_shahed", "sim_rasa", "voice_margin")}, flush=True)
    real_E = emb(val_wavs[:20])
    scores["real_shahed"] = dict(sim_shahed=round(float(np.mean(real_E @ c_tgt)), 3),
                                 sim_rasa=round(float(np.mean(real_E @ c_sup)), 3) if c_sup is not None else None)
    json.dump(scores, open(os.path.join(out, "scores.json"), "w"), indent=1)

    # listening page
    real = []
    tr = os.path.join(paths.EVAL, "shahed_ref", "test_raw.json")
    if os.path.exists(tr):
        for r in json.load(open(tr, encoding="utf-8"))[:8]:
            dst = os.path.join(out, "audio", "real__" + os.path.basename(r["wav"]))
            if not os.path.exists(dst):
                shutil.copy(os.path.join(paths.ROOT, r["wav"]), dst)
            real.append((dst, r["text"]))
    tags = list(renders)
    sec = []
    if real:
        sec.append("<section><h2>Real Sarah Shahed (held-out lecture)</h2>" + "".join(
            f"<div class=row><div class=tx dir=auto>{html.escape(t)}</div><audio controls preload=none "
            f"src='audio/{os.path.basename(p)}'></audio></div>" for p, t in real) + "</section>")
    for cat in ORDER:
        items = suite[cat]["items"][: 12 if cat == "urdu_parity" else None]
        rows = []
        for it in items:
            cells = "".join(
                f"<div class=cell><small>{html.escape(t)}</small><audio controls preload=none "
                f"src='audio/{os.path.basename(renders[t][it['id']])}'></audio></div>"
                for t in tags if it["id"] in renders[t])
            rows.append(f"<div class=row><div class=tx dir=auto>{html.escape(it['text'])}</div>"
                        f"<div class=cells>{cells}</div></div>")
        sec.append(f"<section><h2>{html.escape(suite[cat]['desc'])}</h2>{''.join(rows)}</section>")
    score_rows = "".join(f"<tr><td>{html.escape(t)}</td>" + "".join(
        f"<td>{scores[t].get(k, '')}</td>" for k in ("cer_urdu_parity", "cer_urdu_rare", "en_hit_pure_en",
                                                     "en_hit_en_names", "en_hit_en_business", "sim_shahed", "voice_margin")) + "</tr>" for t in tags)
    page = f"""<!doctype html><html><head><meta charset=utf-8><title>Shahed Model Eval</title>
<meta name=viewport content='width=device-width,initial-scale=1'><style>
:root{{--bg:#fbfaf7;--fg:#1d1d1b;--mu:#6b6b66;--ln:#e2dfd6;--cd:#fff}}
@media (prefers-color-scheme: dark){{:root{{--bg:#161614;--fg:#ecebe6;--mu:#a3a29b;--ln:#34332f;--cd:#1f1f1c}}}}
body{{background:var(--bg);color:var(--fg);font:15px/1.45 system-ui,sans-serif;max-width:1200px;margin:auto;padding:16px}}
section{{background:var(--cd);border:1px solid var(--ln);border-radius:10px;padding:12px;margin:12px 0}}
h2{{font-size:17px;margin:0 0 8px}} .row{{border-top:1px solid var(--ln);padding:8px 0}} .tx{{margin-bottom:4px}}
.cells{{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:6px}} audio{{width:100%}}
small{{color:var(--mu)}} table{{border-collapse:collapse}} td,th{{border:1px solid var(--ln);padding:4px 8px;font-size:13px}}
</style></head><body><h1>Sarah Shahed voice: evaluation</h1>
<p>Scores are automatic (large-v3). Urdu CER: lower is better (base parity reference). English hit only detects
collapse; it rewards American English and cannot judge the Pakistani accent. Your ear decides.</p>
<table><tr><th>system</th><th>Urdu CER (60 held-out)</th><th>Urdu CER rare sounds</th><th>EN hit general</th>
<th>EN hit names</th><th>EN hit business</th><th>voice sim to real Shahed</th><th>voice margin vs Rasa</th></tr>{score_rows}</table>
{''.join(sec)}</body></html>"""
    open(os.path.join(out, "index.html"), "w", encoding="utf-8").write(page)
    print("wrote", os.path.join(out, "index.html"))


if __name__ == "__main__":
    main()
