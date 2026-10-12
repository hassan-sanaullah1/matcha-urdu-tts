"""Phone recogniser + aligner used by pron_eval.py; run on its own, it checks that the training labels'
phone strings say what she actually says.

English words reach the model as phones, not letters: espeak (en-us) then EN_MAP, which is my table of
Pakistani-English choices. Urdu row B replaces some Urdu words with espeak-ur phones. If either table is wrong
for a word, every clip containing it teaches the model a wrong pronunciation -- silently, since the audio and
the label still "agree" as far as any text check can tell.

Method: a phone recogniser (wav2vec2-lv-60-espeak-cv-ft, trained on espeak IPA) transcribes each clip into
phones. For a word we expect to be somewhere in that clip, its expected phone string is aligned against the
best-matching window of the recognised string, and the per-phone substitutions are tallied over the corpus.
A phone whose expected form is usually recognised as something else is a table error, not noise.

Model used (Hugging Face cache, not copied): facebook/wav2vec2-lv-60-espeak-cv-ft.
Input:  data/2_clean/voice_clean
Output: results/phone_check/report.json (top substitutions per channel, worst words)
Usage (from the kiosk-tts folder):  python src/5_evaluate/phone_check.py [--max-clips 150]
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import paths  # noqa: E402  -- kiosk-tts/src/paths.py: every location + the import path

import argparse
import collections
import json
import re

import numpy as np
import soundfile as sf

import phones
import pk_gazetteer                                  # installs the Pakistani-English span front end

CLEAN = paths.VOICE_CLEAN
OUT = os.path.join(paths.RESULTS, "phone_check")
MODEL = "facebook/wav2vec2-lv-60-espeak-cv-ft"
LATIN = re.compile(r"[A-Za-z][A-Za-z'\-]*")
# The recogniser writes English-language espeak symbols and has no retroflex or Urdu-specific phones, so a
# label of ʈ comes back as t and aː as a. Those are spelling differences, not pronunciation errors: phones in
# the same class here are treated as a match, and what remains is a genuine mismatch worth acting on.
EQUIV = [("ʈ", "t", "ɾ", "ʔ"), ("ʈʰ", "tʰ", "th"), ("ɖ", "d", "ɽ"), ("ʋ", "v", "w"), ("r", "ɹ", "ɻ", "ɺ"),
         ("aː", "a", "ɑ", "ɑː", "ʌ", "ɐ"), ("ə", "ʌ", "ɐ", "ɜ", "ɚ"), ("ɛ", "æ", "e", "eː", "ɘ"),
         ("iː", "i", "ɪ", "ᵻ"), ("uː", "u", "ʊ"), ("oː", "o", "ɔ", "ɒ", "ɔː"), ("ɡ", "g"), ("ʧ", "tʃ"),
         ("ʤ", "dʒ"), ("ʃ", "ʂ"), ("ɴ", "n", "ŋ", "ɳ", "ñ", "m"), ("z", "ʒ", "z̠"), ("s", "sʰ"), ("h", "ɦ")]
CANON = {p: cls[0] for cls in EQUIV for p in cls}
URDU_WORD = re.compile(r"[؀-ۿ]{3,}")


def tokens(phone_str):
    """Split a phone string into symbols: our phones first (so 'tʰ' beats 't'), any other IPA char on its own.

    The recogniser emits phones Urdu does not have (ŋ, z̠, ...); keeping them as tokens is what makes a
    systematic mismatch visible instead of silently deleting the evidence.
    """
    rx = re.compile("|".join(sorted(map(re.escape, phones.PHONES), key=len, reverse=True)) + r"|\S")
    return [t for t in rx.findall(phone_str.replace(phones.STRESS, "")) if not t.isspace()]


def canon(seq):
    return [CANON.get(p, p) for p in seq]


def align(exp, got):
    """Align exp against its best-matching window of got; return (distance/len, [(exp_phone, got_phone)])."""
    n, m = len(exp), len(got)
    if not n or not m:
        return 1.0, []
    # free start/end in `got` = local alignment of the whole expected word
    D = np.zeros((n + 1, m + 1))
    D[:, 0] = np.arange(n + 1)
    bt = np.zeros((n + 1, m + 1), dtype=np.int8)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sub = D[i - 1, j - 1] + (exp[i - 1] != got[j - 1])
            dele, ins = D[i - 1, j] + 1, D[i, j - 1] + 1
            D[i, j] = min(sub, dele, ins)
            bt[i, j] = 0 if D[i, j] == sub else (1 if D[i, j] == dele else 2)
    j = int(np.argmin(D[n, 1:])) + 1
    dist = float(D[n, j]) / n
    pairs, i = [], n
    while i > 0 and j > 0:
        if bt[i, j] == 0:
            pairs.append((exp[i - 1], got[j - 1]))
            i, j = i - 1, j - 1
        elif bt[i, j] == 1:
            pairs.append((exp[i - 1], ""))
            i -= 1
        else:
            j -= 1
    return dist, pairs[::-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-clips", type=int, default=150)
    ap.add_argument("--filelist", default="", help="control mode: a Matcha filelist (path|text) to check instead")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a.filelist:
        seen_p = set()
        rows = []
        for l in open(a.filelist, encoding="utf-8"):
            path, text = l.rstrip("\n").split("|", 1)
            if path not in seen_p:
                seen_p.add(path)
                rows.append(dict(clip=path, text=text))
    else:
        rows = [json.loads(l) for l in open(os.path.join(CLEAN, "manifest.jsonl"), encoding="utf-8")]
    eng = [r for r in rows if LATIN.search(r["text"])][: a.max_clips]
    urd = [r for r in rows if not LATIN.search(r["text"])][: a.max_clips]
    print(f"{len(eng)} clips with English, {len(urd)} Urdu-only clips")

    import librosa
    import torch
    from huggingface_hub import hf_hub_download
    from transformers import AutoFeatureExtractor, Wav2Vec2ForCTC
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    fe = AutoFeatureExtractor.from_pretrained(MODEL)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL).to(dev).eval()
    # The matching tokeniser class pulls in the phonemizer library just to build a backend we never use, so
    # the ids are decoded here from the model's own vocab instead (plain greedy CTC: collapse, drop blanks).
    ids = {i: t for t, i in json.load(open(hf_hub_download(MODEL, "vocab.json"), encoding="utf-8")).items()}
    blanks = {"<pad>", "<s>", "</s>", "<unk>"}

    def recognise(clip):
        w, sr = sf.read(clip if a.filelist else os.path.join(CLEAN, "wav", clip), dtype="float32")
        w16 = librosa.resample(w, orig_sr=sr, target_sr=16000)
        with torch.inference_mode():
            logits = model(**fe(w16, sampling_rate=16000, return_tensors="pt").to(dev)).logits
        out, prev = [], None
        for i in logits.argmax(-1)[0].tolist():
            if i != prev:
                t = ids.get(i, "")
                if t not in blanks:
                    out.append(t)
            prev = i
        return " ".join(out)

    report = {}
    for name, clips in (("english", eng), ("urdu_row_b", urd)):
        subs, worst, seen = collections.Counter(), [], collections.Counter()
        per_word = collections.defaultdict(list)
        total, kept_ph = collections.Counter(), collections.Counter()
        for r in clips:
            got = tokens(recognise(r["clip"]))
            if name == "english":
                words = [w for w in LATIN.findall(r["text"])]
                exp_of = {w: pk_gazetteer._english_span_pk(w) for w in words}
            else:
                words = URDU_WORD.findall(r["text"])[:6]
                table = dict(zip(words, phones.urdu_words(words, {})))
                exp_of = {w: (table.get(w) or "") for w in words}
            for w in words:
                exp = tokens(exp_of.get(w) or "")
                if len(exp) < 3:
                    continue
                d, pairs = align(canon(exp), canon(got))
                seen[w] += 1
                subs.update((e, g) for e, g in pairs if e != g)
                total.update(e for e, _ in pairs)
                kept_ph.update(e for e, g in pairs if e == g)
                worst.append((round(d, 2), w, exp_of[w], "".join(got)[:60]))
                per_word[w].append(d)
        worst.sort(reverse=True)
        report[name] = dict(clips=len(clips), words=sum(seen.values()),
                            top_substitutions=[[f"{e} -> {g or 'DROP'}", n] for (e, g), n in subs.most_common(15)],
                            match_rate={p: [round(kept_ph[p] / n, 2), n] for p, n in total.most_common(20)},
                            worst_words=[list(x) for x in worst[:15]],
                            worst_repeated=[[round(float(np.mean(v)), 2), len(v), w]
                                            for w, v in sorted(per_word.items(), key=lambda kv: -np.mean(kv[1]))
                                            if len(v) >= 3][:15])
        print(f"\n=== {name}: {report[name]['words']} word tokens")
        print("  match rate per labelled phone:", report[name]["match_rate"])
        for s, n in report[name]["top_substitutions"]:
            print(f"  {s:20} {n}")
        print("  worst repeated words (mean distance, times seen):")
        for d, n, w in report[name]["worst_repeated"]:
            print(f"    {w:20} {d}  x{n}")
    name_out = "control.json" if a.filelist else "report.json"
    json.dump(report, open(os.path.join(OUT, name_out), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\nwrote", os.path.join(OUT, name_out))


if __name__ == "__main__":
    main()
