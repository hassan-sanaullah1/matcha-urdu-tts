"""STEP 5a -- the main accuracy score: a pronunciation test that hears vowels (Whisper does not: it accepts "cold" for "cooled").

Each item is rendered, and the model's own alignment (predicted durations) says where each target word sits in
the audio. Only that stretch (+-80 ms) is decoded by a phone recogniser (wav2vec2-lv-60-espeak-cv-ft, the one in
5_evaluate/phone_check.py), and the expected phones are aligned against it with phone_check's equivalence classes
(spelling differences such as t/ʈ are not errors; uː vs oː, aːɪ vs iː are).

Groups (MARI first, general-capable):
  PRIMARY  kb_word   English words of the MARI knowledge base, three per carrier sentence
           kb_acr    the knowledge base's acronyms, one per carrier
           kb_sent   knowledge-base sentences (English, code-switched, showcase); every English word scored in context
  GUARD    pair      minimal pairs (cooled/cold, sky/ski, please/police ...): a word counts as right when its own
                     phones fit the heard stretch better than its partner's
           gen_word  words of the general open suite (everyday English, rare words), three per carrier
           gen_acr   general acronyms
  every word is also flagged unseen (never in the target voice's transcripts) so generalisation is reported separately.
Urdu is guarded by eval_target.py (Urdu character error), not here: scoring Urdu against espeak's Urdu phone
labels would test the labels as much as the model.

A word passes when its alignment distance (edits / expected phones) is <= PASS_D. Renders are seed-paired
(seed = item index), so two checkpoints are compared on identical noise.

The test set is frozen in data/4_eval/pron_suite.json (3,277 renders; the shipped model's scores are the
paired numbers in docs/training_report.md). --rebuild regenerates it from the MARI knowledge base
(data/4_eval/knowledge_base/), make_open_suite.py's word lists and the target voice's transcripts, writing
data/4_eval/pron_suite_rebuild.json, never the frozen file. The expected sounds come from today's text front
end, so compare checkpoints only on the same suite.

Model used (Hugging Face cache, not copied): facebook/wav2vec2-lv-60-espeak-cv-ft.
Output: results/pron_<name>/scores.json, words.jsonl, index.html (+ audio of the pairs and the worst words)
        The shipped voice's result is in results/pron_voice_demuffled/.
Usage (from the kiosk-tts folder):
  python src/5_evaluate/pron_eval.py --name voice_demuffled --ckpt checkpoints/voice_demuffled/model.ckpt
  python src/5_evaluate/pron_eval.py --name my_run --ckpt checkpoints/my_run/step=007999.ckpt
  python src/5_evaluate/pron_eval.py --rebuild --build-only                 # just regenerate the suite
  python src/5_evaluate/pron_eval.py --compare voice_demuffled,my_run     # paired differences, no GPU
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import paths  # noqa: E402  -- kiosk-tts/src/paths.py: every location + the import path

import argparse
import collections
import html
import json
import random
import re

import numpy as np

import pk_gazetteer
import tts
from phone_check import MODEL, align, canon, tokens

ROOT = paths.ROOT
KB_DIR = os.path.join(paths.EVAL, "knowledge_base")
KB = os.path.join(KB_DIR, "mari_energies_knowledge_base.md")      # the kiosk's knowledge base (the MARI text)
SUITE = os.path.join(paths.EVAL, "pron_suite.json")                # frozen: the shared suite every checkpoint is scored on
SUITE_REBUILD = os.path.join(paths.EVAL, "pron_suite_rebuild.json")  # what --rebuild writes
HOP_S = 256 / 22050
PASS_D = 0.25
PAD_S = 0.08
URDU = re.compile(r"[؀-ۿ]")
LATIN = re.compile(r"[A-Za-z][A-Za-z'\-]*[A-Za-z]|[A-Za-z]")
CARRIER_1 = "اگلا لفظ {} ہے۔"
CARRIER_3 = "یہ الفاظ سنیں: {}، {}، اور {}۔"
PRIMARY = ("kb_word", "kb_acr", "kb_sent")
GUARD = ("pair", "gen_word", "gen_acr")

PAIRS = [("cooled", "cold"), ("cool", "coal"), ("pool", "pole"), ("fool", "foal"), ("rule", "role"), ("tool", "toll"),
         ("boot", "boat"), ("moon", "moan"), ("shoe", "show"), ("two", "toe"), ("blue", "blow"), ("food", "fold"),
         ("full", "fall"), ("sky", "ski"), ("time", "team"), ("bite", "beat"), ("like", "leak"), ("mile", "meal"),
         ("fine", "fin"), ("price", "prize"), ("please", "police"), ("plan", "plane"), ("man", "main"),
         ("cut", "cat"), ("cup", "cap"), ("hot", "hat"), ("cot", "cut"), ("port", "part"), ("form", "farm"),
         ("load", "loud"), ("now", "no"), ("house", "hose"), ("down", "done"), ("bird", "bed"), ("work", "walk"),
         ("very", "berry"), ("thin", "tin"), ("three", "tree"), ("sell", "shell"), ("save", "shave"),
         ("jet", "yet"), ("chip", "ship"), ("fan", "pan"), ("feel", "peel"), ("zoo", "sue"), ("rice", "lice"),
         ("light", "right"), ("glass", "grass"), ("cloud", "crowd"), ("fly", "fry"), ("sport", "port"),
         ("stop", "top"), ("spin", "pin"), ("school", "cool"), ("small", "mall"), ("skill", "kill"),
         ("play", "pay"), ("grow", "go"), ("flight", "fight"), ("clean", "keen"), ("bread", "bed"),
         ("coat", "caught"), ("boy", "buy"), ("oil", "all"), ("near", "knee"), ("hair", "her")]


# ---------------------------------------------------------------- suite
def expected(word):
    """The phones a word should come out as: its front-end symbols, Urdu-script parts through espeak-ur."""
    ph = pk_gazetteer._phones()
    out = []
    for t in tts.clean(pk_gazetteer.prepare_pk(word)).split():
        if URDU.search(t):
            out.append(ph.urdu_words([t], {})[0] or "")
        else:
            out.append(t)
    return " ".join(out)


def is_phones(word):
    """True if the front end sends this word as English sound symbols (not a name turned into Urdu script)."""
    p = tts.clean(pk_gazetteer.prepare_pk(word))
    return bool(p) and not URDU.search(p)


def kb_sentences(n=40):
    import text_norm
    rnd = random.Random(0)
    out = []
    for line in open(KB, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith(("#", "|", "---")):
            continue
        line = text_norm.strip_markup(line) if hasattr(text_norm, "strip_markup") else line
        for s in re.split(r"(?<=[.!?])\s+", line):
            words = s.split()
            if 8 <= len(words) <= 25 and len(LATIN.findall(s)) >= 6 and not re.search(r"https?://|@", s):
                out.append(s.strip())
    out = sorted(set(out))
    rnd.shuffle(out)
    return out[:n]


def build_suite():
    import make_open_suite as mos
    # English words the target voice actually said in training (the "unseen" flag, and to skip them as rare words).
    # Read from the gate's clean manifest; empty if it is not present.
    seen_words = set()
    manifest = os.path.join(paths.VOICE_CLEAN, "manifest.jsonl")
    if os.path.exists(manifest):
        for l in open(manifest, encoding="utf-8"):
            seen_words |= {w.lower() for w in LATIN.findall(json.loads(l)["text"])}
    items = []

    def add(group, text, targets, **kw):
        items.append(dict(id=f"{group}_{sum(i['group'] == group for i in items):04d}", group=group, text=text,
                          targets=[dict(word=w, exp=expected(w), unseen=w.lower() not in seen_words) for w in targets],
                          **kw))

    for a, b in PAIRS:
        ea, eb = expected(a), expected(b)
        if canon(tokens(ea)) == canon(tokens(eb)):
            continue                                   # the recogniser cannot tell these apart: not a test
        add("pair", CARRIER_1.format(a), [a], rival=eb)
        add("pair", CARRIER_1.format(b), [b], rival=ea)

    kb_text = open(KB, encoding="utf-8").read()
    acr = [a for a in mos.kb_acronyms()]
    kb_w = sorted({w for w in LATIN.findall(kb_text) if len(w) >= 3 and not w.isupper() and w not in acr},
                  key=str.lower)
    seen, kb_words = set(), []
    for w in kb_w:
        if w.lower() not in seen and is_phones(w):
            seen.add(w.lower())
            kb_words.append(w)
    gen_text = " ".join(mos.GEN_EN + mos.GEN_CS)
    gen_words, gseen = [], set(seen)
    for w in LATIN.findall(gen_text):
        if len(w) >= 3 and not w.isupper() and w.lower() not in gseen and is_phones(w):
            gseen.add(w.lower())
            gen_words.append(w)
    # rare dictionary words, as in the open suite (8-14 letters, never said by her)
    rnd = random.Random(1)
    dict_w = [w.strip() for w in open(paths.DICT_US, encoding="utf-8", errors="ignore")
              if w.strip().isalpha() and w.strip().islower() and 5 <= len(w.strip()) <= 12]
    gen_words += [w for w in rnd.sample(dict_w, 400) if w not in seen_words and w not in gseen][:120]

    for group, words in (("kb_word", kb_words), ("gen_word", gen_words)):
        for i in range(0, len(words) - 2, 3):
            add(group, CARRIER_3.format(*words[i:i + 3]), words[i:i + 3])
    for group, words in (("kb_acr", acr), ("gen_acr", mos.ACR_GEN)):
        for w in words:
            add(group, CARRIER_1.format(w), [w])
    showcase = [l.split(":", 1)[1].strip() for l in open(os.path.join(KB_DIR, "showcase_texts.txt"),
                                                          encoding="utf-8") if ":" in l]
    showcase += [open(os.path.join(KB_DIR, "showcase_maryam_sky47_v3.txt"), encoding="utf-8").read().strip()]
    for s in kb_sentences() + mos.MARI_EN + mos.MARI_CS + showcase:
        for part in re.split(r"(?<=[۔.!?])\s+", s):
            words = [w for w in LATIN.findall(part) if len(w) >= 2 and not w.isupper() and is_phones(w)]
            if words:
                add("kb_sent", part, words)
    json.dump(items, open(SUITE_REBUILD, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("wrote", SUITE_REBUILD)
    c = collections.Counter(i["group"] for i in items)
    t = collections.Counter()
    for i in items:
        t[i["group"]] += len(i["targets"])
    print("suite:", {g: f"{c[g]} renders / {t[g]} words" for g in c})
    return items


# ---------------------------------------------------------------- scoring
class Recogniser:
    def __init__(self, dev):
        import torch
        from huggingface_hub import hf_hub_download
        from transformers import AutoFeatureExtractor, Wav2Vec2ForCTC
        self.torch, self.dev = torch, dev
        self.fe = AutoFeatureExtractor.from_pretrained(MODEL)
        self.m = Wav2Vec2ForCTC.from_pretrained(MODEL).to(dev).eval()
        self.ids = {i: t for t, i in json.load(open(hf_hub_download(MODEL, "vocab.json"), encoding="utf-8")).items()}
        self.blank = {"<pad>", "<s>", "</s>", "<unk>"}

    def frames(self, w16):
        with self.torch.inference_mode():
            return self.m(**self.fe(w16, sampling_rate=16000, return_tensors="pt").to(self.dev)).logits[0].argmax(-1).tolist()

    def decode(self, ids, t0, t1, n_total, dur_s):
        a, b = int(max(0, t0) / dur_s * n_total), int(min(dur_s, t1) / dur_s * n_total) + 1
        out, prev = [], None
        for i in ids[a:b]:
            if i != prev:
                t = self.ids.get(i, "")
                if t not in self.blank:
                    out.append(t)
            prev = i
        return " ".join(out)


def word_spans(prepared, dur_frames):
    """Start/end seconds of each space-separated word of `prepared`, from the model's per-token durations."""
    cum = np.concatenate([[0], np.cumsum(dur_frames)]) * HOP_S
    spans, pos = [], 0
    for w in prepared.split(" "):
        spans.append((w, cum[2 * pos], cum[min(len(cum) - 1, 2 * (pos + len(w)) + 1)]))
        pos += len(w) + 1
    return spans


def locate(target_prepared, spans):
    """The span covering the target's prepared symbols (may be several words, e.g. an acronym's letter names)."""
    def key(w):                                       # in a sentence, function words lose their stress mark
        return w.replace("ˈ", "").replace("ˌ", "").strip("،,.۔!?؟:;")
    tw = [key(w) for w in target_prepared.split(" ")]
    words = [key(s[0]) for s in spans]
    for i in range(len(words) - len(tw) + 1):
        if words[i:i + len(tw)] == tw:
            return spans[i][1], spans[i + len(tw) - 1][2]
    return None


def score_ckpt(name, ckpt, items):
    import librosa
    import matcha_torch
    import soundfile as sf
    import spk_cond
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = os.path.join(paths.RESULTS, f"pron_{name}")
    os.makedirs(os.path.join(out_dir, "audio"), exist_ok=True)
    if spk_cond.has_spk_cond(ckpt):
        spk_cond.load_for_inference(ckpt, dev, spk=0)
    c = matcha_torch._load(ckpt, dev)
    rec = Recogniser(dev)
    rows = []
    for n, it in enumerate(items):
        prepared = tts.clean(pk_gazetteer.prepare_pk(it["text"]))
        inter = tts.encode(prepared)
        torch.manual_seed(n)
        with torch.inference_mode():
            o = c["model"].synthesise(torch.tensor([inter], device=dev), torch.tensor([len(inter)], device=dev),
                                      n_timesteps=32, temperature=0.35, length_scale=1.0)
            w = c["vocos"].decode(o["mel"].cpu()).numpy().squeeze().astype("float32")
        dur = o["attn"][0].sum(-1).squeeze().cpu().numpy()
        spans = word_spans(prepared, dur)
        w16 = librosa.resample(w, orig_sr=tts.OUT_SR, target_sr=16000)
        ids = rec.frames(w16)
        dur_s = len(w16) / 16000
        keep_audio = it["group"] == "pair"
        for t in it["targets"]:
            tp = tts.clean(pk_gazetteer.prepare_pk(t["word"]))
            sp = locate(tp, spans)
            if sp is None:
                rows.append(dict(id=it["id"], group=it["group"], word=t["word"], unseen=t["unseen"], d=None,
                                 why="not_located"))
                continue
            heard = rec.decode(ids, sp[0] - PAD_S, sp[1] + PAD_S, len(ids), dur_s)
            got = canon(tokens(heard))
            d, _ = align(canon(tokens(t["exp"])), got)
            r = dict(id=it["id"], group=it["group"], word=t["word"], unseen=t["unseen"], d=round(d, 3),
                     exp=t["exp"], heard=heard.replace(" ", ""), t0=round(sp[0], 2), t1=round(sp[1], 2))
            if it.get("rival"):
                dr, _ = align(canon(tokens(it["rival"])), got)
                r["rival_d"] = round(dr, 3)
                r["pair_ok"] = d < dr
            rows.append(r)
        if keep_audio or it["group"] in PRIMARY:
            sf.write(os.path.join(out_dir, "audio", it["id"] + ".wav"), tts.normalize(w), tts.OUT_SR)
        if (n + 1) % 200 == 0:
            print(f"  {n + 1}/{len(items)}", flush=True)
    with open(os.path.join(out_dir, "words.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary = summarise(rows)
    json.dump(summary, open(os.path.join(out_dir, "scores.json"), "w"), indent=1)
    page(name, rows, summary, out_dir)
    print(json.dumps(summary, indent=1))
    return rows


def summarise(rows):
    s = {}
    for g in PRIMARY + GUARD:
        r = [x for x in rows if x["group"] == g and x["d"] is not None]
        if not r:
            continue
        e = dict(words=len(r), pass_rate=round(float(np.mean([x["d"] <= PASS_D for x in r])), 3),
                 mean_d=round(float(np.mean([x["d"] for x in r])), 3),
                 not_located=sum(1 for x in rows if x["group"] == g and x["d"] is None))
        u = [x for x in r if x["unseen"]]
        if u:
            e["unseen_pass_rate"] = round(float(np.mean([x["d"] <= PASS_D for x in u])), 3)
            e["unseen_words"] = len(u)
        if g == "pair":
            e["pair_accuracy"] = round(float(np.mean([x["pair_ok"] for x in r])), 3)
        s[g] = e
    return s


def page(name, rows, summary, out_dir):
    import subprocess
    for f in os.listdir(os.path.join(out_dir, "audio")):
        if f.endswith(".wav"):
            src = os.path.join(out_dir, "audio", f)
            subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", src, "-b:a", "64k",
                            src[:-4] + ".mp3"], check=True)
            os.remove(src)
    tbl = "".join(f"<tr><td>{g}</td><td>{'primary' if g in PRIMARY else 'guard'}</td><td>{v['words']}</td>"
                  f"<td>{v['pass_rate']:.1%}</td><td>{v.get('pair_accuracy', '')}</td><td>{v.get('unseen_pass_rate', '')}</td></tr>"
                  for g, v in summary.items())

    def row(x):
        a = f"<audio controls preload=none src='audio/{x['id']}.mp3'></audio>" \
            if os.path.exists(os.path.join(out_dir, "audio", x["id"] + ".mp3")) else ""
        extra = f" · rival d {x['rival_d']}" + (" ✓" if x.get("pair_ok") else " ✗") if "rival_d" in x else ""
        return (f"<tr><td><b>{html.escape(x['word'])}</b></td><td><code>{html.escape(x.get('exp', ''))}</code></td>"
                f"<td><code>{html.escape(x.get('heard', ''))}</code></td><td>{x['d']}{extra}</td><td>{a}</td></tr>")
    pairs = [x for x in rows if x["group"] == "pair" and x["d"] is not None]
    worst = sorted([x for x in rows if x["group"] in PRIMARY and x["d"] is not None], key=lambda x: -x["d"])[:60]
    doc = f"""<!doctype html><html><head><meta charset=utf-8><title>Pronunciation Test</title>
<meta name=viewport content='width=device-width,initial-scale=1'><style>
:root{{--bg:#fbfaf7;--fg:#1d1d1b;--mu:#6b6b66;--ln:#e2dfd6}}
@media (prefers-color-scheme: dark){{:root:not([data-theme=light]){{--bg:#161614;--fg:#ecebe6;--mu:#a3a29b;--ln:#34332f}}}}
body{{background:var(--bg);color:var(--fg);font:15px/1.45 system-ui,sans-serif;max-width:1100px;margin:auto;padding:16px}}
table{{border-collapse:collapse;width:100%}} td,th{{border-bottom:1px solid var(--ln);padding:4px 6px;text-align:left;font-size:14px}}
code{{font-size:13px}} audio{{height:30px;max-width:220px}} .wrap{{overflow-x:auto}}</style></head><body>
<h1>Pronunciation test: {html.escape(name)}</h1>
<p>A phone recogniser hears each target word (found by the model's own timing). d = share of expected sounds heard wrong;
a word passes at d ≤ {PASS_D}. Pairs: right when the word fits better than its partner (cooled vs cold).</p>
<div class=wrap><table><tr><th>group</th><th>role</th><th>words</th><th>pass</th><th>pair accuracy</th><th>unseen pass</th></tr>{tbl}</table></div>
<h2>Minimal pairs</h2><div class=wrap><table><tr><th>word</th><th>expected</th><th>heard</th><th>d</th><th></th></tr>{''.join(row(x) for x in pairs)}</table></div>
<h2>Worst knowledge-base words</h2><div class=wrap><table><tr><th>word</th><th>expected</th><th>heard</th><th>d</th><th></th></tr>{''.join(row(x) for x in worst)}</table></div>
</body></html>"""
    open(os.path.join(out_dir, "index.html"), "w", encoding="utf-8").write(doc)


def compare(names):
    """Paired comparison on identical items and seeds: mean change in d with a bootstrap 95% interval."""
    A, B = [{(r["id"], r["word"]): r for r in map(json.loads, open(os.path.join(paths.RESULTS, f"pron_{n}", "words.jsonl")))}
            for n in names]
    rng = np.random.default_rng(0)
    for g in PRIMARY + GUARD:
        keys = [k for k in A if k in B and A[k]["group"] == g and A[k]["d"] is not None and B[k]["d"] is not None]
        if not keys:
            continue
        diff = np.array([B[k]["d"] - A[k]["d"] for k in keys])
        boot = [rng.choice(diff, len(diff)).mean() for _ in range(2000)]
        pa = np.mean([A[k]["d"] <= PASS_D for k in keys])
        pb = np.mean([B[k]["d"] <= PASS_D for k in keys])
        print(f"{g:9s} n={len(keys):5d} pass {pa:.1%} -> {pb:.1%}   mean d change {diff.mean():+.3f} "
              f"[{np.percentile(boot, 2.5):+.3f}, {np.percentile(boot, 97.5):+.3f}]  ({'primary' if g in PRIMARY else 'guard'})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name")
    ap.add_argument("--ckpt")
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--build-only", action="store_true")
    ap.add_argument("--compare", default="")
    a = ap.parse_args()
    if a.compare:
        compare(a.compare.split(","))
        return
    items = build_suite() if a.rebuild else json.load(open(SUITE, encoding="utf-8"))
    if a.build_only:
        return
    score_ckpt(a.name, a.ckpt, items)


if __name__ == "__main__":
    main()
