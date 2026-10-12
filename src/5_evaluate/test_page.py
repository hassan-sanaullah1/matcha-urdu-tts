"""STEP 5c -- kiosk listening page for one checkpoint: kiosk answers and every knowledge-base name, spoken the way the kiosk would.

Inputs:
  data/4_eval/kb_test_sentences.txt   category|visitor question|RAG-style answer (from the MARI knowledge base)
  data/4_eval/kb_test_names.txt       group|name (every person, place, field, company, bank and school in the knowledge base)
Answers are split into sentences, each spoken separately and joined with 0.3 s of silence (the decoder was trained on
4 s windows, so sentence-at-a-time is how the kiosk should speak long replies). Names are spoken as "جی، <name>۔".
Every item shows what the text front end turned it into, and has right / wrong buttons and a note box; the marks
download as JSON so wrong items can be traced back to the front end or the model.

Output: results/<name>/index.html (+ audio). Open it in a browser and mark each item right / wrong; the marks
        download as JSON.
Usage (from the kiosk-tts folder):
  python src/5_evaluate/test_page.py --ckpt checkpoints/voice_demuffled/model.ckpt --name voice_kb_test
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import paths  # noqa: E402  -- kiosk-tts/src/paths.py: every location + the import path

import argparse
import html
import json
import re
import subprocess

import numpy as np
import soundfile as sf
import torch

import pk_gazetteer
import spk_cond
import tts

ROOT = paths.ROOT
DEV = "cuda" if torch.cuda.is_available() else "cpu"
GAP = np.zeros(int(0.3 * tts.OUT_SR), np.float32)
SPLIT = re.compile(r"(?<=[۔.!?؟])\s+|\n+")


def sentences(text):
    return [s.strip(" -*") for s in SPLIT.split(text) if s.strip(" -*")]


def load(path, n):
    rows = []
    for l in open(os.path.join(ROOT, path), encoding="utf-8"):
        if l.startswith("#") or l.count("|") < n - 1:
            continue
        rows.append(l.rstrip("\n").replace("\\n", "\n").split("|", n - 1))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=paths.VOICE)
    ap.add_argument("--name", default="voice_kb_test")
    a = ap.parse_args()
    out = os.path.join(paths.RESULTS, a.name)
    os.makedirs(os.path.join(out, "audio"), exist_ok=True)
    import matcha_torch
    if spk_cond.has_spk_cond(a.ckpt):
        spk_cond.load_for_inference(a.ckpt, DEV, spk=0)
    c = matcha_torch._load(a.ckpt, DEV)

    def say(text, seed):
        parts = []
        for k, s in enumerate(sentences(text)):
            prepared = tts.clean(pk_gazetteer.prepare_pk(s))
            inter = tts.encode(prepared)
            torch.manual_seed(seed * 100 + k)
            with torch.inference_mode():
                o = c["model"].synthesise(torch.tensor([inter], device=DEV), torch.tensor([len(inter)], device=DEV),
                                          n_timesteps=32, temperature=0.35, length_scale=1.0)
                parts += [tts.normalize(c["vocos"].decode(o["mel"].cpu()).numpy().squeeze().astype("float32")), GAP]
        return np.concatenate(parts[:-1])

    def mp3(w, name):
        tmp = os.path.join(out, "audio", name + ".wav")
        sf.write(tmp, w, tts.OUT_SR)
        subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", tmp, "-b:a", "96k", tmp[:-4] + ".mp3"],
                       check=True)
        os.remove(tmp)
        return name + ".mp3"

    answers = load("data/4_eval/kb_test_sentences.txt", 3)
    names = load("data/4_eval/kb_test_names.txt", 2)
    ans_html = []
    for i, (cat, q, ans) in enumerate(answers):
        f = mp3(say(ans, i), f"a{i:02d}")
        reads = " | ".join(tts.clean(pk_gazetteer.prepare_pk(s)) for s in sentences(ans))
        ans_html.append(f"""<article data-k='a{i:02d}'>
<p class=q><span class=cat>{html.escape(cat)}</span> <span dir=auto>{html.escape(q)}</span></p>
<p class=a dir=auto>{html.escape(ans).replace(chr(10), '<br>')}</p>
<audio controls preload=none src='audio/{f}'></audio>
<details><summary>What the model reads</summary><p class=ph dir=auto>{html.escape(reads)}</p></details>
<div class=mk><button data-v=ok>All correct</button><button data-v=bad>Something wrong</button>
<input placeholder='Which word? (optional)' aria-label='which word'></div></article>""")
        print("answer", i, flush=True)
    groups = {}
    for j, (g, n) in enumerate(names):
        f = mp3(say(f"جی، {n}۔", 500 + j), f"n{j:03d}")
        reads = tts.clean(pk_gazetteer.prepare_pk(n))
        groups.setdefault(g, []).append(
            f"<div class=nm data-k='n{j:03d}'><b dir=auto>{html.escape(n)}</b><audio controls preload=none src='audio/{f}'></audio>"
            f"<small dir=auto title='what the model reads'>{html.escape(reads)}</small>"
            f"<div class=mk><button data-v=ok>Right</button><button data-v=bad>Wrong</button></div></div>")
    print("names done", flush=True)
    names_html = "".join(f"<h3>{html.escape(g.capitalize())}</h3><div class=grid>{''.join(v)}</div>" for g, v in groups.items())
    doc = f"""<!doctype html><html lang=en><head><meta charset=utf-8><title>Kiosk Accuracy Test</title>
<meta name=viewport content='width=device-width,initial-scale=1'>
<link rel=stylesheet href='https://fonts.googleapis.com/css2?family=Noto+Nastaliq+Urdu&family=Noto+Sans:wght@400;600&display=swap'>
<style>
:root{{--bg:#f6f7f5;--fg:#1b2320;--mu:#5b6661;--ln:#dde2de;--cd:#fff;--ac:#0e7c66;--ok:#2f7d32;--bad:#b4541e}}
@media (prefers-color-scheme: dark){{:root{{--bg:#121614;--fg:#e6eae7;--mu:#9aa59f;--ln:#2c3531;--cd:#1a201d;--ac:#4fc3a5;--ok:#7bc47f;--bad:#e89464;color-scheme:dark}}}}
body{{background:var(--bg);color:var(--fg);font:15px/1.6 'Noto Sans',system-ui,sans-serif;max-width:1100px;margin:auto;padding:16px}}
[dir=auto],.a,.q{{font-family:'Noto Sans','Noto Nastaliq Urdu',system-ui,sans-serif}}
article{{background:var(--cd);border:1px solid var(--ln);border-radius:8px;padding:12px 14px;margin:10px 0}}
.q{{margin:0;color:var(--mu)}} .a{{margin:6px 0 8px;font-size:16px;line-height:2}} .cat{{font-size:11px;letter-spacing:.06em;text-transform:uppercase;border:1px solid var(--ln);border-radius:99px;padding:1px 8px}}
audio{{width:100%;max-width:420px;height:36px}} details{{margin:6px 0;color:var(--mu);font-size:13px}} .ph{{font-family:monospace;font-size:13px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:8px}}
.nm{{background:var(--cd);border:1px solid var(--ln);border-radius:8px;padding:8px;display:flex;flex-direction:column;gap:4px}} .nm small{{color:var(--mu);font-size:12px}}
.mk{{display:flex;gap:6px;flex-wrap:wrap;align-items:center}} .mk input{{flex:1;min-width:160px;padding:5px;border:1px solid var(--ln);border-radius:6px;background:var(--bg);color:var(--fg)}}
button{{border:1px solid var(--ln);background:var(--bg);color:var(--fg);border-radius:6px;padding:4px 10px;cursor:pointer}}
button.on[data-v=ok]{{background:var(--ok);color:#fff;border-color:var(--ok)}} button.on[data-v=bad]{{background:var(--bad);color:#fff;border-color:var(--bad)}}
.bar{{position:sticky;top:0;background:var(--bg);padding:8px 0;border-bottom:1px solid var(--ln);display:flex;gap:12px;align-items:center;z-index:2}}
.dl{{background:var(--ac);color:#fff;border-color:var(--ac)}} h2{{margin-top:28px}}</style></head><body>
<h1>Kiosk accuracy test: {html.escape(os.path.basename(os.path.dirname(a.ckpt)))}</h1>
<p>Answers written the way the kiosk's RAG chatbot answers from the MARI knowledge base: heavy code-switching, acronyms,
Pakistani names, numbers. Then every name in the knowledge base on its own. Mark each item; for a wrong answer, type the word
that was wrong. "What the model reads" shows the text front end's output, so a wrong word can be traced to the text rules
or to the voice model. Then press Download.</p>
<div class=bar><button class=dl onclick=dl()>Download marks</button><span id=cnt></span></div>
<h2>Kiosk answers</h2>{''.join(ans_html)}
<h2>Every name in the knowledge base</h2>{names_html}
<script>
const K='kiosk_test_{a.name}';function L(){{try{{return JSON.parse(localStorage.getItem(K)||'{{}}')}}catch(e){{return {{}}}}}}
function S(o){{try{{localStorage.setItem(K,JSON.stringify(o))}}catch(e){{}}}}
function paint(){{const o=L();let ok=0,bad=0;document.querySelectorAll('[data-k]').forEach(el=>{{const m=o[el.dataset.k]||{{}};
el.querySelectorAll('button[data-v]').forEach(b=>b.classList.toggle('on',m.v===b.dataset.v));const i=el.querySelector('input');if(i&&m.note!==undefined)i.value=m.note;
if(m.v==='ok')ok++;if(m.v==='bad')bad++}});document.getElementById('cnt').textContent=ok+' right, '+bad+' wrong'}}
document.querySelectorAll('[data-k]').forEach(el=>{{el.querySelectorAll('button[data-v]').forEach(b=>b.addEventListener('click',()=>{{const o=L();
o[el.dataset.k]={{...(o[el.dataset.k]||{{}}),v:b.dataset.v}};S(o);paint()}}));const i=el.querySelector('input');if(i)i.addEventListener('input',()=>{{const o=L();
o[el.dataset.k]={{...(o[el.dataset.k]||{{}}),note:i.value}};S(o)}})}});paint();
function dl(){{const b=new Blob([JSON.stringify(L(),null,1)],{{type:'application/json'}});const x=document.createElement('a');x.href=URL.createObjectURL(b);x.download='kiosk_test_marks.json';x.click()}}
</script></body></html>"""
    open(os.path.join(out, "index.html"), "w", encoding="utf-8").write(doc)
    print("TEST_PAGE_DONE", out)


if __name__ == "__main__":
    main()
