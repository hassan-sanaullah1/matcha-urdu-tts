"""Type-and-listen web page for the shipped voice (checkpoints/voice_demuffled): type text, hear the kiosk voice.

The text is spoken exactly the way the kiosk test page speaks it:
  split into sentences -> pk_gazetteer.prepare_pk (text rules) -> the model as speaker 0, 32 steps, temperature 0.35
  -> Vocos -> sentences joined with 0.3 s of silence.

The code comes from kiosk-tts/src (nothing is written there, not even .pyc).
Python standard library only (no web framework). The page is index.html next to this file.

Usage (from the kiosk-tts folder):
  python demo/server.py                              # then open http://127.0.0.1:8024
  python demo/server.py --port 8024 --host 0.0.0.0   # reachable from other machines on the network
  python demo/server.py --ckpt checkpoints/voice/step=007999.ckpt   # hear the 8k model before the de-muffle
"""
import os
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
import paths  # noqa: E402  -- kiosk-tts/src/paths.py: every location + the import path
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import argparse
import base64
import io
import json
import logging
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import soundfile as sf
import torch

DEFAULT_CKPT = paths.VOICE
SPLIT = re.compile(r"(?<=[۔.!?؟])\s+|\n+")                 # the kiosk's sentence cut (test_page.sentences)
# ...except after an abbreviation or an initial: "Lt. Gen. Anwar", "Dr. Ayesha", "A.F. Ferguson" stay one sentence
ABBR_END = re.compile(r"(?:\b(?:Lt|Gen|Dr|Mr|Mrs|Ms|Col|Maj|Capt|Brig|Prof|Engr|Retd|No|Co|Inc|Ltd|Pvt|St|Jr|Sr|vs|"
                      r"approx|e\.g|i\.e)|(?<![A-Za-z])[A-Z])\.$")
MAX_CHARS = 3000


def sentences(text):
    out = []
    for s in SPLIT.split(text):
        s = s.strip(" -*")
        if not s:
            continue
        if out and ABBR_END.search(out[-1]):
            out[-1] += " " + s
        else:
            out.append(s)
    return out


class Voice:
    def __init__(self, ckpt, device):
        logging.disable(logging.WARNING)
        import tts
        import matcha_torch
        import pk_gazetteer
        import spk_cond
        self.tts, self.prep, self.device = tts, pk_gazetteer.prepare_pk, device
        if spk_cond.has_spk_cond(ckpt):
            spk_cond.load_for_inference(ckpt, device, spk=0)             # speaker 0 = the target voice
        self.c = matcha_torch._load(ckpt, device)
        self.gap = np.zeros(int(0.3 * tts.OUT_SR), np.float32)
        self.lock = threading.Lock()                                     # one render at a time on the GPU
        self.ckpt = ckpt
        self.speak("warm up, 12 sentences.", 1.0, 1)                     # loads NeMo and espeak once

    def speak(self, text, speed, seed):
        sents = sentences(text)
        parts, reads, t_text, t_model = [], [], 0.0, 0.0
        with self.lock:
            for k, s in enumerate(sents):
                t0 = time.time()
                prepared = self.tts.clean(self.prep(s))
                t1 = time.time()
                reads.append(prepared)
                if not prepared:
                    continue
                inter = self.tts.encode(prepared)
                torch.manual_seed(seed * 100 + k)
                with torch.inference_mode():
                    o = self.c["model"].synthesise(torch.tensor([inter], device=self.device),
                                                   torch.tensor([len(inter)], device=self.device),
                                                   n_timesteps=32, temperature=0.35, length_scale=1.0 / speed)
                    w = self.c["vocos"].decode(o["mel"].cpu()).numpy().squeeze().astype("float32")
                parts += [self.tts.normalize(w), self.gap]
                t_text += t1 - t0
                t_model += time.time() - t1
        wav = np.concatenate(parts[:-1]) if parts else np.zeros(1, np.float32)
        buf = io.BytesIO()
        sf.write(buf, wav, self.tts.OUT_SR, format="WAV")
        return dict(audio=base64.b64encode(buf.getvalue()).decode(), sentences=sents, reads=reads,
                    seconds=round(len(wav) / self.tts.OUT_SR, 2),
                    ms_text=round(t_text * 1000), ms_model=round(t_model * 1000))


class Handler(BaseHTTPRequestHandler):
    voice = None

    def _send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            return self._send(200, open(os.path.join(HERE, "index.html"), "rb").read(), "text/html; charset=utf-8")
        if self.path == "/api/info":
            return self._send(200, json.dumps({"ckpt": os.path.relpath(self.voice.ckpt, paths.ROOT)}).encode(),
                              "application/json")
        self._send(404, b"not found", "text/plain")

    def do_POST(self):
        if self.path != "/api/speak":
            return self._send(404, b"not found", "text/plain")
        try:
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            text = str(req.get("text", "")).strip()[:MAX_CHARS]
            if not text:
                return self._send(400, json.dumps({"error": "empty text"}).encode(), "application/json")
            speed = min(1.5, max(0.6, float(req.get("speed", 1.0))))
            seed = int(req.get("seed", 1))
            out = self.voice.speak(text, speed, seed)
            self._send(200, json.dumps(out, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")
        except Exception as e:                                           # show the error on the page, keep serving
            self._send(500, json.dumps({"error": f"{type(e).__name__}: {e}"}).encode(), "application/json")

    def log_message(self, fmt, *args):
        sys.stderr.write("%s %s\n" % (time.strftime("%H:%M:%S"), fmt % args))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None, help="checkpoint to serve (default: checkpoints/voice_demuffled)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8024)
    a = ap.parse_args()
    a.ckpt = a.ckpt or DEFAULT_CKPT
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"loading {a.ckpt} on {dev} ...", flush=True)
    Handler.voice = Voice(os.path.abspath(a.ckpt), dev)
    print(f"READY  http://{a.host}:{a.port}", flush=True)
    ThreadingHTTPServer((a.host, a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
