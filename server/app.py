"""FastAPI TTS server for the Mari Interactive Kiosk -- the kaani male + female Urdu voices, each its own
checkpoint (see engine.py's docstring for why), both loaded once and kept warm. Speaks Urdu with English
words/names/numbers mixed in (the kiosk's own text) -- English-only TTS is served separately, by mari-tts-server.

It implements the contract Mari's built-in "matcha" TTS provider already expects, so Mari reaches it over
HTTP with NO code changes for the FEMALE presenter (see MARI_INTEGRATION.md for the male presenter):

  GET  /healthz                      -> 200 {"ready": true, "voices": [...], "default_voice", ...}
  POST /v1/tts   {"text", "speed"?, "voice"?, "seed"?}  -> audio/wav  (16-bit PCM, mono, 22050 Hz)
  POST /v1/tts/male      {"text", "speed"?, "seed"?}    -> the MALE voice   (ignores any "voice" in the body)
  POST /v1/tts/female    {"text", "speed"?, "seed"?}    -> the FEMALE voice
  POST /v1/normalize {"text"}        -> {"sentences": [{"text","read"}]}   (what the model will actually say)

  "voice" accepts: male|m|0|hamza  or  female|f|1|maryam. Absent -> KIOSK_TTS_DEFAULT_VOICE (default "female").
  "speed": 0.6-1.5, higher = faster (clamped). "seed": int, same text+voice+speed+seed -> same audio.
  /v1/tts also accepts ?voice=male as a query parameter (handy when the caller cannot add a body field).

Response headers on the audio routes: X-Voice, X-Read (url-encoded, " | " between sentences, first 600 chars),
X-Timing-Frontend-Ms, X-Timing-Model-Ms, X-Audio-Seconds, X-Cached.

Run:
  ./run_server.sh                                     # 127.0.0.1:8030 with kaani/envs/venv
  python app.py --host 127.0.0.1 --port 8030 [--vocoder-device cuda]
"""
import os
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))

import argparse
import time
import urllib.parse

import uvicorn
from fastapi import FastAPI, Query, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

import engine

READ_HEADER_CHARS = 600         # X-Read is cut here (url-encoded Urdu is ~6 bytes/char; httpx allows 16 KB headers)

TTS: engine.KaaniTTS | None = None
INFO: dict = {}


class TTSRequest(BaseModel):
    text: str = ""
    speed: float = 1.0
    seed: int = 1
    voice: str | None = None


class NormalizeRequest(BaseModel):
    text: str = ""


app = FastAPI(title="kaani kiosk TTS", docs_url=None, redoc_url=None)


def _validate(req: TTSRequest):
    """-> cleaned text, or a JSONResponse(400) to return instead."""
    text = (req.text or "").strip()
    if not text:
        return None, JSONResponse({"error": "empty text"}, status_code=400)
    if len(text) > engine.MAX_CHARS:
        return None, JSONResponse({"error": f"text longer than {engine.MAX_CHARS} characters"}, status_code=400)
    return text, None


def _audio_response(text, voice, speed, seed):
    name, _ = engine.resolve_voice(voice)
    audio, parts = TTS.synth(text, voice=voice, speed=speed, seed=seed)
    body = engine.wav_bytes(audio)
    headers = {
        "X-Voice": name,
        "X-Read": urllib.parse.quote(" | ".join(p["read"] for p in parts)[:READ_HEADER_CHARS]),
        "X-Timing-Frontend-Ms": str(round(sum(p["ms_text"] for p in parts))),
        "X-Timing-Model-Ms": str(round(sum(p["ms_model"] for p in parts))),
        "X-Audio-Seconds": str(round(len(audio) / engine.SR, 2)),
        "X-Cached": str(int(all(p["cached"] for p in parts if p["audio"] is not None))),
        "Cache-Control": "no-store",
    }
    return Response(content=body, media_type="audio/wav", headers=headers)


@app.get("/healthz")
def healthz():
    return {"ready": TTS is not None, **INFO}


@app.post("/v1/normalize")
def normalize(req: NormalizeRequest):
    text, err = _validate(TTSRequest(text=req.text))
    if err is not None:
        return err
    return {"sentences": [{"text": s, "read": TTS.read(s)} for s in engine.split(text)]}


@app.post("/v1/tts")
def tts_route(req: TTSRequest, voice: str | None = Query(default=None)):
    text, err = _validate(req)
    if err is not None:
        return err
    return _audio_response(text, voice or req.voice, req.speed, req.seed)


@app.post("/v1/tts/male")
def tts_male(req: TTSRequest):
    text, err = _validate(req)
    if err is not None:
        return err
    return _audio_response(text, "male", req.speed, req.seed)


@app.post("/v1/tts/female")
def tts_female(req: TTSRequest):
    text, err = _validate(req)
    if err is not None:
        return err
    return _audio_response(text, "female", req.speed, req.seed)


def main():
    global TTS, INFO
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8030)
    ap.add_argument("--device", default=None, help="model device (default: cuda if available)")
    ap.add_argument("--vocoder-device", default="cpu",
                    help="cpu (default) or cuda (~38 ms faster per sentence, slightly different audio)")
    a = ap.parse_args()
    t0 = time.time()
    import torch
    print("loading the MatchaTTS male and female voices (separate checkpoints) ...", flush=True)
    TTS = engine.KaaniTTS(device=a.device, vocoder_device=a.vocoder_device)
    INFO.update({
        "voices": sorted(engine.VOICES),
        "default_voice": engine.DEFAULT_VOICE if engine.DEFAULT_VOICE in engine.VOICES else "female",
        "sample_rate": engine.SR,
        "device": TTS.device,
        "vocoder_device": a.vocoder_device,
        "warm_up_s": TTS.warm_up_s,
        "checkpoint": {name: os.path.relpath(p, os.path.dirname(HERE)) for name, p in TTS.ckpts.items()},
        "cuda": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    })
    print(f"READY in {time.time() - t0:.1f} s  http://{a.host}:{a.port}  voices={sorted(engine.VOICES)} "
          f"default={INFO['default_voice']}", flush=True)
    # One worker: the model lives in this process and renders one sentence at a time (GPU lock in the engine).
    uvicorn.run(app, host=a.host, port=a.port, log_level="info")


if __name__ == "__main__":
    main()
