# kiosk-tts server — male + female Urdu voice for the Mari kiosk

A small local HTTP TTS server (FastAPI) that the **Mari Interactive Kiosk** calls over HTTP. It speaks Urdu
(with English words, names and numbers mixed in, the way the kiosk writes replies) in **two voices from one
checkpoint**:

| voice    | speaker row | source                                                        |
|----------|-------------|---------------------------------------------------------------|
| `male`   | 0           | kaani Round 29 `kb_male_v2_asis`, step 3999 (UpliftAI prime-time-anchor dataset) |
| `female` | 1           | the Lullaby narrator, carried into the same checkpoint          |

Checkpoint: [`../checkpoints/kaani_male_female/step=003999.ckpt`](../checkpoints/kaani_male_female/step=003999.ckpt)
(a copy of `kaani/pk-r29-kaani-male-voice/checkpoints/male_v2_asis/step=003999.ckpt`, md5 `5def394c…`).

Rendering is byte-identical to `kiosk-tts/src/6_speak/speak.py` and to `pk-r29 src/infer.py` for the same text,
voice and seed. The model, Vocos vocoder, NeMo number grammars and espeak are loaded and warmed **once** at
startup (~11 s), then both speaker rows share the loaded model, so switching voice costs nothing.

## Run

```bash
./run_server.sh                                   # 127.0.0.1:8030, default voice = female
HOST=172.17.0.1 ./run_server.sh                   # for the kiosk backend in Docker (see below)
PORT=8031 ./run_server.sh --vocoder-device cuda   # any app.py flag passes through
KIOSK_TTS_DEFAULT_VOICE=male ./run_server.sh      # make male the voice for bare /v1/tts calls
```

The kiosk backend runs in a container, which cannot reach the host's `127.0.0.1`; bind to the
`docker0` address (`172.17.0.1`) instead, which containers and this machine can reach but the LAN
cannot. See [MARI_INTEGRATION.md](MARI_INTEGRATION.md) for the kiosk side.

It uses the kaani venv (`/home/ubuntu/Desktop/kaani/envs/venv`, Python 3.12) with `fastapi` + `uvicorn` added
(a purely additive install — no pinned package changed). Point `PYTHON=` at another 3.12 env if you prefer.

## API (the contract Mari's `matcha` provider expects)

| method | path            | body                                     | returns |
|--------|-----------------|------------------------------------------|---------|
| GET    | `/healthz`      | —                                        | `200 {"ready": true, "voices": [...], "default_voice", ...}` |
| POST   | `/v1/tts`       | `{"text", "speed"?, "voice"?, "seed"?}`  | `audio/wav` (16-bit PCM, mono, 22050 Hz) |
| POST   | `/v1/tts/male`  | `{"text", "speed"?, "seed"?}`            | `audio/wav` — the male voice |
| POST   | `/v1/tts/female`| `{"text", "speed"?, "seed"?}`            | `audio/wav` — the female voice |
| POST   | `/v1/normalize` | `{"text"}`                               | `{"sentences": [{"text","read"}]}` — what the model will actually say |

- `voice`: `male`/`m`/`0`/`hamza` or `female`/`f`/`1`/`maryam`. Absent → `KIOSK_TTS_DEFAULT_VOICE` (default
  `female`). `/v1/tts` also accepts `?voice=male` as a query parameter.
- `speed`: 0.6–1.5, higher = faster (clamped). `seed`: int; same text+voice+speed+seed → same audio (cached).
- Audio-route response headers: `X-Voice`, `X-Read` (url-encoded), `X-Timing-Frontend-Ms`, `X-Timing-Model-Ms`,
  `X-Audio-Seconds`, `X-Cached`.

**Mari uses the server by routing, not by a request field** — stock Mari posts only `{"text","speed"}`, so a bare
`/v1/tts` call returns the default voice (`female`). To drive **both** avatars from this one server, Mari needs a
tiny change; see [MARI_INTEGRATION.md](MARI_INTEGRATION.md).

## Quick check

```bash
curl -s localhost:8030/healthz | python3 -m json.tool
curl -s -X POST localhost:8030/v1/tts/male   -H 'Content-Type: application/json' \
     -d '{"text":"السلام علیکم، MARI میں خوش آمدید۔"}' -o male.wav
curl -s -X POST localhost:8030/v1/tts/female -H 'Content-Type: application/json' \
     -d '{"text":"السلام علیکم، MARI میں خوش آمدید۔"}' -o female.wav
```
