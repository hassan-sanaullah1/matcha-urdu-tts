# How the Mari kiosk and this TTS server fit together

Context primer for a new chat working in `~/Desktop/kiosk-tts`. Written 2026-10-19, after the kiosk was
wired to this server. Read this first; the two sibling docs are the detail:
`server/MARI_INTEGRATION.md` (how to run and configure) and `server/MARI_CHANGES.md` (every edit made
inside the kiosk repo, and how to revert it).

---

## 1. The two processes

Two separate programs on one machine, talking over plain HTTP. No shared code, no shared Python env.

```
  browser :3000  ──ws──►  Mari backend :8010            ──HTTP──►  this TTS server :8030
  (Next.js)                (FastAPI, in Docker)                     (FastAPI, on the host, owns the GPU)
                                  │
                                  └──HTTPS──►  UpliftAI cloud TTS (paid; all English, and Urdu fallback)
```

- **Mari kiosk**: `~/Desktop/Mari-Interactive-Kiosk-main`, run with `make dev` (frontend `:3000`,
  backend `:8010`). It is the product: STT, LLM, retrieval, avatar lipsync, and TTS routing. It also needs
  Qdrant, an LLM endpoint and Soniox for Urdu STT.
- **This repo**: a Matcha-TTS fine-tuning repo whose `server/` folder exposes the trained voice over HTTP.
  Only `server/` matters to the kiosk; `src/`, `data/` and the training pipeline are irrelevant at serve time.

The kiosk treats this server as just another TTS provider, exactly like the paid one. It is addressed only by
a URL in the kiosk's `.env`.

## 2. This server

Start it (it must be running before the kiosk speaks Urdu locally):

```bash
HOST=172.17.0.1 ~/Desktop/kiosk-tts/server/run_server.sh
```

**`HOST` matters.** The kiosk backend runs inside Docker, and a container cannot reach the host's
`127.0.0.1`. `172.17.0.1` is the host's `docker0` address: reachable from containers and from the host
itself, not from the LAN. For a kiosk backend running directly on the host, the plain `run_server.sh`
(127.0.0.1) is right instead — and the kiosk's `APP_MATCHA_TTS_URL` must match whichever you choose.

| | |
|---|---|
| code | `server/app.py` (routes), `server/engine.py` (model), `server/run_server.sh` (launcher) |
| env | `/home/ubuntu/Desktop/kaani/envs/venv` (override with `PYTHON=`); `HF_HUB_OFFLINE=1` is set in code |
| checkpoint | `checkpoints/kaani_male_female/step=003999.ckpt`, md5 `5def394c…` — **one** checkpoint holding **two** speaker rows: 0 = male, 1 = female |
| provenance | the male rows were fine-tuned from the shipped female checkpoint with the text encoder and the speaker mel-prior frozen, so the female voice and the pronunciation came through unchanged. Measured through the kiosk's own code: female ≈ 215 Hz, male ≈ 123 Hz |
| audio out | 16-bit PCM mono **22050 Hz** WAV |
| inference | 32 ODE steps, temperature 0.35, `length_scale = 1 / speed` |
| startup | loads the model, the number grammars and the lexicons, then renders warm-up lines **for both rows** (~8–13 s). `/healthz` reports `warm_up_s` |
| concurrency | **one sentence at a time**: a `threading.Lock` around text and GPU work, and a single uvicorn worker by design |
| caching | two 256-entry LRUs: normalised text, and audio keyed by (read text, speaker, sentence index, seed, speed). Hits are reported in `X-Cached` |

### API

| method | path | body | returns |
|---|---|---|---|
| GET | `/healthz` | — | `{"ready": true, "voices": [...], "default_voice", "sample_rate", "device", "warm_up_s", "checkpoint", ...}` |
| POST | `/v1/tts` | `{"text", "speed"?, "voice"?, "seed"?}` (also `?voice=`) | `audio/wav` |
| POST | `/v1/tts/male` | `{"text", "speed"?, "seed"?}` | `audio/wav`, forces male (ignores body `voice`) |
| POST | `/v1/tts/female` | same | `audio/wav`, forces female |
| POST | `/v1/normalize` | `{"sentences": [...]}` | what the front end will actually read aloud — useful for debugging pronunciation without audio |

Response headers: `X-Voice`, `X-Read` (the normalised text, url-encoded, truncated at 600 chars),
`X-Timing-Frontend-Ms`, `X-Timing-Model-Ms`, `X-Audio-Seconds`, `X-Cached`.

**Voice naming.** `VOICES = {"male": 0, "female": 1}`, with aliases `m|male|0|hamza` and `f|female|1|maryam`.
The kiosk sends **`f` / `m`** (see §4). An unrecognised value **silently falls back to the default voice**
(female, or `KIOSK_TTS_DEFAULT_VOICE`), so a typo becomes a wrong-gender voice rather than an error — keep the
alias table and the kiosk's value in step.

**Errors.** Empty text or text over `MAX_CHARS` (3000) → `400`. Everything else that fails is a 5xx. This
split is deliberate: the kiosk treats 4xx as *its own* bug and surfaces it, and only falls back to the paid
voice on 5xx and connection errors.

Multi-sentence input is split, rendered sentence by sentence and joined with 0.3 s of silence.

## 3. The kiosk side

All TTS code is in `Mari-Interactive-Kiosk-main/server/`:

| file | role |
|---|---|
| `providers/tts.py` | the adapters: `UpliftTTS` (paid), `MatchaTTS` (this server), `KokoroLocalTTS`/`KokoroRemoteTTS` (English only), and `get_tts_provider()`, the factory |
| `personas.py` + `personas.toml` | the two presenters as **data**: `female` (Maryam) and `male` (Hamza), each with a gender, the engines allowed for its Urdu, and its voice id per engine. `tts_engine(lang, persona)` decides the engine |
| `config.py` | every setting, read from `.env`. `ur_tts_for(persona_id)`, `tts_ready()`, and the `/healthz` snapshot |
| `agent/turn.py` → `agent/speech.py` | a turn renders each sentence as the LLM yields it, then sends the browser a JSON header `{"type":"tts","mime":...}` followed by the raw bytes |
| `avatar/lipsync.py` | decodes whatever the engine returned to 16 kHz PCM for Audio2Face; it already handles this server's WAV |

Request flow for one sentence:

```
turn.py → speech.speak(text, lang, persona_id)
        → providers.get_tts_provider(lang, persona_id)      # engine per language AND presenter
        → MatchaTTS.synthesize(text)
        → POST {APP_MATCHA_TTS_URL}/v1/tts {"text", "speed", "voice"}
        → WAV bytes back to the browser, unchanged
```

Two contract details that are easy to break:

- **`MatchaTTS.wants_spoken_form = False`.** The sentence reaches this server exactly as the LLM wrote it,
  English still in Latin script. This server has its own text front end; applying the paid engine's spoken
  forms first would normalise everything twice and make pronunciation worse.
- **Per-sentence fallback.** On a connection error or 5xx, that one sentence is spoken by the paid engine
  instead, in **that presenter's own** paid voice, and the kiosk logs
  `Matcha-TTS failed (...), speaking this sentence with Uplift`. Controlled by `APP_MATCHA_FALLBACK` (on by
  default). Seeing that line in the backend log is the signal that this server was unreachable.

## 4. Configuration: the per-voice switch

Everything is in the kiosk's `.env`. `APP_UR_TTS` is the default for every presenter and
`APP_UR_TTS_<PRESENTER>` overrides one, where the suffix is a persona id from `personas.toml`:

| want | `.env` |
|---|---|
| both voices local | `APP_UR_TTS=matcha` |
| both voices paid | `APP_UR_TTS=uplift` |
| female local, male paid | `APP_UR_TTS=matcha` + `APP_UR_TTS_MALE=uplift` |
| male local, female paid | `APP_UR_TTS=matcha` + `APP_UR_TTS_FEMALE=uplift` |

```ini
APP_UR_TTS=matcha
APP_MATCHA_TTS_URL=http://172.17.0.1:8030   # 127.0.0.1:8030 if the backend runs on the host
# APP_MATCHA_SPEED=1.0                      # clamped to 0.6–1.5 on both sides
# APP_MATCHA_FALLBACK=1                     # 0 = no paid fallback, failures surface
```

English is a separate setting (`APP_EN_TTS`, currently the paid engine) and is unaffected. A presenter can
never be muted by a bad value: `Persona.urdu_engine()` falls back to the first engine that presenter lists.

The kiosk picks the voice name with `("f" if persona.is_feminine else "m")` — **deliberately not** the
persona id, because a kiosk test forbids presenter ids as literals outside `personas.toml`/`personas.py`.

## 5. Checking it works

```bash
# this server
curl -s http://172.17.0.1:8030/healthz
for v in f m; do curl -s -X POST http://172.17.0.1:8030/v1/tts \
  -H 'content-type: application/json' -d "{\"text\":\"السلام علیکم، یہ ایک test ہے۔\",\"voice\":\"$v\"}" \
  -D- -o /tmp/voice_$v.wav | grep -i x-voice; done

# from inside the backend container (proves the address is right)
cd ~/Desktop/Mari-Interactive-Kiosk-main && docker compose exec backend curl -s http://172.17.0.1:8030/healthz

# the kiosk's own view: tts.ur shows each presenter's provider plus matcha.reachable
curl -s localhost:8010/healthz | python3 -m json.tool

# the kiosk's test suite (954 tests at the time of writing)
cd ~/Desktop/Mari-Interactive-Kiosk-main && .venv/bin/python -m pytest tests/ -q
```

**Verified on 2026-10-19:** server loads and both voices render distinctly; routing through the kiosk's own
`speech.speak` gives each presenter the right voice; all four `.env` modes route correctly; the fallback uses
each presenter's own paid voice; the full kiosk suite passes; the kiosk health snapshot shows both presenters
on this server and it reachable.

**Not verified:** the kiosk running under Docker end to end (container → host reachability was not exercised,
only the host-side equivalent), the browser flow with lipsync, and nobody has judged the kiosk audio by ear.

## 6. Traps worth knowing before editing either side

1. **The kiosk's `.env` holds live API keys in plaintext.** Never `cat`, echo or paste it; read only the lines
   you need. It is gitignored, so it is also **not** covered by a git revert.
2. **Presenter ids are banned as literals** in kiosk code outside the persona files
   (`tests/test_personas_are_data.py`). Read values off the `Persona` object instead.
3. **Every `env("APP_…")` key in `config.py` must appear in `.env.example`**, and nothing may be documented
   there that the code does not read (`tests/test_env_example.py`). A key built with an f-string is invisible
   to that regex, so `APP_UR_TTS_<PRESENTER>` is derived from the persona ids in the test instead.
4. **The golden snapshot (`tests/fixtures/golden.json`) records the exact TTS request bodies and socket
   frames**, so any change to the request shape changes it. Its harness replaces httpx with a stub; that stub
   needed an `is_closed` attribute, because this adapter keeps one keep-alive client and reads `is_closed` to
   decide whether to reopen it. Without it, only the first local-TTS call per process worked and the fixture
   had recorded 34 TTS *failures* as expected behaviour.
5. **One GPU.** This server holds the GPU while running. Any training or dataset work in `kaani/` competes
   with it — stop one before starting the other (`pkill -f "server/app.py"`).
6. `MatchaTTS` uses a module-level `httpx.AsyncClient`, so it is bound to the event loop that created it. Fine
   in the app (one long-lived loop); in a script, do all calls inside a single `asyncio.run`.
7. In the kiosk health output, `tts.ur.<presenter>.voice` is `null` for this engine, because `personas.toml`
   names no voice id for it (the voice comes from the presenter's gender). Cosmetic, and true before too.

## 7. Where things are

```
~/Desktop/kiosk-tts/
  server/app.py, engine.py, run_server.sh      the server
  server/MARI_INTEGRATION.md                   how to run and configure it with the kiosk
  server/MARI_CHANGES.md                       every kiosk-side edit + how to revert
  server/mari_backup_2026-10-19/               the kiosk files as they were, + a reversible patch
  checkpoints/kaani_male_female/               the two-voice checkpoint
  MARI_KIOSK_CONTEXT.md                        this file

~/Desktop/Mari-Interactive-Kiosk-main/
  server/providers/tts.py, personas.*, config.py, app.py
  tests/test_matcha_tts.py                     the adapter's own tests
  .env                                         the switch (secrets — do not print)
```

Nothing is committed on either side: the kiosk's changes are unstaged working-tree edits on branch
`feature/avatar-relight`, revertible with the patch in `mari_backup_2026-10-19/`.
