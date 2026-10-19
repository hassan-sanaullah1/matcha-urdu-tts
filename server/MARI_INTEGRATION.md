# Wiring this TTS server into Mari-Interactive-Kiosk

**Status: applied on 2026-10-19.** Both presenters' Urdu now comes from this server, and `.env` chooses the
engine per voice. What was changed inside `Mari-Interactive-Kiosk-main/`, and how to revert it, is listed in
[MARI_CHANGES.md](MARI_CHANGES.md) — read that before editing the kiosk again.

Mari's `matcha` provider (`server/providers/tts.py` → `MatchaTTS`) speaks this server's contract:
`POST /v1/tts {text, speed, voice}` → 16-bit mono 22050 Hz WAV, and `GET /healthz` → `{"ready": true}`.
It sends `voice` as this server's own alias: `f` for the feminine presenter, `m` for the other.

Uplift was not removed. It still serves all English, stays available for Urdu, and is still the per-sentence
fallback when this server is unreachable (`APP_MATCHA_FALLBACK=1`, the default) — each presenter falling back to
its own Uplift voice.

## 1. Start this server

The kiosk backend runs in Docker, so bind to the address a container sees as "this host":

```bash
HOST=172.17.0.1 ~/Desktop/kiosk-tts/server/run_server.sh      # 172.17.0.1:8030
```

That address is `docker0`: reachable from containers and from this machine, but not from the LAN. For a backend
running directly on the host instead, use `run_server.sh` as it is (127.0.0.1) and set the URL below to match.

Check it, and hear both voices:

```bash
curl -s http://172.17.0.1:8030/healthz
for v in f m; do
  curl -s -X POST http://172.17.0.1:8030/v1/tts -H 'content-type: application/json' \
    -d "{\"text\":\"السلام علیکم، یہ ایک test ہے۔\",\"voice\":\"$v\"}" -o /tmp/voice_$v.wav
done
```

## 2. Choose the engine in Mari's `.env`

| want | `.env` |
|---|---|
| both voices local | `APP_UR_TTS=matcha` |
| both voices paid | `APP_UR_TTS=uplift` |
| female local, male paid | `APP_UR_TTS=matcha` + `APP_UR_TTS_MALE=uplift` |
| male local, female paid | `APP_UR_TTS=matcha` + `APP_UR_TTS_FEMALE=uplift` |

```ini
APP_UR_TTS=matcha
APP_MATCHA_TTS_URL=http://172.17.0.1:8030    # 127.0.0.1:8030 for a backend on the host
# APP_MATCHA_SPEED=1.0                       # 0.6–1.5; 1.0 is the model's own pace
# APP_MATCHA_FALLBACK=1                      # 0 turns the Uplift fallback off
```

Then `make dev` in the kiosk folder. `curl -s localhost:8010/healthz` shows, under `tts.ur`, each presenter's
provider plus `matcha.reachable`.

## Contract notes worth keeping

- `/healthz` must answer inside Mari's 0.5 s budget, which is why the model is warmed at startup here.
- A 4xx is treated by Mari as *its own* bug and surfaces; only 5xx and connection errors fall back to Uplift.
- Mari must not pre-normalize for this server (`wants_spoken_form = False`): it has its own text front end, and
  Uplift's spoken forms would be applied twice.
- One sentence at a time: the engine holds a GPU lock and the server runs a single worker by design.
- An unknown `voice` value silently gets the default (female), so the alias set here and the value Mari sends
  must stay in step.
