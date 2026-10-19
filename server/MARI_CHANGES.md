# Changes made in `Mari-Interactive-Kiosk-main` to wire this TTS server in

Applied **2026-10-19**. Before this, the kiosk sent all Urdu to the paid UpliftAI API; its
`matcha` adapter existed but had no voice field, and its male presenter was pinned to Uplift.

Nothing was removed: Uplift is still a first-class engine, still the per-sentence fallback, and
still serves all English. The kiosk is told nothing about how this server's voices were made — it
knows only "a local Matcha-TTS server at `APP_MATCHA_TTS_URL` with voices `f` and `m`".

## The switch

| want | `.env` |
|---|---|
| both voices local | `APP_UR_TTS=matcha` |
| both voices paid | `APP_UR_TTS=uplift` |
| female local, male paid | `APP_UR_TTS=matcha` + `APP_UR_TTS_MALE=uplift` |
| male local, female paid | `APP_UR_TTS=matcha` + `APP_UR_TTS_FEMALE=uplift` |

`APP_UR_TTS_<PRESENTER>` takes the presenter ids from `server/personas.toml`. English is separate
(`APP_EN_TTS`) and untouched.

## Files changed

| file | change |
|---|---|
| `server/personas.toml` | `[male] urdu_engines = ["matcha", "uplift"]` (was `["uplift"]`); dropped two comments claiming the local voice is female-only |
| `server/config.py` | added `ur_tts_for(persona_id)` → `APP_UR_TTS_<ID>` else `APP_UR_TTS`; `_ur_tts_status()` now probes the server when **any** presenter uses it (was: only when the global setting was `matcha`); refreshed the `UR_TTS` comment |
| `server/personas.py` | `tts_engine()` asks `config.ur_tts_for(persona.id)` instead of reading `config.UR_TTS`; docstrings refreshed |
| `server/providers/tts.py` | `MatchaTTS` takes `voice` (default `"f"`) and `uplift_voice`; sends `"voice"` in the request body; `_fallback` uses that presenter's own Uplift voice; docstring refreshed |
| `server/providers/tts.py` | `get_tts_provider()` builds `MatchaTTS(voice="f"/"m", uplift_voice=persona.voice("ur","uplift"))` |
| `server/providers/__init__.py` | `warm()` takes an optional `avatar` |
| `server/app.py` | startup warms the first presenter actually routed to this server (was: only the default presenter, and only when the global setting was `matcha`) |
| `.env.example` | matcha block rewritten: the table above, both addresses, the two new keys (commented) |
| `.env` | **not in git.** Only added the two commented override lines and a comment about the address. No existing line changed |
| `tests/test_matcha_tts.py` | the "male keeps Uplift" test became "each presenter gets its own voice, and its own Uplift voice to fall back to"; added a per-presenter-override test; the request-body assertion now expects `voice` |
| `tests/test_personas_are_data.py` | "matcha is only offered to a feminine presenter" → "a matcha presenter can still fall back to Uplift" |
| `tests/test_env_example.py` | taught the documentation guard about `APP_UR_TTS_<PRESENTER>`, whose key is built at call time and so cannot be found by its regex |
| `tests/golden_harness.py` | its fake HTTP client now exposes `is_closed`, which the real `httpx` client has |
| `tests/fixtures/golden.json` | the 20 Urdu turn snapshots re-recorded — see below |

### Why the golden fixture changed

The fake client above lacked `is_closed`, which the TTS adapter reads to decide whether to re-open
its keep-alive client. Only the **first** call per process avoided that attribute, so every Urdu
sentence after the first was recorded as a failure: the old fixture contained **34 warn frames
reading `tts: '_Client' object has no attribute 'is_closed'`**. With the stub fixed, all 20 Urdu
turns record real audio and **0 warn frames**, each presenter with its own voice. Only `turns`
entries whose label starts with `ws/ur/` were rewritten; every other section was asserted
byte-identical first, and the whole suite passes (954 tests).

## Reverting

Everything except `.env` is tracked by git, and no TTS file was dirty beforehand.

```bash
cd ~/Desktop/Mari-Interactive-Kiosk-main
# whole thing, in one step
git apply -R ~/Desktop/kiosk-tts/server/mari_backup_2026-10-19/mari_tts_wiring.patch
# or discard the tracked files outright
git checkout -- server/ tests/ .env.example
# .env (not in git): drop the four commented lines added under APP_UR_TTS, or
cp ~/Desktop/kiosk-tts/server/mari_backup_2026-10-19/.env .env   # overwrites later .env edits
```
`mari_backup_2026-10-19/` holds every file exactly as it was before the change.

To stop using the local voices without touching code at all, set `APP_UR_TTS=uplift`.
