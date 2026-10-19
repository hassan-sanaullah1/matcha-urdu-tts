"""Settings for the web server and providers, read once from ``.env`` and the environment.

Real environment variables win over ``.env``. Retrieval settings are separate, in
server/rag/settings.py (prefix ``MARI_RAG_``).
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_env(path: Path) -> dict:
    env: dict[str, str] = {}
    if not path.exists():
        return env
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        val = val.split(" #")[0].strip().strip('"').strip("'")
        env[key.strip()] = val
    return env


# real env vars take precedence over .env file
ENV = {**_load_env(ROOT / ".env"), **os.environ}


def env(key: str, default: str = "") -> str:
    v = ENV.get(key, "")
    return v if v else default


FORCE_DEMO = ENV.get("MARI_FORCE_DEMO") == "1"

LANGS = ("en", "ur")

HOST = env("MARI_HOST", "127.0.0.1")
PORT = int(env("MARI_PORT", "8010"))

# Reported by /health/ready so anyone can confirm which deployment and which commit
# a running server is: local, dev, staging or prod, and the deployed git commit.
APP_ENV = env("APP_ENV", "local")
APP_COMMIT = env("APP_COMMIT", "local")

KNOWLEDGE_FILE = Path(
    env("APP_KNOWLEDGE_FILE", str(ROOT / "server" / "data" / "mari_energies_knowledge_base.md"))
)

# ── LLM — prefer DeepSeek (APP_LLM_*) when configured, else vLLM Qwen (APP_VLLM_*).
#    Force either with APP_LLM_PROVIDER = "deepseek" | "vllm". ─────────
_LLM_FORCED = env("APP_LLM_PROVIDER").lower()
_DS_BASE, _DS_KEY = env("APP_LLM_API_BASE"), env("APP_LLM_API_KEY")
_USE_DEEPSEEK = _LLM_FORCED == "deepseek" or (_LLM_FORCED != "vllm" and bool(_DS_BASE and _DS_KEY))

if _USE_DEEPSEEK:
    LLM_PROVIDER = "deepseek"
    LLM_BASE = _DS_BASE.rstrip("/")
    LLM_KEY = _DS_KEY
    LLM_MODEL = env("APP_LLM_MODEL", "deepseek-chat")
else:
    LLM_PROVIDER = "vllm"
    LLM_BASE = env("APP_VLLM_API_BASE").rstrip("/")
    LLM_KEY = env("APP_VLLM_API_KEY")
    LLM_MODEL = env("APP_VLLM_MODEL", "qwen3.5")

# Low on purpose: the reply must restate figures and dates from the retrieved knowledge,
# and at 0.7 the model swapped them ("share price 54.25" for EPS, July 2026 as May).
LLM_TEMPERATURE = float(env("APP_LLM_TEMPERATURE", "0.3"))

# ── Urdu STT — Soniox realtime (websocket) ──────────────────────────
SONIOX_KEY = env("APP_SONIOX_API_KEY")
SONIOX_URL = env("APP_SONIOX_BASE_URL", "wss://stt-rt.jp.soniox.com/transcribe-websocket")
SONIOX_MODEL = env("APP_SONIOX_MODEL", "stt-rt-v4")

# ── Urdu TTS — UpliftAI (REST, returns mp3) ─────────────────────────
UPLIFT_KEY = env("APP_UPLIFT_API_KEY")
UPLIFT_BASE = env("APP_UPLIFT_TTS_BASE_URL", "https://ap-southeast-1.api.upliftai.org").rstrip("/")
UPLIFT_PATH = env("APP_UPLIFT_TTS_API_PATH", "/v1/synthesis/text-to-speech")
UPLIFT_VOICE = env("APP_UPLIFT_VOICE_ID", "v_8eelc901v6")
# English through Uplift uses the same voice by default, so the persona sounds the same.
UPLIFT_VOICE_EN = env("APP_UPLIFT_VOICE_ID_EN", UPLIFT_VOICE)
# The male presenter's voice, for both languages. Ids: https://docs.upliftai.org/orator_voices
UPLIFT_VOICE_MALE = env("APP_UPLIFT_VOICE_ID_MALE", UPLIFT_VOICE)
UPLIFT_FORMAT = env("APP_UPLIFT_OUTPUT_FORMAT", "MP3_22050_32")
# Below 1.0 so short unstressed words are not clipped; measured, and not monotonic
# (docs/pronunciation-notes.md).
try:
    UPLIFT_SPEED = float(env("APP_UPLIFT_SPEED", "0.9"))
except ValueError:  # a typo in the env must not take the voice down
    UPLIFT_SPEED = 0.9

# English has its own rate, tuned by ear on the deployed kiosk.
try:
    UPLIFT_SPEED_EN = float(env("APP_UPLIFT_SPEED_EN", "0.98"))
except ValueError:
    UPLIFT_SPEED_EN = 0.98

# ── Urdu TTS for the female presenter: "uplift" (default) or "matcha", a local
#    server (kaani/kisok-integrated-tts) reached over HTTP only. Matcha-TTS is a female
#    voice, so the male presenter's Urdu stays on Uplift either way. ──
UR_TTS = env("APP_UR_TTS", "uplift")
MATCHA_URL = env("APP_MATCHA_TTS_URL", "http://127.0.0.1:8030").rstrip("/")
# 1.0 is the model's own pace; the server clamps to 0.6–1.5 too.
try:
    MATCHA_SPEED = min(1.5, max(0.6, float(env("APP_MATCHA_SPEED", "1.0"))))
except ValueError:
    MATCHA_SPEED = 1.0
# A sentence the local server can't render goes to Uplift rather than being silent.
MATCHA_FALLBACK = env("APP_MATCHA_FALLBACK", "1").lower() not in ("0", "false", "no", "off")

# ── English STT/TTS. Urdu uses Soniox, and Uplift or Matcha-TTS (UR_TTS). English TTS
#    defaults to Uplift because no Kokoro instance is deployed; set
#    APP_EN_TTS=local|remote when one is. ──
EN_STT = env("APP_EN_STT", "local")  # "local" (faster-whisper) | "remote"
EN_TTS = env("APP_EN_TTS", "uplift")  # "uplift" | "local" (kokoro) | "remote"

# local faster-whisper
WHISPER_LOCAL_MODEL = env("APP_ASR_MODEL_SIZE", "base.en")

# remote Whisper (fallback if APP_EN_STT=remote)
WHISPER_URL = env("APP_REMOTE_WHISPER_URL")
WHISPER_TOKEN = env("APP_REMOTE_WHISPER_TOKEN")
WHISPER_MODEL = env("APP_REMOTE_WHISPER_MODEL", "whisper-1")

# remote Kokoro (fallback if APP_EN_TTS=remote)
KOKORO_BASE = env("APP_REMOTE_TTS_API_BASE").rstrip("/")
KOKORO_PATH = env("APP_REMOTE_TTS_API_PATH", "/v1/audio/speech")
KOKORO_TOKEN = env("APP_REMOTE_TTS_TOKEN")
KOKORO_MODEL = env("APP_REMOTE_TTS_MODEL", "kokoro")
KOKORO_FORMAT = env("APP_REMOTE_TTS_RESPONSE_FORMAT", "mp3")

# ── Audio2Face-3D (avatar lipsync) — NVIDIA NIM over gRPC ───────────
#    Unset APP_A2F_URL ⇒ no lipsync frames; everything else runs unchanged.
#    Forms: host:port · grpc://host:port · grpcs://host[:port] (TLS + API key)
A2F_URL = env("APP_A2F_URL")
A2F_API_KEY = env("APP_A2F_API_KEY")
A2F_TLS_CA = env("APP_A2F_TLS_CA")
# How many reply sentences may be in flight at the NIM at once. Must not exceed
# the NIM's own `stream_number` (deploy/a2f/configs/deployment_config.yaml),
# which is 1 on a 6GB GPU and 3 on a full-size one.
A2F_MAX_CLIPS = max(1, int(env("APP_A2F_MAX_CLIPS", "1")))

# ── Voices ──────────────────────────────────────────────────────────
# Kokoro voice names. Urdu uses Uplift (UPLIFT_VOICE) or Matcha-TTS and English uses Uplift
# too while APP_EN_TTS=uplift (UPLIFT_VOICE_EN), so these only apply when either
# language is routed to Kokoro.
VOICE_EN = env("APP_TTS_VOICE_ENGLISH", "af_heart")
VOICE_EN_MALE = env("APP_TTS_VOICE_ENGLISH_MALE", "am_michael")
VOICE_UR = env("APP_TTS_VOICE_URDU", "af_heart")


def pitch_only() -> bool:
    """MARI_PITCH_ONLY: a recording aid that answers every turn with the presenter's pitch line."""
    return ENV.get("MARI_PITCH_ONLY", "0") in ("1", "true", "yes", "on")


def llm_ready() -> bool:
    return bool(LLM_BASE and LLM_KEY) and not FORCE_DEMO


def stt_ready(lang: str) -> bool:
    if lang == "ur":
        return bool(SONIOX_KEY)
    return True if EN_STT == "local" else bool(WHISPER_URL)


def tts_ready(lang: str, avatar: str | None = None) -> bool:
    """Whether this presenter can be spoken in this language right now."""
    from .personas import get_persona, tts_engine

    engine = tts_engine(lang, get_persona(avatar))
    if engine == "matcha":
        # MatchaTTS falls back to Uplift by itself when the local server is down.
        return True
    if engine == "uplift":
        return bool(UPLIFT_KEY)
    return True if engine == "local" else bool(KOKORO_BASE)


async def status() -> dict:
    """Compact config snapshot for /healthz: no secrets and no internal hostnames."""
    en_stt = f"whisper-local:{WHISPER_LOCAL_MODEL}" if EN_STT == "local" else "whisper-remote"
    return {
        "llm": {"ready": llm_ready(), "provider": LLM_PROVIDER, "model": LLM_MODEL},
        "stt": {
            "ur": {"provider": "soniox", "ready": stt_ready("ur"), "model": SONIOX_MODEL},
            "en": {"provider": en_stt, "ready": stt_ready("en")},
        },
        "tts": {
            "ur": await _ur_tts_status(),
            "en": await _en_tts_status(),
        },
        "avatar": await _avatar_status(),
    }


async def _en_tts_status() -> dict:
    """English TTS per presenter."""
    from .personas import all_personas, tts_engine

    engine_label = {"uplift": "uplift", "local": "kokoro-local"}.get(EN_TTS, "kokoro-remote")
    return {
        persona.id: {
            "provider": engine_label,
            "ready": tts_ready("en", persona.id),
            "voice": persona.voice("en", tts_engine("en", persona)) or None,
        }
        for persona in all_personas().values()
    }


async def _ur_tts_status() -> dict:
    """Urdu TTS per presenter; with matcha, whether its local server answers now."""
    from .personas import all_personas, tts_engine

    per_persona = {
        persona.id: {
            "provider": tts_engine("ur", persona),
            "ready": tts_ready("ur", persona.id),
            "voice": persona.voice("ur", tts_engine("ur", persona)) or None,
        }
        for persona in all_personas().values()
    }
    if UR_TTS != "matcha":
        return per_persona
    import httpx

    try:
        async with httpx.AsyncClient(timeout=0.5) as client:
            r = await client.get(f"{MATCHA_URL}/healthz")
            reachable = r.status_code == 200 and r.json().get("ready") is True
    except Exception:
        reachable = False
    return {
        **per_persona,
        "matcha": {"reachable": reachable, "fallback": "uplift" if MATCHA_FALLBACK else None},
    }


async def _avatar_status() -> dict:
    """Lipsync readiness — imported lazily so config stays dependency-free."""
    from .avatar import a2f_status

    return {"a2f": await a2f_status()}
