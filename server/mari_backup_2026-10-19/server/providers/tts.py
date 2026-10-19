"""TTS adapters, each implementing :class:`server.providers.base.TTSProvider`.

  UpliftTTS        Urdu, and English while APP_EN_TTS=uplift (UpliftAI REST, mp3)
  MatchaTTS        the female presenter's Urdu while APP_UR_TTS=matcha (local server, wav)
  KokoroLocalTTS   English, local Kokoro (GPU if available)
  KokoroRemoteTTS  English, remote OpenAI-compatible /v1/audio/speech

Adapters send text as given. Rewriting a reply into its spoken form is done by
server/normalization.py, for adapters that set ``wants_spoken_form``. Matcha-TTS never
does: its server has its own normalizer, and Uplift's spoken forms make it worse.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx

from .. import config
from ..personas import all_personas, get_persona, tts_engine
from .base import TTSProvider

logger = logging.getLogger(__name__)


class UpliftTTS(TTSProvider):
    """UpliftAI REST synthesis. One voice handles Urdu and English."""

    wants_spoken_form = True
    engine = "uplift"

    def __init__(
        self,
        api_key: str = "",
        base: str = "",
        path: str = "",
        voice: str = "",
        output_format: str = "",
        speed: float = 1.0,
        lang: str = "ur",
    ):
        self.api_key = api_key or config.UPLIFT_KEY
        self.base = (base or config.UPLIFT_BASE).rstrip("/")
        self.path = path or config.UPLIFT_PATH
        self.voice = voice or config.UPLIFT_VOICE
        self.output_format = output_format or config.UPLIFT_FORMAT
        self.lang = lang
        # 1.0 means "not set by the caller": use the configured per-language rate.
        self.speed = (
            speed
            if speed != 1.0
            else (config.UPLIFT_SPEED_EN if lang == "en" else config.UPLIFT_SPEED)
        )

    async def synthesize(self, text: str) -> tuple[bytes, str]:
        if not (text or "").strip():
            return b"", "audio/mpeg"
        path = self.path if self.path.startswith("/") else f"/{self.path}"
        url = f"{self.base}{path}"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {
            "voiceId": self.voice,
            "text": text,
            "speed": self.speed,
            "outputFormat": self.output_format,
        }
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(url, json=payload, headers=headers)
            r.raise_for_status()
            mime = r.headers.get("content-type", "audio/mpeg").split(";")[0]
            return r.content, mime or "audio/mpeg"


# One keep-alive client for every Matcha-TTS sentence: a new connection per sentence
# would cost more than the ~2 ms the server takes for a cached line.
_matcha_http: httpx.AsyncClient | None = None


def _matcha_client() -> httpx.AsyncClient:
    global _matcha_http
    if _matcha_http is None or _matcha_http.is_closed:
        _matcha_http = httpx.AsyncClient(timeout=httpx.Timeout(15.0, connect=0.5))
    return _matcha_http


class MatchaTTS(TTSProvider):
    """A local Matcha-TTS server (kaani/kisok-integrated-tts): Urdu with English mixed
    in, 16-bit mono 22050 Hz WAV. Reached over HTTP only; its env is separate from ours.

    It gets the reply sentence as the responder wrote it, English still in Latin script.
    When the server is down or fails, the sentence goes to Uplift (APP_MATCHA_FALLBACK).
    """

    wants_spoken_form = False
    engine = "matcha"
    lang = "ur"

    def __init__(self, base: str = "", client: httpx.AsyncClient | None = None):
        self.base = (base or config.MATCHA_URL).rstrip("/")
        self._client = client
        self._uplift: UpliftTTS | None = None

    async def synthesize(self, text: str) -> tuple[bytes, str]:
        text = (text or "").strip()
        if not text:
            return b"", "audio/wav"
        client = self._client or _matcha_client()
        try:
            r = await client.post(
                f"{self.base}/v1/tts", json={"text": text, "speed": config.MATCHA_SPEED}
            )
        except httpx.TransportError as exc:
            if not config.MATCHA_FALLBACK:
                raise
            return await self._fallback(text, exc)
        if r.status_code >= 500 and config.MATCHA_FALLBACK:
            return await self._fallback(text, f"HTTP {r.status_code}")
        # A 4xx means we sent something wrong; hiding it behind Uplift would hide the bug.
        r.raise_for_status()
        logger.debug(
            "matcha-tts: frontend %s ms, model %s ms, %s s of audio, cached %s",
            r.headers.get("x-timing-frontend-ms"),
            r.headers.get("x-timing-model-ms"),
            r.headers.get("x-audio-seconds"),
            r.headers.get("x-cached"),
        )
        return r.content, "audio/wav"

    async def _fallback(self, text: str, reason: object) -> tuple[bytes, str]:
        from ..normalization import normalize_for_tts

        logger.warning("Matcha-TTS failed (%s), speaking this sentence with Uplift", reason)
        if self._uplift is None:
            # Only a persona that lists matcha reaches this adapter, so its own Uplift
            # Urdu voice is the right stand-in.
            persona = next(p for p in all_personas().values() if "matcha" in p.urdu_engines)
            self._uplift = UpliftTTS(voice=persona.voice("ur", "uplift"))
        # speech.speak skipped normalization for this adapter; Uplift needs it.
        return await self._uplift.synthesize(normalize_for_tts(text, "ur", "uplift"))

    def warm(self) -> None:
        """Check the server is up; the model itself is loaded by the server at its start."""
        try:
            r = httpx.get(f"{self.base}/healthz", timeout=2.0)
            ready = r.status_code == 200 and r.json().get("ready") is True
        except Exception:
            ready = False
        if ready:
            logger.info("Matcha-TTS ready at %s", self.base)
        elif config.MATCHA_FALLBACK:
            logger.warning(
                "Matcha-TTS not reachable at %s, Urdu will fall back to Uplift", self.base
            )
        else:
            logger.warning("Matcha-TTS not reachable at %s, and fallback is off", self.base)


class KokoroLocalTTS(TTSProvider):
    """In-process Kokoro pipeline (GPU if available)."""

    wants_spoken_form = True
    engine = "kokoro"
    _pipe = None  # one pipeline per process

    def __init__(self, voice: str = ""):
        self.voice = voice or config.VOICE_EN

    def _get_pipe(self) -> Any:
        if KokoroLocalTTS._pipe is None:
            from kokoro import KPipeline

            KokoroLocalTTS._pipe = KPipeline(lang_code="a")  # 'a' = American English
        return KokoroLocalTTS._pipe

    def warm(self) -> None:
        self._get_pipe()

    async def synthesize(self, text: str) -> tuple[bytes, str]:
        import io

        import numpy as np
        import soundfile as sf

        text = (text or "").strip()
        if not text:
            return b"", "audio/wav"
        pipe = self._get_pipe()

        def run() -> tuple[bytes, str]:
            parts = []
            for _, _, audio in pipe(text, voice=self.voice):
                if hasattr(audio, "detach"):
                    audio = audio.detach().cpu().numpy()
                parts.append(np.asarray(audio, dtype=np.float32).reshape(-1))
            if not parts:
                return b"", "audio/wav"
            buf = io.BytesIO()
            sf.write(buf, np.concatenate(parts), 24000, format="WAV", subtype="PCM_16")
            return buf.getvalue(), "audio/wav"

        return await asyncio.to_thread(run)


class KokoroRemoteTTS(TTSProvider):
    wants_spoken_form = True
    engine = "kokoro"

    def __init__(
        self,
        base: str = "",
        path: str = "",
        token: str = "",
        model: str = "",
        voice: str = "",
        fmt: str = "",
    ):
        self.base = (base or config.KOKORO_BASE).rstrip("/")
        self.path = path or config.KOKORO_PATH
        self.token = token or config.KOKORO_TOKEN
        self.model = model or config.KOKORO_MODEL
        self.voice = voice or config.VOICE_EN
        self.fmt = fmt or config.KOKORO_FORMAT

    async def synthesize(self, text: str) -> tuple[bytes, str]:
        text = (text or "").strip()
        if not text:
            return b"", "audio/mpeg"
        url = f"{self.base}{self.path}"
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        payload = {
            "model": self.model,
            "input": text,
            "voice": self.voice,
            "response_format": self.fmt,
        }
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(url, json=payload, headers=headers)
            r.raise_for_status()
            mime = r.headers.get("content-type", "audio/mpeg").split(";")[0]
            return r.content, mime or "audio/mpeg"


_tts_cache: dict[str, TTSProvider] = {}


def get_tts_provider(lang: str, avatar: str | None = None) -> TTSProvider:
    """The configured TTS adapter for a language and presenter, cached per voice.

    The persona supplies both the engine and the voice id (server/personas.toml);
    nothing here knows which presenters exist.
    """
    persona = get_persona(avatar)
    engine = tts_engine(lang, persona)
    key = f"{lang}:{engine}:{persona.id}"
    if key not in _tts_cache:
        if engine == "matcha":
            _tts_cache[key] = MatchaTTS()
        elif engine == "uplift":
            _tts_cache[key] = UpliftTTS(voice=persona.voice(lang, "uplift"), lang=lang)
        else:
            kokoro = KokoroLocalTTS if engine == "local" else KokoroRemoteTTS
            _tts_cache[key] = kokoro(voice=persona.voice(lang, "kokoro"))
    return _tts_cache[key]
