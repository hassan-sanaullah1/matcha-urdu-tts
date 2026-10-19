"""Provider adapters for the external speech and language services.

  stt(wav, lang) -> str            transcribe with the configured STT adapter
  get_tts_provider(lang, avatar)   the configured TTS adapter (used by server/agent/speech.py)
  llm                              chat-completions client (server/providers/llm.py)
  warm(lang)                       preload local-model adapters at startup
  SonioxStream                     live streaming STT for /ws
  wav_to_pcm16 / pcm16_to_wav      WAV <-> PCM helpers

Adapters implement the Protocols in ``base.py``.
"""

from __future__ import annotations

from .base import STTProvider, TTSProvider
from .stt import (
    SonioxStream,
    SonioxSTT,
    WhisperLocalSTT,
    WhisperRemoteSTT,
    get_stt_provider,
    pcm16_to_wav,
    stt,
    wav_to_pcm16,
)
from .tts import KokoroLocalTTS, KokoroRemoteTTS, MatchaTTS, UpliftTTS, get_tts_provider

__all__ = [
    "KokoroLocalTTS",
    "KokoroRemoteTTS",
    "MatchaTTS",
    "STTProvider",
    "SonioxSTT",
    "SonioxStream",
    "TTSProvider",
    "UpliftTTS",
    "WhisperLocalSTT",
    "WhisperRemoteSTT",
    "get_stt_provider",
    "get_tts_provider",
    "pcm16_to_wav",
    "stt",
    "warm",
    "wav_to_pcm16",
]


async def warm(lang: str = "en") -> None:
    """Preload local-model adapters (Whisper/Kokoro) in the background so the first
    turn in that language isn't slowed by model load, and check the Matcha-TTS server is
    up. No-op for remote adapters."""
    import asyncio

    stt_provider = get_stt_provider(lang)
    if hasattr(stt_provider, "warm"):
        try:
            await asyncio.to_thread(stt_provider.warm)
        except Exception:
            pass
    tts_provider = get_tts_provider(lang)
    if hasattr(tts_provider, "warm"):
        try:
            await asyncio.to_thread(tts_provider.warm)
        except Exception:
            pass
