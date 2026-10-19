"""The local Matcha-TTS voice for the female presenter's Urdu, and synthesis ahead
of playback.

Matcha-TTS is a female voice, so APP_UR_TTS=matcha moves only the female presenter's Urdu
off Uplift. Its server has its own normalizer, so the reply goes to it as the responder
wrote it; Uplift's spoken forms would make it worse. When the server is down, the
sentence is spoken by Uplift instead of being lost.

The turn renders each sentence's audio as soon as the responder yields it, but the
browser must still get reply → tts → bytes for one sentence before the next begins.
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from server import avatar
from server import config as C
from server.agent import speech
from server.agent import turn as turn_mod
from server.normalization import normalize_for_tts
from server.providers import tts
from server.providers.tts import MatchaTTS, UpliftTTS, get_tts_provider

WAV = b"RIFF\x24\x00\x00\x00WAVEfmt fake-matcha-audio"
SENTENCE = "MARI کا revenue FY2024 میں PKR 65bn تھا۔"


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    monkeypatch.setattr(tts, "_tts_cache", {})
    monkeypatch.setattr(C, "UR_TTS", "uplift")
    monkeypatch.setattr(C, "MATCHA_FALLBACK", True)
    monkeypatch.setattr(C, "UPLIFT_KEY", "test-key")


def _matcha(handler) -> MatchaTTS:
    return MatchaTTS(
        base="http://matcha.test", client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )


def _record_uplift(monkeypatch) -> list[str]:
    said: list[str] = []

    async def fake(self, text: str) -> tuple[bytes, str]:
        said.append(text)
        return b"mp3", "audio/mpeg"

    monkeypatch.setattr(UpliftTTS, "synthesize", fake)
    return said


# ── routing ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("avatar_id", ["female", "male"])
def test_by_default_urdu_stays_on_uplift(avatar_id: str) -> None:
    assert isinstance(get_tts_provider("ur", avatar_id), UpliftTTS)
    assert C.tts_ready("ur", avatar_id)


def test_matcha_speaks_the_female_presenters_urdu(monkeypatch) -> None:
    monkeypatch.setattr(C, "UR_TTS", "matcha")
    provider = get_tts_provider("ur", "female")
    assert isinstance(provider, MatchaTTS)
    assert get_tts_provider("ur", "female") is provider
    # Ready without an Uplift key: the server answers or the adapter falls back.
    monkeypatch.setattr(C, "UPLIFT_KEY", "")
    assert C.tts_ready("ur", "female")


def test_the_male_presenter_keeps_uplift(monkeypatch) -> None:
    """Matcha-TTS is a female voice."""
    monkeypatch.setattr(C, "UR_TTS", "matcha")
    provider = get_tts_provider("ur", "male")
    assert isinstance(provider, UpliftTTS)
    assert provider.voice == C.UPLIFT_VOICE_MALE
    monkeypatch.setattr(C, "UPLIFT_KEY", "")
    assert not C.tts_ready("ur", "male")


@pytest.mark.parametrize("avatar_id", ["female", "male"])
def test_english_is_unaffected(monkeypatch, avatar_id: str) -> None:
    before = type(get_tts_provider("en", avatar_id))
    monkeypatch.setattr(tts, "_tts_cache", {})
    monkeypatch.setattr(C, "UR_TTS", "matcha")
    assert type(get_tts_provider("en", avatar_id)) is before
    assert not isinstance(get_tts_provider("en", avatar_id), MatchaTTS)


def test_speak_sends_matcha_the_raw_sentence(monkeypatch) -> None:
    """No Uplift spoken forms: no Roman respellings, no Urdu letter names for acronyms."""
    monkeypatch.setattr(C, "UR_TTS", "matcha")
    got: list[str] = []

    async def fake(self, text: str) -> tuple[bytes, str]:
        got.append(text)
        return WAV, "audio/wav"

    monkeypatch.setattr(MatchaTTS, "synthesize", fake)
    asyncio.run(speech.speak(f"  {SENTENCE}  ", "ur", "female"))
    assert got == [SENTENCE]
    assert normalize_for_tts(SENTENCE, "ur", "uplift") != SENTENCE


# ── the adapter ────────────────────────────────────────────────────────────


def test_returns_the_servers_wav(monkeypatch) -> None:
    monkeypatch.setattr(C, "MATCHA_SPEED", 1.1)
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, content=WAV, headers={"content-type": "audio/wav"})

    assert asyncio.run(_matcha(handler).synthesize(SENTENCE)) == (WAV, "audio/wav")
    assert str(seen[0].url) == "http://matcha.test/v1/tts"
    assert json.loads(seen[0].content) == {"text": SENTENCE, "speed": 1.1}


def test_empty_text_makes_no_request() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("no request expected")

    assert asyncio.run(_matcha(handler).synthesize("   ")) == (b"", "audio/wav")


def _connect_error(request: httpx.Request) -> httpx.Response:
    raise httpx.ConnectError("connection refused", request=request)


def _server_error(request: httpx.Request) -> httpx.Response:
    return httpx.Response(500, json={"error": "CUDA out of memory"})


@pytest.mark.parametrize("handler", [_connect_error, _server_error], ids=["down", "500"])
def test_a_failing_server_falls_back_to_uplift(monkeypatch, handler) -> None:
    """Uplift gets its own spoken form, since speech.speak skipped it for Matcha-TTS."""
    said = _record_uplift(monkeypatch)
    assert asyncio.run(_matcha(handler).synthesize(SENTENCE)) == (b"mp3", "audio/mpeg")
    assert said == [normalize_for_tts(SENTENCE, "ur", "uplift")]


@pytest.mark.parametrize("handler", [_connect_error, _server_error], ids=["down", "500"])
def test_with_fallback_off_the_failure_surfaces(monkeypatch, handler) -> None:
    monkeypatch.setattr(C, "MATCHA_FALLBACK", False)
    said = _record_uplift(monkeypatch)
    with pytest.raises(httpx.HTTPError):
        asyncio.run(_matcha(handler).synthesize(SENTENCE))
    assert said == []


def test_a_bad_request_is_a_bug_not_a_fallback(monkeypatch) -> None:
    said = _record_uplift(monkeypatch)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": "text too long"})

    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(_matcha(handler).synthesize(SENTENCE))
    assert said == []


def test_warm_never_raises_when_the_server_is_down(monkeypatch) -> None:
    monkeypatch.setattr(
        httpx, "get", lambda *a, **k: (_ for _ in ()).throw(httpx.ConnectError("x"))
    )
    MatchaTTS(base="http://matcha.test").warm()


# ── synthesis ahead, delivery in order ─────────────────────────────────────


class _Sock:
    """Records what run_reply sends. ``close_on_bytes``: the visitor leaves as the
    first audio goes out, and every send after that fails, as on a real socket."""

    def __init__(self, close_on_bytes: bool = False) -> None:
        self.sent: list = []
        self.close_on_bytes = close_on_bytes
        self.closed = False

    async def send_json(self, payload: dict) -> None:
        if self.closed:
            raise RuntimeError("socket closed")
        self.sent.append(payload)

    async def send_bytes(self, data: bytes) -> None:
        if self.close_on_bytes:
            self.closed = True
        if self.closed:
            raise RuntimeError("socket closed")
        self.sent.append(data)


def _fake_turn(monkeypatch, delays: dict[str, float]) -> list[str]:
    """Sentences s1.. from the responder, each rendering for its own delay.
    Returns the render events in order: "start s1", "end s1", "cancel s2", ..."""
    log: list[str] = []

    async def respond_sentences(text, lang, history, avatar_id):
        for s in delays:
            yield s, False

    async def speak(text: str, lang: str, avatar_id: str = "female") -> tuple[bytes, str]:
        log.append(f"start {text}")
        try:
            await asyncio.sleep(delays[text])
        except asyncio.CancelledError:
            log.append(f"cancel {text}")
            raise
        log.append(f"end {text}")
        return f"audio:{text}".encode(), "audio/wav"

    monkeypatch.setattr(turn_mod, "respond_sentences", respond_sentences)
    monkeypatch.setattr(speech, "speak", speak)
    monkeypatch.setattr(avatar, "get_a2f_client", lambda: None)
    return log


def test_sentences_go_out_in_order_even_when_a_later_one_renders_first(monkeypatch) -> None:
    log = _fake_turn(monkeypatch, {"s1": 0.05, "s2": 0.0, "s3": 0.0})
    sock = _Sock()
    asyncio.run(turn_mod.run_reply(sock, "q", "ur", echo_transcript=False))

    assert log.index("end s2") < log.index("end s1")  # rendered ahead, during s1
    expected: list = []
    for s in ("s1", "s2", "s3"):
        expected += [
            {"type": "reply", "text": s, "demo": False},
            {"type": "tts", "mime": "audio/wav"},
            f"audio:{s}".encode(),
        ]
    assert sock.sent == [*expected, {"type": "done", "spoken": True}]


def test_no_more_than_two_sentences_render_ahead(monkeypatch) -> None:
    log = _fake_turn(monkeypatch, {"s1": 0.05, "s2": 0.0, "s3": 0.0, "s4": 0.0})
    asyncio.run(turn_mod.run_reply(_Sock(), "q", "ur", echo_transcript=False))
    # s2 renders alongside s1, but s3 may only start once s1 has been sent.
    assert log.index("start s2") < log.index("end s1") < log.index("start s3")


def test_a_closed_socket_cancels_the_pending_renders(monkeypatch) -> None:
    log = _fake_turn(monkeypatch, {"s1": 0.0, "s2": 5.0, "s3": 5.0})

    async def main() -> None:
        with pytest.raises(RuntimeError, match="socket closed"):
            await turn_mod.run_reply(_Sock(close_on_bytes=True), "q", "ur", echo_transcript=False)
        await asyncio.sleep(0)  # let the cancellation land; asyncio.run would cancel later anyway
        assert "cancel s2" in log

    asyncio.run(main())
    assert "end s2" not in log and "end s3" not in log
