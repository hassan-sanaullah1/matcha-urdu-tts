"""MARI kiosk backend: the API the Next.js kiosk in frontend/ talks to.

    GET  /health/live    the process is up (container liveness)
    GET  /health/ready   can it answer: environment, commit, llm + qdrant checks; 503 if not
    GET  /healthz        developer diagnostics: provider, avatar and retrieval status
    POST /chat           {text, lang, history?, avatar?} → reply text only
    WS   /ws             the streaming turn the UI uses (see server/agent/turn.py)

Any stage that isn't configured or reachable degrades: the response carries what
succeeded plus an ``error`` note, and a canned line stands in for the LLM.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable

from fastapi import FastAPI, Request, Response, WebSocket
from pydantic import BaseModel

from . import avatar, config, providers, rag
from .agent.policy import failure_reply, fallback_reply
from .agent.responder import respond
from .agent.session import VoiceSession
from .logging_setup import configure_logging, is_quiet, new_request_id, request_id
from .personas import get_persona
from .prompts import get_prompts

configure_logging()
logger = logging.getLogger(__name__)

app = FastAPI(title="MARI · Voice")

# Container health probes; their INFO lines are dropped (see logging_setup.is_quiet).
_PROBE_PATHS = frozenset({"/health/live", "/health/ready"})


@app.middleware("http")
async def log_request(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """Give every HTTP request its own ID, and log one line when it finishes."""
    token = request_id.set(new_request_id())
    quiet_token = is_quiet.set(request.url.path in _PROBE_PATHS)
    started = time.perf_counter()
    try:
        response = await call_next(request)
        logger.info(
            "%s %s %d %.0f ms",
            request.method,
            request.url.path,
            response.status_code,
            (time.perf_counter() - started) * 1000,
        )
        return response
    except Exception:
        logger.exception("%s %s failed", request.method, request.url.path)
        raise
    finally:
        is_quiet.reset(quiet_token)
        request_id.reset(token)


@app.on_event("startup")
async def _startup() -> None:
    get_prompts()  # a missing prompt file stops startup here
    asyncio.create_task(providers.warm("en"))
    if config.UR_TTS == "matcha":
        # Only checks the local TTS server; Soniox has nothing to preload.
        asyncio.create_task(providers.warm("ur"))
    asyncio.create_task(avatar.warm())
    # Minutes on a cold container; turns are answered from the core brief meanwhile.
    asyncio.create_task(rag.startup())


@app.on_event("shutdown")
async def _shutdown() -> None:
    await rag.shutdown()


class LiveOut(BaseModel):
    status: str


class ReadyOut(BaseModel):
    status: str
    environment: str
    commit: str
    checks: dict[str, str]


@app.get("/health/live", response_model=LiveOut)
async def health_live() -> LiveOut:
    """The process is up. Checks nothing else, so it never fails while the app runs."""
    return LiveOut(status="ok")


# Check states that still mean "can answer a visitor": retrieval warming falls back to
# the core brief, and demo mode answers with the canned line on purpose.
_SERVING_STATES = frozenset({"ok", "warming", "demo"})


def _llm_state() -> str:
    """From configuration alone: a health check must never call the paid LLM."""
    if config.FORCE_DEMO:
        return "demo"
    return "ok" if config.llm_ready() else "fail"


@app.get("/health/ready", response_model=ReadyOut, responses={503: {"model": ReadyOut}})
async def health_ready(response: Response) -> ReadyOut:
    """Can this server answer a visitor? 503 if not."""
    checks = {"llm": _llm_state(), "qdrant": await rag.readiness()}
    is_ready = all(state in _SERVING_STATES for state in checks.values())
    if not is_ready:
        response.status_code = 503
    return ReadyOut(
        status="ok" if is_ready else "fail",
        environment=config.APP_ENV,
        commit=config.APP_COMMIT,
        checks=checks,
    )


@app.get("/healthz")
async def healthz() -> dict:
    """Developer diagnostics: every provider's configuration and reachability."""
    return {
        "ok": True,
        **await config.status(),
        "rag": await rag.health(),
    }


class ChatIn(BaseModel):
    text: str
    lang: str = "en"
    # Prior turns, oldest first, as [{"role": "user"|"assistant", "text": ...}].
    history: list[dict] | None = None
    # A presenter id from server/personas.toml; anything else becomes the default.
    avatar: str | None = None


class ChatOut(BaseModel):
    reply: str
    demo: bool
    # Only present when the LLM call failed and `reply` is the canned stand-in.
    error: str | None = None


# FastAPI validates the returned dict against ChatOut; the function itself keeps
# returning a dict because the golden harness calls it directly. exclude_none keeps
# `error` out of the JSON unless it is set (tests/test_chat_endpoint.py pins the shape).
@app.post("/chat", response_model=ChatOut, response_model_exclude_none=True)
async def chat(body: ChatIn) -> dict:
    text = (body.text or "").strip()
    lang = body.lang if body.lang in config.LANGS else "en"
    avatar_id = get_persona(body.avatar).id
    if not text:
        return {"reply": "", "demo": not config.llm_ready()}
    if (stand_in := fallback_reply(lang, avatar_id)) is not None:
        return {"reply": stand_in.text, "demo": stand_in.is_demo}
    try:
        return {"reply": await respond(text, lang, body.history, avatar_id), "demo": False}
    except Exception as exc:
        logger.warning("/chat: LLM call failed, falling back to demo line: %s", exc)
        failed = failure_reply(lang, avatar_id)
        return {"reply": failed.text, "demo": failed.is_demo, "error": str(exc)}


@app.websocket("/ws")
async def ws(sock: WebSocket) -> None:
    """One streaming turn per socket; the state and handlers are in VoiceSession."""
    await VoiceSession(sock).run()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server.app:app", host=config.HOST, port=config.PORT, reload=False)
