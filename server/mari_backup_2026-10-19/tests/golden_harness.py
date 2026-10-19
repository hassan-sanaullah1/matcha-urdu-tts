"""Collects the kiosk's observable behaviour for the golden snapshot test.

Everything here goes through the outermost seams — the text the TTS request carries, the
messages the LLM request carries, the frames the socket receives — so the snapshot stays
valid while the code behind those seams is reorganised. Only the ``API`` section below
names internal modules.

Regenerate ONLY on a deliberate behaviour change:  python -m tests.golden_harness --write
"""

from __future__ import annotations

import ast
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "tests" / "fixtures" / "golden.json"
LANGS = ("en", "ur")
AVATARS = ("female", "male")

# ── API: the only lines that name internal modules ──────────────────


def api():
    from server import app as A
    from server import config as C
    from server import rag
    from server.agent.greeting import is_greeting
    from server.agent.replies import canned, pitch_override
    from server.agent.reply_fixes import force_salam, gender_agreement
    from server.agent.turn import run_reply
    from server.normalization import normalize_for_tts
    from server.prompts import get_prompts
    from server.rag.expansion import expand_for_lexical
    from server.rag.generation import GenerationService
    from server.rag.retriever import RetrievalResult

    return {
        "normalize": normalize_for_tts,
        "force_salam": force_salam,
        "gender_agreement": gender_agreement,
        "is_greeting": is_greeting,
        "expand": expand_for_lexical,
        "canned": canned,
        "pitch": pitch_override,
        "core_brief": get_prompts().core_brief,
        "GenerationService": GenerationService,
        "RetrievalResult": RetrievalResult,
        "app": A,
        "config": C,
        "rag": rag,
        "run_reply": run_reply,
        "chat": A.chat,
        "ChatIn": A.ChatIn,
    }


# ── inputs ──────────────────────────────────────────────────────────


def _string_literals(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            s = node.value.strip()
            if 2 <= len(s) <= 2000 and not s.startswith(("\n", "The ", "A ", "An ")):
                out.append(node.value)
    return out


def collect_inputs(a) -> dict:
    tests = ROOT / "tests"
    tts_sources = [
        "test_tts_spoken.py",
        "test_english_normalise.py",
        "test_addresses_and_symbols.py",
        "test_names_and_ranks.py",
        "test_places.py",
        "test_personas.py",
        "test_prompts_and_glossary.py",
    ]
    literals: list[str] = []
    for name in tts_sources:
        literals += _string_literals(tests / name)

    kb = (ROOT / "server" / "data" / "mari_energies_knowledge_base.md").read_text(encoding="utf-8")
    paragraphs = [p.strip() for p in kb.split("\n\n") if p.strip()]

    prompt_lines: list[str] = []
    for md in sorted((ROOT / "server" / "prompts").glob("*/*.md")):
        prompt_lines += [
            ln.strip() for ln in md.read_text(encoding="utf-8").splitlines() if ln.strip()
        ]

    brief_lines = [ln for ln in a["core_brief"].splitlines() if ln.strip()]
    tts_texts = sorted(set(literals + paragraphs + prompt_lines + brief_lines))

    reply_sources = ["test_greetings.py", "test_personas.py", "test_persona_gender.py"]
    replies: list[str] = []
    for name in reply_sources:
        replies += _string_literals(tests / name)
    replies = sorted(set(replies + prompt_lines))

    from tests.retrieval_cases import CASES

    queries = sorted({q for q, _ in CASES} | set(replies))
    return {"tts": tts_texts, "replies": replies, "queries": queries}


# ── fake network ────────────────────────────────────────────────────


def _digest_system(payload: dict) -> dict:
    """System prompts are snapshotted in full under "prompts"; per turn a hash is enough."""
    import copy
    import hashlib

    payload = copy.deepcopy(payload)
    for msg in payload.get("messages", []):
        if msg.get("role") == "system":
            body = msg["content"].encode("utf-8")
            msg["content"] = f"sha256:{hashlib.sha256(body).hexdigest()} len:{len(msg['content'])}"
    return payload


class FakeNet:
    """Stands in for httpx.AsyncClient: answers the LLM and Uplift, records payloads."""

    def __init__(self) -> None:
        self.llm_reply = ""
        self.llm_mode = "ok"  # ok | error_frame | midstream_fail | connect_fail
        self.calls: list[dict] = []

    def client_class(net):  # noqa: N805
        import httpx

        class _Resp:
            def __init__(self, url, payload, content=b"", ctype="application/json"):
                self.url, self.payload = url, payload
                self.content = content
                self.headers = {"content-type": ctype}
                self.status_code = 200

            def raise_for_status(self):
                return None

            def json(self):
                return {"choices": [{"message": {"content": net.llm_reply}}]}

            async def aiter_lines(self):
                if net.llm_mode == "error_frame":
                    yield 'data: {"error": {"message": "context length exceeded"}}'
                    yield "data: [DONE]"
                    return
                text = net.llm_reply
                cut = len(text) // 2 if net.llm_mode == "midstream_fail" else len(text)
                for i in range(0, cut, 3):
                    frame = {"choices": [{"delta": {"content": text[i : i + 3]}}]}
                    yield "data: " + json.dumps(frame, ensure_ascii=False)
                if net.llm_mode == "midstream_fail":
                    raise httpx.ReadTimeout("mid-stream")
                yield "data: [DONE]"

        class _Stream:
            def __init__(self, resp):
                self.resp = resp

            async def __aenter__(self):
                return self.resp

            async def __aexit__(self, *exc):
                return False

        class _Client:
            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

            def _record(self, url, json_payload):
                kind = "llm" if "chat/completions" in url else "tts"
                net.calls.append({"kind": kind, "payload": _digest_system(json_payload)})
                if kind == "llm" and net.llm_mode == "connect_fail":
                    raise httpx.ConnectError("unreachable")

            async def post(self, url, json=None, headers=None, **kw):
                self._record(url, json)
                if "chat/completions" in url:
                    return _Resp(url, json)
                return _Resp(url, json, content=b"ID3fake", ctype="audio/mpeg")

            def stream(self, method, url, json=None, headers=None, **kw):
                self._record(url, json)
                return _Stream(_Resp(url, json))

        return _Client


class Sock:
    def __init__(self) -> None:
        self.sent: list = []

    async def send_json(self, payload: dict) -> None:
        self.sent.append(payload)

    async def send_bytes(self, data: bytes) -> None:
        self.sent.append({"bytes": len(data)})


# ── scenarios ───────────────────────────────────────────────────────

LLM_REPLIES = {
    "en": [
        "Hello! I'm Maryam, a representative of Mari Energies. Our FY2024-25 net profit was "
        "PKR 65.14bn, and EPS was 54.25 rupees. Sky47 runs Huawei Ascend NPU and NVIDIA-compatible "
        "GPU clusters at up to 50kW per rack. Call (+92) 51-111 410 410 or visit marienergies.com.pk.",
        "MD/CEO Faheem Haider leads the company; the Chairman is Lt Gen (R) Anwar Ali Hyder, HI(M). "
        "Head office: 21 Mauve Area, 3rd Road, G-10/4, Islamabad 44000. We produce 127 KBOEPD, "
        "reserves are 952 MMBOE, and CO2 and CH4 are part of GEM Energy's methane mitigation work!",
    ],
    "ur": [
        "وعلیکم السلام، میں مریم ہوں، Mari Energies کا نمائندہ۔ ہم نے 1954 میں ڈہرکی میں Mari Gas Field "
        "دریافت کیا تھا۔ Sky47 کے کلاؤڈ انفرا اسٹرکچر میں Huawei Ascend NPU کلسٹرز استعمال ہوتے ہیں؟ "
        "مزید معلومات کے لیے marienergies.com.pk دیکھیں۔",
        "میں بتا سکتا ہوں کہ PKR 65.14bn منافع ہوا، اور کمپنی کی سبسڈیری Mari Minerals کاپر اور گولڈ پر کام "
        "کرتی ہے۔ MD/CEO Faheem Haider ہیں… میتھین مٹیگیشن GEM Energy کا کام ہے۔",
    ],
}

TRANSCRIPTS = {
    "en": [
        "hello",
        "What is Mari's net profit?",
        "Please introduce yourself and tell me about Sky47",
    ],
    "ur": ["السلام علیکم", "ماری انرجیز کا منافع کتنا ہے؟", "آپ کون ہیں؟"],
}

HISTORY = [
    {"role": "user", "text": "Tell me about the verticals"},
    {
        "role": "assistant",
        "text": "We have four verticals: Mari Services, Mari Minerals, Sky47 and GEM Energy.",
    },
    {"role": "system", "text": "ignored"},
    {"role": "user", "text": "x" * 900},
]


def _configure(a, monkeypatch_setattr) -> FakeNet:
    import httpx

    from server import avatar

    C = a["config"]
    net = FakeNet()
    monkeypatch_setattr(httpx, "AsyncClient", net.client_class())
    for key, value in {
        "LLM_BASE": "http://llm.test/v1",
        "LLM_KEY": "k",
        "LLM_MODEL": "qwen3.5",
        "LLM_PROVIDER": "vllm",
        "FORCE_DEMO": False,
        "UPLIFT_KEY": "u",
        "EN_TTS": "uplift",
        "SONIOX_KEY": "",
        "A2F_URL": "",
        # Voice ids land in every TTS payload, so a developer's .env must not leak in.
        "UPLIFT_VOICE": "v_8eelc901v6",
        "UPLIFT_VOICE_EN": "v_8eelc901v6",
        "UPLIFT_VOICE_MALE": "v_8eelc901v6",
    }.items():
        monkeypatch_setattr(C, key, value)
    from server.providers import tts

    monkeypatch_setattr(tts, "_tts_cache", {})
    monkeypatch_setattr(avatar, "_client", None)
    monkeypatch_setattr(avatar, "_resolved", False)
    rag = a["rag"]
    monkeypatch_setattr(rag, "_retriever", None)

    async def _no_sleep(*_a, **_k):
        return None

    monkeypatch_setattr(asyncio, "sleep", _no_sleep)
    return net


def run_turns(a, monkeypatch_setattr, set_env) -> dict:
    net = _configure(a, monkeypatch_setattr)
    out: dict = {}

    def turn(label: str, transcript: str, lang: str, avatar: str, **kw):
        net.calls.clear()
        sock = Sock()
        asyncio.run(a["run_reply"](sock, transcript, lang, avatar_id=avatar, **kw))
        out[label] = {"socket": sock.sent, "net": list(net.calls)}

    for lang in LANGS:
        for avatar in AVATARS:
            for i, transcript in enumerate(TRANSCRIPTS[lang]):
                net.llm_mode = "ok"
                net.llm_reply = LLM_REPLIES[lang][i % 2]
                turn(
                    f"ws/{lang}/{avatar}/{i}",
                    transcript,
                    lang,
                    avatar,
                    history=HISTORY if i == 1 else None,
                )
            for mode in ("error_frame", "midstream_fail", "connect_fail"):
                net.llm_mode = mode
                net.llm_reply = LLM_REPLIES[lang][0]
                turn(f"ws/{lang}/{avatar}/{mode}", TRANSCRIPTS[lang][1], lang, avatar)
            net.llm_mode = "ok"
            turn(f"ws/{lang}/{avatar}/empty_spoke", "", lang, avatar, spoke=True)
            turn(f"ws/{lang}/{avatar}/empty_silent", "", lang, avatar, spoke=False)
            turn(
                f"ws/{lang}/{avatar}/typed",
                TRANSCRIPTS[lang][2],
                lang,
                avatar,
                echo_transcript=False,
            )

            for i, transcript in enumerate(TRANSCRIPTS[lang]):
                net.calls.clear()
                net.llm_reply = LLM_REPLIES[lang][i % 2]
                body = a["ChatIn"](
                    text=transcript, lang=lang, avatar=avatar, history=HISTORY if i == 1 else None
                )
                reply = asyncio.run(a["chat"](body))
                out[f"chat/{lang}/{avatar}/{i}"] = {"reply": reply, "net": list(net.calls)}

    C = a["config"]
    monkeypatch_setattr(C, "FORCE_DEMO", True)
    for lang in LANGS:
        for avatar in AVATARS:
            turn(f"ws/{lang}/{avatar}/demo", TRANSCRIPTS[lang][1], lang, avatar)
    monkeypatch_setattr(C, "FORCE_DEMO", False)

    set_env("MARI_PITCH_ONLY", "1")
    for lang in LANGS:
        for avatar in AVATARS:
            turn(f"ws/{lang}/{avatar}/pitch", TRANSCRIPTS[lang][1], lang, avatar)
    set_env("MARI_PITCH_ONLY", None)
    return out


def collect(monkeypatch_setattr, set_env, inputs: dict | None = None) -> dict:
    sys.path.insert(0, str(ROOT))
    a = api()
    inputs = inputs or collect_inputs(a)

    normalize = {lang: [a["normalize"](t, lang) for t in inputs["tts"]] for lang in LANGS}
    fixes = {
        f"{lang}/{avatar}": [
            [a["force_salam"](r, lang), a["gender_agreement"](r, lang, avatar)]
            for r in inputs["replies"]
        ]
        for lang in LANGS
        for avatar in AVATARS
    }
    greetings = [a["is_greeting"](q) for q in inputs["queries"]]
    expansion = [list(a["expand"](q)) for q in inputs["queries"]]

    gen = a["GenerationService"]()
    RR = a["RetrievalResult"]
    with_context = RR(
        context="## 1.5 Shareholding\nFauji Foundation 40 percent.",
        sources=["kb#1.5"],
        glossary_block="ABBREVIATIONS:\nMPCL = Mari Petroleum",
    )
    prompts = {}
    for avatar in (*AVATARS, "unknown"):
        for lang in (*LANGS, "xx"):
            for greeting in (False, True):
                for label, result in (("context", with_context), ("empty", RR())):
                    p = gen.build_prompt(
                        result, lang, core_brief=a["core_brief"], greeting=greeting, avatar=avatar
                    )
                    prompts[f"{avatar}/{lang}/{greeting}/{label}"] = [
                        p.system,
                        p.sources,
                        p.grounded,
                    ]

    canned = {}
    for name in ("demo", "no_speech"):
        for avatar in (*AVATARS, "unknown"):
            for lang in (*LANGS, "xx"):
                canned[f"{name}/{avatar}/{lang}"] = a["canned"](name, lang, avatar)
    set_env("MARI_PITCH_ONLY", "1")
    for avatar in AVATARS:
        for lang in (*LANGS, "xx"):
            canned[f"pitch/{avatar}/{lang}"] = a["pitch"](lang, avatar)
    set_env("MARI_PITCH_ONLY", None)

    return {
        "inputs": inputs,
        "normalize": normalize,
        "reply_fixes": fixes,
        "is_greeting": greetings,
        "expansion": expansion,
        "prompts": prompts,
        "canned": canned,
        "turns": run_turns(a, monkeypatch_setattr, set_env),
    }


def _standalone_patchers():
    import os

    def setattr_(obj, name, value):
        setattr(obj, name, value)

    def set_env(key, value):
        from server import config as C

        if value is None:
            os.environ.pop(key, None)
            C.ENV.pop(key, None)
        else:
            os.environ[key] = value
            C.ENV[key] = value

    return setattr_, set_env


if __name__ == "__main__":
    if "--write" not in sys.argv:
        sys.exit("refusing to overwrite the golden fixture without --write")
    data = collect(*_standalone_patchers())
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {FIXTURE} ({FIXTURE.stat().st_size // 1024} KB)")
