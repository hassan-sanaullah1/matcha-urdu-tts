"""Presenters must stay data, not branches.

The kiosk shows one of two presenters, and before this they differed by about ten
`if avatar == "male"` branches and twenty `"female"` defaults spread across TTS, A2F,
the prompts and the routes. Adding a third would have meant finding every one.

Now `server/personas.toml` is the only place a presenter is described. These tests
fail if that starts leaking back out, and they are the reason the ARCHITECTURE
invariant "no persona branches" can be stated as a fact.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from server.personas import PERSONAS_FILE, all_personas, default_persona, get_persona, persona_ids

ROOT = Path(__file__).resolve().parent.parent
SERVER = ROOT / "server"

# personas.py reads the ids, so it is the one module allowed to name them.
ALLOWED_TO_NAME_PRESENTERS = {SERVER / "personas.py"}


def _python_sources() -> list[Path]:
    return [p for p in SERVER.rglob("*.py") if "__pycache__" not in p.parts]


def test_no_presenter_id_outside_personas() -> None:
    """A presenter id in code is a branch waiting to be forgotten."""
    ids = persona_ids()
    pattern = re.compile("|".join(rf'"{re.escape(i)}"' for i in ids))
    offenders: list[str] = []
    for path in _python_sources():
        if path in ALLOWED_TO_NAME_PRESENTERS:
            continue
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if pattern.search(line):
                offenders.append(f"{path.relative_to(ROOT)}:{number}: {line.strip()}")
    assert not offenders, (
        "presenter ids must only appear in personas.toml/personas.py — read the value "
        "off the Persona instead:\n  " + "\n  ".join(offenders)
    )


def test_every_presenter_defines_the_same_fields() -> None:
    """One presenter quietly missing a field is how the two drift apart."""
    personas = list(all_personas().values())
    assert len(personas) >= 2, "there should be at least two presenters to compare"

    voice_keys = [set(p.voice_settings) for p in personas]
    assert all(keys == voice_keys[0] for keys in voice_keys), (
        "every presenter needs a voice for the same engine/language pairs; "
        f"got {[sorted(k) for k in voice_keys]}"
    )


@pytest.mark.parametrize("persona_id", persona_ids())
def test_presenter_is_fully_specified(persona_id: str) -> None:
    persona = get_persona(persona_id)
    assert persona.display_name, "a presenter needs a name for the prompts"
    assert persona.grammatical_gender in {"f", "m"}, "Urdu agreement needs a gender"
    assert persona.urdu_engines, "a presenter with no Urdu engine cannot speak Urdu"


@pytest.mark.parametrize("persona_id", persona_ids())
def test_every_presenter_can_speak_both_languages(persona_id: str) -> None:
    """A presenter with no voice for a language would be mute on that tab."""
    persona = get_persona(persona_id)
    for lang in ("en", "ur"):
        engine = persona.urdu_engine("uplift") if lang == "ur" else "uplift"
        assert persona.voice(lang, engine), f"{persona_id} has no {lang} voice for {engine}"


def test_unknown_presenter_falls_back_rather_than_failing() -> None:
    """The kiosk must always speak, even if the browser sends something odd."""
    assert get_persona("robot").id == default_persona().id
    assert get_persona(None).id == default_persona().id
    assert get_persona("").id == default_persona().id


def test_default_presenter_is_the_first_in_the_file() -> None:
    assert default_persona().id == persona_ids()[0]


def test_matcha_is_offered_only_to_a_feminine_presenter() -> None:
    """Matcha-TTS is a female voice; a male presenter routed to it would sound wrong."""
    for persona in all_personas().values():
        if "matcha" in persona.urdu_engines:
            assert persona.is_feminine, (
                f"{persona.id} lists matcha but is not feminine — it is a female voice"
            )


def test_urdu_engine_falls_back_within_the_persona() -> None:
    """A deployment choosing an engine a presenter cannot use must not mute them."""
    for persona in all_personas().values():
        chosen = persona.urdu_engine("matcha")
        assert chosen in persona.urdu_engines
        assert persona.urdu_engine("not-an-engine") in persona.urdu_engines


def test_a2f_tuning_is_none_when_unmeasured() -> None:
    """None means "let the NIM use its defaults", which is not the same as {}."""
    for persona in all_personas().values():
        face, multipliers = persona.a2f_tuning()
        assert face is None or face, "an empty override must be None, not {}"
        assert multipliers is None or multipliers


def test_prompt_and_reply_files_exist_for_every_presenter() -> None:
    """personas.toml and the prompt files have to agree or startup fails mid-turn."""
    from server.prompts import get_prompts

    prompts = get_prompts()
    for persona_id in persona_ids():
        for lang in ("en", "ur"):
            assert (persona_id, lang) in prompts.system, f"no system prompt for {persona_id}/{lang}"
            assert (persona_id, lang) in prompts.greeting, f"no greeting for {persona_id}/{lang}"
        for name in ("demo", "no_speech"):
            assert persona_id in prompts.replies[name], f"no {name} line for {persona_id}"


def test_frontend_and_backend_agree_on_grammatical_gender() -> None:
    """Urdu marks gender on verbs, so the spoken reply and the UI have to agree.

    The server picks the agreement pass for the reply; the browser picks the status
    copy. Two sources, so this checks they still say the same thing — the male
    presenter used to speak masculine Urdu under feminine UI labels.
    """
    registry = (ROOT / "frontend" / "components" / "avatar" / "models.ts").read_text(
        encoding="utf-8"
    )
    # `id: "x",` ... `grammaticalGender: "y",` within the same object literal.
    entries = dict(
        re.findall(
            r'id:\s*"(\w+)"[\s\S]*?grammaticalGender:\s*"([fm])"',
            registry,
        )
    )
    assert entries, "could not read grammaticalGender out of the frontend registry"
    for persona_id, persona in all_personas().items():
        assert entries.get(persona_id) == persona.grammatical_gender, (
            f"{persona_id}: personas.toml says {persona.grammatical_gender!r} but the "
            f"frontend says {entries.get(persona_id)!r}"
        )


def test_frontend_profiles_match_the_backend_presenters() -> None:
    """The id travels over the WebSocket, so the two sides have to list the same set."""
    registry = (ROOT / "frontend" / "components" / "avatar" / "models.ts").read_text(
        encoding="utf-8"
    )
    match = re.search(r"export type AvatarId\s*=\s*([^;]+);", registry)
    assert match, "could not find AvatarId in the frontend avatar registry"
    frontend_ids = set(re.findall(r'"([^"]+)"', match.group(1)))
    assert frontend_ids == set(persona_ids()), (
        f"frontend offers {sorted(frontend_ids)} but {PERSONAS_FILE.name} defines "
        f"{sorted(persona_ids())}; a presenter the server does not know falls back silently"
    )
