"""The presenters, loaded from personas.toml.

The one place a presenter id is read. Everything else takes a ``Persona`` and asks it
for the voice, the A2F tuning or the grammatical gender it needs, so a third presenter
is a table in the TOML rather than another branch in every module that speaks.

    persona = get_persona(message.avatar)     # validates once, at the edge
    voice = persona.voice("ur", "uplift")     # resolved against the environment
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from . import config

PERSONAS_FILE = Path(__file__).with_name("personas.toml")


@dataclass(frozen=True, slots=True)
class Persona:
    """One presenter: who they are, how they sound, how their rig is tuned."""

    id: str
    display_name: str
    # "f" or "m". Urdu marks gender on verbs, so replies and UI copy agree with this.
    grammatical_gender: str
    # Urdu TTS engines this persona may use, best first.
    urdu_engines: tuple[str, ...]
    # "<engine>:<lang>" -> the name of the config setting holding the voice id.
    voice_settings: dict[str, str] = field(default_factory=dict)
    # Audio2Face per-clip tuning. Empty means the NIM's defaults.
    a2f_face_params: dict[str, float] = field(default_factory=dict)
    a2f_blendshape_multipliers: dict[str, float] = field(default_factory=dict)

    @property
    def is_feminine(self) -> bool:
        return self.grammatical_gender == "f"

    def voice(self, lang: str, engine: str) -> str:
        """The voice id for this persona on that engine, read from the environment.

        The id lives in config rather than the TOML because it is deployment-specific
        (the Uplift ones are tied to an account), while which setting to read is a
        fact about the persona.
        """
        setting = self.voice_settings.get(f"{engine}:{lang}")
        return getattr(config, setting, "") if setting else ""

    def a2f_tuning(self) -> tuple[dict[str, float] | None, dict[str, float] | None]:
        """(face_params, blendshape_multipliers) for this persona's clips.

        None means "send nothing and let the NIM use its own defaults", which is
        different from an empty override and is what a persona with no measured
        tuning wants.
        """
        return (self.a2f_face_params or None), (self.a2f_blendshape_multipliers or None)

    def urdu_engine(self, configured: str) -> str:
        """Which engine actually speaks this persona's Urdu.

        ``configured`` is the deployment's preference (APP_UR_TTS). A persona that
        cannot use it — Matcha-TTS is a female voice — falls back to its own first
        supported engine rather than being silent.
        """
        if configured in self.urdu_engines:
            return configured
        return self.urdu_engines[0]


def _parse(raw: dict) -> dict[str, Persona]:
    personas = {}
    for persona_id, table in raw.items():
        a2f = table.get("a2f", {})
        personas[persona_id] = Persona(
            id=persona_id,
            display_name=table["display_name"],
            grammatical_gender=table["grammatical_gender"],
            urdu_engines=tuple(table["urdu_engines"]),
            voice_settings=dict(table.get("voices", {})),
            a2f_face_params=dict(a2f.get("face_params", {})),
            a2f_blendshape_multipliers=dict(a2f.get("blendshape_multipliers", {})),
        )
    if not personas:
        raise RuntimeError(f"{PERSONAS_FILE} defines no presenters")
    return personas


@lru_cache(maxsize=1)
def all_personas() -> dict[str, Persona]:
    """Every presenter, by id. Read once; a malformed file fails at startup."""
    return _parse(tomllib.loads(PERSONAS_FILE.read_text(encoding="utf-8")))


def persona_ids() -> tuple[str, ...]:
    """The ids the browser may send as ``avatar``. Declaration order."""
    return tuple(all_personas())


def default_persona() -> Persona:
    """The presenter a turn uses when the browser names none, or names an unknown one.

    The first table in personas.toml, so the file decides rather than a literal
    scattered through the code.
    """
    return next(iter(all_personas().values()))


def tts_engine(lang: str, persona: Persona) -> str:
    """Which engine speaks this language for this presenter.

    Urdu is the interesting case: the deployment's preference (APP_UR_TTS) only applies
    if the persona supports it, because Matcha-TTS is a female voice.

    Here rather than in providers/tts.py so that config.tts_ready() can ask the same
    question without the bottom layer importing the adapter layer.
    """
    if lang == "ur":
        return persona.urdu_engine(config.UR_TTS)
    return config.EN_TTS


def get_persona(persona_id: str | None) -> Persona:
    """Validate an id from outside and turn it into a Persona.

    Called once per turn at the API edge. Anything unrecognised becomes the default
    rather than an error, because a presenter the server does not know is no reason to
    refuse to speak.
    """
    if persona_id:
        persona = all_personas().get(persona_id)
        if persona is not None:
            return persona
    return default_persona()
