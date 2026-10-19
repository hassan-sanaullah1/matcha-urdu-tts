"""`.env.example` has to list every setting the code reads.

It is the only description of a deployment's environment, and it had drifted badly:
24 of the settings `server/config.py` read were undocumented, including every DeepSeek
key and the host and port, while one it did document (`APP_SONIOX_API_BASE`) was read
by nothing — so anyone copying the file configured the wrong key and the server quietly
used its default.

A missing entry is a setting nobody deploying this can know about, so it fails the
build rather than waiting to be noticed.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
ENV_EXAMPLE = ROOT / ".env.example"
CONFIG = ROOT / "server" / "config.py"

# Read from the environment by the retrieval layer, which uses pydantic-settings with
# this prefix rather than config.py's reader.
RAG_PREFIX = "MARI_RAG_"

# Set by the test harnesses and the recording aids, not by a deployment.
NOT_FOR_DEPLOYMENTS = frozenset({"MARI_PITCH_ONLY"})


def _settings_config_reads() -> set[str]:
    """Every environment key `server/config.py` looks up."""
    source = CONFIG.read_text(encoding="utf-8")
    return set(re.findall(r'env\(\s*"([A-Z0-9_]+)"', source)) | set(
        re.findall(r'ENV\.get\(\s*"([A-Z0-9_]+)"', source)
    )


def _rag_setting_names() -> set[str]:
    """Every `MARI_RAG_*` key, from the RagSettings field names."""
    from server.rag.settings import RagSettings

    return {f"{RAG_PREFIX}{name.upper()}" for name in RagSettings.model_fields}


def _declared_in_example() -> set[str]:
    """Keys `.env.example` lists, whether commented out or not."""
    text = ENV_EXAMPLE.read_text(encoding="utf-8")
    return set(re.findall(r"^#?\s*([A-Z][A-Z0-9_]*)=", text, re.M))


def test_every_provider_setting_is_documented() -> None:
    """Everything server/config.py reads: the providers, the keys, the host and port.

    These are the settings a deployment has to get right and cannot discover any other
    way. The retrieval tunables are held to a weaker rule below, because they have
    measured defaults and their own documentation.
    """
    missing = sorted(_settings_config_reads() - _declared_in_example() - NOT_FOR_DEPLOYMENTS)
    assert not missing, (
        "these settings are read by server/config.py but missing from .env.example, so "
        f"nobody deploying this can know they exist: {missing}"
    )


def test_documented_retrieval_settings_are_real() -> None:
    """A MARI_RAG_* key in the file that is not a RagSettings field does nothing.

    pydantic-settings ignores an unknown key silently, so a typo here reads as a
    tunable that simply has no effect.
    """
    documented = {k for k in _declared_in_example() if k.startswith(RAG_PREFIX)}
    unknown = sorted(documented - _rag_setting_names())
    assert not unknown, (
        f".env.example lists retrieval settings that are not RagSettings fields, so "
        f"they are ignored: {unknown}"
    )


def test_no_documented_setting_is_dead() -> None:
    """A key in the file that nothing reads sends people to configure the wrong thing."""
    known = _settings_config_reads() | _rag_setting_names() | NOT_FOR_DEPLOYMENTS
    dead = sorted(
        key for key in _declared_in_example() - known if key.startswith(("APP_", "MARI_"))
    )
    assert not dead, (
        f".env.example documents settings the code never reads: {dead}. Either wire "
        "them up or delete the lines."
    )


def test_no_secret_has_a_value() -> None:
    """The example file is committed, so every credential in it must be blank."""
    offenders = []
    for line in ENV_EXAMPLE.read_text(encoding="utf-8").splitlines():
        if line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.split(" #")[0].strip()
        if not value:
            continue
        if re.search(r"(KEY|TOKEN|SECRET|PASSWORD)$", key):
            offenders.append(key)
    assert not offenders, f"credentials with values in a committed file: {offenders}"


@pytest.mark.parametrize("required", ["MARI_RAG_QDRANT_API_KEY"])
def test_settings_compose_requires_are_documented(required: str) -> None:
    """compose refuses to start without these, so the file has to mention them."""
    assert required in _declared_in_example()
