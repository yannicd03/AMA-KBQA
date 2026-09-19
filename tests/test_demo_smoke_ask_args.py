"""Tests for the smoke script's "provider:model" argument.

The booth build smoke-tests non-KIT endpoints through this script, so the
parsing is what decides whether a run exercises OpenRouter or silently posts
an OpenRouter model id to KIT. Every pre-existing invocation passes a bare
KIT id and must keep working.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "demo_smoke_ask.py"


def _load_script():
    """Import the script by path: scripts/ is not a package."""
    spec = importlib.util.spec_from_file_location("demo_smoke_ask", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def script():
    return _load_script()


@pytest.mark.parametrize(
    "arg,expected",
    [
        ("openrouter:deepseek/deepseek-v4-flash", ("openrouter", "deepseek/deepseek-v4-flash")),
        ("deepseek:deepseek-flash", ("deepseek", "deepseek-flash")),
        ("kit:kit.glm-5.3", ("kit", "kit.glm-5.3")),
        # Bare ids keep meaning KIT.
        ("kit.mistral-small-4-119b-a8b", ("kit", "kit.mistral-small-4-119b-a8b")),
        # Unknown prefix: the whole string is the model id, not a provider.
        ("vendor:model", ("kit", "vendor:model")),
        # Trailing colon with no model is not a provider split either.
        ("openrouter:", ("kit", "openrouter:")),
    ],
)
def test_parse_model_arg(script, arg, expected):
    assert script.parse_model_arg(arg) == expected


def test_default_model_argument_parses_as_kit(script):
    from ama_kbqa.frontend.utils.chat_controls import DEFAULT_MODEL_PREFERENCE

    provider, model = script.parse_model_arg(DEFAULT_MODEL_PREFERENCE[0])
    assert provider == "kit"
    assert model == DEFAULT_MODEL_PREFERENCE[0]
