"""Tests for the AMA_KBQA_CHAT_MODEL / AMA_KBQA_CHAT_TEMPERATURE env-var
override in get_chat_model_name() / get_chat_temperature().

Background: the demo's model picker (chat_controls.apply_chat_settings)
mutates this process's in-memory config cache, but MCP tool-server
subprocesses are spawned with env=os.environ.copy() and load their own
config.toml independently — they never see that in-memory mutation. These
two functions must honor an env-var override so the picked model/temperature
reaches those subprocesses too, while falling back to config.toml exactly as
before when the override is unset.
"""

from __future__ import annotations

import pytest

import ama_kbqa.config as cfg


@pytest.fixture(autouse=True)
def _fixed_config_cache(monkeypatch):
    """Point load_config() at a fixed in-memory config so these tests never
    touch the real config.toml on disk."""
    monkeypatch.setattr(
        cfg,
        "_config_cache",
        {
            "llm": {"chat_provider": "kit", "chat_temperature": 0.42},
            "kit": {"chat_model": "kit.glm-5.3"},
        },
    )


def test_get_chat_model_name_uses_config_when_no_override(monkeypatch):
    monkeypatch.delenv(cfg.CHAT_MODEL_OVERRIDE_ENV_VAR, raising=False)
    assert cfg.get_chat_model_name() == "kit.glm-5.3"


def test_get_chat_model_name_honors_override(monkeypatch):
    monkeypatch.setenv(cfg.CHAT_MODEL_OVERRIDE_ENV_VAR, "kit.deepseek-v4-flash")
    assert cfg.get_chat_model_name() == "kit.deepseek-v4-flash"


def test_get_chat_model_name_ignores_empty_override(monkeypatch):
    # An empty string is falsy — treat it like "unset" rather than picking an
    # empty model id.
    monkeypatch.setenv(cfg.CHAT_MODEL_OVERRIDE_ENV_VAR, "")
    assert cfg.get_chat_model_name() == "kit.glm-5.3"


def test_get_chat_temperature_uses_config_when_no_override(monkeypatch):
    monkeypatch.delenv(cfg.CHAT_TEMPERATURE_OVERRIDE_ENV_VAR, raising=False)
    assert cfg.get_chat_temperature() == pytest.approx(0.42)


def test_get_chat_temperature_honors_override(monkeypatch):
    monkeypatch.setenv(cfg.CHAT_TEMPERATURE_OVERRIDE_ENV_VAR, "1.35")
    assert cfg.get_chat_temperature() == pytest.approx(1.35)


def test_get_chat_temperature_falls_back_on_invalid_override(monkeypatch):
    monkeypatch.setenv(cfg.CHAT_TEMPERATURE_OVERRIDE_ENV_VAR, "not-a-float")
    # Malformed override must not raise — fall back to config.toml's value.
    assert cfg.get_chat_temperature() == pytest.approx(0.42)
