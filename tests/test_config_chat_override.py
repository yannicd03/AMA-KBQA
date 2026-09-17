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


# ---------------------------------------------------------------------------
# AMA_KBQA_CHAT_PROVIDER
# ---------------------------------------------------------------------------
# One level up from the model override: which *endpoint* the model lives on.
# The demo picker offers KIT models next to OpenRouter ones, and the id alone
# does not say where to post it. Without this channel the MCP tool servers
# would build a KIT client for an OpenRouter id — and, because they would then
# answer from a KIT model instead of failing loudly, the wrong endpoint would
# be invisible.

def test_get_chat_provider_uses_config_when_no_override(monkeypatch):
    monkeypatch.delenv(cfg.CHAT_PROVIDER_OVERRIDE_ENV_VAR, raising=False)
    assert cfg.get_chat_provider() == "kit"


def test_get_chat_provider_honors_override(monkeypatch):
    monkeypatch.setenv(cfg.CHAT_PROVIDER_OVERRIDE_ENV_VAR, "openrouter")
    assert cfg.get_chat_provider() == "openrouter"


def test_get_chat_provider_ignores_empty_override(monkeypatch):
    monkeypatch.setenv(cfg.CHAT_PROVIDER_OVERRIDE_ENV_VAR, "")
    assert cfg.get_chat_provider() == "kit"


def test_chat_model_name_follows_the_provider_override(monkeypatch):
    """The pair has to move together: with the provider switched and no model
    override, the model must come from the *new* provider's section."""
    cfg._config_cache["openrouter"] = {"chat_model": "deepseek/deepseek-v4-pro"}
    monkeypatch.delenv(cfg.CHAT_MODEL_OVERRIDE_ENV_VAR, raising=False)
    monkeypatch.setenv(cfg.CHAT_PROVIDER_OVERRIDE_ENV_VAR, "openrouter")
    assert cfg.get_chat_model_name() == "deepseek/deepseek-v4-pro"


def test_embedding_provider_is_not_affected_by_the_chat_override(monkeypatch):
    # Embeddings, reranking and synthesis stay where config.toml puts them no
    # matter which chat model the visitor picks.
    cfg._config_cache["llm"]["embedding_provider"] = "kit"
    monkeypatch.setenv(cfg.CHAT_PROVIDER_OVERRIDE_ENV_VAR, "openrouter")
    assert cfg._config_cache["llm"]["embedding_provider"] == "kit"


# ---------------------------------------------------------------------------
# [[frontend.chat_models]]
# ---------------------------------------------------------------------------

def test_frontend_chat_models_is_empty_without_the_section():
    # The pinned cache above has no [frontend] table at all.
    assert cfg.get_frontend_chat_models() == []


def test_frontend_chat_models_normalizes_entries():
    cfg._config_cache["frontend"] = {"chat_models": [
        {"provider": "openrouter", "id": "deepseek/deepseek-v4-pro",
         "name": "DeepSeek V4 Pro"},
        {"provider": "openrouter", "id": "a/b", "prompt_usd_per_m": 1.5,
         "completion_usd_per_m": "3.0", "default": True},
    ]}
    entries = cfg.get_frontend_chat_models()
    assert entries[0] == {
        "provider": "openrouter", "id": "deepseek/deepseek-v4-pro",
        "name": "DeepSeek V4 Pro", "prompt_usd_per_m": None,
        "completion_usd_per_m": None, "default": False,
    }
    assert entries[1]["prompt_usd_per_m"] == pytest.approx(1.5)
    assert entries[1]["completion_usd_per_m"] == pytest.approx(3.0)
    assert entries[1]["default"] is True


def test_frontend_chat_models_drops_bad_entries_without_raising():
    # A hand-edited config must degrade the picker to "KIT only" rather than
    # take the demo down.
    cfg._config_cache["frontend"] = {"chat_models": [
        {"provider": "openrouter", "id": "good/one"},
        # "deepseek" was an unsupported provider on demo-v2-int, which routed
        # its DeepSeek presets through OpenRouter. demo-booth ships the direct
        # [deepseek] endpoint, so this entry is now legitimately KEPT — the
        # assertion moved because the build's provider list really changed.
        {"provider": "deepseek", "id": "direct"},
        {"provider": "acme", "id": "x/y"},             # unknown provider
        {"provider": "openrouter", "id": "   "},       # empty id
        {"provider": "openrouter"},                    # no id at all
        "not-a-table",
        {"provider": "openrouter", "id": "priced/one", "prompt_usd_per_m": "abc"},
    ]}
    entries = cfg.get_frontend_chat_models()
    assert [e["id"] for e in entries] == ["good/one", "direct", "priced/one"]
    # A non-numeric price is dropped, the entry survives.
    assert entries[2]["prompt_usd_per_m"] is None


def test_frontend_chat_models_ignores_a_non_list_section():
    cfg._config_cache["frontend"] = {"chat_models": {"provider": "openrouter"}}
    assert cfg.get_frontend_chat_models() == []
