"""Tests for the AMA_KBQA_CHAT_PROVIDER override and the DeepSeek endpoint.

Background: the booth build lets the presenter switch the chat *endpoint*, not
just the model. MCP tool-server subprocesses are spawned with
env=os.environ.copy() and build their own chat client from their own
config.toml, so a provider switch that lives only in this process's config
cache would leave them posting an OpenRouter/DeepSeek model id to KIT ("Model
not found"). get_chat_provider() reads the env override first so the switch
crosses the process boundary, and every reader of [llm].chat_provider goes
through it.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

import ama_kbqa.config as cfg

REPO_ROOT = Path(__file__).resolve().parents[1]
SHIPPED_TOMLS = ("config.toml", "config.docker.toml")


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
            "openrouter": {
                "base_url": "https://openrouter.ai/api/v1",
                "chat_model": "deepseek/deepseek-v4-flash",
                "chat_model_provider": "",
            },
            "deepseek": {
                "base_url": "https://api.deepseek.com",
                "chat_model": "deepseek-flash",
            },
        },
    )
    monkeypatch.delenv(cfg.CHAT_PROVIDER_OVERRIDE_ENV_VAR, raising=False)
    monkeypatch.delenv(cfg.CHAT_MODEL_OVERRIDE_ENV_VAR, raising=False)


class TestChatProviderOverride:
    def test_falls_back_to_config_when_unset(self):
        assert cfg.get_chat_provider() == "kit"

    def test_env_override_wins(self, monkeypatch):
        monkeypatch.setenv(cfg.CHAT_PROVIDER_OVERRIDE_ENV_VAR, "deepseek")
        assert cfg.get_chat_provider() == "deepseek"

    def test_empty_override_is_treated_as_unset(self, monkeypatch):
        monkeypatch.setenv(cfg.CHAT_PROVIDER_OVERRIDE_ENV_VAR, "")
        assert cfg.get_chat_provider() == "kit"

    def test_model_name_follows_the_override_provider(self, monkeypatch):
        # No AMA_KBQA_CHAT_MODEL: the model must come from the *override*
        # provider's section, not from the config's chat_provider.
        monkeypatch.setenv(cfg.CHAT_PROVIDER_OVERRIDE_ENV_VAR, "deepseek")
        assert cfg.get_chat_model_name() == "deepseek-flash"

    def test_explicit_model_override_still_wins(self, monkeypatch):
        monkeypatch.setenv(cfg.CHAT_PROVIDER_OVERRIDE_ENV_VAR, "deepseek")
        monkeypatch.setenv(cfg.CHAT_MODEL_OVERRIDE_ENV_VAR, "deepseek-v4-pro")
        assert cfg.get_chat_model_name() == "deepseek-v4-pro"

    def test_openrouter_only_helpers_follow_the_override(self, monkeypatch):
        # get_chat_model_provider / get_provider_preferences short-circuit for
        # non-OpenRouter providers; they must key off the override too.
        monkeypatch.setenv(cfg.CHAT_PROVIDER_OVERRIDE_ENV_VAR, "deepseek")
        assert cfg.get_chat_model_provider() is None
        assert cfg.get_provider_preferences() is None
        monkeypatch.setenv(cfg.CHAT_PROVIDER_OVERRIDE_ENV_VAR, "openrouter")
        assert cfg.get_chat_model_provider() == ""


class TestDeepSeekClient:
    def test_builds_client_from_config_and_env_key(self, monkeypatch):
        monkeypatch.setenv(cfg.CHAT_PROVIDER_OVERRIDE_ENV_VAR, "deepseek")
        monkeypatch.setenv("DEEPSEEK_API_KEY", "ds-test-key")
        client = cfg.get_chat_client()
        assert str(client.base_url).rstrip("/") == "https://api.deepseek.com"
        assert client.api_key == "ds-test-key"

    def test_missing_key_raises_keyerror_naming_the_var(self, monkeypatch):
        monkeypatch.setenv(cfg.CHAT_PROVIDER_OVERRIDE_ENV_VAR, "deepseek")
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        with pytest.raises(KeyError, match="DEEPSEEK_API_KEY"):
            cfg.get_chat_client()

    def test_deepseek_key_alone_satisfies_the_startup_assertion(self, monkeypatch):
        monkeypatch.delenv("KIT_API_KEY", raising=False)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.setenv("DEEPSEEK_API_KEY", "ds-test-key")
        cfg.assert_provider_api_key_present()  # must not raise


class TestChatExtraBody:
    """DeepSeek's thinking mode is incompatible with our tool loop: it demands
    each assistant message be returned with its reasoning_content, which the
    fast path's synthesised tool_calls messages do not have."""

    def test_deepseek_disables_thinking_by_default(self, monkeypatch):
        monkeypatch.setenv(cfg.CHAT_PROVIDER_OVERRIDE_ENV_VAR, "deepseek")
        assert cfg.get_chat_extra_body() == {"thinking": {"type": "disabled"}}

    def test_deepseek_thinking_can_be_re_enabled_in_config(self, monkeypatch):
        cfg._config_cache["deepseek"]["thinking_enabled"] = True
        monkeypatch.setenv(cfg.CHAT_PROVIDER_OVERRIDE_ENV_VAR, "deepseek")
        assert cfg.get_chat_extra_body() == {}

    @pytest.mark.parametrize("provider", ["kit", "openrouter"])
    def test_other_providers_get_nothing(self, monkeypatch, provider):
        # Byte-identical request bodies to before this helper existed.
        monkeypatch.setenv(cfg.CHAT_PROVIDER_OVERRIDE_ENV_VAR, provider)
        assert cfg.get_chat_extra_body() == {}

    def test_openrouter_routing_preferences_are_untouched(self, monkeypatch):
        # The OpenRouter "provider" key stays in its own helper; the two must
        # not start shadowing each other.
        monkeypatch.setenv(cfg.CHAT_PROVIDER_OVERRIDE_ENV_VAR, "openrouter")
        cfg._config_cache["openrouter"]["chat_model_provider"] = "deepinfra/turbo"
        assert cfg.get_provider_preferences() == {
            "order": ["deepinfra/turbo"],
            "allow_fallbacks": False,
        }
        assert "provider" not in cfg.get_chat_extra_body()

    def test_shipped_configs_default_thinking_off(self):
        for name in SHIPPED_TOMLS:
            data = tomllib.loads((REPO_ROOT / name).read_text(encoding="utf-8"))
            assert data["deepseek"]["thinking_enabled"] is False, name


class TestFrontendChatModels:
    def test_reads_and_normalizes_entries(self, monkeypatch):
        monkeypatch.setattr(
            cfg,
            "get_frontend_config",
            lambda: {
                "chat_models": [
                    {"provider": "openrouter", "id": "vendor/model"},
                    {
                        "provider": "deepseek",
                        "id": "deepseek-flash",
                        "name": "DeepSeek Flash (direct)",
                        "prompt_usd_per_m": 0.30,
                        "completion_usd_per_m": 1.20,
                        "default": True,
                    },
                ]
            },
        )
        entries = cfg.get_frontend_chat_models()
        assert entries[0] == {
            "provider": "openrouter",
            "id": "vendor/model",
            "name": "",
            "prompt_usd_per_m": None,
            "completion_usd_per_m": None,
            "default": False,
        }
        assert entries[1]["prompt_usd_per_m"] == pytest.approx(0.30)
        assert entries[1]["default"] is True

    def test_drops_invalid_entries_without_raising(self, monkeypatch):
        monkeypatch.setattr(
            cfg,
            "get_frontend_config",
            lambda: {
                "chat_models": [
                    {"provider": "anthropic", "id": "claude"},  # unknown provider
                    {"provider": "openrouter"},                 # no id
                    {"provider": "openrouter", "id": "  "},     # blank id
                    "not-a-table",
                    {"provider": "openrouter", "id": "ok/model",
                     "prompt_usd_per_m": "free"},               # unparseable price
                ]
            },
        )
        entries = cfg.get_frontend_chat_models()
        assert [e["id"] for e in entries] == ["ok/model"]
        # A bad price costs that price, not the entry.
        assert entries[0]["prompt_usd_per_m"] is None

    @pytest.mark.parametrize("section", [{}, {"chat_models": "nonsense"}, {"chat_models": []}])
    def test_absent_or_malformed_section_yields_empty_list(self, monkeypatch, section):
        monkeypatch.setattr(cfg, "get_frontend_config", lambda: section)
        assert cfg.get_frontend_chat_models() == []


class TestShippedConfigs:
    @pytest.mark.parametrize("name", SHIPPED_TOMLS)
    def test_parses_and_declares_the_deepseek_endpoint(self, name):
        data = tomllib.loads((REPO_ROOT / name).read_text(encoding="utf-8"))
        assert data["deepseek"]["base_url"] == "https://api.deepseek.com"
        assert data["deepseek"]["chat_model"]

    @pytest.mark.parametrize("name", SHIPPED_TOMLS)
    def test_offers_at_least_one_entry_per_billed_provider(self, name):
        data = tomllib.loads((REPO_ROOT / name).read_text(encoding="utf-8"))
        entries = data["frontend"]["chat_models"]
        providers = {e["provider"] for e in entries}
        assert "openrouter" in providers
        assert "deepseek" in providers
        # Every shipped entry must survive the validator, or the booth picker
        # would silently drop it.
        assert all(e.get("id") for e in entries)
