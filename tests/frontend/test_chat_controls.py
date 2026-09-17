"""Tests for the demo chat sidebar helpers in chat_controls."""

from __future__ import annotations

import os

import pytest

from ama_kbqa.frontend.utils import chat_controls


def _entry(
    id_,
    *,
    name=None,
    connection_type="local",
    preset=False,
    tags=None,
    hidden=False,
    capabilities=None,
):
    """Build a normalized catalog entry shaped like
    ``config_editor.fetch_provider_models_meta``'s output."""
    return {
        "id": id_,
        "name": name or id_,
        "connection_type": connection_type,
        "preset": preset,
        "tags": tags or [],
        "hidden": hidden,
        "capabilities": capabilities if capabilities is not None else {"vision": False},
    }


# A normalized catalog shaped like a real KIT /models response: 3 locally-hosted
# chat LLMs, 5 non-chat local models (all hidden, 3 also with null capabilities),
# 3 local routing aliases, a local preset bundle, and 2 external models. Only the
# 3 chat LLMs must survive the filter.
_LIVE_KIT_CATALOG = [
    _entry("kit.deepseek-v4-flash", name="DeepSeek V4 Flash"),
    _entry("kit.glm-5.3", name="GLM-5.3"),
    _entry("kit.mistral-small-4-119b-a8b", name="Mistral Small 4"),
    _entry("kit.qwen3-reranker-8b", hidden=True, capabilities={"vision": True}),
    _entry("kit.qwen3-embedding-8b", hidden=True, capabilities={"vision": False}),
    _entry("kit.flux.2-dev", hidden=True, capabilities=None),
    _entry("kit.voxtral-4b-tts-2603", hidden=True, capabilities=None),
    _entry("kit.whisper-large-v3", hidden=True, capabilities=None),
    _entry("alias.simple-local", name="Alias: Simple Model (Local)", tags=["Alias"]),
    _entry("alias.medium-local", name="Alias: Medium Model (Local)", tags=["Alias"]),
    _entry("alias.complex-local", name="Alias: Complex Model (Local)", tags=["Alias"]),
    _entry("standard-local", name="Standard", preset=True, tags=["Standard"]),
    _entry("azure.gpt-5", name="GPT-5", connection_type="external", tags=["GPT"]),
    _entry(
        "google.gemini-3.5-flash",
        name="Gemini 3.5 Flash",
        connection_type="external",
        tags=["Gemini"],
    ),
]

_EXPECTED_CHAT_MODELS = [
    "kit.deepseek-v4-flash",
    "kit.glm-5.3",
    "kit.mistral-small-4-119b-a8b",
]


def test_filter_returns_exactly_the_local_chat_models():
    # The whole point: feed a realistic KIT catalog, get back only the
    # locally-hosted chat LLMs.
    out = chat_controls.filter_selectable_models(_LIVE_KIT_CATALOG)
    assert out == _EXPECTED_CHAT_MODELS


def test_filter_drops_non_chat_kit_models():
    # Image-gen, embeddings, rerankers, TTS and speech-to-text must never show,
    # even though some carry non-null capabilities (the reranker).
    non_chat = [e for e in _LIVE_KIT_CATALOG if e["id"].startswith("kit.") and e["id"] not in _EXPECTED_CHAT_MODELS]
    assert non_chat  # sanity: fixture actually has non-chat kit.* entries
    assert chat_controls.filter_selectable_models(non_chat) == []


def test_filter_drops_aliases_and_presets():
    aliasy = [
        e for e in _LIVE_KIT_CATALOG
        if e["id"].startswith("alias.") or e.get("preset")
    ]
    assert aliasy
    assert chat_controls.filter_selectable_models(aliasy) == []


def test_filter_drops_external_models():
    external = [e for e in _LIVE_KIT_CATALOG if e["connection_type"] == "external"]
    assert external
    assert chat_controls.filter_selectable_models(external) == []


def test_filter_picks_up_new_local_chat_model_with_no_code_change():
    # A hypothetical new KIT chat model must be surfaced automatically.
    catalog = _LIVE_KIT_CATALOG + [_entry("kit.newmodel-x", name="New Model X")]
    out = chat_controls.filter_selectable_models(catalog)
    assert out == sorted(_EXPECTED_CHAT_MODELS + ["kit.newmodel-x"])


def test_filter_dedupes_and_sorts():
    out = chat_controls.filter_selectable_models(
        [
            _entry("kit.glm-5.3"),
            _entry("kit.deepseek-v4-flash"),
            _entry("kit.glm-5.3"),
        ]
    )
    assert out == ["kit.deepseek-v4-flash", "kit.glm-5.3"]


def test_default_model_preference_order():
    assert chat_controls.DEFAULT_MODEL_PREFERENCE == (
        "kit.mistral-small-4-119b-a8b",
        "kit.deepseek-v4-flash",
        "kit.glm-5.3",
    )
    # First-listed preference wins regardless of input ordering.
    assert chat_controls.default_model(
        ["kit.glm-5.3", "kit.deepseek-v4-flash", "kit.mistral-small-4-119b-a8b"]
    ) == "kit.mistral-small-4-119b-a8b"
    # Falls through to the next preference when an earlier one is unavailable.
    assert chat_controls.default_model(
        ["kit.glm-5.3", "kit.deepseek-v4-flash"]
    ) == "kit.deepseek-v4-flash"


def test_default_model_falls_back_when_no_preference_available():
    assert chat_controls.default_model(["kit.newmodel-x"]) == "kit.newmodel-x"


def test_default_model_empty_list_returns_first_preference():
    assert chat_controls.default_model([]) == chat_controls.DEFAULT_MODEL_PREFERENCE[0]


def test_available_models_uses_offline_fallback_on_fetch_failure(monkeypatch):
    def _boom():
        raise RuntimeError("KIT /models unreachable")

    monkeypatch.setattr(chat_controls, "_fetch_kit_models_meta", _boom)
    out = chat_controls.available_models()
    assert out == sorted(chat_controls._OFFLINE_FALLBACK_MODELS)
    # The fallback is the current local chat ids, NOT pricing.known_models()
    # (which still contains models KIT has since removed).
    assert "kit.gemma4-31b-it" not in out
    assert "kit.qwen3.5-397b-A17b" not in out


def test_available_models_filters_live_catalog(monkeypatch):
    monkeypatch.setattr(
        chat_controls, "_fetch_kit_models_meta", lambda: _LIVE_KIT_CATALOG
    )
    out = chat_controls.available_models()
    assert out == _EXPECTED_CHAT_MODELS


def test_price_caption_for_known_model():
    cap = chat_controls.price_caption("kit.gemma4-31b-it")
    assert cap is not None
    assert "per 1M in" in cap and "per 1M out" in cap
    # gemma-4-31b: 1.2e-7 * 1e6 = $0.120 input, 3.7e-7 * 1e6 = $0.370 output
    assert "$0.120" in cap
    assert "$0.370" in cap
    # Dollar signs must be escaped so Streamlit markdown does not render the
    # text between them as LaTeX math.
    assert r"\$0.120" in cap
    assert r"\$0.370" in cap


def test_display_model_name_uses_endpoint_name_when_known(monkeypatch):
    monkeypatch.setattr(
        chat_controls, "_DISPLAY_NAMES", {"kit.deepseek-v4-flash": "DeepSeek V4 Flash"}
    )
    assert chat_controls.display_model_name("kit.deepseek-v4-flash") == "DeepSeek V4 Flash"


def test_display_model_name_falls_back_to_stripped_id(monkeypatch):
    monkeypatch.setattr(chat_controls, "_DISPLAY_NAMES", {})
    assert chat_controls.display_model_name("kit.some-unseen-model") == "some-unseen-model"
    # Non-kit ids are shown verbatim.
    assert chat_controls.display_model_name("azure.gpt-5") == "azure.gpt-5"


def test_available_models_remembers_display_names(monkeypatch):
    monkeypatch.setattr(chat_controls, "_DISPLAY_NAMES", {})
    monkeypatch.setattr(
        chat_controls, "_fetch_kit_models_meta", lambda: _LIVE_KIT_CATALOG
    )
    chat_controls.available_models()
    assert chat_controls.display_model_name("kit.deepseek-v4-flash") == "DeepSeek V4 Flash"
    assert chat_controls.display_model_name("kit.glm-5.3") == "GLM-5.3"


def test_price_caption_unknown_model_is_none():
    assert chat_controls.price_caption("kit.not-a-real-model") is None
    assert chat_controls.price_caption(None) is None


def test_apply_chat_settings_updates_config_cache(monkeypatch):
    import ama_kbqa.config as cfg_module

    base = {"llm": {"chat_provider": "openrouter", "chat_temperature": 0.2}, "kit": {"chat_model": "old"}}
    monkeypatch.setattr(cfg_module, "_config_cache", base)
    monkeypatch.setattr(cfg_module, "load_config", lambda: cfg_module._config_cache)
    monkeypatch.delenv(cfg_module.CHAT_PROVIDER_OVERRIDE_ENV_VAR, raising=False)
    monkeypatch.delenv(cfg_module.CHAT_MODEL_OVERRIDE_ENV_VAR, raising=False)
    monkeypatch.delenv(cfg_module.CHAT_TEMPERATURE_OVERRIDE_ENV_VAR, raising=False)

    chat_controls.apply_chat_settings("kit.deepseek-v4-flash", 1.0)

    assert cfg_module._config_cache["llm"]["chat_provider"] == "kit"
    assert cfg_module._config_cache["kit"]["chat_model"] == "kit.deepseek-v4-flash"
    assert cfg_module._config_cache["llm"]["chat_temperature"] == pytest.approx(1.0)


def test_apply_chat_settings_sets_env_overrides_for_mcp_subprocesses(monkeypatch):
    # MCP tool-server subprocesses (orchestrator + specialists) are spawned
    # with env=os.environ.copy() and load their own config.toml — they never
    # see the _config_cache mutation above, only these env vars.
    import ama_kbqa.config as cfg_module

    base = {"llm": {"chat_provider": "kit", "chat_temperature": 1.0}, "kit": {"chat_model": "old"}}
    monkeypatch.setattr(cfg_module, "_config_cache", base)
    monkeypatch.setattr(cfg_module, "load_config", lambda: cfg_module._config_cache)
    # All three are process-global writes, so every one of them must be
    # restored after this test — a leaked AMA_KBQA_CHAT_PROVIDER would
    # silently override config.toml for every test that runs later.
    monkeypatch.delenv(cfg_module.CHAT_PROVIDER_OVERRIDE_ENV_VAR, raising=False)
    monkeypatch.delenv(cfg_module.CHAT_MODEL_OVERRIDE_ENV_VAR, raising=False)
    monkeypatch.delenv(cfg_module.CHAT_TEMPERATURE_OVERRIDE_ENV_VAR, raising=False)

    chat_controls.apply_chat_settings("kit.glm-5.3", 0.55)

    assert os.environ[cfg_module.CHAT_PROVIDER_OVERRIDE_ENV_VAR] == "kit"
    assert os.environ[cfg_module.CHAT_MODEL_OVERRIDE_ENV_VAR] == "kit.glm-5.3"
    assert os.environ[cfg_module.CHAT_TEMPERATURE_OVERRIDE_ENV_VAR] == "0.55"
    # The functions a freshly-spawned subprocess would call must honor it.
    assert cfg_module.get_chat_provider() == "kit"
    assert cfg_module.get_chat_model_name() == "kit.glm-5.3"
    assert cfg_module.get_chat_temperature() == pytest.approx(0.55)


# ── Multi-provider picker (booth build) ──────────────────────────────────────

# Hand-written entries shaped like OpenRouter's GET /api/v1/models payload
# after ``config_editor.fetch_provider_models_pricing`` normalizes it: prices
# in USD per token, tool support from ``supported_parameters``.
_OPENROUTER_CATALOG = {
    "deepseek/deepseek-v4-flash": {
        "name": "DeepSeek: DeepSeek V4 Flash 0423",
        "prompt": 7.784e-08,
        "completion": 1.5568e-07,
        "supports_tools": True,
    },
    "anthropic/claude-sonnet-4.5": {
        "name": "Anthropic: Claude Sonnet 4.5",
        "prompt": 3e-06,
        "completion": 1.5e-05,
        "supports_tools": True,
    },
}

_CONFIG_ENTRIES = [
    {
        "provider": "openrouter",
        "id": "deepseek/deepseek-v4-flash",
        "name": "DeepSeek V4 Flash",
        "prompt_usd_per_m": None,
        "completion_usd_per_m": None,
        "default": False,
    },
    {
        # No name: the catalog must supply one.
        "provider": "openrouter",
        "id": "anthropic/claude-sonnet-4.5",
        "name": "",
        "prompt_usd_per_m": None,
        "completion_usd_per_m": None,
        "default": False,
    },
    {
        # Retired / mistyped id: present in config, absent from the catalog.
        "provider": "openrouter",
        "id": "vendor/retired-model",
        "name": "Retired Model",
        "prompt_usd_per_m": None,
        "completion_usd_per_m": None,
        "default": False,
    },
    {
        "provider": "deepseek",
        "id": "deepseek-flash",
        "name": "DeepSeek Flash (direct)",
        "prompt_usd_per_m": 0.30,
        "completion_usd_per_m": 1.20,
        "default": False,
    },
]


@pytest.fixture
def picker_env(monkeypatch):
    """KIT catalog stubbed, no provider keys set, config entries fixed."""
    monkeypatch.setattr(
        chat_controls, "_fetch_kit_models_meta", lambda: _LIVE_KIT_CATALOG
    )
    monkeypatch.setattr(
        chat_controls, "get_frontend_chat_models", lambda: list(_CONFIG_ENTRIES)
    )
    monkeypatch.setattr(
        chat_controls, "_fetch_openrouter_catalog", lambda: dict(_OPENROUTER_CATALOG)
    )
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)


def test_available_choices_without_keys_is_kit_only_plus_notices(picker_env):
    choices, notices = chat_controls.available_choices()
    assert [c.key for c in choices] == [f"kit:{m}" for m in _EXPECTED_CHAT_MODELS]
    assert notices == [
        "OpenRouter models hidden: OPENROUTER_API_KEY not set",
        "DeepSeek models hidden: DEEPSEEK_API_KEY not set",
    ]


def test_available_choices_validates_openrouter_against_the_catalog(
    picker_env, monkeypatch
):
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-test-key")
    choices, notices = chat_controls.available_choices()
    keys = [c.key for c in choices]

    assert "openrouter:deepseek/deepseek-v4-flash" in keys
    assert "openrouter:anthropic/claude-sonnet-4.5" in keys
    # Unknown id dropped, and the presenter is told why.
    assert "openrouter:vendor/retired-model" not in keys
    assert any("vendor/retired-model" in n for n in notices)
    # DeepSeek still hidden (no key), exactly one notice for it.
    assert "DeepSeek models hidden: DEEPSEEK_API_KEY not set" in notices

    by_key = {c.key: c for c in choices}
    # Config name wins where given; the catalog fills a missing one.
    assert by_key["openrouter:deepseek/deepseek-v4-flash"].name == "DeepSeek V4 Flash"
    assert (
        by_key["openrouter:anthropic/claude-sonnet-4.5"].name
        == "Anthropic: Claude Sonnet 4.5"
    )
    # Catalog prices fill the (omitted) config prices.
    assert by_key["openrouter:anthropic/claude-sonnet-4.5"].prompt_usd_per_token == (
        pytest.approx(3e-06)
    )
    assert by_key[
        "openrouter:anthropic/claude-sonnet-4.5"
    ].completion_usd_per_token == pytest.approx(1.5e-05)


def test_available_choices_hides_models_that_cannot_call_tools(
    picker_env, monkeypatch
):
    # Every agent here answers by calling tools, so a non-tool model would
    # fail on the presenter's first question.
    catalog = {
        k: dict(v) for k, v in _OPENROUTER_CATALOG.items()
    }
    catalog["anthropic/claude-sonnet-4.5"]["supports_tools"] = False
    monkeypatch.setattr(chat_controls, "_fetch_openrouter_catalog", lambda: catalog)
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-test-key")

    choices, notices = chat_controls.available_choices()
    keys = [c.key for c in choices]

    assert "openrouter:anthropic/claude-sonnet-4.5" not in keys
    assert "openrouter:deepseek/deepseek-v4-flash" in keys
    assert any(
        "anthropic/claude-sonnet-4.5" in n and "cannot call tools" in n
        for n in notices
    )


def test_available_choices_keeps_config_entries_when_catalog_unreachable(
    picker_env, monkeypatch
):
    def _boom():
        raise RuntimeError("openrouter /models unreachable")

    monkeypatch.setenv("OPENROUTER_API_KEY", "or-test-key")
    monkeypatch.setattr(chat_controls, "_fetch_openrouter_catalog", _boom)

    choices, notices = chat_controls.available_choices()
    keys = [c.key for c in choices]
    # No validation means no drops, including the id the catalog would reject.
    assert "openrouter:vendor/retired-model" in keys
    assert not any("catalog" in n for n in notices)
    by_key = {c.key: c for c in choices}
    assert by_key["openrouter:vendor/retired-model"].prompt_usd_per_token is None


def test_available_choices_offers_deepseek_with_config_prices(picker_env, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "ds-test-key")
    choices, notices = chat_controls.available_choices()
    by_key = {c.key: c for c in choices}

    entry = by_key["deepseek:deepseek-flash"]
    assert entry.name == "DeepSeek Flash (direct)"
    # 0.30 USD per 1M tokens = 3e-07 per token.
    assert entry.prompt_usd_per_token == pytest.approx(3e-07)
    assert entry.completion_usd_per_token == pytest.approx(1.2e-06)
    assert "OpenRouter models hidden: OPENROUTER_API_KEY not set" in notices


def test_available_choices_registers_runtime_prices(picker_env, monkeypatch):
    from ama_kbqa import pricing

    monkeypatch.setattr(pricing, "_RUNTIME_PRICING", {})
    monkeypatch.setenv("DEEPSEEK_API_KEY", "ds-test-key")
    chat_controls.available_choices()
    # The chat footer prices a message by its bare model id.
    assert pricing.estimate_cost_usd("deepseek-flash", 1_000_000, 0) == pytest.approx(0.30)


def test_default_choice_honours_the_config_default(picker_env, monkeypatch):
    entries = [dict(e) for e in _CONFIG_ENTRIES]
    entries[0]["default"] = True
    monkeypatch.setattr(chat_controls, "get_frontend_chat_models", lambda: entries)
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-test-key")

    choices, _ = chat_controls.available_choices()
    assert chat_controls.default_choice(choices).key == (
        "openrouter:deepseek/deepseek-v4-flash"
    )


def test_default_choice_falls_back_to_the_kit_preference(picker_env):
    choices, _ = chat_controls.available_choices()
    # No default = true anywhere: the fast free KIT model wins, so a booth
    # session never starts spending money by accident.
    assert chat_controls.default_choice(choices).key == (
        "kit:kit.mistral-small-4-119b-a8b"
    )


def test_default_choice_on_empty_list_returns_a_kit_placeholder():
    choice = chat_controls.default_choice([])
    assert choice.provider == "kit"
    assert choice.model == chat_controls.DEFAULT_MODEL_PREFERENCE[0]


def test_display_choice_names_the_endpoint():
    choice = chat_controls.ChatModelChoice(
        provider="openrouter", model="anthropic/claude-sonnet-4.5",
        name="Claude Sonnet 4.5",
    )
    assert chat_controls.display_choice(choice) == "Claude Sonnet 4.5 · OpenRouter"
    kit = chat_controls.ChatModelChoice(
        provider="kit", model="kit.glm-5.3", name="GLM-5.3"
    )
    assert chat_controls.display_choice(kit) == "GLM-5.3 · KIT"


def test_price_caption_for_billed_model_says_billed():
    choice = chat_controls.ChatModelChoice(
        provider="deepseek", model="deepseek-flash", name="DeepSeek Flash (direct)",
        prompt_usd_per_token=3e-07, completion_usd_per_token=1.2e-06,
    )
    cap = chat_controls.price_caption_for(choice)
    assert cap.endswith("(billed)")
    assert r"\$0.300" in cap and r"\$1.20" in cap


def test_price_caption_for_unpriced_model_is_none():
    choice = chat_controls.ChatModelChoice(
        provider="openrouter", model="vendor/model", name="Model"
    )
    assert chat_controls.price_caption_for(choice) is None


def test_price_caption_for_kit_uses_the_illustrative_table():
    choice = chat_controls.ChatModelChoice(
        provider="kit", model="kit.gemma4-31b-it", name="Gemma 4"
    )
    cap = chat_controls.price_caption_for(choice)
    assert cap is not None
    # KIT is free to us, so the caption must NOT claim it is billed.
    assert "(billed)" not in cap


def test_apply_chat_settings_with_provider_sets_all_three_overrides(monkeypatch):
    import ama_kbqa.config as cfg_module

    base = {
        "llm": {"chat_provider": "kit", "chat_temperature": 1.0},
        "kit": {"chat_model": "kit.glm-5.3"},
        "deepseek": {"base_url": "https://api.deepseek.com", "chat_model": "old"},
    }
    monkeypatch.setattr(cfg_module, "_config_cache", base)
    monkeypatch.setattr(cfg_module, "load_config", lambda: cfg_module._config_cache)

    chat_controls.apply_chat_settings("deepseek-v4-pro", 0.7, provider="deepseek")

    assert os.environ[cfg_module.CHAT_PROVIDER_OVERRIDE_ENV_VAR] == "deepseek"
    assert os.environ[cfg_module.CHAT_MODEL_OVERRIDE_ENV_VAR] == "deepseek-v4-pro"
    assert cfg_module._config_cache["llm"]["chat_provider"] == "deepseek"
    assert cfg_module._config_cache["deepseek"]["chat_model"] == "deepseek-v4-pro"
    # What a freshly-spawned MCP subprocess would resolve.
    assert cfg_module.get_chat_provider() == "deepseek"
    assert cfg_module.get_chat_model_name() == "deepseek-v4-pro"


def test_apply_chat_settings_two_argument_call_still_pins_kit(monkeypatch):
    import ama_kbqa.config as cfg_module

    base = {"llm": {"chat_provider": "openrouter"}, "kit": {"chat_model": "old"}}
    monkeypatch.setattr(cfg_module, "_config_cache", base)
    monkeypatch.setattr(cfg_module, "load_config", lambda: cfg_module._config_cache)

    chat_controls.apply_chat_settings("kit.glm-5.3", 1.0)

    assert os.environ[cfg_module.CHAT_PROVIDER_OVERRIDE_ENV_VAR] == "kit"
    assert cfg_module._config_cache["llm"]["chat_provider"] == "kit"
    assert cfg_module._config_cache["kit"]["chat_model"] == "kit.glm-5.3"
