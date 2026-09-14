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
    monkeypatch.delenv(cfg_module.CHAT_MODEL_OVERRIDE_ENV_VAR, raising=False)
    monkeypatch.delenv(cfg_module.CHAT_TEMPERATURE_OVERRIDE_ENV_VAR, raising=False)

    chat_controls.apply_chat_settings("kit.glm-5.3", 0.55)

    assert os.environ[cfg_module.CHAT_MODEL_OVERRIDE_ENV_VAR] == "kit.glm-5.3"
    assert os.environ[cfg_module.CHAT_TEMPERATURE_OVERRIDE_ENV_VAR] == "0.55"
    # The functions a freshly-spawned subprocess would call must honor it.
    assert cfg_module.get_chat_model_name() == "kit.glm-5.3"
    assert cfg_module.get_chat_temperature() == pytest.approx(0.55)
