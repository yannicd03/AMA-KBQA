"""Tests for the demo chat sidebar helpers in chat_controls."""

from __future__ import annotations

import pytest

from ama_kbqa.frontend.utils import chat_controls


# The exact set of model ids KIT's /models endpoint advertises today: 5 LLM
# chat models plus a pile of non-chat models, Azure-routed models, and routing
# aliases. The picker must surface only the 5 chat models.
_LIVE_KIT_MODELS = [
    "azure.gpt-4.1",
    "azure.gpt-5",
    "azure.o4-mini",
    "kit.flux.2-dev",
    "kit.gemma4-31b-it",
    "kit.gpt-oss-120b",
    "kit.minimax-m2.7-229b",
    "kit.mistral-small-4-119b-a8b",
    "kit.qwen3-embedding-8b",
    "kit.qwen3-reranker-8b",
    "kit.qwen3.5-397b-A17b",
    "kit.voxtral-4b-tts-2603",
    "kit.whisper-large-v3",
    "standard-extern",
    "standard-local",
]

_EXPECTED_WHITELIST = [
    "kit.gemma4-31b-it",
    "kit.gpt-oss-120b",
    "kit.minimax-m2.7-229b",
    "kit.mistral-small-4-119b-a8b",
    "kit.qwen3.5-397b-A17b",
]


def test_filter_returns_exactly_the_whitelisted_llms():
    # The whole point: feed the real KIT catalog, get back only chat LLMs.
    out = chat_controls.filter_selectable_models(_LIVE_KIT_MODELS)
    assert out == _EXPECTED_WHITELIST


def test_filter_drops_non_llm_kit_models():
    # Image-gen, embeddings, rerankers, TTS and speech-to-text must never show.
    non_llm = [
        "kit.flux.2-dev",
        "kit.qwen3-embedding-8b",
        "kit.qwen3-reranker-8b",
        "kit.voxtral-4b-tts-2603",
        "kit.whisper-large-v3",
    ]
    assert chat_controls.filter_selectable_models(non_llm) == []


def test_filter_drops_all_azure_models():
    # Azure-routed models are excluded wholesale (no gpt-oss carve-out needed —
    # the open-source gpt-oss is served under the kit. prefix).
    azure = ["azure.gpt-4.1-mini", "azure.o4-mini", "azure.gpt-5", "azure.gpt-oss-120b"]
    assert chat_controls.filter_selectable_models(azure) == []


def test_filter_dedupes_and_sorts():
    out = chat_controls.filter_selectable_models(
        ["kit.gpt-oss-120b", "kit.gemma4-31b-it", "kit.gpt-oss-120b"]
    )
    assert out == ["kit.gemma4-31b-it", "kit.gpt-oss-120b"]


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


def test_filter_drops_standard_routing_aliases():
    raw = [
        "kit.standard-extern",
        "kit.standard-local",
        "standard_local",
        "kit.gemma4-31b-it",
    ]
    out = chat_controls.filter_selectable_models(raw)
    assert out == ["kit.gemma4-31b-it"]


def test_display_model_name_strips_kit_prefix():
    assert chat_controls.display_model_name("kit.gemma4-31b-it") == "gemma4-31b-it"
    # Non-kit names are shown verbatim.
    assert chat_controls.display_model_name("azure.gpt-oss-120b") == "azure.gpt-oss-120b"


def test_price_caption_unknown_model_is_none():
    assert chat_controls.price_caption("kit.not-a-real-model") is None
    assert chat_controls.price_caption(None) is None


def test_apply_chat_settings_updates_config_cache(monkeypatch):
    import ama_kbqa.config as cfg_module

    base = {"llm": {"chat_provider": "openrouter", "chat_temperature": 0.2}, "kit": {"chat_model": "old"}}
    monkeypatch.setattr(cfg_module, "_config_cache", base)
    monkeypatch.setattr(cfg_module, "load_config", lambda: cfg_module._config_cache)

    chat_controls.apply_chat_settings("kit.gemma4-31b-it", 1.0)

    assert cfg_module._config_cache["llm"]["chat_provider"] == "kit"
    assert cfg_module._config_cache["kit"]["chat_model"] == "kit.gemma4-31b-it"
    assert cfg_module._config_cache["llm"]["chat_temperature"] == pytest.approx(1.0)
