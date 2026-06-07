"""Tests for the demo chat sidebar helpers in chat_controls."""

from __future__ import annotations

import pytest

from ama_kbqa.frontend.utils import chat_controls


def test_filter_drops_azure_except_gpt_oss():
    raw = [
        "kit.gemma4-31b-it",
        "azure.gpt-4.1-mini",
        "azure.o4-mini",
        "kit.gpt-oss-120b",
        "kit.qwen3.5-397b-A17b",
    ]
    out = chat_controls.filter_selectable_models(raw)
    assert "azure.gpt-4.1-mini" not in out
    assert "azure.o4-mini" not in out
    assert "kit.gpt-oss-120b" in out
    assert "kit.gemma4-31b-it" in out
    assert "kit.qwen3.5-397b-A17b" in out


def test_filter_keeps_azure_gpt_oss_carveout():
    # If the endpoint ever serves gpt-oss under an azure prefix, keep it.
    out = chat_controls.filter_selectable_models(["azure.gpt-oss-120b", "azure.o4-mini"])
    assert out == ["azure.gpt-oss-120b"]


def test_filter_drops_embedding_models():
    out = chat_controls.filter_selectable_models(
        ["kit.qwen3-embedding-8b", "kit.gemma4-31b-it"]
    )
    assert out == ["kit.gemma4-31b-it"]


def test_filter_dedupes_and_sorts():
    out = chat_controls.filter_selectable_models(
        ["kit.b", "kit.a", "kit.b"]
    )
    assert out == ["kit.a", "kit.b"]


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


def test_apply_retrieval_settings_updates_cache_and_env(monkeypatch):
    import os

    import ama_kbqa.config as cfg_module

    base = {"retrieval": {"hybrid_enabled": False, "fusion": "rrf"}}
    monkeypatch.setattr(cfg_module, "_config_cache", base)
    monkeypatch.setattr(cfg_module, "load_config", lambda: cfg_module._config_cache)
    # Register the env vars with monkeypatch so they are restored after the
    # test even though the helper writes os.environ directly.
    for suffix in ("HYBRID_ENABLED", "FUSION", "RERANKER_ENABLED"):
        monkeypatch.delenv(cfg_module.RETRIEVAL_ENV_PREFIX + suffix, raising=False)

    chat_controls.apply_retrieval_settings(True, "dbsf", True)

    # In-process channel: cached config dict.
    retrieval = cfg_module._config_cache["retrieval"]
    assert retrieval["hybrid_enabled"] is True
    assert retrieval["fusion"] == "dbsf"
    assert retrieval["reranker_enabled"] is True
    # Cross-process channel: env vars inherited by MCP server subprocesses.
    prefix = cfg_module.RETRIEVAL_ENV_PREFIX
    assert os.environ[prefix + "HYBRID_ENABLED"] == "true"
    assert os.environ[prefix + "FUSION"] == "dbsf"
    assert os.environ[prefix + "RERANKER_ENABLED"] == "true"

    chat_controls.apply_retrieval_settings(False, "rrf", False)

    assert os.environ[prefix + "HYBRID_ENABLED"] == "false"
    assert os.environ[prefix + "RERANKER_ENABLED"] == "false"


def test_reranker_available_matches_import(monkeypatch):
    import importlib.util

    expected = importlib.util.find_spec("sentence_transformers") is not None
    assert chat_controls.reranker_available() is expected
