"""``ama_kbqa.graph.model.build_chat_model`` builds the right LangChain class
per provider, without making any network calls.

Model-object construction (``ChatOpenAI(...)`` / ``chatkit.client.get_kit_model(...)``)
never opens a socket by itself — the underlying ``openai.OpenAI`` client
connects lazily on first request. The "kit" branch additionally passes
``preflight=False`` specifically so the VPN-reachability check
(``chatkit.client._vpn_preflight``) is skipped here.
"""

from __future__ import annotations

import pytest
from langchain_openai import ChatOpenAI

from ama_kbqa.graph.model import build_chat_model


@pytest.fixture(autouse=True)
def _dummy_api_keys(monkeypatch):
    monkeypatch.setenv("KIT_API_KEY", "test-kit-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")


def test_kit_provider_returns_chat_kit_without_network(monkeypatch):
    from chatkit.client import _get_chat_kit_class

    model = build_chat_model("kit", purpose="chat", max_retries=0)

    assert isinstance(model, _get_chat_kit_class())
    assert model.model_name == "kit.gemma4-31b-it"


def test_kit_provider_wires_the_passed_retry_instance():
    from chatkit.retry import TransientRetry

    retry = TransientRetry()
    model = build_chat_model("kit", purpose="chat", max_retries=0, retry=retry)

    assert model._retry is retry


def test_openrouter_provider_returns_plain_chat_openai_without_network():
    model = build_chat_model("openrouter", purpose="synthesis", max_retries=0)

    assert isinstance(model, ChatOpenAI)
    # Synthesis model name comes from [synthesis].synthesis_model, independent
    # of the openrouter section's own chat_model — matching
    # get_synthesis_model_name()'s existing (provider-independent) lookup.
    assert model.model_name == "deepseek/deepseek-v3.2"


def test_openrouter_provider_sets_ranking_headers():
    model = build_chat_model("openrouter", purpose="chat", max_retries=0)

    assert model.default_headers is not None
    assert model.default_headers.get("X-Title") == "ama-kbqa"


def test_unknown_provider_raises_value_error():
    with pytest.raises(ValueError, match="not found in config.toml"):
        build_chat_model("not-a-real-provider", purpose="chat")
