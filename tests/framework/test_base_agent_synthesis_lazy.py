"""Regression tests for lazy synthesis-client initialization.

Background: on the bwcloud deployment the chat provider (KIT) had a valid key
while the synthesis provider (OpenRouter) did not, and synthesis was disabled
in config. Eagerly building the synthesis client in __init__ made the KQAPro
fallback agent fail to even construct ("Failed to initialize synthesis LLM
client ... OPENROUTER_API_KEY ... not set"). The client is now built lazily so
an agent stays constructible when synthesis is off, and the key requirement is
only enforced if synthesis actually runs.
"""

from __future__ import annotations

import pytest

from ama_kbqa.framework import base_agent
from ama_kbqa.framework.base_agent import BaseKBQAAgent
from ama_kbqa.framework.config import KnowledgeGraphConfig


class _ConcreteAgent(BaseKBQAAgent):
    def get_config(self) -> KnowledgeGraphConfig:  # pragma: no cover - unused
        raise NotImplementedError

    def get_mcp_server_path(self) -> str:  # pragma: no cover - unused
        raise NotImplementedError


@pytest.fixture
def patched_clients(monkeypatch):
    """Chat client builds fine; synthesis raises like a missing API key would."""
    monkeypatch.setattr(base_agent, "get_chat_client", lambda: object())
    monkeypatch.setattr(base_agent, "get_chat_model_name", lambda: "chat-model")

    def _raise_synthesis():
        raise KeyError("Environment variable 'OPENROUTER_API_KEY' ... is not set.")

    monkeypatch.setattr(base_agent, "get_synthesis_client", _raise_synthesis)
    monkeypatch.setattr(base_agent, "get_synthesis_model_name", lambda: "synth-model")


def test_agent_constructs_when_synthesis_client_unavailable(patched_clients):
    # Must NOT raise: this is exactly the bwcloud failure the fix addresses.
    agent = _ConcreteAgent(name="kqapro_agent", session_id="t")
    assert agent.client is not None


def test_synthesis_client_raises_lazily_only_on_access(patched_clients):
    agent = _ConcreteAgent(name="kqapro_agent", session_id="t")
    with pytest.raises(RuntimeError, match="synthesis LLM client"):
        _ = agent.synthesis_client


def test_synthesis_client_builds_when_provider_available(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(base_agent, "get_chat_client", lambda: object())
    monkeypatch.setattr(base_agent, "get_chat_model_name", lambda: "chat-model")
    monkeypatch.setattr(base_agent, "get_synthesis_client", lambda: sentinel)
    monkeypatch.setattr(base_agent, "get_synthesis_model_name", lambda: "synth-model")

    agent = _ConcreteAgent(name="kqapro_agent", session_id="t")
    assert agent.synthesis_client is sentinel
    assert agent.synthesis_model == "synth-model"
