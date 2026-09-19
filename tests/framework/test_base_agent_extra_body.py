"""The chat provider's extra request body must reach the tool-loop call sites.

DeepSeek's direct API runs thinking mode by default and then rejects any
tools-carrying request whose assistant messages lack ``reasoning_content``, which
ours do because the fast path synthesises them. config.get_chat_extra_body()
turns thinking off; these tests pin that it actually lands in the request, and
that OpenRouter's routing preferences are not clobbered when both apply.
"""

from __future__ import annotations

import types

import pytest

from ama_kbqa.framework import base_agent
from ama_kbqa.framework.base_agent import BaseKBQAAgent
from ama_kbqa.framework.config import KnowledgeGraphConfig
from ama_kbqa.framework.trace import TraceRecorder
from chatkit import TransientRetry


class _ConcreteAgent(BaseKBQAAgent):
    def get_config(self) -> KnowledgeGraphConfig:  # pragma: no cover - unused
        raise NotImplementedError

    def get_mcp_server_path(self) -> str:  # pragma: no cover - unused
        raise NotImplementedError


def _completion(content: str = "hi"):
    message = types.SimpleNamespace(content=content, tool_calls=None)
    choice = types.SimpleNamespace(message=message, finish_reason="stop")
    return types.SimpleNamespace(choices=[choice], usage=None)


@pytest.fixture
def captured(monkeypatch):
    """An agent whose fake client records the kwargs of each request."""
    monkeypatch.setattr(base_agent, "get_chat_temperature", lambda: 0.2)
    monkeypatch.setattr(base_agent, "get_chat_max_tokens", lambda: 100)
    monkeypatch.setattr(base_agent, "get_chat_seed", lambda: None)
    monkeypatch.setattr(base_agent, "get_provider_preferences", lambda: None)

    calls: list[dict] = []

    def create(**kw):
        calls.append(kw)
        return _completion()

    agent = object.__new__(_ConcreteAgent)
    agent._retry = TransientRetry()
    agent._trace = lambda *a, **k: None
    agent.recorder = TraceRecorder()
    agent.model = "deepseek-flash"
    agent.client = types.SimpleNamespace(
        chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create))
    )
    agent._messages = [{"role": "system", "content": "sys"}]
    agent.request_timeout = 30
    agent.token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    return agent, calls


_THINKING_OFF = {"thinking": {"type": "disabled"}}


def test_tool_loop_call_carries_the_thinking_switch(captured, monkeypatch):
    agent, calls = captured
    monkeypatch.setattr(base_agent, "get_chat_extra_body", lambda: _THINKING_OFF)

    agent._llm_call(tools=[{"type": "function", "function": {"name": "FindNode"}}])

    assert calls[0]["extra_body"] == _THINKING_OFF


def test_text_only_call_carries_the_thinking_switch(captured, monkeypatch):
    agent, calls = captured
    monkeypatch.setattr(base_agent, "get_chat_extra_body", lambda: _THINKING_OFF)

    agent._llm_call_text_only()

    assert calls[0]["extra_body"] == _THINKING_OFF


def test_openrouter_routing_preferences_are_not_clobbered(captured, monkeypatch):
    agent, calls = captured
    prefs = {"order": ["deepinfra/turbo"], "allow_fallbacks": False}
    monkeypatch.setattr(base_agent, "get_provider_preferences", lambda: prefs)
    monkeypatch.setattr(base_agent, "get_chat_extra_body", lambda: _THINKING_OFF)

    agent._llm_call()

    assert calls[0]["extra_body"] == {"provider": prefs, **_THINKING_OFF}


def test_no_extra_body_when_the_provider_needs_none(captured, monkeypatch):
    # KIT / OpenRouter requests must stay byte-identical to before the helper.
    agent, calls = captured
    monkeypatch.setattr(base_agent, "get_chat_extra_body", dict)

    agent._llm_call()

    assert "extra_body" not in calls[0]
