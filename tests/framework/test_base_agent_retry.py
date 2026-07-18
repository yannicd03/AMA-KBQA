"""Tests for the stepped-backoff transient-retry layer wired into BaseKBQAAgent.

Ported/adapted from wikikgqa-2026's tests/framework/test_base_agent_retry.py onto
dev's current base_agent API (dev's `_create_with_retry` takes a `call_params` dict,
same as the reference; only the surrounding call sites and the addition of the
classification call path differ).

`_create_with_retry` is the sole defense against provider flakiness (KIT "Server
Connection Error", 5xx, rate limits) for every chat-completions call this agent
makes: classification, the tool loop, text-only, and synthesis. It must: retry
transient errors with an escalating backoff, reset the ramp after a success, and
re-raise deterministic errors (auth/bad-request) immediately.

Beyond the reference (which left classification unwrapped, tolerating any failure
via its existing "default to Query" fallback), this repo's task explicitly requires
classification to go through the same retry layer too — see the integration tests
below for that call path, plus the SciQA-agent classification override and the
Orchestrator's own probe/decision + LLM-fallback calls.
"""

from __future__ import annotations

import types

import pytest

from ama_kbqa.framework import base_agent
from ama_kbqa.framework.base_agent import BaseKBQAAgent
from ama_kbqa.framework.config import KnowledgeGraphConfig
from ama_kbqa.framework.trace import TraceRecorder
from ama_kbqa.llm.retry import DEFAULT_MAX_ATTEMPTS, TransientRetry


class _ConcreteAgent(BaseKBQAAgent):
    def get_config(self) -> KnowledgeGraphConfig:  # pragma: no cover - unused
        raise NotImplementedError

    def get_mcp_server_path(self) -> str:  # pragma: no cover - unused
        raise NotImplementedError


def _bare_agent():
    """An agent instance without the heavy __init__ (we only exercise retry wiring)."""
    agent = object.__new__(_ConcreteAgent)
    agent._retry = TransientRetry()
    agent._trace = lambda *a, **k: None
    return agent


def _client(create):
    return types.SimpleNamespace(
        chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create))
    )


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    waits: list[float] = []
    monkeypatch.setattr("time.sleep", lambda s: waits.append(s))
    return waits


# ---------------------------------------------------------------------------
# Core retry mechanics (ported from wikikgqa-2026)
# ---------------------------------------------------------------------------


def test_is_transient_classification():
    assert BaseKBQAAgent._is_transient_error(Exception("Open WebUI: Server Connection Error"))
    assert BaseKBQAAgent._is_transient_error(Exception("HTTP 503 Service Unavailable"))
    assert BaseKBQAAgent._is_transient_error(Exception("Rate limit exceeded (429)"))
    # Deterministic: retrying is futile.
    assert not BaseKBQAAgent._is_transient_error(Exception("401 Unauthorized: invalid api key"))
    assert not BaseKBQAAgent._is_transient_error(Exception("model does not exist"))


def test_retries_then_succeeds_with_escalating_backoff(_no_sleep):
    calls = {"n": 0}

    def create(**kw):
        calls["n"] += 1
        if calls["n"] <= 3:
            raise Exception("server connection error")
        return "OK"

    agent = _bare_agent()
    out = agent._create_with_retry(_client(create), {}, label="LLM")
    assert out == "OK"
    assert calls["n"] == 4  # 3 failures + 1 success
    # Escalating steps, then reset after success.
    assert _no_sleep == [2.0, 5.0, 10.0]
    assert agent._retry.level == 0


def test_backoff_level_persists_across_calls_until_success(_no_sleep):
    """The ramp is stateful: a later call starts where earlier ones left the level,
    and only a success resets it — so a still-flaky endpoint waits longer, not from
    scratch."""
    agent = _bare_agent()
    agent._retry.level = 2  # as if two prior transient failures already ramped it

    def fail_once_then_ok(**kw):
        fail_once_then_ok.n += 1
        if fail_once_then_ok.n == 1:
            raise Exception("connection reset")
        return "OK"
    fail_once_then_ok.n = 0

    agent._create_with_retry(_client(fail_once_then_ok), {}, label="LLM")
    # Starts at the persisted step (index 2 = 10s), not from scratch (2s).
    assert _no_sleep == [10.0]
    assert agent._retry.level == 0  # success resets the ramp


def test_deterministic_error_raises_immediately(_no_sleep):
    def create(**kw):
        raise Exception("401 Unauthorized")

    agent = _bare_agent()
    with pytest.raises(Exception, match="401"):
        agent._create_with_retry(_client(create), {}, label="LLM")
    assert _no_sleep == []  # no backoff for a non-transient error


def test_gives_up_after_max_attempts(_no_sleep):
    def create(**kw):
        raise Exception("temporarily unavailable")

    agent = _bare_agent()
    with pytest.raises(Exception, match="temporarily unavailable"):
        agent._create_with_retry(_client(create), {}, label="LLM")
    # Bounded: it does not retry forever even while errors stay transient.
    assert len(_no_sleep) == DEFAULT_MAX_ATTEMPTS - 1


def test_on_retry_hook_logs_via_trace(_no_sleep):
    """Each retry emits a trace line (benchmark logs show backoff activity)."""
    logged: list[str] = []

    def create(**kw):
        create.n += 1
        if create.n == 1:
            raise Exception("503 Service Unavailable")
        return "OK"
    create.n = 0

    agent = _bare_agent()
    agent._trace = lambda msg, color=None: logged.append(msg)
    agent._create_with_retry(_client(create), {}, label="LLM")

    assert len(logged) == 1
    assert "Transient LLM error" in logged[0]
    assert "attempt 1" in logged[0]


# ---------------------------------------------------------------------------
# Integration: each real BaseKBQAAgent call site goes through the retry layer
# ---------------------------------------------------------------------------


def _full_agent(create, seed=None):
    """A BaseKBQAAgent built enough to exercise _llm_call / _llm_call_text_only /
    _llm_call_synthesis / _classify_and_extract for real, with a fake client."""
    agent = object.__new__(_ConcreteAgent)
    agent._retry = TransientRetry()
    agent._trace = lambda *a, **k: None
    agent.recorder = TraceRecorder()
    agent.model = "test-model"
    agent.client = _client(create)
    agent._synthesis_client = _client(create)
    agent._synthesis_model = "test-model"
    agent._messages = [{"role": "system", "content": "sys"}]
    agent.request_timeout = 30
    agent.token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    return agent


def _completion(content: str = "hi", tool_calls=None):
    message = types.SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = types.SimpleNamespace(message=message, finish_reason="stop")
    return types.SimpleNamespace(choices=[choice], usage=None)


def test_llm_call_retries_transient_then_succeeds(monkeypatch, _no_sleep):
    monkeypatch.setattr(base_agent, "get_chat_temperature", lambda: 0.2)
    monkeypatch.setattr(base_agent, "get_chat_max_tokens", lambda: 100)
    monkeypatch.setattr(base_agent, "get_chat_seed", lambda: None)
    monkeypatch.setattr(base_agent, "get_provider_preferences", lambda: None)

    def create(**kw):
        create.n += 1
        if create.n == 1:
            raise Exception("502 Bad Gateway")
        return _completion("hello")
    create.n = 0

    agent = _full_agent(create)
    response = agent._llm_call()

    assert response.choices[0].message.content == "hello"
    assert create.n == 2
    assert _no_sleep == [2.0]


def test_llm_call_text_only_retries_transient_then_succeeds(monkeypatch, _no_sleep):
    monkeypatch.setattr(base_agent, "get_chat_temperature", lambda: 0.2)
    monkeypatch.setattr(base_agent, "get_chat_max_tokens", lambda: 100)
    monkeypatch.setattr(base_agent, "get_chat_seed", lambda: None)
    monkeypatch.setattr(base_agent, "get_provider_preferences", lambda: None)

    def create(**kw):
        create.n += 1
        if create.n == 1:
            raise Exception("connection reset")
        return _completion("plain text answer")
    create.n = 0

    agent = _full_agent(create)
    content = agent._llm_call_text_only()

    assert content == "plain text answer"
    assert create.n == 2
    assert _no_sleep == [2.0]


def test_llm_call_synthesis_retries_transient_then_succeeds(monkeypatch, _no_sleep):
    monkeypatch.setattr(base_agent, "get_synthesis_temperature", lambda: 0.2)
    monkeypatch.setattr(base_agent, "get_synthesis_max_tokens", lambda: 100)
    monkeypatch.setattr(base_agent, "get_synthesis_provider_preferences", lambda: None)

    def create(**kw):
        create.n += 1
        if create.n == 1:
            raise Exception("rate limit exceeded (429)")
        return _completion("final answer")
    create.n = 0

    agent = _full_agent(create)
    content = agent._llm_call_synthesis(messages_override=[{"role": "user", "content": "q"}])

    assert content == "final answer"
    assert create.n == 2
    assert _no_sleep == [2.0]


def test_llm_call_synthesis_deterministic_error_returns_empty_without_retry(monkeypatch, _no_sleep):
    """Synthesis never propagates: a deterministic failure returns '' immediately
    (no backoff wasted on a hopeless call)."""
    monkeypatch.setattr(base_agent, "get_synthesis_temperature", lambda: 0.2)
    monkeypatch.setattr(base_agent, "get_synthesis_max_tokens", lambda: 100)
    monkeypatch.setattr(base_agent, "get_synthesis_provider_preferences", lambda: None)

    def create(**kw):
        create.n += 1
        raise Exception("401 Unauthorized")
    create.n = 0

    agent = _full_agent(create)
    content = agent._llm_call_synthesis(messages_override=[{"role": "user", "content": "q"}])

    assert content == ""
    assert create.n == 1
    assert _no_sleep == []


def test_classify_and_extract_retries_transient_then_succeeds(monkeypatch, _no_sleep):
    """Dev-specific: unlike the wikikgqa-2026 reference (which left classification
    unwrapped), this repo's task requires classification to go through the same
    retry layer, so a transient hiccup is retried instead of silently degrading
    to the 'Query' default on the first blip."""
    monkeypatch.setattr(base_agent, "get_chat_seed", lambda: None)
    monkeypatch.setattr(base_agent, "get_chat_temperature", lambda: 0.2)

    def create(**kw):
        create.n += 1
        if create.n == 1:
            raise Exception("500 Internal Server Error")
        return _completion('{"question_type": "Count", "entities": ["a"], "relations": []}')
    create.n = 0

    agent = _full_agent(create)
    result = agent._classify_and_extract("How many X are there?")

    assert result["question_type"] == "Count"
    assert create.n == 2
    assert _no_sleep == [2.0]


def test_classify_and_extract_gives_up_and_defaults_to_query(monkeypatch, _no_sleep):
    """A deterministic (or exhausted) failure still degrades to the existing
    'Query' default instead of raising out of the whole question."""
    monkeypatch.setattr(base_agent, "get_chat_seed", lambda: None)
    monkeypatch.setattr(base_agent, "get_chat_temperature", lambda: 0.2)

    def create(**kw):
        raise Exception("401 Unauthorized")

    agent = _full_agent(create)
    result = agent._classify_and_extract("How many X are there?")

    assert result == {"question_type": "Query", "entities": [], "relations": []}
    assert _no_sleep == []
