"""Tests for the stepped-backoff transient-retry layer (base_agent).

With the per-question wall-clock removed, _create_with_retry is the sole defense
against provider flakiness (KIT "Server Connection Error", 5xx, rate limits). It
must: retry transient errors with an escalating backoff, reset the ramp after a
success, and re-raise deterministic errors (auth/bad-request) immediately.
"""

from __future__ import annotations

import types

import pytest

from ama_kbqa.framework.base_agent import BaseKBQAAgent
from ama_kbqa.framework.config import KnowledgeGraphConfig
from chatkit import DEFAULT_MAX_ATTEMPTS, TransientRetry


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
