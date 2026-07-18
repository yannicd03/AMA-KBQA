"""SciQAAgent overrides `_classify_question` with its own chat.completions.create
call (fewshot injection needs the raw question_type before the base class's combined
classify+extract runs). That override must go through the same TransientRetry-backed
`_create_with_retry` as everything else in BaseKBQAAgent, not a bare client call that
would bypass the resilience layer for one whole call path.
"""

from __future__ import annotations

import types

import pytest

from ama_kbqa.agents.sciqa_agent.agent import SciQAAgent
from chatkit import TransientRetry


def _completion(content: str):
    message = types.SimpleNamespace(content=content)
    choice = types.SimpleNamespace(message=message)
    return types.SimpleNamespace(choices=[choice], usage=None)


def _client(create):
    return types.SimpleNamespace(
        chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create))
    )


def _bare_sciqa_agent(create):
    agent = object.__new__(SciQAAgent)
    agent._retry = TransientRetry()
    agent._trace = lambda *a, **k: None
    agent.client = _client(create)
    agent.model = "test-model"
    agent.use_fewshot = False
    return agent


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    waits: list[float] = []
    monkeypatch.setattr("time.sleep", lambda s: waits.append(s))
    return waits


def test_sciqa_classify_question_retries_transient_then_succeeds(monkeypatch, _no_sleep):
    monkeypatch.setattr(
        "ama_kbqa.agents.sciqa_agent.agent.get_chat_temperature", lambda: 0.2
    )

    def create(**kw):
        create.n += 1
        if create.n == 1:
            raise Exception("Open WebUI: Server Connection Error")
        return _completion('{"question_type": "Aggregation"}')
    create.n = 0

    agent = _bare_sciqa_agent(create)
    result = agent._classify_question("How many papers use BERT?")

    assert result["question_type"] == "Aggregation"
    assert create.n == 2
    assert _no_sleep == [2.0]


def test_sciqa_classify_question_deterministic_error_defaults_to_general(monkeypatch, _no_sleep):
    monkeypatch.setattr(
        "ama_kbqa.agents.sciqa_agent.agent.get_chat_temperature", lambda: 0.2
    )

    def create(**kw):
        raise Exception("401 Unauthorized")

    agent = _bare_sciqa_agent(create)
    result = agent._classify_question("How many papers use BERT?")

    assert result["question_type"] == "General"
    assert _no_sleep == []
