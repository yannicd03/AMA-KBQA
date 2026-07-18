"""Hermetic unit tests for Orchestrator._route_autonomously.

No network calls, no API keys, no MCP subprocess. The Orchestrator is
constructed via __new__ so __init__ (which calls assert_provider_api_key_present
and get_chat_client) is bypassed; all needed attributes are set by hand.

Routing contract (one round-trip): the probing tool is called directly via
MCP with the question verbatim (no LLM), then a SINGLE LLM call with the
evidence in the user message commits to an agent through a forced
`select_agent` tool call. A failed probe degrades to a domain-only decision;
a failed decision returns None.
"""
import asyncio
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from ama_kbqa.agents.orchestrator_agent.agent import Orchestrator
from ama_kbqa.llm.retry import TransientRetry


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    """Retry backoff uses real time.sleep; keep these hermetic tests instant."""
    monkeypatch.setattr("time.sleep", lambda s: None)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(coro):
    return asyncio.run(coro)


def _make_tool_call(id_="tc_001", name="select_agent", arguments="{}"):
    """Build a SimpleNamespace that looks like an OpenAI ToolCall."""
    return SimpleNamespace(
        id=id_,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _make_completion(tool_calls=None, content=None):
    """Build a minimal completion SimpleNamespace with one choice."""
    message = SimpleNamespace(tool_calls=tool_calls, content=content)
    choice = SimpleNamespace(message=message)
    return SimpleNamespace(choices=[choice])


class _FakeMcp:
    """Minimal MCP stub: records call_tool invocations."""

    def __init__(self, tool_result_json):
        self._tool_result = tool_result_json
        self.calls = []

    async def call_tool(self, name, args):
        self.calls.append((name, args))
        return self._tool_result


class _BrokenProbeMcp:
    async def call_tool(self, name, args):
        raise RuntimeError("qdrant down")


_EVIDENCE_JSON = json.dumps({
    "semantics": {"subject": "who", "predicate": "invented", "objects": ["quantum computing"]},
    "kg_evidence": {
        "kqapro": {"terms_probed": 1, "terms_matched": 0, "avg_score": 0.0, "matches": {}},
        "sciqa": {"terms_probed": 1, "terms_matched": 1, "avg_score": 0.82,
                  "matches": {"quantum computing": {"id": "R123", "label": "Quantum Computing", "score": 0.82}}},
    },
    "degraded": False,
})


def _make_orchestrator(mcp, client, agent_config=None):
    """Instantiate Orchestrator without running __init__."""
    o = Orchestrator.__new__(Orchestrator)
    o.name = "TEST_ORCHESTRATOR"
    o.mcp = mcp
    o.client = client
    o.model = "test-model"
    o.last_routing_reason = None
    o._retry = TransientRetry()
    o._agent_config = agent_config or {
        "kqapro_agent": {
            "module": "ama_kbqa.agents.kqapro_agent.agent",
            "class": "KQAProAgent",
            "description": "General world knowledge.",
        },
        "sciqa_agent": {
            "module": "ama_kbqa.agents.sciqa_agent.agent",
            "class": "SciQAAgent",
            "description": "Scholarly knowledge.",
        },
    }
    return o


def _one_call_client(tool_calls, captured_kwargs=None):
    """Stub whose .chat.completions.create() returns one canned completion."""
    def _create(**kwargs):
        if captured_kwargs is not None:
            captured_kwargs.append(kwargs)
        return _make_completion(tool_calls=tool_calls)

    client = MagicMock()
    client.chat.completions.create.side_effect = _create
    return client


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRouteAutonomouslyHappyPath:

    def test_routes_to_sciqa_agent_and_sets_reason(self):
        decision_tc = [_make_tool_call(
            arguments='{"agent": "sciqa_agent", "reason": "scholarly"}',
        )]
        mcp = _FakeMcp(_EVIDENCE_JSON)
        client = _one_call_client(decision_tc)
        o = _make_orchestrator(mcp, client)

        result = _run(o._route_autonomously("Who invented quantum computing?"))

        assert result == "sciqa_agent"
        assert o.last_routing_reason == "scholarly"

    def test_probe_is_called_directly_with_verbatim_question(self):
        decision_tc = [_make_tool_call(
            arguments='{"agent": "sciqa_agent", "reason": "scholarly"}',
        )]
        mcp = _FakeMcp(_EVIDENCE_JSON)
        client = _one_call_client(decision_tc)
        o = _make_orchestrator(mcp, client)

        _run(o._route_autonomously("Who invented quantum computing?"))

        assert mcp.calls == [
            ("analyze_query_recommend_db", {"question": "Who invented quantum computing?"}),
        ]

    def test_single_llm_call_with_evidence_and_forced_select_agent(self):
        """Exactly ONE LLM call; evidence is in the user message; select_agent
        is forced via tool_choice."""
        decision_tc = [_make_tool_call(
            arguments='{"agent": "kqapro_agent", "reason": "world knowledge"}',
        )]
        captured = []
        mcp = _FakeMcp(_EVIDENCE_JSON)
        client = _one_call_client(decision_tc, captured_kwargs=captured)
        o = _make_orchestrator(mcp, client)

        result = _run(o._route_autonomously("Some question"))

        assert result == "kqapro_agent"
        assert len(captured) == 1, "client should be called exactly once"

        kwargs = captured[0]
        user_msgs = [m for m in kwargs["messages"] if m["role"] == "user"]
        assert len(user_msgs) == 1
        assert "Some question" in user_msgs[0]["content"]
        assert _EVIDENCE_JSON in user_msgs[0]["content"]

        assert kwargs["tool_choice"] == {
            "type": "function", "function": {"name": "select_agent"}
        }
        tool_names = [t["function"]["name"] for t in kwargs["tools"]]
        assert tool_names == ["select_agent"]


class TestRouteAutonomouslyDegradedAndFailurePaths:

    def test_returns_none_when_mcp_is_none(self):
        client = MagicMock()
        o = _make_orchestrator(mcp=None, client=client)

        result = _run(o._route_autonomously("Anything"))

        assert result is None
        client.chat.completions.create.assert_not_called()

    def test_probe_failure_degrades_to_domain_only_decision(self):
        """A dead probe must NOT abort routing; the LLM decides from the
        question's domain with degraded evidence."""
        decision_tc = [_make_tool_call(
            arguments='{"agent": "sciqa_agent", "reason": "research domain"}',
        )]
        captured = []
        client = _one_call_client(decision_tc, captured_kwargs=captured)
        o = _make_orchestrator(_BrokenProbeMcp(), client)

        result = _run(o._route_autonomously("Which papers evaluate BERT?"))

        assert result == "sciqa_agent"
        user_content = captured[0]["messages"][1]["content"]
        evidence = json.loads(user_content.split("knowledge graphs:\n", 1)[1])
        assert evidence["degraded"] is True
        assert "probe unavailable" in evidence["note"]

    def test_returns_none_when_decision_has_no_tool_calls(self):
        mcp = _FakeMcp(_EVIDENCE_JSON)
        client = _one_call_client(tool_calls=None)

        o = _make_orchestrator(mcp, client)
        result = _run(o._route_autonomously("Anything"))

        assert result is None

    def test_returns_none_when_selected_agent_is_unknown(self):
        decision_tc = [_make_tool_call(
            arguments='{"agent": "ghost_agent", "reason": "unknown"}',
        )]
        mcp = _FakeMcp(_EVIDENCE_JSON)
        client = _one_call_client(decision_tc)

        o = _make_orchestrator(mcp, client)
        result = _run(o._route_autonomously("Anything"))

        assert result is None
        assert o.last_routing_reason is None

    def test_returns_none_when_client_raises(self):
        mcp = _FakeMcp(_EVIDENCE_JSON)

        client = MagicMock()
        client.chat.completions.create.side_effect = Exception("API timeout")

        o = _make_orchestrator(mcp, client)
        result = _run(o._route_autonomously("Anything"))

        assert result is None


class TestOrchestratorRetry:
    """The routing decision call and the last-resort LLM fallback both go through
    Orchestrator._create_with_retry (self._retry, shared core in
    ama_kbqa.llm.retry) — same stepped-backoff resilience as BaseKBQAAgent, and
    the same persistent-per-instance ramp (a still-flaky endpoint waits longer on
    the next call, not from scratch)."""

    def test_route_autonomously_retries_transient_error_then_succeeds(self, monkeypatch):
        waits: list[float] = []
        monkeypatch.setattr("time.sleep", lambda s: waits.append(s))

        decision_tc = [_make_tool_call(
            arguments='{"agent": "sciqa_agent", "reason": "scholarly"}',
        )]
        mcp = _FakeMcp(_EVIDENCE_JSON)

        calls = {"n": 0}

        def _create(**kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise Exception("Open WebUI: Server Connection Error")
            return _make_completion(tool_calls=decision_tc)

        client = MagicMock()
        client.chat.completions.create.side_effect = _create
        o = _make_orchestrator(mcp, client)

        result = _run(o._route_autonomously("Who invented quantum computing?"))

        assert result == "sciqa_agent"
        assert calls["n"] == 2
        assert waits == [2.0]
        assert o._retry.level == 0  # reset after the eventual success

    def test_fallback_llm_goes_through_retry_and_shares_the_agent_level(self, monkeypatch):
        waits: list[float] = []
        monkeypatch.setattr("time.sleep", lambda s: waits.append(s))

        calls = {"n": 0}

        def _create(**kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise Exception("503 Service Unavailable")
            message = SimpleNamespace(content="fallback answer")
            return SimpleNamespace(choices=[SimpleNamespace(message=message)])

        client = MagicMock()
        client.chat.completions.create.side_effect = _create
        o = _make_orchestrator(mcp=None, client=client)
        o._retry.level = 1  # as if a prior call on this instance already ramped it

        answer = o._fallback_llm("Some question")

        assert answer == "fallback answer"
        assert calls["n"] == 2
        # Starts at the persisted level (index 1 = 5s), not from scratch.
        assert waits == [5.0]

    def test_deterministic_error_in_route_autonomously_does_not_retry(self, monkeypatch):
        waits: list[float] = []
        monkeypatch.setattr("time.sleep", lambda s: waits.append(s))

        mcp = _FakeMcp(_EVIDENCE_JSON)
        client = MagicMock()
        client.chat.completions.create.side_effect = Exception("401 Unauthorized")
        o = _make_orchestrator(mcp, client)

        result = _run(o._route_autonomously("Anything"))

        assert result is None
        assert client.chat.completions.create.call_count == 1
        assert waits == []
