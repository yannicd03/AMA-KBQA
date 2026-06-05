"""Hermetic unit tests for Orchestrator._route_autonomously.

No network calls, no API keys, no MCP subprocess. The Orchestrator is
constructed via __new__ so __init__ (which calls assert_provider_api_key_present
and get_chat_client) is bypassed; all needed attributes are set by hand.

Async pattern: asyncio.run() helpers, mirroring tests/server/test_orchestrator_server.py
and tests/framework/test_base_agent_multiturn.py.
"""
import asyncio
import json
from types import SimpleNamespace
from typing import Optional
from unittest.mock import MagicMock

import pytest

from ama_kbqa.agents.orchestrator_agent.agent import Orchestrator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(coro):
    return asyncio.run(coro)


def _make_tool_call(id_="tc_001", name="analyze_query_recommend_db", arguments='{"question":"test"}'):
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
    """Minimal MCP stub: list_tools + call_tool, both async."""

    def __init__(self, tools, tool_result_json):
        self._tools = tools
        self._tool_result = tool_result_json

    async def list_tools(self):
        return self._tools

    async def call_tool(self, name, args):
        return self._tool_result


def _fake_tool(name="analyze_query_recommend_db", description="Probe KGs"):
    """A fake MCP tool with the attributes _mcp_tool_to_openai needs."""
    return SimpleNamespace(
        name=name,
        description=description,
        inputSchema={
            "type": "object",
            "properties": {"question": {"type": "string"}},
            "required": ["question"],
        },
    )


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


# ---------------------------------------------------------------------------
# Two-call client stub factory
# ---------------------------------------------------------------------------

def _two_call_client(step1_tool_calls, step2_tool_calls, captured_kwargs=None):
    """
    Returns a stub whose .chat.completions.create() callable:
    - first call  -> completion with step1_tool_calls
    - second call -> completion with step2_tool_calls

    If captured_kwargs is a list, each call's kwargs dict is appended to it.
    """
    call_count = [0]

    def _create(**kwargs):
        if captured_kwargs is not None:
            captured_kwargs.append(kwargs)
        call_count[0] += 1
        if call_count[0] == 1:
            return _make_completion(tool_calls=step1_tool_calls)
        return _make_completion(tool_calls=step2_tool_calls)

    client = MagicMock()
    client.chat.completions.create.side_effect = _create
    return client


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRouteAutonomouslyHappyPath:

    def test_routes_to_sciqa_agent_and_sets_reason(self):
        step1_tc = [_make_tool_call()]
        step2_tc = [_make_tool_call(
            id_="tc_002",
            name="select_agent",
            arguments='{"agent": "sciqa_agent", "reason": "scholarly"}',
        )]
        mcp = _FakeMcp([_fake_tool()], _EVIDENCE_JSON)
        client = _two_call_client(step1_tc, step2_tc)
        o = _make_orchestrator(mcp, client)

        result = _run(o._route_autonomously("Who invented quantum computing?"))

        assert result == "sciqa_agent"
        assert o.last_routing_reason == "scholarly"

    def test_second_call_receives_tool_result_and_forced_select_agent(self):
        """Step-2 messages must include the tool result and use forced tool_choice."""
        step1_tc = [_make_tool_call(id_="tc_abc")]
        step2_tc = [_make_tool_call(
            id_="tc_def",
            name="select_agent",
            arguments='{"agent": "kqapro_agent", "reason": "world knowledge"}',
        )]
        captured = []
        mcp = _FakeMcp([_fake_tool()], _EVIDENCE_JSON)
        client = _two_call_client(step1_tc, step2_tc, captured_kwargs=captured)
        o = _make_orchestrator(mcp, client)

        result = _run(o._route_autonomously("Some question"))

        assert result == "kqapro_agent"
        assert len(captured) == 2, "client should be called exactly twice"

        # Second call must include the tool result message.
        second_msgs = captured[1]["messages"]
        tool_result_msgs = [m for m in second_msgs if m.get("role") == "tool"]
        assert len(tool_result_msgs) == 1
        assert tool_result_msgs[0]["content"] == _EVIDENCE_JSON
        assert tool_result_msgs[0]["tool_call_id"] == "tc_abc"

        # Second call must force select_agent via tool_choice.
        tc = captured[1]["tool_choice"]
        assert tc == {"type": "function", "function": {"name": "select_agent"}}


class TestRouteAutonomouslyFailurePaths:

    def test_returns_none_when_mcp_is_none(self):
        client = MagicMock()
        o = _make_orchestrator(mcp=None, client=client)

        result = _run(o._route_autonomously("Anything"))

        assert result is None
        client.chat.completions.create.assert_not_called()

    def test_returns_none_when_list_tools_raises(self):
        class _BrokenMcp:
            async def list_tools(self):
                raise RuntimeError("transport error")

            async def call_tool(self, name, args):
                raise AssertionError("should not be reached")

        client = MagicMock()
        o = _make_orchestrator(mcp=_BrokenMcp(), client=client)

        result = _run(o._route_autonomously("Anything"))

        assert result is None

    def test_returns_none_when_step1_has_no_tool_calls(self):
        # LLM replies without calling any tool (no tool_calls).
        mcp = _FakeMcp([_fake_tool()], _EVIDENCE_JSON)

        call_count = [0]
        def _create(**kwargs):
            call_count[0] += 1
            return _make_completion(tool_calls=None, content="I cannot decide.")

        client = MagicMock()
        client.chat.completions.create.side_effect = _create

        o = _make_orchestrator(mcp, client)
        result = _run(o._route_autonomously("Anything"))

        assert result is None
        # Only one LLM call should have been made (step 1).
        assert call_count[0] == 1

    def test_returns_none_when_step2_has_no_tool_calls(self):
        step1_tc = [_make_tool_call()]
        # Second LLM call returns no tool_calls.
        mcp = _FakeMcp([_fake_tool()], _EVIDENCE_JSON)
        client = _two_call_client(step1_tc, step2_tool_calls=None)

        o = _make_orchestrator(mcp, client)
        result = _run(o._route_autonomously("Anything"))

        assert result is None

    def test_returns_none_when_selected_agent_is_unknown(self):
        step1_tc = [_make_tool_call()]
        step2_tc = [_make_tool_call(
            id_="tc_x",
            name="select_agent",
            arguments='{"agent": "ghost_agent", "reason": "unknown"}',
        )]
        mcp = _FakeMcp([_fake_tool()], _EVIDENCE_JSON)
        client = _two_call_client(step1_tc, step2_tc)

        o = _make_orchestrator(mcp, client)
        result = _run(o._route_autonomously("Anything"))

        assert result is None
        assert o.last_routing_reason is None

    def test_returns_none_when_client_raises(self):
        mcp = _FakeMcp([_fake_tool()], _EVIDENCE_JSON)

        client = MagicMock()
        client.chat.completions.create.side_effect = Exception("API timeout")

        o = _make_orchestrator(mcp, client)
        result = _run(o._route_autonomously("Anything"))

        assert result is None
