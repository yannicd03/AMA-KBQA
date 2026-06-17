"""Hermetic unit tests for the Orchestrator's routing and dispatch.

Covers _route_autonomously (probe -> select_agents), _federate (concurrent
fan-out with per-specialist failure isolation) and _fuse_answers (answer
fusion with degrade paths).

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
from ama_kbqa.framework.trace import TraceRecorder


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
    """Instantiate Orchestrator without running __init__.

    Federation flags fall back to the class-level defaults (disabled);
    federated tests flip o._federation_enabled per instance.
    """
    o = Orchestrator.__new__(Orchestrator)
    o.name = "TEST_ORCHESTRATOR"
    o.mcp = mcp
    o.client = client
    o.model = "test-model"
    o.last_routing_reason = None
    o.last_routing_evidence = None
    o.recorder = TraceRecorder()
    o.journal_snapshots = []
    o.token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    o._agents = {}
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


class _FakeAgent:
    """Minimal sub-agent stub matching the surface _run_specialist touches."""

    def __init__(self, name, answer="", error=None, delay=0.0, journal_state=None):
        self.name = name
        self._answer = answer
        self._error = error
        self._delay = delay
        # `journal_state` is the JournalState dump the orchestrator renders
        # into the handoff scratchpad; default is an empty stub.
        state = journal_state if journal_state is not None else {"agent": name}
        self.journal_snapshots = [{"ts": 1.0, "trigger": "test", "state": state}]
        self.token_usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        self.recorder = None
        self._parent_span_id_override = None

    async def ask(self, query):
        if self._delay:
            await asyncio.sleep(self._delay)
        if self._error is not None:
            raise self._error
        return self._answer


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
            name="select_agents",
            arguments='{"agents": ["sciqa_agent"], "reason": "scholarly"}',
        )]
        mcp = _FakeMcp([_fake_tool()], _EVIDENCE_JSON)
        client = _two_call_client(step1_tc, step2_tc)
        o = _make_orchestrator(mcp, client)

        result = _run(o._route_autonomously("Who invented quantum computing?"))

        assert result == ["sciqa_agent"]
        assert o.last_routing_reason == "scholarly"
        assert o.last_routing_evidence == _EVIDENCE_JSON

    def test_second_call_receives_tool_result_and_forced_select_agents(self):
        """Step-2 messages must include the tool result and use forced tool_choice."""
        step1_tc = [_make_tool_call(id_="tc_abc")]
        step2_tc = [_make_tool_call(
            id_="tc_def",
            name="select_agents",
            arguments='{"agents": ["kqapro_agent"], "reason": "world knowledge"}',
        )]
        captured = []
        mcp = _FakeMcp([_fake_tool()], _EVIDENCE_JSON)
        client = _two_call_client(step1_tc, step2_tc, captured_kwargs=captured)
        o = _make_orchestrator(mcp, client)

        result = _run(o._route_autonomously("Some question"))

        assert result == ["kqapro_agent"]
        assert len(captured) == 2, "client should be called exactly twice"

        # Second call must include the tool result message.
        second_msgs = captured[1]["messages"]
        tool_result_msgs = [m for m in second_msgs if m.get("role") == "tool"]
        assert len(tool_result_msgs) == 1
        assert tool_result_msgs[0]["content"] == _EVIDENCE_JSON
        assert tool_result_msgs[0]["tool_call_id"] == "tc_abc"

        # Second call must force select_agents via tool_choice.
        tc = captured[1]["tool_choice"]
        assert tc == {"type": "function", "function": {"name": "select_agents"}}

    def test_single_dispatch_schema_caps_agents_at_one(self):
        """With federation disabled the select_agents schema must cap maxItems at 1."""
        o = _make_orchestrator(mcp=None, client=MagicMock())

        schema = o._select_agents_tool()["function"]["parameters"]["properties"]["agents"]

        assert schema["maxItems"] == 1
        assert schema["minItems"] == 1
        assert set(schema["items"]["enum"]) == {"kqapro_agent", "sciqa_agent"}


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
            name="select_agents",
            arguments='{"agents": ["ghost_agent"], "reason": "unknown"}',
        )]
        mcp = _FakeMcp([_fake_tool()], _EVIDENCE_JSON)
        client = _two_call_client(step1_tc, step2_tc)

        o = _make_orchestrator(mcp, client)
        result = _run(o._route_autonomously("Anything"))

        assert result is None
        assert o.last_routing_reason is None

    def test_returns_none_when_agents_list_is_empty(self):
        step1_tc = [_make_tool_call()]
        step2_tc = [_make_tool_call(
            id_="tc_x",
            name="select_agents",
            arguments='{"agents": [], "reason": "indecisive"}',
        )]
        mcp = _FakeMcp([_fake_tool()], _EVIDENCE_JSON)
        client = _two_call_client(step1_tc, step2_tc)

        o = _make_orchestrator(mcp, client)
        result = _run(o._route_autonomously("Anything"))

        assert result is None

    def test_returns_none_when_client_raises(self):
        mcp = _FakeMcp([_fake_tool()], _EVIDENCE_JSON)

        client = MagicMock()
        client.chat.completions.create.side_effect = Exception("API timeout")

        o = _make_orchestrator(mcp, client)
        result = _run(o._route_autonomously("Anything"))

        assert result is None


class TestRouteAutonomouslyFederated:

    def test_federated_selection_returns_both_agents(self):
        step1_tc = [_make_tool_call()]
        step2_tc = [_make_tool_call(
            id_="tc_fed",
            name="select_agents",
            arguments='{"agents": ["kqapro_agent", "sciqa_agent"], "reason": "spans both domains"}',
        )]
        mcp = _FakeMcp([_fake_tool()], _EVIDENCE_JSON)
        client = _two_call_client(step1_tc, step2_tc)
        o = _make_orchestrator(mcp, client)
        o._federation_enabled = True

        result = _run(o._route_autonomously("Papers about the city of Berlin?"))

        assert result == ["kqapro_agent", "sciqa_agent"]
        assert o.last_routing_reason == "spans both domains"

    def test_multi_selection_is_capped_to_one_when_federation_disabled(self):
        """Defense in depth: even if the provider ignores maxItems, a
        multi-agent selection must degrade to single dispatch when
        federation is off."""
        step1_tc = [_make_tool_call()]
        step2_tc = [_make_tool_call(
            id_="tc_fed",
            name="select_agents",
            arguments='{"agents": ["sciqa_agent", "kqapro_agent"], "reason": "greedy"}',
        )]
        mcp = _FakeMcp([_fake_tool()], _EVIDENCE_JSON)
        client = _two_call_client(step1_tc, step2_tc)
        o = _make_orchestrator(mcp, client)
        assert o._federation_enabled is False  # class default

        result = _run(o._route_autonomously("Anything"))

        assert result == ["sciqa_agent"]

    def test_duplicate_agents_are_deduped_in_router_order(self):
        step1_tc = [_make_tool_call()]
        step2_tc = [_make_tool_call(
            id_="tc_fed",
            name="select_agents",
            arguments='{"agents": ["sciqa_agent", "sciqa_agent"], "reason": "stutter"}',
        )]
        mcp = _FakeMcp([_fake_tool()], _EVIDENCE_JSON)
        client = _two_call_client(step1_tc, step2_tc)
        o = _make_orchestrator(mcp, client)
        o._federation_enabled = True

        result = _run(o._route_autonomously("Anything"))

        assert result == ["sciqa_agent"]

    def test_federated_schema_allows_max_specialists(self):
        o = _make_orchestrator(mcp=None, client=MagicMock())
        o._federation_enabled = True
        o._federation_max_specialists = 2

        schema = o._select_agents_tool()["function"]["parameters"]["properties"]["agents"]

        assert schema["maxItems"] == 2


# ---------------------------------------------------------------------------
# Federated dispatch (_federate) and fusion (_fuse_answers)
# ---------------------------------------------------------------------------

def _fusion_client(content="fused answer", captured_kwargs=None, error=None):
    """Client stub for the single fusion LLM call."""
    def _create(**kwargs):
        if captured_kwargs is not None:
            captured_kwargs.append(kwargs)
        if error is not None:
            raise error
        return _make_completion(tool_calls=None, content=content)

    client = MagicMock()
    client.chat.completions.create.side_effect = _create
    return client


class TestFederate:

    def test_runs_both_specialists_and_fuses(self):
        captured = []
        o = _make_orchestrator(mcp=None, client=_fusion_client(captured_kwargs=captured))
        o._agents = {
            "kqapro_agent": _FakeAgent("kqapro_agent", answer="42 films"),
            "sciqa_agent": _FakeAgent("sciqa_agent", answer="3 papers"),
        }

        answer = _run(o._federate(["kqapro_agent", "sciqa_agent"], "Q?"))

        assert answer == "fused answer"
        # Fusion prompt must carry both specialist answers.
        user_msg = captured[0]["messages"][-1]["content"]
        assert "42 films" in user_msg
        assert "3 papers" in user_msg
        # Both sub-agents' token usage is hoisted (fusion adds none: stub
        # completion has no usage attribute).
        assert o.token_usage["total_tokens"] == 30

    def test_specialist_scratchpad_reaches_fusion_prompt(self):
        """The handoff now carries each specialist's journal: its verified
        facts must surface in the fusion prompt as grounding evidence."""
        captured = []
        o = _make_orchestrator(mcp=None, client=_fusion_client(captured_kwargs=captured))
        kqapro_state = {
            "kg_name": "KQAPro",
            "verified_facts": [
                {"subject": "Q42", "predicate": "born_in", "object": "1879", "source": "RunSPARQL"}
            ],
        }
        sciqa_state = {
            "kg_name": "SciQA",
            "found_values": {"R7": {"citation_count": 12}},
        }
        o._agents = {
            "kqapro_agent": _FakeAgent("kqapro_agent", answer="1879", journal_state=kqapro_state),
            "sciqa_agent": _FakeAgent("sciqa_agent", answer="12 papers", journal_state=sciqa_state),
        }

        _run(o._federate(["kqapro_agent", "sciqa_agent"], "Q?"))

        user_msg = captured[0]["messages"][-1]["content"]
        # Working-notes block present, with structured journal content.
        assert "Working notes" in user_msg
        assert "born_in" in user_msg          # kqapro verified fact
        assert "citation_count" in user_msg   # sciqa found value

    def test_concurrent_delegate_spans_nest_as_siblings(self):
        """Each specialist gets its own delegate span; neither parents
        under the other despite running concurrently."""
        o = _make_orchestrator(mcp=None, client=_fusion_client())
        o._agents = {
            "kqapro_agent": _FakeAgent("kqapro_agent", answer="a", delay=0.01),
            "sciqa_agent": _FakeAgent("sciqa_agent", answer="b", delay=0.01),
        }

        _run(o._federate(["kqapro_agent", "sciqa_agent"], "Q?"))

        delegates = [e for e in o.recorder.events if e.kind == "delegate"]
        assert len(delegates) == 2
        assert {d.name for d in delegates} == {"kqapro_agent", "sciqa_agent"}
        delegate_ids = {d.span_id for d in delegates}
        assert all(d.parent_span_id not in delegate_ids for d in delegates)

    def test_journal_snapshots_are_tagged_with_source_agent(self):
        o = _make_orchestrator(mcp=None, client=_fusion_client())
        o._agents = {
            "kqapro_agent": _FakeAgent("kqapro_agent", answer="a"),
            "sciqa_agent": _FakeAgent("sciqa_agent", answer="b"),
        }

        _run(o._federate(["kqapro_agent", "sciqa_agent"], "Q?"))

        sources = sorted(s["source_agent"] for s in o.journal_snapshots)
        assert sources == ["kqapro_agent", "sciqa_agent"]

    def test_one_failure_degrades_to_surviving_answer_without_fusion(self):
        client = _fusion_client()
        o = _make_orchestrator(mcp=None, client=client)
        o._agents = {
            "kqapro_agent": _FakeAgent("kqapro_agent", error=RuntimeError("boom")),
            "sciqa_agent": _FakeAgent("sciqa_agent", answer="3 papers"),
        }

        answer = _run(o._federate(["kqapro_agent", "sciqa_agent"], "Q?"))

        assert answer == "3 papers"
        client.chat.completions.create.assert_not_called()

    def test_all_failures_fall_back_to_kqapro(self):
        o = _make_orchestrator(mcp=None, client=_fusion_client())
        o._agents = {
            "kqapro_agent": _FakeAgent("kqapro_agent", error=RuntimeError("boom")),
            "sciqa_agent": _FakeAgent("sciqa_agent", error=RuntimeError("crash")),
        }
        fallback_calls = []

        async def _fake_fallback(query):
            fallback_calls.append(query)
            return "fallback answer"

        o._fallback_kqapro = _fake_fallback

        answer = _run(o._federate(["kqapro_agent", "sciqa_agent"], "Q?"))

        assert answer == "fallback answer"
        assert fallback_calls == ["Q?"]


class TestFuseAnswers:

    _ANSWERS = [
        {"agent": "kqapro_agent", "answer": "Born in 1879."},
        {"agent": "sciqa_agent", "answer": "Cited in 12 papers."},
    ]

    def test_forwards_routing_evidence_to_fusion_prompt(self):
        captured = []
        o = _make_orchestrator(mcp=None, client=_fusion_client(captured_kwargs=captured))
        o.last_routing_evidence = _EVIDENCE_JSON

        _run(o._fuse_answers("Q?", self._ANSWERS))

        user_msg = captured[0]["messages"][-1]["content"]
        assert _EVIDENCE_JSON in user_msg
        # System prompt carries the conflict policy.
        system_msg = captured[0]["messages"][0]["content"]
        assert "CONFLICT" in system_msg

    def test_scratchpad_block_included_when_present(self):
        captured = []
        o = _make_orchestrator(mcp=None, client=_fusion_client(captured_kwargs=captured))
        answers = [
            {"agent": "kqapro_agent", "answer": "Born in 1879.",
             "scratchpad": "Verified Facts:\n  - [Q42] --born_in--> [1879] (from: RunSPARQL)"},
            {"agent": "sciqa_agent", "answer": "Cited in 12 papers.", "scratchpad": None},
        ]

        _run(o._fuse_answers("Q?", answers))

        user_msg = captured[0]["messages"][-1]["content"]
        assert "Working notes" in user_msg
        assert "born_in" in user_msg
        # System prompt explains how to use the notes.
        system_msg = captured[0]["messages"][0]["content"]
        assert "working notes" in system_msg.lower()

    def test_missing_scratchpad_omits_block(self):
        """Answers without a scratchpad (e.g. legacy/None) add no notes block."""
        captured = []
        o = _make_orchestrator(mcp=None, client=_fusion_client(captured_kwargs=captured))

        _run(o._fuse_answers("Q?", self._ANSWERS))  # fixtures have no scratchpad key

        user_msg = captured[0]["messages"][-1]["content"]
        assert "Working notes" not in user_msg

    def test_empty_fusion_degrades_to_first_answer(self):
        o = _make_orchestrator(mcp=None, client=_fusion_client(content=""))

        answer = _run(o._fuse_answers("Q?", self._ANSWERS))

        assert answer == "Born in 1879."

    def test_fusion_error_degrades_to_first_answer(self):
        o = _make_orchestrator(
            mcp=None, client=_fusion_client(error=Exception("API down"))
        )

        answer = _run(o._fuse_answers("Q?", self._ANSWERS))

        assert answer == "Born in 1879."

    def test_records_synthesis_span_with_agents(self):
        o = _make_orchestrator(mcp=None, client=_fusion_client())

        _run(o._fuse_answers("Q?", self._ANSWERS))

        spans = [e for e in o.recorder.events if e.kind == "synthesis"]
        assert len(spans) == 1
        assert spans[0].attributes["agents"] == "kqapro_agent, sciqa_agent"


class TestExtractScratchpad:

    def test_renders_journal_state_to_summary(self):
        o = _make_orchestrator(mcp=None, client=MagicMock())
        agent = _FakeAgent("kqapro_agent", journal_state={
            "kg_name": "KQAPro",
            "verified_facts": [
                {"subject": "Q42", "predicate": "born_in", "object": "1879", "source": "RunSPARQL"}
            ],
        })

        rendered = o._extract_scratchpad(agent)

        assert rendered is not None
        assert "born_in" in rendered

    def test_no_snapshots_returns_none(self):
        o = _make_orchestrator(mcp=None, client=MagicMock())
        agent = _FakeAgent("kqapro_agent")
        agent.journal_snapshots = []

        assert o._extract_scratchpad(agent) is None

    def test_missing_attribute_returns_none(self):
        o = _make_orchestrator(mcp=None, client=MagicMock())

        assert o._extract_scratchpad(object()) is None
