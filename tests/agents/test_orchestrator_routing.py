"""Hermetic unit tests for the Orchestrator's routing and dispatch.

Covers _route_autonomously (probe -> select_agent / select_agents),
_delegate / _run_specialist (single dispatch + MCP-close-in-own-task),
_federate (concurrent fan-out with per-specialist failure isolation), and
_fuse_answers (answer fusion with degrade paths).

No network calls, no API keys, no MCP subprocess. The Orchestrator is
constructed via __new__ so __init__ (which calls assert_provider_api_key_present
and get_chat_client) is bypassed; all needed attributes are set by hand.

Routing contract (one round-trip, both modes): the probing tool is called
directly via MCP with the question verbatim (no LLM), then a SINGLE LLM call
with the evidence in the user message commits to agent(s) through a forced
tool call. Router mode (federation disabled) sends dev's original
`select_agent` (string enum) byte-for-byte; Federated mode sends
`select_agents` (array, capped at max_specialists). A failed probe degrades
to a domain-only decision; a failed decision returns None.
"""
import asyncio
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from chatkit import TransientRetry

from ama_kbqa.agents.orchestrator_agent.agent import Orchestrator
from ama_kbqa.framework.trace import TraceRecorder


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


class _FakeAgent:
    """Minimal sub-agent stub matching the surface _run_specialist touches."""

    def __init__(self, name, answer="", error=None, delay=0.0, journal_state=None,
                 has_close=True, close_calls=None):
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
        self._close_calls = close_calls if close_calls is not None else []
        if not has_close:
            # Simulate a sub-agent surface with no close() (getattr(..., None)
            # path in _run_specialist must tolerate this).
            self.close = None
            del self.close

    async def ask(self, query):
        if self._delay:
            await asyncio.sleep(self._delay)
        if self._error is not None:
            raise self._error
        return self._answer

    async def close(self):
        import asyncio as _asyncio
        self._close_calls.append(_asyncio.current_task())


# ---------------------------------------------------------------------------
# Tests: single-dispatch routing (Router mode, unchanged contract)
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

        assert result == ["sciqa_agent"]
        assert o.last_routing_reason == "scholarly"
        assert o.last_routing_evidence == _EVIDENCE_JSON

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

        assert result == ["kqapro_agent"]
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

        assert result == ["sciqa_agent"]
        user_content = captured[0]["messages"][1]["content"]
        evidence = json.loads(user_content.split("knowledge graphs:\n", 1)[1])
        assert evidence["degraded"] is True
        assert "probe unavailable" in evidence["note"]
        assert o.last_routing_evidence is not None
        assert json.loads(o.last_routing_evidence)["degraded"] is True

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
    """The routing decision call, the fusion call and the last-resort LLM
    fallback all go through Orchestrator._create_with_retry (self._retry,
    shared core in chatkit.retry) — same stepped-backoff resilience as
    BaseKBQAAgent, and the same persistent-per-instance ramp (a still-flaky
    endpoint waits longer on the next call, not from scratch)."""

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

        assert result == ["sciqa_agent"]
        assert calls["n"] == 2
        assert waits == [2.0]
        assert o._retry.level == 0  # reset after the eventual success

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


# ---------------------------------------------------------------------------
# Tests: federated routing (mode-gated tool/prompt)
# ---------------------------------------------------------------------------

class TestRouteAutonomouslyFederated:

    def test_federated_selection_returns_both_agents(self):
        decision_tc = [_make_tool_call(
            name="select_agents",
            arguments='{"agents": ["kqapro_agent", "sciqa_agent"], "reason": "spans both domains"}',
        )]
        mcp = _FakeMcp(_EVIDENCE_JSON)
        client = _one_call_client(decision_tc)
        o = _make_orchestrator(mcp, client)
        o._federation_enabled = True
        o._federation_max_specialists = 2

        result = _run(o._route_autonomously("Papers about the city of Berlin?"))

        assert result == ["kqapro_agent", "sciqa_agent"]
        assert o.last_routing_reason == "spans both domains"

    def test_federated_call_uses_select_agents_tool_and_addendum_prompt(self):
        decision_tc = [_make_tool_call(
            name="select_agents",
            arguments='{"agents": ["sciqa_agent"], "reason": "scholarly"}',
        )]
        captured = []
        mcp = _FakeMcp(_EVIDENCE_JSON)
        client = _one_call_client(decision_tc, captured_kwargs=captured)
        o = _make_orchestrator(mcp, client)
        o._federation_enabled = True
        o._federation_max_specialists = 2

        _run(o._route_autonomously("Anything"))

        kwargs = captured[0]
        assert kwargs["tool_choice"] == {
            "type": "function", "function": {"name": "select_agents"}
        }
        tool_names = [t["function"]["name"] for t in kwargs["tools"]]
        assert tool_names == ["select_agents"]
        system_msg = kwargs["messages"][0]["content"]
        assert "MULTIPLE agents" in system_msg

    def test_multi_selection_is_capped_to_one_when_federation_disabled(self):
        """Defense in depth: even if the provider ignores maxItems, a
        multi-agent selection must degrade to single dispatch when
        federation is off. (Router mode never sends select_agents at all,
        so this exercises the defensive cap on a misbehaving/legacy client.)"""
        decision_tc = [_make_tool_call(
            arguments='{"agent": "sciqa_agent", "reason": "greedy"}',
        )]
        mcp = _FakeMcp(_EVIDENCE_JSON)
        client = _one_call_client(decision_tc)
        o = _make_orchestrator(mcp, client)
        assert o._federation_enabled is False  # class default

        result = _run(o._route_autonomously("Anything"))

        assert result == ["sciqa_agent"]

    def test_duplicate_agents_are_deduped_in_router_order(self):
        decision_tc = [_make_tool_call(
            name="select_agents",
            arguments='{"agents": ["sciqa_agent", "sciqa_agent"], "reason": "stutter"}',
        )]
        mcp = _FakeMcp(_EVIDENCE_JSON)
        client = _one_call_client(decision_tc)
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
        assert schema["minItems"] == 1
        assert set(schema["items"]["enum"]) == {"kqapro_agent", "sciqa_agent"}

    def test_returns_none_when_agents_list_is_empty(self):
        decision_tc = [_make_tool_call(
            name="select_agents",
            arguments='{"agents": [], "reason": "indecisive"}',
        )]
        mcp = _FakeMcp(_EVIDENCE_JSON)
        client = _one_call_client(decision_tc)
        o = _make_orchestrator(mcp, client)
        o._federation_enabled = True

        result = _run(o._route_autonomously("Anything"))

        assert result is None

    def test_returns_none_when_unknown_agent_in_federated_selection(self):
        decision_tc = [_make_tool_call(
            name="select_agents",
            arguments='{"agents": ["ghost_agent"], "reason": "unknown"}',
        )]
        mcp = _FakeMcp(_EVIDENCE_JSON)
        client = _one_call_client(decision_tc)
        o = _make_orchestrator(mcp, client)
        o._federation_enabled = True

        result = _run(o._route_autonomously("Anything"))

        assert result is None
        assert o.last_routing_reason is None


# ---------------------------------------------------------------------------
# Tests: _delegate / _run_specialist (single dispatch)
# ---------------------------------------------------------------------------

class TestDelegateAndRunSpecialist:

    def test_delegate_returns_specialist_answer(self, monkeypatch):
        o = _make_orchestrator(mcp=None, client=MagicMock())
        agent = _FakeAgent("kqapro_agent", answer="42 films")
        monkeypatch.setattr(o, "_load_agent", lambda name: agent)

        answer = _run(o._delegate("kqapro_agent", "Q?"))

        assert answer == "42 films"

    def test_delegate_falls_back_to_kqapro_on_specialist_error(self, monkeypatch):
        o = _make_orchestrator(mcp=None, client=MagicMock())
        agent = _FakeAgent("sciqa_agent", error=RuntimeError("boom"))
        monkeypatch.setattr(o, "_load_agent", lambda name: agent)

        async def _fake_fallback(query):
            return "fallback answer"

        o._fallback_kqapro = _fake_fallback

        answer = _run(o._delegate("sciqa_agent", "Q?"))

        assert answer == "fallback answer"

    def test_run_specialist_raises_when_agent_cannot_be_loaded(self, monkeypatch):
        o = _make_orchestrator(mcp=None, client=MagicMock())
        monkeypatch.setattr(o, "_load_agent", lambda name: None)

        with pytest.raises(RuntimeError):
            _run(o._run_specialist("kqapro_agent", "Q?"))

    def test_run_specialist_tolerates_agent_without_close(self, monkeypatch):
        o = _make_orchestrator(mcp=None, client=MagicMock())
        agent = _FakeAgent("kqapro_agent", answer="ok", has_close=False)
        monkeypatch.setattr(o, "_load_agent", lambda name: agent)

        result = _run(o._run_specialist("kqapro_agent", "Q?"))

        assert result["answer"] == "ok"

    def test_run_specialist_close_error_does_not_mask_answer(self, monkeypatch):
        o = _make_orchestrator(mcp=None, client=MagicMock())
        agent = _FakeAgent("kqapro_agent", answer="ok")

        async def _boom_close():
            raise RuntimeError("close blew up")

        agent.close = _boom_close
        monkeypatch.setattr(o, "_load_agent", lambda name: agent)

        result = _run(o._run_specialist("kqapro_agent", "Q?"))

        assert result["answer"] == "ok"


# ---------------------------------------------------------------------------
# Tests: federated dispatch (_federate) and fusion (_fuse_answers)
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
        """The handoff carries each specialist's journal: its verified
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

    def test_each_specialist_closes_mcp_in_its_own_task(self):
        """Federated dispatch runs specialists concurrently via
        asyncio.gather (one Task each); each must close its own MCP
        connection from within its own task, not a shared/foreign one."""
        o = _make_orchestrator(mcp=None, client=_fusion_client())
        kqapro_closes = []
        sciqa_closes = []
        o._agents = {
            "kqapro_agent": _FakeAgent("kqapro_agent", answer="a", delay=0.01, close_calls=kqapro_closes),
            "sciqa_agent": _FakeAgent("sciqa_agent", answer="b", delay=0.01, close_calls=sciqa_closes),
        }

        _run(o._federate(["kqapro_agent", "sciqa_agent"], "Q?"))

        assert len(kqapro_closes) == 1
        assert len(sciqa_closes) == 1
        # Distinct tasks (asyncio.gather schedules one Task per coroutine).
        assert kqapro_closes[0] is not sciqa_closes[0]

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

    def test_fusion_call_goes_through_create_with_retry(self, monkeypatch):
        """The fusion LLM call must go through self._create_with_retry, same
        stepped-backoff resilience as the routing decision call."""
        waits: list[float] = []
        monkeypatch.setattr("time.sleep", lambda s: waits.append(s))

        calls = {"n": 0}

        def _create(**kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise Exception("Open WebUI: Server Connection Error")
            return _make_completion(tool_calls=None, content="fused")

        client = MagicMock()
        client.chat.completions.create.side_effect = _create
        o = _make_orchestrator(mcp=None, client=client)

        answer = _run(o._fuse_answers("Q?", self._ANSWERS))

        assert answer == "fused"
        assert calls["n"] == 2
        assert waits == [2.0]


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
