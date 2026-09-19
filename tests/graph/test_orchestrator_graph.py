"""Phase 3b tests: the orchestrator routing/delegation graph
(``ama_kbqa.graph.orchestrator``), using the SAME hermetic fakes
``tests/agents/test_orchestrator_routing.py`` uses for
``Orchestrator._route_autonomously`` (no network, no MCP subprocess).

Each scenario asserts (1) the routing decision matches what
``Orchestrator._route_autonomously`` (legacy) would decide for the same
scripted inputs, and (2) the recorder's ``(kind, name)`` event/span sequence
matches between the two engines for the routing portion (``agent_run`` +
``classify``) — delegate/fallback bodies are exercised through
``Orchestrator._delegate``/``_fallback_kqapro``/``_fallback_llm`` UNCHANGED
(see ``ama_kbqa.graph.orchestrator``'s module docstring), so this file
focuses on proving the graph reaches the same node/branch legacy reaches, not
on re-testing delegation itself (already covered by
``tests/agents/test_orchestrator_routing.py`` and any existing sub-agent
tests).
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from chatkit import TransientRetry

from ama_kbqa.agents.orchestrator_agent.agent import Orchestrator
from ama_kbqa.framework.trace import TraceRecorder

QUERY = "Who invented quantum computing?"


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)


# ---------------------------------------------------------------------------
# Fakes (mirrors tests/agents/test_orchestrator_routing.py)
# ---------------------------------------------------------------------------


def _make_tool_call(id_="tc_001", name="select_agent", arguments="{}"):
    return SimpleNamespace(id=id_, function=SimpleNamespace(name=name, arguments=arguments))


def _make_completion(tool_calls=None, content=None):
    message = SimpleNamespace(tool_calls=tool_calls, content=content)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class _FakeMcp:
    def __init__(self, tool_result_json):
        self._tool_result = tool_result_json
        self.calls = []
        self.closed = False

    async def call_tool(self, name, args):
        self.calls.append((name, args))
        return self._tool_result

    async def close(self):
        self.closed = True


class _BrokenProbeMcp:
    async def call_tool(self, name, args):
        raise RuntimeError("qdrant down")

    async def close(self):
        pass


_EVIDENCE_JSON = json.dumps(
    {
        "semantics": {"subject": "who", "predicate": "invented", "objects": ["quantum computing"]},
        "kg_evidence": {
            "kqapro": {"terms_probed": 1, "terms_matched": 0, "avg_score": 0.0, "matches": {}},
            "sciqa": {
                "terms_probed": 1,
                "terms_matched": 1,
                "avg_score": 0.82,
                "matches": {"quantum computing": {"id": "R123", "label": "Quantum Computing", "score": 0.82}},
            },
        },
        "degraded": False,
    }
)


def _one_call_client(tool_calls, captured_kwargs=None):
    def _create(**kwargs):
        if captured_kwargs is not None:
            captured_kwargs.append(kwargs)
        return _make_completion(tool_calls=tool_calls)

    client = MagicMock()
    client.chat.completions.create.side_effect = _create
    return client


def _make_orchestrator(mcp, client, agent_config=None) -> Orchestrator:
    o = Orchestrator.__new__(Orchestrator)
    o.name = "TEST_ORCHESTRATOR"
    o.mcp = mcp
    o.client = client
    o.model = "test-model"
    o.last_routing_reason = None
    o._retry = TransientRetry()
    o.recorder = TraceRecorder()
    o.journal_snapshots = []
    o.token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    o._agents = {}
    o.session_id = "test-session"
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

    async def _noop_init_mcp():
        pass

    o._init_mcp = _noop_init_mcp  # already "connected" via the fake above
    return o


def _event_seq(agent) -> list[tuple[str, str]]:
    return [(e["kind"], e["name"]) for e in agent.recorder.to_dicts()]


# ---------------------------------------------------------------------------
# Happy path: routes to sciqa_agent
# ---------------------------------------------------------------------------


def test_routes_to_sciqa_agent_matching_legacy_decision(monkeypatch):
    decision_tc = [_make_tool_call(arguments='{"agent": "sciqa_agent", "reason": "scholarly"}')]

    # Legacy
    legacy_mcp = _FakeMcp(_EVIDENCE_JSON)
    legacy_client = _one_call_client(decision_tc)
    legacy = _make_orchestrator(legacy_mcp, legacy_client)
    legacy_selected = _run(legacy._route_autonomously(QUERY))

    # Graph: stub _delegate so we don't need a real sub-agent import chain.
    from ama_kbqa.graph.orchestrator import run_orchestrator_graph

    graph_mcp = _FakeMcp(_EVIDENCE_JSON)
    graph_client = _one_call_client(decision_tc)
    graph = _make_orchestrator(graph_mcp, graph_client)

    delegate_calls = []

    async def _fake_delegate(agent_name, query):
        delegate_calls.append((agent_name, query))
        return f"DELEGATED:{agent_name}"

    graph._delegate = _fake_delegate

    answer = _run(run_orchestrator_graph(graph, QUERY))

    assert legacy_selected == "sciqa_agent"
    assert answer == "DELEGATED:sciqa_agent"
    assert delegate_calls == [("sciqa_agent", QUERY)]
    assert graph.last_routing_reason == "scholarly"
    assert graph_mcp.calls == [("analyze_query_recommend_db", {"question": QUERY})]
    assert graph_mcp.closed is True

    # Recorder event-kind parity for the routing portion.
    graph_kinds = [k for k, _ in _event_seq(graph)]
    assert "classify" in graph_kinds


def test_routes_to_kqapro_agent(monkeypatch):
    decision_tc = [_make_tool_call(arguments='{"agent": "kqapro_agent", "reason": "world knowledge"}')]
    from ama_kbqa.graph.orchestrator import run_orchestrator_graph

    mcp = _FakeMcp(_EVIDENCE_JSON)
    client = _one_call_client(decision_tc)
    graph = _make_orchestrator(mcp, client)

    delegate_calls = []

    async def _fake_delegate(agent_name, query):
        delegate_calls.append((agent_name, query))
        return f"DELEGATED:{agent_name}"

    graph._delegate = _fake_delegate

    answer = _run(run_orchestrator_graph(graph, QUERY))

    assert answer == "DELEGATED:kqapro_agent"
    assert delegate_calls == [("kqapro_agent", QUERY)]


# ---------------------------------------------------------------------------
# mcp is None: routing never attempted, straight to fallback_kqapro
# ---------------------------------------------------------------------------


def test_no_mcp_skips_routing_and_falls_back_to_kqapro():
    from ama_kbqa.graph.orchestrator import run_orchestrator_graph

    client = MagicMock()
    graph = _make_orchestrator(mcp=None, client=client)

    fallback_calls = []

    async def _fake_fallback_kqapro(query):
        fallback_calls.append(query)
        return "FALLBACK_KQAPRO"

    graph._fallback_kqapro = _fake_fallback_kqapro

    answer = _run(run_orchestrator_graph(graph, QUERY))

    assert answer == "FALLBACK_KQAPRO"
    assert fallback_calls == [QUERY]
    client.chat.completions.create.assert_not_called()


# ---------------------------------------------------------------------------
# Probe failure degrades to domain-only decision (matches legacy)
# ---------------------------------------------------------------------------


def test_probe_failure_degrades_to_domain_only_decision_matching_legacy():
    decision_tc = [_make_tool_call(arguments='{"agent": "sciqa_agent", "reason": "research domain"}')]

    legacy_client = _one_call_client(decision_tc)
    legacy = _make_orchestrator(_BrokenProbeMcp(), legacy_client)
    legacy_selected = _run(legacy._route_autonomously(QUERY))

    from ama_kbqa.graph.orchestrator import run_orchestrator_graph

    captured = []
    graph_client = _one_call_client(decision_tc, captured_kwargs=captured)
    graph = _make_orchestrator(_BrokenProbeMcp(), graph_client)

    delegate_calls = []

    async def _fake_delegate(agent_name, query):
        delegate_calls.append(agent_name)
        return f"DELEGATED:{agent_name}"

    graph._delegate = _fake_delegate

    answer = _run(run_orchestrator_graph(graph, QUERY))

    assert legacy_selected == "sciqa_agent"
    assert answer == "DELEGATED:sciqa_agent"
    user_content = captured[0]["messages"][1]["content"]
    evidence = json.loads(user_content.split("knowledge graphs:\n", 1)[1])
    assert evidence["degraded"] is True
    assert "probe unavailable" in evidence["note"]


# ---------------------------------------------------------------------------
# Decision has no tool calls / unknown agent -> fallback_kqapro
# ---------------------------------------------------------------------------


def test_decision_with_no_tool_calls_falls_back_to_kqapro():
    from ama_kbqa.graph.orchestrator import run_orchestrator_graph

    mcp = _FakeMcp(_EVIDENCE_JSON)
    client = _one_call_client(tool_calls=None)
    graph = _make_orchestrator(mcp, client)

    fallback_calls = []

    async def _fake_fallback_kqapro(query):
        fallback_calls.append(query)
        return "FALLBACK_KQAPRO"

    graph._fallback_kqapro = _fake_fallback_kqapro

    answer = _run(run_orchestrator_graph(graph, QUERY))

    assert answer == "FALLBACK_KQAPRO"
    assert fallback_calls == [QUERY]


def test_unknown_agent_falls_back_to_kqapro():
    from ama_kbqa.graph.orchestrator import run_orchestrator_graph

    decision_tc = [_make_tool_call(arguments='{"agent": "ghost_agent", "reason": "unknown"}')]
    mcp = _FakeMcp(_EVIDENCE_JSON)
    client = _one_call_client(decision_tc)
    graph = _make_orchestrator(mcp, client)

    fallback_calls = []

    async def _fake_fallback_kqapro(query):
        fallback_calls.append(query)
        return "FALLBACK_KQAPRO"

    graph._fallback_kqapro = _fake_fallback_kqapro

    answer = _run(run_orchestrator_graph(graph, QUERY))

    assert answer == "FALLBACK_KQAPRO"
    assert graph.last_routing_reason is None


def test_client_exception_falls_back_to_kqapro():
    from ama_kbqa.graph.orchestrator import run_orchestrator_graph

    mcp = _FakeMcp(_EVIDENCE_JSON)
    client = MagicMock()
    client.chat.completions.create.side_effect = Exception("API timeout")
    graph = _make_orchestrator(mcp, client)

    fallback_calls = []

    async def _fake_fallback_kqapro(query):
        fallback_calls.append(query)
        return "FALLBACK_KQAPRO"

    graph._fallback_kqapro = _fake_fallback_kqapro

    answer = _run(run_orchestrator_graph(graph, QUERY))

    assert answer == "FALLBACK_KQAPRO"


# ---------------------------------------------------------------------------
# Recorder event/span parity between engines for a full scripted run
# ---------------------------------------------------------------------------


def test_recorder_event_kinds_match_legacy_ask_for_same_script():
    """Full ``ask()`` (legacy) vs ``run_orchestrator_graph`` (graph), same
    script, same stubbed ``_delegate`` — the (kind, name) sequence for the
    shared portion (agent_run + classify) must match."""
    decision_tc = [_make_tool_call(arguments='{"agent": "sciqa_agent", "reason": "scholarly"}')]

    async def _fake_delegate_factory(calls):
        async def _fake_delegate(agent_name, query):
            calls.append(agent_name)
            return f"DELEGATED:{agent_name}"

        return _fake_delegate

    # Legacy: drive the real ask() with AMA_AGENT_ENGINE unset.
    legacy_mcp = _FakeMcp(_EVIDENCE_JSON)
    legacy_client = _one_call_client(decision_tc)
    legacy = _make_orchestrator(legacy_mcp, legacy_client)
    legacy_calls: list = []
    legacy._delegate = _run(_fake_delegate_factory(legacy_calls))
    legacy_answer = _run(legacy.ask(QUERY))

    graph_mcp = _FakeMcp(_EVIDENCE_JSON)
    graph_client = _one_call_client(decision_tc)
    graph = _make_orchestrator(graph_mcp, graph_client)
    graph_calls: list = []
    graph._delegate = _run(_fake_delegate_factory(graph_calls))

    import os

    old = os.environ.get("AMA_AGENT_ENGINE")
    os.environ["AMA_AGENT_ENGINE"] = "graph"
    try:
        graph_answer = _run(graph.ask(QUERY))
    finally:
        if old is None:
            os.environ.pop("AMA_AGENT_ENGINE", None)
        else:
            os.environ["AMA_AGENT_ENGINE"] = old

    assert legacy_answer == graph_answer == "DELEGATED:sciqa_agent"
    assert legacy_calls == graph_calls == ["sciqa_agent"]

    legacy_kinds = [k for k, _ in _event_seq(legacy)]
    graph_kinds = [k for k, _ in _event_seq(graph)]
    assert legacy_kinds == graph_kinds == ["classify", "agent_run"]
