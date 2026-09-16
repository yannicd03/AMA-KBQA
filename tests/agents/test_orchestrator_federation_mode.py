"""Hermetic unit tests for the per-instance Router/Federated mode switch.

Covers the surface introduced by the federated-dispatch port that is NOT a
rewrite of fed's pre-existing routing/fusion tests (those live in
tests/agents/test_orchestrator_routing.py):

- Orchestrator(federation=...) per-instance override vs. config default.
- Router mode sends dev's exact `select_agent` tool/prompt (the departure
  from fed: this is what the paper's benchmark numbers were measured with,
  and what the LangGraph rewrite's probe/select_agent nodes mirror).
- Federated mode sends `select_agents` capped at max_specialists.
- The `route` span's `route_mode` attribute ("single" | "federated").
- The fusion LLM call goes through Orchestrator._create_with_retry.
- The MCP teardown risk: _run_specialist must close a specialist's MCP
  connection from within the SAME asyncio task that opened it, in both
  single and federated (asyncio.gather) dispatch.

No network calls, no API keys, no MCP subprocess.
"""
import asyncio
from unittest.mock import MagicMock

import pytest

from ama_kbqa.agents.orchestrator_agent.agent import Orchestrator
from ama_kbqa.framework.trace import TraceRecorder


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FakeMcp:
    def __init__(self, tool_result_json):
        self._tool_result = tool_result_json

    async def call_tool(self, name, args):
        return self._tool_result

    async def close(self):
        pass


class _FakeAgent:
    """Minimal sub-agent stub matching the surface _run_specialist touches."""

    def __init__(self, name, answer="", error=None, delay=0.0, close_calls=None):
        self.name = name
        self._answer = answer
        self._error = error
        self._delay = delay
        self.journal_snapshots = []
        self.token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self.recorder = None
        self._parent_span_id_override = None
        self._close_calls = close_calls if close_calls is not None else []

    async def ask(self, query):
        if self._delay:
            await asyncio.sleep(self._delay)
        if self._error is not None:
            raise self._error
        return self._answer

    async def close(self):
        self._close_calls.append(asyncio.current_task())


def _make_orchestrator(mcp=None, client=None):
    """Instantiate Orchestrator without running __init__ (mirrors the
    pattern in test_orchestrator_routing.py)."""
    from chatkit import TransientRetry

    o = Orchestrator.__new__(Orchestrator)
    o.name = "TEST_ORCHESTRATOR"
    o.mcp = mcp
    o.client = client if client is not None else MagicMock()
    o.model = "test-model"
    o.last_routing_reason = None
    o.last_routing_evidence = None
    o.recorder = TraceRecorder()
    o.journal_snapshots = []
    o.token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    o._agents = {}
    o._retry = TransientRetry()
    o._agent_config = {
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


def _patch_init(monkeypatch, federation_enabled, max_specialists=2):
    monkeypatch.setattr(
        "ama_kbqa.agents.orchestrator_agent.agent.get_federation_enabled",
        lambda: federation_enabled,
    )
    monkeypatch.setattr(
        "ama_kbqa.agents.orchestrator_agent.agent.get_federation_max_specialists",
        lambda: max_specialists,
    )
    monkeypatch.setattr(
        "ama_kbqa.agents.orchestrator_agent.agent.assert_provider_api_key_present",
        lambda: None,
    )
    monkeypatch.setattr(
        "ama_kbqa.agents.orchestrator_agent.agent.get_chat_client",
        lambda **kwargs: MagicMock(),
    )
    monkeypatch.setattr(
        "ama_kbqa.agents.orchestrator_agent.agent.get_chat_model_name",
        lambda: "test-model",
    )


# ---------------------------------------------------------------------------
# Constructor override
# ---------------------------------------------------------------------------

class TestFederationConstructorOverride:

    def test_federation_none_follows_config_enabled(self, monkeypatch):
        _patch_init(monkeypatch, federation_enabled=True, max_specialists=3)

        o = Orchestrator(federation=None)

        assert o._federation_enabled is True
        assert o._federation_max_specialists == 3

    def test_federation_none_follows_config_disabled(self, monkeypatch):
        _patch_init(monkeypatch, federation_enabled=False)

        o = Orchestrator(federation=None)

        assert o._federation_enabled is False

    def test_federation_true_overrides_config_false(self, monkeypatch):
        _patch_init(monkeypatch, federation_enabled=False)

        o = Orchestrator(federation=True)

        assert o._federation_enabled is True

    def test_federation_false_overrides_config_true(self, monkeypatch):
        _patch_init(monkeypatch, federation_enabled=True)

        o = Orchestrator(federation=False)

        assert o._federation_enabled is False

    def test_default_call_with_no_federation_arg_follows_config(self, monkeypatch):
        """Orchestrator() with no `federation` kwarg at all must behave
        identically to federation=None (config-driven)."""
        _patch_init(monkeypatch, federation_enabled=True, max_specialists=5)

        o = Orchestrator()

        assert o._federation_enabled is True
        assert o._federation_max_specialists == 5


# ---------------------------------------------------------------------------
# Router vs Federated tool/prompt gating
# ---------------------------------------------------------------------------

class TestRouterVsFederatedToolGating:
    """Router mode (federation off) must send the LLM EXACTLY the tool
    schema and system prompt from demo-v2's pre-existing single-dispatch
    orchestrator (bc9d6e2, before this port): the paper's benchmark numbers
    were measured with this exact contract, and the LangGraph rewrite's
    probe/select_agent nodes mirror it too."""

    _SNAPSHOT_SELECT_AGENT_TOOL = {
        "type": "function",
        "function": {
            "name": "select_agent",
            "description": (
                "Commit to the specialist agent that should answer the "
                "user's question."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "agent": {
                        "type": "string",
                        "enum": ["kqapro_agent", "sciqa_agent"],
                        "description": "The agent to route the question to."
                    },
                    "reason": {
                        "type": "string",
                        "description": "One sentence explaining the routing decision."
                    },
                },
                "required": ["agent", "reason"],
            },
        },
    }

    _SNAPSHOT_ROUTING_PROMPT = (
        "You route user questions to one of these specialist agents:\n"
        "- kqapro_agent: General world knowledge.\n"
        "- sciqa_agent: Scholarly knowledge.\n\n"
        "Alongside the question you receive entity-linking evidence probed "
        "from both knowledge graphs. Strong, on-topic matches in one graph "
        "are a good signal, but judge the question's domain yourself: "
        "generic terms can match spuriously in either graph, so check the "
        "matched labels, not just the scores. If the evidence is weak, "
        "missing, or degraded, decide from the question's domain alone. "
        "Prefer kqapro_agent only when the question is genuinely ambiguous "
        "between the two."
    )

    def test_router_mode_select_agent_tool_schema_is_unchanged_snapshot(self):
        o = _make_orchestrator()
        assert o._federation_enabled is False

        assert o._select_agent_tool() == self._SNAPSHOT_SELECT_AGENT_TOOL

    def test_router_mode_prompt_is_unchanged_snapshot(self):
        o = _make_orchestrator()
        assert o._federation_enabled is False

        assert o._routing_system_prompt() == self._SNAPSHOT_ROUTING_PROMPT

    def test_federated_mode_select_agents_tool_caps_at_max_specialists(self):
        o = _make_orchestrator()
        o._federation_enabled = True
        o._federation_max_specialists = 4

        schema = o._select_agents_tool()["function"]["parameters"]["properties"]["agents"]

        assert schema["maxItems"] == 4
        assert o._select_agents_tool()["function"]["name"] == "select_agents"

    def test_federated_mode_select_agents_tool_caps_at_least_one(self):
        """A misconfigured max_specialists <= 0 must still allow one agent."""
        o = _make_orchestrator()
        o._federation_enabled = True
        o._federation_max_specialists = 0

        schema = o._select_agents_tool()["function"]["parameters"]["properties"]["agents"]

        assert schema["maxItems"] == 1


# ---------------------------------------------------------------------------
# route span: route_mode
# ---------------------------------------------------------------------------

class TestRouteSpanMode:

    def test_route_span_reports_single_mode(self, monkeypatch):
        o = _make_orchestrator(mcp=None, client=MagicMock())

        async def _fake_init_mcp():
            o.mcp = _FakeMcp("{}")

        async def _fake_route(query):
            o.last_routing_reason = "world knowledge"
            return ["kqapro_agent"]

        async def _fake_delegate(name, query, cancel_token=None):
            return "answer"

        monkeypatch.setattr(o, "_init_mcp", _fake_init_mcp)
        monkeypatch.setattr(o, "_route_autonomously", _fake_route)
        monkeypatch.setattr(o, "_delegate", _fake_delegate)

        _run(o.ask("Q?"))

        route_spans = [e for e in o.recorder.events if e.kind == "classify" and e.name == "route"]
        assert len(route_spans) == 1
        assert route_spans[0].attributes["selected_agent"] == "kqapro_agent"
        assert route_spans[0].attributes["route_mode"] == "single"

    def test_route_span_reports_federated_mode(self, monkeypatch):
        o = _make_orchestrator(mcp=None, client=MagicMock())

        async def _fake_init_mcp():
            o.mcp = _FakeMcp("{}")

        async def _fake_route(query):
            o.last_routing_reason = "spans both"
            return ["kqapro_agent", "sciqa_agent"]

        async def _fake_federate(names, query, cancel_token=None):
            return "fused answer"

        monkeypatch.setattr(o, "_init_mcp", _fake_init_mcp)
        monkeypatch.setattr(o, "_route_autonomously", _fake_route)
        monkeypatch.setattr(o, "_federate", _fake_federate)

        result = _run(o.ask("Q?"))

        assert result == "fused answer"
        route_spans = [e for e in o.recorder.events if e.kind == "classify" and e.name == "route"]
        assert len(route_spans) == 1
        assert route_spans[0].attributes["selected_agent"] == "kqapro_agent, sciqa_agent"
        assert route_spans[0].attributes["route_mode"] == "federated"

    def test_route_span_reports_single_mode_on_routing_failure(self, monkeypatch):
        o = _make_orchestrator(mcp=None, client=MagicMock())

        async def _fake_init_mcp():
            o.mcp = _FakeMcp("{}")

        async def _fake_route(query):
            return None

        async def _fake_fallback(query, cancel_token=None):
            return "fallback answer"

        monkeypatch.setattr(o, "_init_mcp", _fake_init_mcp)
        monkeypatch.setattr(o, "_route_autonomously", _fake_route)
        monkeypatch.setattr(o, "_fallback_kqapro", _fake_fallback)

        result = _run(o.ask("Q?"))

        assert result == "fallback answer"
        route_spans = [e for e in o.recorder.events if e.kind == "classify" and e.name == "route"]
        assert route_spans[0].attributes["selected_agent"] == "<none>"
        assert route_spans[0].attributes["route_mode"] == "single"


# ---------------------------------------------------------------------------
# Fusion goes through the shared retry
# ---------------------------------------------------------------------------

class TestFusionUsesCreateWithRetry:

    def test_fusion_retries_transient_error_then_succeeds(self, monkeypatch):
        from types import SimpleNamespace

        waits: list[float] = []
        monkeypatch.setattr("time.sleep", lambda s: waits.append(s))

        calls = {"n": 0}

        def _create(**kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise Exception("Open WebUI: Server Connection Error")
            message = SimpleNamespace(tool_calls=None, content="fused answer")
            return SimpleNamespace(choices=[SimpleNamespace(message=message)])

        client = MagicMock()
        client.chat.completions.create.side_effect = _create
        o = _make_orchestrator(mcp=None, client=client)

        answer = _run(o._fuse_answers("Q?", [
            {"agent": "kqapro_agent", "answer": "a"},
            {"agent": "sciqa_agent", "answer": "b"},
        ]))

        assert answer == "fused answer"
        assert calls["n"] == 2
        assert waits == [2.0]
        assert o._retry.level == 0


# ---------------------------------------------------------------------------
# MCP teardown risk: close() awaited from the specialist's own task
# ---------------------------------------------------------------------------

class TestMcpCloseInOwnTask:
    """Running several MCP-backed sub-agents in one event loop can trip
    anyio's "Attempted to exit cancel scope in a different task" bug in the
    mcp stdio_client teardown path if a connection is closed (or garbage
    collected) from a different task than the one that opened it.
    _run_specialist must call agent.close() from within its own
    coroutine/task, immediately after agent.ask() returns, in both single
    and federated dispatch."""

    def test_single_dispatch_closes_in_the_calling_task(self, monkeypatch):
        o = _make_orchestrator(mcp=None, client=MagicMock())
        close_calls = []
        agent = _FakeAgent("kqapro_agent", answer="ok", close_calls=close_calls)
        monkeypatch.setattr(o, "_load_agent", lambda name: agent)

        async def _runner():
            this_task = asyncio.current_task()
            result = await o._run_specialist("kqapro_agent", "Q?")
            return result, this_task

        result, caller_task = _run(_runner())

        assert result["answer"] == "ok"
        assert len(close_calls) == 1
        assert close_calls[0] is caller_task

    def test_federated_dispatch_closes_each_specialist_in_its_own_task(self):
        """asyncio.gather schedules one Task per specialist; each specialist
        must close its own MCP connection from within its own Task, not a
        shared one and not the gathering coroutine's task."""
        o = _make_orchestrator(mcp=None, client=MagicMock())

        def _fusion_create(**kwargs):
            from types import SimpleNamespace
            message = SimpleNamespace(tool_calls=None, content="fused")
            return SimpleNamespace(choices=[SimpleNamespace(message=message)])

        o.client.chat.completions.create.side_effect = _fusion_create

        kqapro_closes = []
        sciqa_closes = []
        o._agents = {
            "kqapro_agent": _FakeAgent("kqapro_agent", answer="a", delay=0.01, close_calls=kqapro_closes),
            "sciqa_agent": _FakeAgent("sciqa_agent", answer="b", delay=0.01, close_calls=sciqa_closes),
        }

        _run(o._federate(["kqapro_agent", "sciqa_agent"], "Q?"))

        assert len(kqapro_closes) == 1
        assert len(sciqa_closes) == 1
        assert kqapro_closes[0] is not sciqa_closes[0]

    def test_close_awaited_even_when_specialist_raises(self, monkeypatch):
        o = _make_orchestrator(mcp=None, client=MagicMock())
        close_calls = []
        agent = _FakeAgent("kqapro_agent", error=RuntimeError("boom"), close_calls=close_calls)
        monkeypatch.setattr(o, "_load_agent", lambda name: agent)

        with pytest.raises(RuntimeError):
            _run(o._run_specialist("kqapro_agent", "Q?"))

        assert len(close_calls) == 1
