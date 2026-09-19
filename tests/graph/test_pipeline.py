"""Phase 3 tests: the outer per-question pipeline graph
(``ama_kbqa.graph.pipeline``), driven with a scripted ``PipelineAgentDouble``
(a ``GraphAgentDouble`` extended with the MCP/config/classification surface
``_ask_impl``'s body touches before it ever reaches the tool loop).

Scenario coverage (see the Phase 3 task description):

(a) fresh question -> classify -> assemble -> tool loop -> finalize; the
    ``_messages`` handed to the tool loop must byte-match what legacy
    ``_ask_impl`` builds for the same inputs — verified by running the SAME
    scripted inputs through a `_HookAgent`-style legacy double (stubbing
    ``_run_tool_loop`` to capture ``self._messages`` instead of running it)
    and diffing.
(b) follow-up turn skips classification and tool filtering.
(c) fast-path success skips the tool loop entirely.
(d) fast-path failure falls through to the tool loop.
(e) an exception raised inside the tool loop propagates out of
    ``run_pipeline_graph`` UNCHANGED (mirrors ``_ask_impl``'s own
    ``except Exception: trace(...); raise`` — it does NOT swallow into a
    string, see ``ama_kbqa.graph.pipeline``'s module docstring), while
    ``_finalize_question`` still ran and the ``agent_run`` span still closed
    (with ``status="error"``) — both invariants hold via ``ask()``'s span
    context manager and ``run_pipeline_graph``'s ``finally``, not by
    swallowing the exception into an answer string.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

from ama_kbqa.framework.base_agent import BaseKBQAAgent

from ._fakes import GraphAgentDouble

QUERY = "Who directed Inception?"


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Shared fakes
# ---------------------------------------------------------------------------


class _FakeMcpTool:
    def __init__(self, name: str):
        self.name = name


class _FakeMcp:
    """Minimal MCP double: lists a fixed tool set, converts to OpenAI shape
    the same way ``ama_kbqa.framework.mcp_client.MCPClient`` does."""

    def __init__(self, tool_names=("FindNode", "GetAttributeDetails", "GetRelationDetails")):
        self._tool_names = list(tool_names)

    async def list_tools(self):
        return [_FakeMcpTool(n) for n in self._tool_names]

    def convert_tools_to_openai_format(self, tools):
        return [
            {"type": "function", "function": {"name": t.name, "description": "", "parameters": {}}}
            for t in tools
        ]

    async def call_tool(self, name, args=None):
        return ""


class PipelineAgentDouble(GraphAgentDouble):
    """``GraphAgentDouble`` extended with the MCP/config/classification
    surface ``ama_kbqa.graph.pipeline`` nodes call — ``prepare``, ``classify``,
    ``assemble_prompt``, ``fast_path``. Subclasses/tests override the
    instrumented hooks to script a scenario the same way ``_HookAgent``
    (``tests/framework/test_base_agent_multiturn.py``) does for the legacy
    path.
    """

    def __init__(self, known_tools=("FindNode", "GetAttributeDetails", "GetRelationDetails")):
        super().__init__(known_tools=known_tools)
        self._messages = [{"role": "system", "content": "sys"}]
        self.mcp = _FakeMcp(known_tools)
        self._domain_settings: Dict[str, Any] = {}

        # Instrumentation
        self.classify_calls = 0
        self.filter_calls = 0
        self.static_context_calls = 0
        self.question_context_calls = 0
        self.finalize_question_calls = 0

        self._classify_result = {
            "question_type": "QueryName",
            "entities": [],
            "relations": [],
            "fewshot_examples": "",
        }

    async def _init_mcp(self) -> None:
        pass  # self.mcp is already set

    def get_config(self):
        return SimpleNamespace(domain_settings=self._domain_settings)

    def get_mcp_server_path(self) -> str:  # pragma: no cover - unused
        raise NotImplementedError

    async def _finalize_question(self) -> None:
        self.finalize_question_calls += 1

    def _classify_question(self, query):
        self.classify_calls += 1
        return dict(self._classify_result)

    def _build_static_qtype_context(self, qtype, fewshot_examples=""):
        self.static_context_calls += 1
        return "STATIC_QTYPE_CONTEXT"

    def _build_question_context(self, qtype, entities, relations, query=""):
        self.question_context_calls += 1
        return "QUESTION_CONTEXT"

    def _get_allowed_tools_for_qtype(self, qtype):
        self.filter_calls += 1
        return None

    def _get_denied_tool_names(self):
        return set()


class _ToolLoopStubMixin:
    """Mixin: stub the Phase 1/2 tool loop out so pipeline tests don't need a
    real LLM. Records the ``(query, tools, qtype)`` it was called with and the
    ``self._messages`` snapshot at call time (what the loop would have seen)."""

    tool_loop_calls: List[Dict[str, Any]]

    async def _run_tool_loop(self, query, openai_tools, max_iterations, refresh_interval, qtype="", cancel_token=None):
        self.tool_loop_calls.append(
            {
                "query": query,
                "tools": [t["function"]["name"] for t in openai_tools],
                "qtype": qtype,
                "messages_snapshot": [dict(m) for m in self._messages],
            }
        )
        return f"ANSWER: {query}"


# ---------------------------------------------------------------------------
# Legacy comparison double (for the message-shape parity assertion)
# ---------------------------------------------------------------------------


class _LegacyHookAgent(BaseKBQAAgent):
    """Legacy-engine counterpart to ``PipelineAgentDouble``: drives the real
    ``_ask_impl`` body (engine="legacy") with the same scripted classify/
    context-builder hooks, capturing ``self._messages`` right before the tool
    loop would run (mirrors ``tests/framework/test_base_agent_multiturn.py::_HookAgent``).
    """

    def __init__(self, known_tools=("FindNode", "GetAttributeDetails", "GetRelationDetails")):
        self.name = "legacy_hook_agent"
        self.model = "test-model"
        self._parent_span_id_override = None
        from ama_kbqa.framework.trace import TraceRecorder

        self.recorder = TraceRecorder()
        self.mcp = _FakeMcp(known_tools)
        self._text_tool_call_mode = False
        self._messages = [{"role": "system", "content": "sys"}]
        self.tool_loop_calls: List[Dict[str, Any]] = []
        self._classify_result = {
            "question_type": "QueryName",
            "entities": [],
            "relations": [],
            "fewshot_examples": "",
        }

    async def _init_mcp(self) -> None:
        pass

    def get_config(self):
        return SimpleNamespace(domain_settings={})

    def get_mcp_server_path(self) -> str:  # pragma: no cover - unused
        raise NotImplementedError

    def _trace(self, message: str, color: str = "") -> None:
        pass

    async def _finalize_question(self) -> None:
        pass

    def _classify_question(self, query):
        return dict(self._classify_result)

    def _build_static_qtype_context(self, qtype, fewshot_examples=""):
        return "STATIC_QTYPE_CONTEXT"

    def _build_question_context(self, qtype, entities, relations, query=""):
        return "QUESTION_CONTEXT"

    def _get_allowed_tools_for_qtype(self, qtype):
        return None

    async def _run_tool_loop(self, query, openai_tools, max_iterations, refresh_interval, qtype="", cancel_token=None):
        self.tool_loop_calls.append(
            {
                "query": query,
                "tools": [t["function"]["name"] for t in openai_tools],
                "qtype": qtype,
                "messages_snapshot": [dict(m) for m in self._messages],
            }
        )
        return f"ANSWER: {query}"


# ---------------------------------------------------------------------------
# (a) fresh question: classify -> assemble -> tool loop; message-shape parity
# ---------------------------------------------------------------------------


def test_fresh_question_runs_classify_assemble_then_tool_loop(monkeypatch):
    monkeypatch.setenv("AMA_AGENT_ENGINE", "graph")

    class Agent(_ToolLoopStubMixin, PipelineAgentDouble):
        def __init__(self):
            PipelineAgentDouble.__init__(self)
            self.tool_loop_calls = []

    agent = Agent()
    answer = _run(agent.ask(QUERY))

    assert answer == f"ANSWER: {QUERY}"
    assert agent.classify_calls == 1
    assert agent.filter_calls == 1
    assert agent.static_context_calls == 1
    assert agent.question_context_calls == 1
    assert agent.finalize_question_calls == 1
    assert len(agent.tool_loop_calls) == 1
    call = agent.tool_loop_calls[0]
    assert call["qtype"] == "QueryName"
    contents = [m.get("content") for m in call["messages_snapshot"]]
    assert "STATIC_QTYPE_CONTEXT" in contents
    assert QUERY in contents
    assert "QUESTION_CONTEXT" in contents
    # Prefix-cache ordering: static block, then raw query, then per-question context.
    assert contents.index("STATIC_QTYPE_CONTEXT") < contents.index(QUERY) < contents.index(
        "QUESTION_CONTEXT"
    )

    # agent_run span recorded and closed ok.
    events = agent.recorder.to_dicts()
    assert any(e["kind"] == "agent_run" and e["status"] == "ok" for e in events)


def test_message_shape_entering_tool_loop_matches_legacy(monkeypatch):
    """The graph pipeline's ``_messages`` snapshot right before the tool loop
    must equal what legacy ``_ask_impl`` builds for the same inputs."""

    class GraphAgent(_ToolLoopStubMixin, PipelineAgentDouble):
        def __init__(self):
            PipelineAgentDouble.__init__(self)
            self.tool_loop_calls = []

    monkeypatch.setenv("AMA_AGENT_ENGINE", "graph")
    graph_agent = GraphAgent()
    _run(graph_agent.ask(QUERY))

    monkeypatch.delenv("AMA_AGENT_ENGINE", raising=False)
    legacy_agent = _LegacyHookAgent()
    _run(legacy_agent.ask(QUERY))

    graph_call = graph_agent.tool_loop_calls[0]
    legacy_call = legacy_agent.tool_loop_calls[0]

    assert graph_call["messages_snapshot"] == legacy_call["messages_snapshot"]
    assert graph_call["tools"] == legacy_call["tools"]
    assert graph_call["qtype"] == legacy_call["qtype"]
    assert graph_call["query"] == legacy_call["query"]


# ---------------------------------------------------------------------------
# (b) follow-up turn skips classify + tool filtering
# ---------------------------------------------------------------------------


def test_followup_turn_skips_classify_and_tool_filtering(monkeypatch):
    monkeypatch.setenv("AMA_AGENT_ENGINE", "graph")

    class Agent(_ToolLoopStubMixin, PipelineAgentDouble):
        def __init__(self):
            PipelineAgentDouble.__init__(self)
            self.tool_loop_calls = []

    agent = Agent()
    # Preserved stack from a completed prior turn (as reset(keep_history=True)
    # would leave it).
    agent._messages += [
        {"role": "user", "content": "Who directed Inception?"},
        {"role": "user", "content": "STATIC_QTYPE_CONTEXT"},
        {"role": "user", "content": "QUESTION_CONTEXT"},
        {"role": "assistant", "content": "Christopher Nolan."},
    ]

    answer = _run(agent.ask("Where was he born?"))

    assert answer == "ANSWER: Where was he born?"
    assert agent.classify_calls == 0
    assert agent.filter_calls == 0
    assert agent.static_context_calls == 0
    assert agent.question_context_calls == 0

    call = agent.tool_loop_calls[0]
    assert call["qtype"] == "Query"
    # Follow-up keeps the FULL unfiltered tool set (no qtype filter, no denylist).
    assert set(call["tools"]) == {"FindNode", "GetAttributeDetails", "GetRelationDetails"}
    assert call["messages_snapshot"][-1] == {"role": "user", "content": "Where was he born?"}


# ---------------------------------------------------------------------------
# (c) fast-path success skips the tool loop
# ---------------------------------------------------------------------------


def test_fast_path_success_skips_tool_loop(monkeypatch):
    monkeypatch.setenv("AMA_AGENT_ENGINE", "graph")

    class Agent(_ToolLoopStubMixin, PipelineAgentDouble):
        def __init__(self):
            PipelineAgentDouble.__init__(self)
            self.tool_loop_calls = []
            self._classify_result = {
                "question_type": "QueryAttr",
                "entities": ["Inception"],
                "relations": ["release date"],
                "fewshot_examples": "",
            }

        async def _try_fast_path(self, query, qtype, entities, relations):
            return "2010-07-16"

    agent = Agent()
    answer = _run(agent.ask(QUERY))

    assert answer == "2010-07-16"
    assert agent.tool_loop_calls == []
    assert agent._messages[-1] == {"role": "assistant", "content": "2010-07-16"}
    events = agent.recorder.to_dicts()
    fast_path_events = [e for e in events if e["kind"] == "fast_path"]
    assert len(fast_path_events) == 1
    assert fast_path_events[0]["attributes"]["succeeded"] is True


# ---------------------------------------------------------------------------
# (d) fast-path failure falls through to the tool loop
# ---------------------------------------------------------------------------


def test_fast_path_failure_falls_through_to_tool_loop(monkeypatch):
    monkeypatch.setenv("AMA_AGENT_ENGINE", "graph")

    class Agent(_ToolLoopStubMixin, PipelineAgentDouble):
        def __init__(self):
            PipelineAgentDouble.__init__(self)
            self.tool_loop_calls = []
            self._classify_result = {
                "question_type": "QueryAttr",
                "entities": ["Inception"],
                "relations": ["release date"],
                "fewshot_examples": "",
            }

        async def _try_fast_path(self, query, qtype, entities, relations):
            return None

    agent = Agent()
    answer = _run(agent.ask(QUERY))

    assert answer == f"ANSWER: {QUERY}"
    assert len(agent.tool_loop_calls) == 1
    events = agent.recorder.to_dicts()
    fast_path_events = [e for e in events if e["kind"] == "fast_path"]
    assert len(fast_path_events) == 1
    assert fast_path_events[0]["attributes"]["succeeded"] is False
    # Tool filtering ran even though fast path was attempted-and-failed.
    assert agent.filter_calls == 1


def test_fast_path_ineligible_qtype_falls_through_to_tool_loop(monkeypatch):
    """qtype outside the fast-path set never even attempts _try_fast_path."""
    monkeypatch.setenv("AMA_AGENT_ENGINE", "graph")

    class Agent(_ToolLoopStubMixin, PipelineAgentDouble):
        def __init__(self):
            PipelineAgentDouble.__init__(self)
            self.tool_loop_calls = []
            self.fast_path_attempted = False
            self._classify_result = {
                "question_type": "Count",
                "entities": ["Inception"],
                "relations": [],
                "fewshot_examples": "",
            }

        async def _try_fast_path(self, query, qtype, entities, relations):
            self.fast_path_attempted = True
            return "should not be called"

    agent = Agent()
    answer = _run(agent.ask(QUERY))

    assert answer == f"ANSWER: {QUERY}"
    assert agent.fast_path_attempted is False
    assert len(agent.tool_loop_calls) == 1
    events = agent.recorder.to_dicts()
    assert not [e for e in events if e["kind"] == "fast_path"]


# ---------------------------------------------------------------------------
# (e) exception inside the tool loop
# ---------------------------------------------------------------------------


def test_exception_inside_tool_loop_propagates_but_still_finalizes(monkeypatch):
    """Mirrors _ask_impl's own `except Exception: trace(...); raise` — the
    graph engine does not swallow tool-loop exceptions into a string answer,
    matching legacy exactly. `_finalize_question` still runs (the `finally`
    in `run_pipeline_graph`) and the `agent_run` span still closes with
    status="error" (TraceRecorder.span's own finally)."""
    monkeypatch.setenv("AMA_AGENT_ENGINE", "graph")

    class Agent(PipelineAgentDouble):
        async def _run_tool_loop(self, query, openai_tools, max_iterations, refresh_interval, qtype="", cancel_token=None):
            raise RuntimeError("boom")

    agent = Agent()

    with pytest.raises(RuntimeError, match="boom"):
        _run(agent.ask(QUERY))

    assert agent.finalize_question_calls == 1
    events = agent.recorder.to_dicts()
    agent_run_events = [e for e in events if e["kind"] == "agent_run"]
    assert len(agent_run_events) == 1
    assert agent_run_events[0]["status"] == "error"
    assert "boom" in (agent_run_events[0]["error"] or "")


# ---------------------------------------------------------------------------
# Cancellation: the graph pipeline has no mid-run checkpoints yet, but a token
# that is already cancelled must stop the run before anything is dispatched.
# ---------------------------------------------------------------------------

def test_cancelled_token_is_honoured_before_graph_pipeline_dispatch(monkeypatch):
    from ama_kbqa.framework.cancellation import CancellationToken, cancelled_answer

    class GraphAgent(_ToolLoopStubMixin, PipelineAgentDouble):
        def __init__(self):
            PipelineAgentDouble.__init__(self)
            self.tool_loop_calls = []

    async def _must_not_run(*_args, **_kwargs):
        raise AssertionError("run_pipeline_graph must not be reached")

    monkeypatch.setenv("AMA_AGENT_ENGINE", "graph")
    monkeypatch.setattr("ama_kbqa.graph.pipeline.run_pipeline_graph", _must_not_run)

    token = CancellationToken()
    token.cancel("user pressed stop")
    agent = GraphAgent()

    assert _run(agent.ask(QUERY, cancel_token=token)) == cancelled_answer(token)
    assert agent.tool_loop_calls == []
