"""Phase 4 tests: the opt-in graph-engine checkpointer
(``ama_kbqa.graph.pipeline``'s ``_resolve_checkpointer``/``_next_thread_id``,
wired into ``build_pipeline_graph``/``run_pipeline_graph``).

Coverage:

(a) ``checkpointer = "none"`` (the default): ``build_pipeline_graph`` compiles
    with no checkpointer attached, byte-identical to Phase 3.
(b) ``checkpointer = "memory"``: the scripted Phase 3 scenario from
    ``tests/graph/test_pipeline.py`` still runs end to end, AND a checkpoint
    is left behind, retrievable via ``graph.aget_state({"configurable":
    {"thread_id": ...}})``.
(c) two consecutive ``ask()`` calls on the same agent instance (mirroring
    ``soft_reset()`` between benchmark questions) get two different
    ``thread_id``s, each with its own retrievable checkpoint.
"""

from __future__ import annotations

import asyncio

from ama_kbqa.graph.pipeline import (
    _next_thread_id,
    _resolve_checkpointer,
    build_pipeline_graph,
)

from .test_pipeline import QUERY, PipelineAgentDouble, _ToolLoopStubMixin


def _run(coro):
    return asyncio.run(coro)


class _Agent(_ToolLoopStubMixin, PipelineAgentDouble):
    def __init__(self):
        PipelineAgentDouble.__init__(self)
        self.tool_loop_calls = []


# ---------------------------------------------------------------------------
# (a) "none" — no checkpointer attached
# ---------------------------------------------------------------------------


def test_none_checkpointer_resolves_to_none(monkeypatch):
    monkeypatch.delenv("AMA_AGENT_CHECKPOINTER", raising=False)
    agent = _Agent()

    checkpointer = _run(_resolve_checkpointer(agent))

    assert checkpointer is None
    graph = build_pipeline_graph(agent, checkpointer=checkpointer)
    # No checkpointer compiled in: the compiled graph has no checkpointer
    # attribute set to a real saver.
    assert graph.checkpointer is None


def test_none_checkpointer_end_to_end_unaffected(monkeypatch):
    monkeypatch.setenv("AMA_AGENT_ENGINE", "graph")
    monkeypatch.setenv("AMA_AGENT_CHECKPOINTER", "none")
    agent = _Agent()

    answer = _run(agent.ask(QUERY))

    assert answer == f"ANSWER: {QUERY}"
    assert getattr(agent, "_graph_checkpointer", "unset") is None


# ---------------------------------------------------------------------------
# (b) "memory" — scripted scenario still runs, checkpoint retrievable
# ---------------------------------------------------------------------------


def test_memory_checkpointer_runs_scripted_scenario_and_leaves_checkpoint(monkeypatch):
    monkeypatch.setenv("AMA_AGENT_ENGINE", "graph")
    monkeypatch.setenv("AMA_AGENT_CHECKPOINTER", "memory")
    agent = _Agent()

    answer = _run(agent.ask(QUERY))

    assert answer == f"ANSWER: {QUERY}"
    assert len(agent.tool_loop_calls) == 1

    from langgraph.checkpoint.memory import MemorySaver

    checkpointer = agent._graph_checkpointer
    assert isinstance(checkpointer, MemorySaver)

    # thread_id was consumed once (counter starts at 1).
    assert agent._graph_thread_counter == 1
    thread_id = f"{agent.session_id}::1" if hasattr(agent, "session_id") else "default::1"

    graph = build_pipeline_graph(agent, checkpointer=checkpointer)
    state = _run(graph.aget_state({"configurable": {"thread_id": thread_id}}))
    assert state is not None
    assert state.values.get("answer") == f"ANSWER: {QUERY}"


def test_memory_checkpointer_gives_each_question_its_own_thread(monkeypatch):
    """Two consecutive ``ask()`` calls on the same agent instance (mirroring
    two benchmark questions with ``soft_reset()`` in between) get distinct
    thread_ids and distinct retrievable checkpoints, while sharing the same
    ``MemorySaver`` instance."""
    monkeypatch.setenv("AMA_AGENT_ENGINE", "graph")
    monkeypatch.setenv("AMA_AGENT_CHECKPOINTER", "memory")
    agent = _Agent()

    first = _run(agent.ask("Who directed Inception?"))
    second = _run(agent.ask("Who directed Tenet?"))

    assert first == "ANSWER: Who directed Inception?"
    assert second == "ANSWER: Who directed Tenet?"
    assert agent._graph_thread_counter == 2

    checkpointer = agent._graph_checkpointer
    graph = build_pipeline_graph(agent, checkpointer=checkpointer)

    session_id = getattr(agent, "session_id", None) or "default"
    state1 = _run(graph.aget_state({"configurable": {"thread_id": f"{session_id}::1"}}))
    state2 = _run(graph.aget_state({"configurable": {"thread_id": f"{session_id}::2"}}))

    assert state1.values.get("answer") == "ANSWER: Who directed Inception?"
    assert state2.values.get("answer") == "ANSWER: Who directed Tenet?"


# ---------------------------------------------------------------------------
# _next_thread_id
# ---------------------------------------------------------------------------


def test_next_thread_id_increments_and_uses_session_id():
    agent = _Agent()
    agent.session_id = "sess-1"

    assert _next_thread_id(agent) == "sess-1::1"
    assert _next_thread_id(agent) == "sess-1::2"


def test_next_thread_id_defaults_session_id_when_absent():
    agent = _Agent()
    assert not hasattr(agent, "session_id")

    assert _next_thread_id(agent) == "default::1"
