"""Drives ``ama_kbqa.graph.builder.build_graph`` with a scripted fake chat
model and a fake tool executor — no real LLM provider, no real MCP server.

Covers the Phase 1 core-loop contract: two tool rounds then a final answer,
the tool_choice schedule ("required" for iteration <= 3, "auto" after),
malformed-JSON tool args producing the legacy error ToolMessage text,
concurrent execution of independent tool calls in one turn, and the
max_iterations cap setting ``exit_reason``.
"""

from __future__ import annotations

import asyncio

from chatkit import TransientRetry
from langchain_core.messages import AIMessage, ToolMessage

from ama_kbqa.framework.trace import TraceRecorder
from ama_kbqa.graph.builder import build_graph

from ._fakes import ScriptedChatModel


class FakeAgent:
    """Minimal stand-in for the pieces of ``BaseKBQAAgent`` the graph nodes
    call back into: tracing, retry, token accounting, and tool execution."""

    def __init__(self, known_tools=("FindNode", "GetRelationDetails", "ToolA", "ToolB", "ToolC")):
        self.recorder = TraceRecorder()
        self.model = "test-model"
        self._retry = TransientRetry()
        self.token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self.tool_call_counts: dict[str, int] = {}
        self.tool_call_durations: list[dict] = []
        self._known_tool_names = set(known_tools)
        self.executed: list[tuple[str, dict]] = []
        self.tool_result_fn = lambda name, args: f"result:{name}"

    def _trace(self, message: str, color: str = "") -> None:
        pass

    async def _execute_single_tool(self, func_name: str, func_args: dict) -> str:
        self.tool_call_counts[func_name] = self.tool_call_counts.get(func_name, 0) + 1
        self.executed.append((func_name, func_args))
        result = self.tool_result_fn(func_name, func_args)
        if asyncio.iscoroutine(result):
            result = await result
        return result


def _run(coro):
    return asyncio.run(coro)


def _initial_state():
    return {
        "messages": [],
        "iteration": 0,
        "total_tool_calls": 0,
        "exit_reason": None,
        "tool_call_history": [],
    }


def test_two_tool_rounds_then_final_answer():
    responses = [
        AIMessage(
            content="",
            tool_calls=[
                {"name": "FindNode", "args": {"semantic_node_name": "Inception"}, "id": "call_1", "type": "tool_call"}
            ],
        ),
        AIMessage(
            content="",
            tool_calls=[
                {"name": "GetRelationDetails", "args": {"base_node_id": "Q1"}, "id": "call_2", "type": "tool_call"}
            ],
        ),
        AIMessage(content="Christopher Nolan directed Inception.", tool_calls=[]),
    ]
    model = ScriptedChatModel(responses)
    agent = FakeAgent()
    graph = build_graph(agent, model, "test-provider", tools=[{"type": "function"}], max_iterations=10)

    final_state = _run(graph.ainvoke(_initial_state()))

    assert final_state["exit_reason"] == "final_answer"
    assert agent.executed == [
        ("FindNode", {"semantic_node_name": "Inception"}),
        ("GetRelationDetails", {"base_node_id": "Q1"}),
    ]
    tool_messages = [m for m in final_state["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 2
    ai_messages = [m for m in final_state["messages"] if isinstance(m, AIMessage)]
    assert ai_messages[-1].content == "Christopher Nolan directed Inception."


def test_tool_choice_required_through_iteration_three_then_auto():
    # 4 tool-call rounds followed by a final answer: iterations 1-3 must see
    # tool_choice="required", iteration 4 (and the final, 5th, call) "auto".
    responses = [
        AIMessage(content="", tool_calls=[{"name": "ToolA", "args": {}, "id": f"call_{i}", "type": "tool_call"}])
        for i in range(1, 5)
    ] + [AIMessage(content="done", tool_calls=[])]
    model = ScriptedChatModel(responses)
    agent = FakeAgent()
    graph = build_graph(agent, model, "test-provider", tools=[{"type": "function"}], max_iterations=10)

    _run(graph.ainvoke(_initial_state()))

    assert model.tool_choice_calls == ["required", "required", "required", "auto", "auto"]


def test_malformed_json_args_produce_same_error_message_as_legacy():
    responses = [
        AIMessage(
            content="",
            invalid_tool_calls=[
                {
                    "name": "FindNode",
                    "args": "{name: 'Inception'",  # not valid JSON
                    "id": "call_bad",
                    "error": None,
                    "type": "invalid_tool_call",
                }
            ],
        ),
        AIMessage(content="done", tool_calls=[]),
    ]
    model = ScriptedChatModel(responses)
    agent = FakeAgent()
    graph = build_graph(agent, model, "test-provider", tools=[{"type": "function"}], max_iterations=10)

    final_state = _run(graph.ainvoke(_initial_state()))

    # The malformed call must NOT have reached _execute_single_tool.
    assert agent.executed == []
    tool_messages = [m for m in final_state["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1
    assert "could not parse arguments as JSON" in tool_messages[0].content
    assert tool_messages[0].tool_call_id == "call_bad"


def test_concurrent_tool_calls_execute_in_parallel():
    running = {"n": 0, "max": 0}

    async def _slow_tool_result(name, args):
        running["n"] += 1
        running["max"] = max(running["max"], running["n"])
        await asyncio.sleep(0.01)
        running["n"] -= 1
        return f"result:{name}"

    responses = [
        AIMessage(
            content="",
            tool_calls=[
                {"name": "ToolA", "args": {}, "id": "call_1", "type": "tool_call"},
                {"name": "ToolB", "args": {}, "id": "call_2", "type": "tool_call"},
                {"name": "ToolC", "args": {}, "id": "call_3", "type": "tool_call"},
            ],
        ),
        AIMessage(content="done", tool_calls=[]),
    ]
    model = ScriptedChatModel(responses)
    agent = FakeAgent()
    agent.tool_result_fn = _slow_tool_result
    graph = build_graph(agent, model, "test-provider", tools=[{"type": "function"}], max_iterations=10)

    final_state = _run(graph.ainvoke(_initial_state()))

    assert running["max"] == 3
    tool_messages = [m for m in final_state["messages"] if isinstance(m, ToolMessage)]
    # Results append in emission order with matching tool_call_ids.
    assert [m.tool_call_id for m in tool_messages] == ["call_1", "call_2", "call_3"]
    assert [m.content for m in tool_messages] == ["result:ToolA", "result:ToolB", "result:ToolC"]


def test_unknown_tool_short_circuits_but_does_not_call_execute_single_tool():
    responses = [
        AIMessage(
            content="",
            tool_calls=[{"name": "GhostTool", "args": {}, "id": "call_1", "type": "tool_call"}],
        ),
        AIMessage(content="done", tool_calls=[]),
    ]
    model = ScriptedChatModel(responses)
    agent = FakeAgent(known_tools=("FindNode",))
    graph = build_graph(agent, model, "test-provider", tools=[{"type": "function"}], max_iterations=10)

    final_state = _run(graph.ainvoke(_initial_state()))

    assert agent.executed == []
    tool_messages = [m for m in final_state["messages"] if isinstance(m, ToolMessage)]
    assert "does not exist" in tool_messages[0].content


def test_max_iterations_sets_exit_reason_without_calling_model():
    # A model that would always keep calling tools forever if allowed.
    responses = [
        AIMessage(content="", tool_calls=[{"name": "ToolA", "args": {}, "id": f"call_{i}", "type": "tool_call"}])
        for i in range(1, 50)
    ]
    model = ScriptedChatModel(responses)
    agent = FakeAgent()
    graph = build_graph(agent, model, "test-provider", tools=[{"type": "function"}], max_iterations=2)

    final_state = _run(graph.ainvoke(_initial_state(), {"recursion_limit": 50}))

    assert final_state["exit_reason"] == "max_iterations"
    assert final_state["iteration"] == 3  # checked BEFORE the 3rd LLM call
    # Only 2 tool-call rounds actually ran before the cap tripped.
    assert len(agent.executed) == 2


def test_max_iterations_zero_exits_without_any_llm_call():
    model = ScriptedChatModel([])  # would raise IndexError if ever invoked
    agent = FakeAgent()
    graph = build_graph(agent, model, "test-provider", tools=[{"type": "function"}], max_iterations=0)

    final_state = _run(graph.ainvoke(_initial_state()))

    assert final_state["exit_reason"] == "max_iterations"
    assert final_state["messages"] == []
