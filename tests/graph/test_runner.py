"""``ama_kbqa.graph.runner.run_tool_loop_graph`` — the bridge back into
``BaseKBQAAgent``. Exercises the two exit paths (max_iterations, final
answer) against a real ``BaseKBQAAgent`` subclass double, with
``build_chat_model`` monkeypatched to a scripted fake (no network, no real
provider config needed).
"""

from __future__ import annotations

import asyncio

from chatkit import TransientRetry
from langchain_core.messages import AIMessage

import ama_kbqa.graph.runner as runner_module
from ama_kbqa.framework.base_agent import BaseKBQAAgent
from ama_kbqa.framework.trace import TraceRecorder

from ._fakes import ScriptedChatModel


def _run(coro):
    return asyncio.run(coro)


class _RunnerTestAgent(BaseKBQAAgent):
    """Bypasses BaseKBQAAgent.__init__'s real client/MCP setup; carries only
    the attributes run_tool_loop_graph and its downstream calls touch."""

    def __init__(self):
        self.name = "runner_test_agent"
        self.model = "test-model"
        self.recorder = TraceRecorder()
        self._retry = TransientRetry()
        self.token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self.tool_call_counts: dict[str, int] = {}
        self.tool_call_durations: list[dict] = []
        self._known_tool_names = {"FindNode"}
        self._messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "Who directed Inception?"},
        ]
        self.synthesis_calls: list[dict] = []

    def get_config(self):  # pragma: no cover - unused
        raise NotImplementedError

    def get_mcp_server_path(self) -> str:  # pragma: no cover - unused
        raise NotImplementedError

    def _trace(self, message: str, color: str = "") -> None:
        pass

    async def _execute_single_tool(self, func_name: str, func_args: dict) -> str:
        self.tool_call_counts[func_name] = self.tool_call_counts.get(func_name, 0) + 1
        return "tool result"

    async def _run_synthesis(self, query: str, qtype: str = "") -> str:
        self.synthesis_calls.append({"query": query, "qtype": qtype})
        return "SYNTHESIZED_SENTINEL"


def test_max_iterations_reaches_synthesis_via_runner(monkeypatch):
    # A model that would keep calling tools forever if the cap didn't stop it.
    responses = [
        AIMessage(content="", tool_calls=[{"name": "FindNode", "args": {}, "id": f"call_{i}", "type": "tool_call"}])
        for i in range(1, 50)
    ]
    model = ScriptedChatModel(responses)
    monkeypatch.setattr(runner_module, "build_chat_model", lambda *a, **k: model)

    agent = _RunnerTestAgent()

    result = _run(
        runner_module.run_tool_loop_graph(
            agent, "Who directed Inception?", tools=[{"type": "function"}],
            max_iterations=2, refresh_interval=5, qtype="Query",
        )
    )

    assert result == "SYNTHESIZED_SENTINEL"
    assert agent.synthesis_calls == [{"query": "Who directed Inception?", "qtype": "Query"}]
    # An "intervention" event was recorded, same kind/name as the legacy path.
    events = agent.recorder.to_dicts()
    assert any(
        e["kind"] == "intervention" and e["name"] == "max_iterations_reached"
        for e in events
    )


def test_final_answer_reaches_synthesis_when_synthesis_enabled(monkeypatch):
    monkeypatch.setattr(runner_module, "get_synthesis_enabled", lambda: True)
    responses = [AIMessage(content="Christopher Nolan.", tool_calls=[])]
    model = ScriptedChatModel(responses)
    monkeypatch.setattr(runner_module, "build_chat_model", lambda *a, **k: model)

    agent = _RunnerTestAgent()

    result = _run(
        runner_module.run_tool_loop_graph(
            agent, "Who directed Inception?", tools=[{"type": "function"}],
            max_iterations=10, refresh_interval=5, qtype="Query",
        )
    )

    assert result == "SYNTHESIZED_SENTINEL"
    assert agent.synthesis_calls == [{"query": "Who directed Inception?", "qtype": "Query"}]


def test_final_answer_bypasses_synthesis_when_disabled(monkeypatch):
    monkeypatch.setattr(runner_module, "get_synthesis_enabled", lambda: False)
    responses = [AIMessage(content="Christopher Nolan.", tool_calls=[])]
    model = ScriptedChatModel(responses)
    monkeypatch.setattr(runner_module, "build_chat_model", lambda *a, **k: model)

    agent = _RunnerTestAgent()

    result = _run(
        runner_module.run_tool_loop_graph(
            agent, "Who directed Inception?", tools=[{"type": "function"}],
            max_iterations=10, refresh_interval=5, qtype="Query",
        )
    )

    assert result == "Christopher Nolan."
    assert agent.synthesis_calls == []
