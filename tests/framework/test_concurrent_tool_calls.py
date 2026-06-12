"""Tests for concurrent execution of batched tool calls in one LLM turn."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

from ama_kbqa.framework.base_agent import BaseKBQAAgent


def _run(coro):
    return asyncio.run(coro)


def _tool_call(id_: str, name: str, arguments: dict | None = None):
    return SimpleNamespace(
        id=id_,
        function=SimpleNamespace(name=name, arguments=json.dumps(arguments or {})),
    )


class ConcurrencyAgent(BaseKBQAAgent):
    """Test double whose tool execution records overlap."""

    def __init__(self, known_tools=("ToolA", "ToolB", "ToolC", "GetJournalSummary")):
        self._messages = []
        self._known_tool_names = set(known_tools)
        self.executed = []
        self.running = 0
        self.max_concurrent = 0

    def get_config(self):
        raise NotImplementedError

    def get_mcp_server_path(self) -> str:
        raise NotImplementedError

    def _trace(self, message: str, color: str = "") -> None:
        pass

    def _detect_loops(self, func_name: str, func_args: dict):
        return False, ""

    async def _execute_single_tool(self, func_name: str, func_args: dict) -> str:
        self.running += 1
        self.max_concurrent = max(self.max_concurrent, self.running)
        await asyncio.sleep(0.01)
        self.running -= 1
        self.executed.append(func_name)
        return f"result:{func_name}"


def test_batched_calls_run_concurrently():
    agent = ConcurrencyAgent()
    calls = [
        _tool_call("c1", "ToolA"),
        _tool_call("c2", "ToolB"),
        _tool_call("c3", "ToolC"),
    ]
    _run(agent._execute_tool_calls(calls))
    assert agent.max_concurrent == 3


def test_single_call_stays_sequential():
    agent = ConcurrencyAgent()
    _run(agent._execute_tool_calls([_tool_call("c1", "ToolA")]))
    assert agent.max_concurrent == 1
    assert agent.executed == ["ToolA"]


def test_results_append_in_emission_order_with_matching_ids():
    agent = ConcurrencyAgent()
    calls = [
        _tool_call("c1", "ToolC"),
        _tool_call("c2", "ToolA"),
        _tool_call("c3", "ToolB"),
    ]
    _run(agent._execute_tool_calls(calls))
    tool_msgs = [m for m in agent._messages if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in tool_msgs] == ["c1", "c2", "c3"]
    assert [m["name"] for m in tool_msgs] == ["ToolC", "ToolA", "ToolB"]
    assert [m["content"] for m in tool_msgs] == [
        "result:ToolC", "result:ToolA", "result:ToolB",
    ]


def test_loop_detected_call_skips_execution_but_others_run():
    class LoopOnB(ConcurrencyAgent):
        def _detect_loops(self, func_name, func_args):
            if func_name == "ToolB":
                return True, "ToolB looping"
            return False, ""

        async def _handle_loop_detected(self, func_name, loop_reason):
            return f"INTERVENTION:{loop_reason}"

    agent = LoopOnB()
    calls = [
        _tool_call("c1", "ToolA"),
        _tool_call("c2", "ToolB"),
        _tool_call("c3", "ToolC"),
    ]
    _run(agent._execute_tool_calls(calls))
    assert sorted(agent.executed) == ["ToolA", "ToolC"]
    tool_msgs = [m for m in agent._messages if m["role"] == "tool"]
    assert tool_msgs[1]["content"] == "INTERVENTION:ToolB looping"
    assert tool_msgs[0]["content"] == "result:ToolA"
    assert tool_msgs[2]["content"] == "result:ToolC"


def test_unknown_tool_short_circuits_but_others_run():
    agent = ConcurrencyAgent(known_tools=("ToolA",))
    calls = [
        _tool_call("c1", "ToolA"),
        _tool_call("c2", "GhostTool"),
    ]
    _run(agent._execute_tool_calls(calls))
    assert agent.executed == ["ToolA"]
    tool_msgs = [m for m in agent._messages if m["role"] == "tool"]
    assert "does not exist" in tool_msgs[1]["content"]


def test_journal_summary_flag_survives_batching():
    agent = ConcurrencyAgent()
    calls = [
        _tool_call("c1", "ToolA"),
        _tool_call("c2", "GetJournalSummary"),
    ]
    called = _run(agent._execute_tool_calls(calls))
    assert called is True
