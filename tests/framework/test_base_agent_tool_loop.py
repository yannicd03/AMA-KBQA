"""Tests for shared agent tool-loop behavior."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

from ama_kbqa.framework.base_agent import BaseKBQAAgent
from ama_kbqa.framework.trace import TraceRecorder


def _run(coro):
    return asyncio.run(coro)


def _tool_call(name: str, arguments: dict | None = None):
    return SimpleNamespace(
        id="call-1",
        function=SimpleNamespace(
            name=name,
            arguments=json.dumps(arguments or {}),
        ),
    )


class DummyAgent(BaseKBQAAgent):
    """Minimal BaseKBQAAgent test double that bypasses production client setup."""

    def __init__(self):
        self._messages = []
        self._known_tool_names = {"GetJournalSummary"}
        self.executed_calls = []

    def get_config(self):
        raise NotImplementedError

    def get_mcp_server_path(self) -> str:
        raise NotImplementedError

    def _trace(self, message: str, color: str = "") -> None:
        pass

    def _detect_loops(self, func_name: str, func_args: dict):
        return False, ""

    async def _execute_single_tool(self, func_name: str, func_args: dict) -> str:
        self.executed_calls.append((func_name, func_args))
        return "journal summary"


def test_get_journal_summary_without_args_triggers_answer_prompt_flag():
    agent = DummyAgent()

    called = _run(agent._execute_tool_calls([_tool_call("GetJournalSummary")]))

    assert called is True
    assert agent.executed_calls == [("GetJournalSummary", {})]


def test_get_journal_summary_with_args_does_not_trigger_answer_prompt_flag():
    agent = DummyAgent()

    called = _run(agent._execute_tool_calls([
        _tool_call("GetJournalSummary", {"action": "read"})
    ]))

    assert called is False
    assert agent.executed_calls == [("GetJournalSummary", {"action": "read"})]


def _malformed_tool_call(name: str, raw_arguments: str):
    """A tool call whose function.arguments is not valid JSON."""
    return SimpleNamespace(
        id="call-bad",
        function=SimpleNamespace(name=name, arguments=raw_arguments),
    )


class RecordingDummyAgent(DummyAgent):
    """DummyAgent variant with a broader known-tool set so both a malformed
    and a well-formed call can be exercised in the same batch."""

    def __init__(self):
        super().__init__()
        self._known_tool_names = {"GetJournalSummary", "FindNode"}


def test_malformed_tool_arguments_are_not_executed_and_yield_parse_error_message():
    agent = RecordingDummyAgent()

    _run(agent._execute_tool_calls([
        _malformed_tool_call("FindNode", "{name: 'Inception'")  # invalid JSON
    ]))

    # The tool must NOT have been invoked with a silently-defaulted {} arg set.
    assert agent.executed_calls == []

    # A tool-result message carrying the parse-error text was appended instead.
    assert len(agent._messages) == 1
    result_message = agent._messages[0]
    assert result_message["role"] == "tool"
    assert result_message["name"] == "FindNode"
    assert "could not parse arguments as JSON" in result_message["content"]


def test_well_formed_call_still_executes_alongside_a_malformed_one():
    agent = RecordingDummyAgent()

    _run(agent._execute_tool_calls([
        _malformed_tool_call("FindNode", "not json at all"),
        _tool_call("GetJournalSummary"),
    ]))

    # Only the well-formed call actually executed.
    assert agent.executed_calls == [("GetJournalSummary", {})]

    # Both calls produced a tool-result message (error for the bad one, real
    # result for the good one), in order.
    assert len(agent._messages) == 2
    assert "could not parse arguments as JSON" in agent._messages[0]["content"]
    assert agent._messages[1]["name"] == "GetJournalSummary"
    assert agent._messages[1]["content"] == "journal summary"


class _MaxIterAgent(BaseKBQAAgent):
    """Minimal double for driving `_run_tool_loop`'s max-iterations guard
    without needing an LLM client, MCP connection, or any tool wiring."""

    def __init__(self):
        self._messages = []
        self.recorder = TraceRecorder()
        self.tool_call_counts = {}
        self.synthesis_calls = []

    def get_config(self):
        raise NotImplementedError

    def get_mcp_server_path(self) -> str:
        raise NotImplementedError

    def _trace(self, message: str, color: str = "") -> None:
        pass

    async def _run_synthesis(self, query: str, qtype: str = "") -> str:
        self.synthesis_calls.append({"query": query, "qtype": qtype})
        return "SYNTHESIZED_SENTINEL"


def test_max_iterations_exits_through_synthesis_not_error_string():
    agent = _MaxIterAgent()

    # max_iterations=0 means the very first loop pass (iteration_count=1)
    # already exceeds the cap, so the guard fires without needing to stub
    # an LLM call or any tools.
    result = _run(agent._run_tool_loop(
        query="Who directed Inception?",
        tools=[],
        max_iterations=0,
        refresh_interval=5,
        qtype="Query",
    ))

    assert result == "SYNTHESIZED_SENTINEL"
    assert result != "Error: Agent reached maximum iteration limit."
    assert agent.synthesis_calls == [{"query": "Who directed Inception?", "qtype": "Query"}]
