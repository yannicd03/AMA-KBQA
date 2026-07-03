"""Tests for shared agent tool-loop behavior."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

from ama_kbqa.framework.base_agent import BaseKBQAAgent


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
        # Mirrors the real agent's budget-accounting state: only
        # _execute_single_tool (below) increments it, exactly like production.
        self.tool_call_counts: dict = {}

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
        self.tool_call_counts[func_name] = self.tool_call_counts.get(func_name, 0) + 1
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


def _malformed_tool_call(name: str):
    """A tool call whose function.arguments is not valid JSON."""
    return SimpleNamespace(
        id="call-bad",
        function=SimpleNamespace(name=name, arguments="{not valid json"),
    )


def test_malformed_json_arguments_do_not_execute_or_burn_budget():
    """Malformed JSON must not execute with {} (that burns a call on a
    guaranteed error) and must not increment tool_call_counts, since budget
    accounting counts EXECUTED calls only. Uses a name already in
    _known_tool_names so the malformed-JSON branch is what's under test,
    not the unknown-tool-name branch that runs before it."""
    agent = DummyAgent()

    called = _run(agent._execute_tool_calls([_malformed_tool_call("GetJournalSummary")]))

    assert called is False
    # The tool was never actually invoked.
    assert agent.executed_calls == []
    assert agent.tool_call_counts == {}
    # A tool result explaining the problem was appended instead.
    assert len(agent._messages) == 1
    result_msg = agent._messages[0]
    assert result_msg["role"] == "tool"
    assert result_msg["name"] == "GetJournalSummary"
    assert "not valid JSON" in result_msg["content"]
