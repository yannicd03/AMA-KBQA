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
