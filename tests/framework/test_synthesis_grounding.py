"""Tests for grounded-answer / "I don't know" handling in synthesis.

The system must only answer from knowledge-graph data recorded in the journal.
Two guarantees are exercised here:

1. If the synthesis step says "I don't know" but the journal actually contains
   discovered values, the verification pass re-prompts so a real answer is not
   lost (regression for adding "i don't know" to failure_phrases).
2. If the journal genuinely has no data, an "I don't know" answer is returned
   as-is and is NOT re-prompted into a fabricated answer.
"""

from __future__ import annotations

import asyncio

from ama_kbqa.framework.base_agent import BaseKBQAAgent


def _run(coro):
    return asyncio.run(coro)


class _Span:
    def set_attribute(self, *_a, **_k):
        pass

    def set_payload(self, *_a, **_k):
        pass


class _MCP:
    def __init__(self, journal_summary: str):
        self._journal = journal_summary

    async def call_tool(self, name: str, _args):
        assert name == "GetJournalSummary"
        return self._journal


class _SynthAgent(BaseKBQAAgent):
    """Base agent double that drives _run_synthesis_impl deterministically."""

    def __init__(self, journal_summary: str, llm_responses):
        self._messages = []
        self.mcp = _MCP(journal_summary)
        self._llm_responses = list(llm_responses)
        self.synth_calls = []

    # --- abstract / production hooks stubbed out ---
    def get_config(self):
        raise NotImplementedError

    def get_mcp_server_path(self) -> str:
        raise NotImplementedError

    def _trace(self, message: str, color: str = "") -> None:
        pass

    async def _snapshot_journal(self, trigger: str = "") -> None:
        pass

    def _get_synthesis_prompt_template(self) -> str:
        return "{journal_summary}\n\nQ: {query}"

    def _get_synthesis_system_prompt(self) -> str:
        return "system"

    def _llm_call_synthesis(self, messages_override=None):
        self.synth_calls.append(messages_override)
        return self._llm_responses.pop(0)


def test_idk_with_journal_data_triggers_reprompt():
    journal = "Discovered Values:\n- director = Christopher Nolan"
    agent = _SynthAgent(journal, ["I don't know.", "Christopher Nolan"])

    answer = _run(agent._run_synthesis_impl("Who directed it?", _Span()))

    # The first IDK was re-prompted because the journal had discovered values.
    assert len(agent.synth_calls) == 2
    assert answer == "Christopher Nolan"


def test_idk_with_empty_journal_is_kept():
    journal = "No values discovered yet."
    agent = _SynthAgent(journal, ["I don't know."])

    answer = _run(agent._run_synthesis_impl("Who directed it?", _Span()))

    # Empty journal: no re-prompt, the honest "I don't know" stands.
    assert len(agent.synth_calls) == 1
    assert answer == "I don't know."
