"""Tests for the synthesis-exit funnel (commit f6b313c).

Every terminal condition in the tool loop must break to synthesis instead of
returning/raising an error string, so a journal with validated queries is
never discarded. Synthesis itself is guarded end to end: a MCP failure at
the journal-fetch step degrades to the last local snapshot, and any
synthesis failure falls back to the best validated query recorded in the
local journal snapshots (`best_snapshot_query`), shared with the WikiKGQA
generator's own recovery path.
"""

from __future__ import annotations

import asyncio

import pytest

from ama_kbqa.framework import base_agent
from ama_kbqa.framework.base_agent import BaseKBQAAgent, best_snapshot_query
from ama_kbqa.framework.config import KnowledgeGraphConfig
from ama_kbqa.framework.trace import TraceRecorder


def _run(coro):
    return asyncio.run(coro)


# --------------------------------------------------------------------------- #
# best_snapshot_query — pure function
# --------------------------------------------------------------------------- #

def _snap(found_values: dict) -> dict:
    return {"state": {"found_values": found_values}}


def test_best_snapshot_query_picks_highest_numbered_non_empty_result():
    snapshots = [
        _snap({
            "sparql_result_1": {"query": "SELECT ?a WHERE { ?a a wd:Q1 }", "result_count": 3},
        }),
        _snap({
            "sparql_result_2": {"query": "SELECT ?b WHERE { ?b a wd:Q2 }", "result_count": 5},
            "sparql_result_3": {"query": "SELECT ?c WHERE { ?c a wd:Q3 }", "result_count": 0},
        }),
    ]

    assert best_snapshot_query(snapshots) == "SELECT ?b WHERE { ?b a wd:Q2 }"


def test_best_snapshot_query_ignores_zero_result_count():
    snapshots = [
        _snap({
            "sparql_result_1": {"query": "SELECT ?a WHERE { ?a a wd:Q1 }", "result_count": 0},
        }),
    ]

    assert best_snapshot_query(snapshots) is None


def test_best_snapshot_query_ignores_non_sparql_result_keys():
    snapshots = [
        _snap({
            "visited_node_1": {"query": "not a sparql result", "result_count": 5},
        }),
    ]

    assert best_snapshot_query(snapshots) is None


def test_best_snapshot_query_returns_none_for_empty_input():
    assert best_snapshot_query([]) is None
    assert best_snapshot_query(None) is None


def test_best_snapshot_query_returns_none_when_no_valid_query_present():
    snapshots = [_snap({}), {"state": {}}, {}]
    assert best_snapshot_query(snapshots) is None


# --------------------------------------------------------------------------- #
# _run_synthesis_guarded — fallback to best_snapshot_query on failure
# --------------------------------------------------------------------------- #

class _ConcreteAgent(BaseKBQAAgent):
    def get_config(self) -> KnowledgeGraphConfig:  # pragma: no cover - unused
        raise NotImplementedError

    def get_mcp_server_path(self) -> str:  # pragma: no cover - unused
        raise NotImplementedError


def _guarded_agent(journal_snapshots):
    agent = object.__new__(_ConcreteAgent)
    agent.recorder = TraceRecorder()
    agent.journal_snapshots = journal_snapshots
    agent._trace = lambda *a, **k: None

    async def _raising_run_synthesis(query, qtype=""):
        raise RuntimeError("synthesis LLM call blew up")

    agent._run_synthesis = _raising_run_synthesis
    return agent


def test_run_synthesis_guarded_falls_back_to_best_snapshot_query_on_failure():
    snapshots = [
        _snap({
            "sparql_result_1": {"query": "SELECT ?a WHERE { ?a a wd:Q1 }", "result_count": 2},
        }),
    ]
    agent = _guarded_agent(snapshots)

    answer = _run(agent._run_synthesis_guarded("some question"))

    assert answer == "SELECT ?a WHERE { ?a a wd:Q1 }"


def test_run_synthesis_guarded_returns_empty_string_when_no_snapshots():
    agent = _guarded_agent([])

    answer = _run(agent._run_synthesis_guarded("some question"))

    assert answer == ""


# --------------------------------------------------------------------------- #
# _run_synthesis_impl — GetJournalSummary failure degrades to local snapshot
# --------------------------------------------------------------------------- #

class _Span:
    def set_attribute(self, *_a, **_k):
        pass

    def set_payload(self, *_a, **_k):
        pass


class _RaisingMCP:
    """MCP stub whose GetJournalSummary call always fails, like a dead
    session at the finish line."""

    async def call_tool(self, name: str, _args):
        raise ConnectionError(f"{name} unavailable: session dead")


class _JournalFallbackAgent(BaseKBQAAgent):
    """Base agent double that drives _run_synthesis_impl deterministically,
    with a failing MCP and a captured synthesis prompt."""

    def __init__(self, journal_snapshots):
        self._messages = []
        self.mcp = _RaisingMCP()
        self.journal_snapshots = journal_snapshots
        self.synth_calls = []

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
        return "Christopher Nolan"


def test_journal_summary_failure_degrades_to_last_local_snapshot():
    snapshots = [
        {"state": {"found_values": {"director": "Christopher Nolan"}}},
    ]
    agent = _JournalFallbackAgent(snapshots)

    # Must not raise despite GetJournalSummary always failing.
    answer = _run(agent._run_synthesis_impl("Who directed it?", _Span()))

    assert answer == "Christopher Nolan"
    assert len(agent.synth_calls) == 1
    # The synthesis prompt was built from the local snapshot state, not an
    # empty/absent journal.
    prompt_content = agent.synth_calls[0][-1]["content"]
    assert "director" in prompt_content


def test_journal_summary_failure_with_no_snapshots_still_completes():
    agent = _JournalFallbackAgent(journal_snapshots=[])

    answer = _run(agent._run_synthesis_impl("Who directed it?", _Span()))

    assert answer == "Christopher Nolan"
    assert len(agent.synth_calls) == 1


# --------------------------------------------------------------------------- #
# Tool-loop terminal conditions break to synthesis instead of returning an
# error string / raising.
# --------------------------------------------------------------------------- #

class _LoopTerminalAgent(BaseKBQAAgent):
    """Drives the real _run_tool_loop far enough to hit a terminal
    condition, with the LLM call and synthesis guard stubbed out."""

    def __init__(self, llm_call=None):
        self._messages = [{"role": "system", "content": "sys"}]
        self.recorder = TraceRecorder()
        self.tool_call_counts: dict = {}
        self._text_tool_call_mode = False
        self.synthesis_guard_calls = []
        self._llm_call_override = llm_call

    def get_config(self):
        raise NotImplementedError

    def get_mcp_server_path(self) -> str:
        raise NotImplementedError

    def _trace(self, message: str, color: str = "") -> None:
        pass

    def _llm_call(self, tools=None, tool_choice=None):
        if self._llm_call_override is not None:
            return self._llm_call_override()
        raise AssertionError("_llm_call should not be reached in this test")

    async def _run_synthesis_guarded(self, query, qtype=""):
        self.synthesis_guard_calls.append((query, qtype))
        return "SYNTHESIZED_ANSWER"


@pytest.fixture
def _synthesis_config(monkeypatch):
    monkeypatch.setattr(base_agent, "get_synthesis_enabled", lambda: True)
    monkeypatch.setattr(base_agent, "get_auto_inject_journal", lambda: True)


def test_max_iterations_exceeded_breaks_to_synthesis_not_error_string(_synthesis_config):
    agent = _LoopTerminalAgent()

    answer = _run(agent._run_tool_loop("some question", [], max_iterations=0, refresh_interval=5))

    assert answer == "SYNTHESIZED_ANSWER"
    assert answer != "Error: Agent reached maximum iteration limit."
    assert agent.synthesis_guard_calls == [("some question", "")]
    # The intervention was recorded on the trace.
    event_names = [e.name for e in agent.recorder.events]
    assert "max_iterations_reached" in event_names


def test_mid_loop_llm_failure_breaks_to_synthesis_instead_of_raising(_synthesis_config):
    def _boom():
        raise ConnectionError("provider is dead")

    agent = _LoopTerminalAgent(llm_call=_boom)

    answer = _run(agent._run_tool_loop("some question", [], max_iterations=5, refresh_interval=5))

    assert answer == "SYNTHESIZED_ANSWER"
    assert agent.synthesis_guard_calls == [("some question", "")]
    event_names = [e.name for e in agent.recorder.events]
    assert "llm_call_failed_break_to_synthesis" in event_names
