"""Tests for multiturn support: history-preserving reset and catalog dedup.

Multiturn (for directly-selected sub-agents) keeps ONE agent instance alive
across a conversation. Between turns the frontend calls
``reset(keep_history=True)``, which must preserve the accumulated message stack
(so a follow-up can resolve coreference against prior turns) while still
resetting per-turn state (tokens, tool counters, loop detection, recorder,
journal). It must also leave the text-mode tool-catalog dedup flag intact so the
catalog is not re-injected onto the preserved stack.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from ama_kbqa.framework.base_agent import BaseKBQAAgent
from ama_kbqa.framework.trace import TraceRecorder


def _run(coro):
    return asyncio.run(coro)


class _ResetAgent(BaseKBQAAgent):
    """Minimal test double that bypasses production client/MCP setup but carries
    the exact attribute surface that ``reset()`` touches."""

    def __init__(self):
        self._parent_span_id_override = None
        self.mcp = None
        self.recorder = TraceRecorder()
        self._messages = [{"role": "system", "content": self._get_system_prompt()}]
        self.token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self.tool_call_counts = {}
        self.tool_call_durations = []
        self.tool_call_history = []
        self.tool_sequence = []
        self.empty_result_count = 0
        self.last_journal_state = None
        self.journal_snapshots = []
        self._last_journal_summary_hash = None
        self._catalog_injected = False

    def get_config(self):  # pragma: no cover - unused
        raise NotImplementedError

    def get_mcp_server_path(self) -> str:  # pragma: no cover - unused
        raise NotImplementedError


def _dirty(agent: _ResetAgent) -> None:
    """Simulate the end-of-turn state of a completed question."""
    agent._messages.extend([
        {"role": "user", "content": "Who directed Inception?"},
        {"role": "assistant", "content": "Christopher Nolan directed Inception."},
    ])
    agent.token_usage = {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
    agent.tool_call_counts = {"FindResource": 2}
    agent.tool_call_durations = [{"name": "FindResource", "seconds": 0.3}]
    agent.tool_call_history = [("FindResource", "Inception")]
    agent.tool_sequence = ["FindResource"]
    agent.empty_result_count = 1
    agent.last_journal_state = "some state"
    agent.journal_snapshots = [{"snap": 1}]
    agent._last_journal_summary_hash = "abc123"
    agent._catalog_injected = True


def test_keep_history_preserves_messages_but_resets_per_turn_state():
    agent = _ResetAgent()
    _dirty(agent)
    messages_before = list(agent._messages)
    recorder_before = agent.recorder

    _run(agent.reset(keep_history=True))

    # Message stack is carried forward verbatim.
    assert agent._messages == messages_before
    assert len(agent._messages) == 3

    # Everything ephemeral is reset.
    assert agent.token_usage == {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    assert agent.tool_call_counts == {}
    assert agent.tool_call_durations == []
    assert agent.tool_call_history == []
    assert agent.tool_sequence == []
    assert agent.empty_result_count == 0
    assert agent.last_journal_state is None
    assert agent.journal_snapshots == []
    assert agent._last_journal_summary_hash is None
    # Fresh trace for the new turn.
    assert agent.recorder is not recorder_before

    # The catalog stays in the preserved stack, so the dedup flag must remain
    # set or _ask_impl would inject a duplicate catalog.
    assert agent._catalog_injected is True


def test_keep_history_false_is_legacy_full_reset():
    agent = _ResetAgent()
    _dirty(agent)

    _run(agent.reset(keep_history=False))

    # Stack wiped back to just the system prompt.
    assert agent._messages == [
        {"role": "system", "content": agent._get_system_prompt()}
    ]
    # Catalog flag cleared so the next ask re-injects it onto the fresh stack.
    assert agent._catalog_injected is False
    assert agent.token_usage == {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def test_reset_defaults_to_full_reset():
    """Default call (no kwargs) must behave like the pre-multiturn reset."""
    agent = _ResetAgent()
    _dirty(agent)

    _run(agent.reset())

    assert len(agent._messages) == 1
    assert agent._messages[0]["role"] == "system"
    assert agent._catalog_injected is False


class _FakeMCP:
    async def list_tools(self):
        return []

    def convert_tools_to_openai_format(self, tools):
        return []


class _HookAgent(BaseKBQAAgent):
    """Drives _ask_impl with the heavy async deps stubbed so we can assert
    whether the pre-agent classification hook runs for a given turn."""

    def __init__(self):
        self.name = "test_agent"
        self.model = "test-model"
        self._parent_span_id_override = None
        self.recorder = TraceRecorder()
        self.mcp = None
        self._text_tool_call_mode = False
        self._messages = [{"role": "system", "content": "sys"}]
        # Instrumentation
        self.classify_calls = 0
        self.filter_calls = 0
        self.tool_loop_calls = []
        self.static_context_calls = 0
        self.question_context_calls = 0

    # --- abstract / external surface, stubbed -------------------------------
    def get_config(self):
        return SimpleNamespace(domain_settings={})

    def get_mcp_server_path(self) -> str:  # pragma: no cover - unused
        raise NotImplementedError

    def _trace(self, message: str, color: str = "") -> None:
        pass

    async def _init_mcp(self) -> None:
        self.mcp = _FakeMCP()

    async def _finalize_question(self) -> None:
        pass

    # --- pre-agent hook surface (instrumented) ------------------------------
    def _classify_question(self, query):
        self.classify_calls += 1
        return {
            "question_type": "QueryName",
            "entities": [],
            "relations": [],
            "fewshot_examples": "",
        }

    def _build_static_qtype_context(self, qtype, fewshot_examples=""):
        self.static_context_calls += 1
        return "STATIC_QTYPE_CONTEXT"

    def _build_question_context(self, qtype, entities, relations, query=""):
        self.question_context_calls += 1
        return "QUESTION_CONTEXT"

    def _get_allowed_tools_for_qtype(self, qtype):
        self.filter_calls += 1
        return None

    async def _run_tool_loop(self, query, openai_tools, max_iterations, refresh_interval, qtype=""):
        self.tool_loop_calls.append({"query": query, "qtype": qtype})
        return f"ANSWER: {query}"


def test_first_turn_runs_classification():
    agent = _HookAgent()  # fresh stack: only the system message

    answer = _run(agent.ask("Who directed Inception?"))

    assert answer == "ANSWER: Who directed Inception?"
    # Fresh turn: full pre-agent hook runs.
    assert agent.classify_calls == 1
    assert agent.filter_calls == 1
    # Both the static qtype context and the per-question context were built
    # and injected exactly once.
    assert agent.static_context_calls == 1
    assert agent.question_context_calls == 1
    assert any(m.get("content") == "STATIC_QTYPE_CONTEXT" for m in agent._messages)
    assert any(m.get("content") == "QUESTION_CONTEXT" for m in agent._messages)


def test_static_context_precedes_query_and_question_context():
    """Prompt-cache ordering: the static, qtype-only context must sit BEFORE
    both the raw query and the per-question context in the message stack, so
    it forms a stable, shareable prefix across same-qtype questions (see
    .agent/Tasks/active/prompt-cache-utilization.md, item 2)."""
    agent = _HookAgent()

    _run(agent.ask("Who directed Inception?"))

    contents = [m.get("content") for m in agent._messages]
    static_idx = contents.index("STATIC_QTYPE_CONTEXT")
    query_idx = contents.index("Who directed Inception?")
    question_idx = contents.index("QUESTION_CONTEXT")

    assert static_idx < query_idx < question_idx


def test_followup_skips_classification_and_filtering():
    agent = _HookAgent()
    # Simulate a preserved stack from a completed prior turn.
    agent._messages += [
        {"role": "user", "content": "Who directed Inception?"},
        {"role": "user", "content": "STATIC_QTYPE_CONTEXT"},
        {"role": "user", "content": "QUESTION_CONTEXT"},
        {"role": "assistant", "content": "Christopher Nolan."},
    ]

    answer = _run(agent.ask("Where was he born?"))

    assert answer == "ANSWER: Where was he born?"
    # Follow-up: classification, fast path, and tool filtering are all skipped.
    assert agent.classify_calls == 0
    assert agent.filter_calls == 0
    # No NEW static/per-question context built or injected for the follow-up
    # (only the prior turn's, already in the preserved stack).
    assert agent.static_context_calls == 0
    assert agent.question_context_calls == 0
    assert sum(1 for m in agent._messages if m.get("content") == "STATIC_QTYPE_CONTEXT") == 1
    assert sum(1 for m in agent._messages if m.get("content") == "QUESTION_CONTEXT") == 1
    # Full loop ran with the generic default qtype.
    assert agent.tool_loop_calls == [{"query": "Where was he born?", "qtype": "Query"}]
