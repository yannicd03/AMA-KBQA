"""Tests for prefix-cache-friendly message history.

Two invariants protect provider prompt caching:
1. Journal refreshes are APPENDED; prior messages are never removed/shifted.
2. Context trimming happens as discrete compaction events with hysteresis:
   between compactions the history is append-only (no interior mutation).
"""

from __future__ import annotations

import asyncio
import copy

from ama_kbqa.framework.base_agent import BaseKBQAAgent


def _run(coro):
    return asyncio.run(coro)


class _Recorder:
    def __init__(self):
        self.events = []

    def event(self, kind, name, attributes=None):
        self.events.append((kind, name, attributes or {}))


class _FakeMcp:
    def __init__(self, summary="journal summary v2"):
        self._summary = summary

    async def call_tool(self, name, args):
        return self._summary


class HistoryAgent(BaseKBQAAgent):
    """Test double exposing only history-management state."""

    MARKER = BaseKBQAAgent._JOURNAL_REFRESH_MARKER

    def __init__(self, messages=None, context_limit=100000):
        self._messages = messages if messages is not None else [
            {"role": "system", "content": "sys"}
        ]
        self._context_limit = context_limit
        self._next_trim_trigger = 0.5
        self.recorder = _Recorder()
        self.last_journal_state = None
        self.mcp = _FakeMcp()

    def get_config(self):
        raise NotImplementedError

    def get_mcp_server_path(self) -> str:
        raise NotImplementedError

    def _trace(self, message: str, color: str = "") -> None:
        pass

    async def _snapshot_journal(self, trigger: str) -> None:
        pass


def _tool_msg(content):
    return {"role": "tool", "tool_call_id": "x", "name": "T", "content": content}


# ---------------------------------------------------------------------------
# Journal refresh: append-only
# ---------------------------------------------------------------------------

def test_journal_refresh_appends_without_removing_prior_refresh():
    old_refresh = {
        "role": "user",
        "content": HistoryAgent.MARKER + "WORKING MEMORY REFRESH (Iteration 5)\nold",
    }
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "assistant", "content": "thought"},
        old_refresh,
        {"role": "assistant", "content": "more"},
    ]
    before = copy.deepcopy(messages)
    agent = HistoryAgent(messages=messages)

    _run(agent._inject_journal_refresh(10))

    # All prior messages unchanged and in place (stable prefix).
    assert agent._messages[: len(before)] == before
    # New refresh appended at the end.
    assert agent._messages[-1]["content"].startswith(HistoryAgent.MARKER)
    assert "Iteration 10" in agent._messages[-1]["content"]
    assert "journal summary v2" in agent._messages[-1]["content"]


# ---------------------------------------------------------------------------
# Context compaction with hysteresis
# ---------------------------------------------------------------------------

def _build_heavy_agent(n_tool_msgs=20, msg_chars=350, context_limit=3500):
    """~20*350/3.5 = 2000 estimated tokens vs limit 3500 -> ~57% usage."""
    messages = [{"role": "system", "content": "sys"}]
    messages += [_tool_msg("x" * msg_chars) for _ in range(n_tool_msgs)]
    return HistoryAgent(messages=messages, context_limit=context_limit)


def test_no_mutation_below_trigger():
    agent = _build_heavy_agent(n_tool_msgs=4)  # ~400 tokens, far below 50%
    before = copy.deepcopy(agent._messages)
    agent._manage_context_window()
    assert agent._messages == before
    assert agent._next_trim_trigger == 0.5


def test_compaction_trims_and_arms_next_trigger():
    agent = _build_heavy_agent()
    agent._manage_context_window()

    # Interior tool results trimmed; protected tail (last 8) untouched.
    assert agent._messages[1]["content"].endswith("...[trimmed]")
    assert agent._messages[-1]["content"] == "x" * 350
    # Next trigger armed above the post-compaction usage ratio, so an
    # immediate second pass is a no-op (hysteresis).
    post_chars = sum(len(m.get("content", "") or "") for m in agent._messages)
    assert agent._next_trim_trigger > (post_chars / 3.5) / 3500
    after = copy.deepcopy(agent._messages)
    agent._manage_context_window()
    assert agent._messages == after
    kinds = [k for k, _, _ in agent.recorder.events]
    assert "context_trim" in kinds


def test_history_is_stable_between_compactions():
    agent = _build_heavy_agent()
    agent._manage_context_window()  # first compaction
    after_compaction = copy.deepcopy(agent._messages)

    # History grows a little, but stays below the newly armed trigger.
    agent._messages.append(_tool_msg("y" * 200))
    agent._manage_context_window()

    # Every pre-existing message is byte-identical: append-only prefix.
    assert agent._messages[: len(after_compaction)] == after_compaction


def test_recompaction_after_regrowth_past_next_trigger():
    agent = _build_heavy_agent()
    agent._manage_context_window()
    armed = agent._next_trim_trigger

    # Regrow well past the armed trigger.
    agent._messages += [_tool_msg("z" * 400) for _ in range(30)]
    agent._manage_context_window()
    assert agent._next_trim_trigger >= armed


def test_superseded_refreshes_stubbed_on_compaction_only():
    marker = HistoryAgent.MARKER
    messages = [{"role": "system", "content": "sys"}]
    messages.append({"role": "user", "content": marker + "REFRESH 5 " + "a" * 300})
    messages += [_tool_msg("x" * 350) for _ in range(10)]
    messages.append({"role": "user", "content": marker + "REFRESH 10 " + "b" * 300})
    messages += [_tool_msg("x" * 350) for _ in range(10)]
    agent = HistoryAgent(messages=messages, context_limit=3500)

    agent._manage_context_window()

    # Old refresh (interior) stubbed; the newest refresh survives.
    assert agent._messages[1]["content"] == marker + "[superseded by a later WORKING MEMORY REFRESH]"
    assert "REFRESH 10" in agent._messages[12]["content"]


def test_aggressive_tier_trims_user_injections():
    messages = [{"role": "system", "content": "sys"}]
    messages += [_tool_msg("x" * 350) for _ in range(2)]
    # User injections are only trimmed past index 2.
    messages.append({"role": "user", "content": "guidance " * 100})  # >500 chars
    messages += [_tool_msg("x" * 350) for _ in range(20)]
    # ~8600 chars -> ~2460 tokens vs limit 2800 -> ~88% usage (aggressive tier)
    agent = HistoryAgent(messages=messages, context_limit=2800)

    agent._manage_context_window()

    assert agent._messages[3]["content"].endswith("...[trimmed]")
    assert len(agent._messages[3]["content"]) <= 200 + len("...[trimmed]")
