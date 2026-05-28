"""Smoke tests for the background lifecycle runner."""

from __future__ import annotations

import queue
import threading
import time

from ama_kbqa.framework.trace import TraceRecorder
from ama_kbqa.frontend.utils.lifecycle_runner import (
    LiveLifecycleState,
    drain_into,
    start_run,
)


class _StubAgent:
    """Minimal agent that emits the canonical span set then returns an answer."""

    def __init__(self, *, fail: bool = False) -> None:
        self.recorder = TraceRecorder()
        self.fail = fail

    async def ask(self, question: str) -> str:
        async with self.recorder.span("agent_run", "ask"):
            async with self.recorder.span("classify", "classify"):
                pass
            async with self.recorder.span("llm_call", "gpt-4o"):
                async with self.recorder.span("tool_call", "FindNode"):
                    pass
                async with self.recorder.span("tool_call", "VerifyFact"):
                    pass
            self.recorder.event("loop_detected", "x")
            async with self.recorder.span("synthesis", "final"):
                pass
        if self.fail:
            raise RuntimeError("boom")
        return f"answer to: {question}"


class _StubMultiturnAgent:
    """Agent double that mimics the real history-preserving reset + ask so the
    multiturn continuation path can be driven end-to-end through start_run."""

    def __init__(self) -> None:
        self.recorder = TraceRecorder()
        self._messages = [{"role": "system", "content": "sys"}]
        self._catalog_injected = False
        self.mcp = object()  # pretend an MCP connection is open
        self.reset_calls: list[dict] = []

    async def reset(self, keep_mcp_open: bool = False, keep_history: bool = False) -> None:
        self.reset_calls.append({"keep_mcp_open": keep_mcp_open, "keep_history": keep_history})
        if not keep_history:
            self._messages = [{"role": "system", "content": "sys"}]
            self._catalog_injected = False
        self.recorder = TraceRecorder()  # fresh trace each turn
        if not keep_mcp_open and self.mcp is not None:
            self.mcp = None

    async def ask(self, question: str) -> str:
        if self.mcp is None:  # mimic _init_mcp rebuilding on this loop
            self.mcp = object()
        if not self._catalog_injected:  # one-time catalog injection
            self._messages.append({"role": "system", "content": "CATALOG"})
            self._catalog_injected = True
        self._messages.append({"role": "user", "content": question})
        async with self.recorder.span("agent_run", "ask"):
            pass
        answer = f"answer to: {question}"
        self._messages.append({"role": "assistant", "content": answer})
        return answer


def _wait_until_terminal(q: queue.Queue, state: LiveLifecycleState, timeout: float = 5.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if drain_into(q, state):
            return
        time.sleep(0.02)
    raise AssertionError(f"runner did not terminate; state={state.status}")


class TestRunner:
    def test_successful_run_transitions_to_done(self):
        agent = _StubAgent()
        q: queue.Queue = queue.Queue()
        state = LiveLifecycleState()
        t = start_run(agent=agent, question="hi", q=q)
        assert isinstance(t, threading.Thread)

        _wait_until_terminal(q, state)

        assert state.status == "done"
        assert state.answer == "answer to: hi"
        # Tool calls (traversal + summary) all light the single Tool Call box.
        assert "main_tool_call" in state.visited_node_ids
        assert "main_llm_reason" in state.visited_node_ids
        assert "post_synthesis" in state.visited_node_ids
        assert "agent_invocation" in state.visited_node_ids
        # No spans should be active after completion.
        assert state.active_node_ids == set()

    def test_failing_run_transitions_to_error(self):
        agent = _StubAgent(fail=True)
        q: queue.Queue = queue.Queue()
        state = LiveLifecycleState()
        start_run(agent=agent, question="hi", q=q)

        _wait_until_terminal(q, state)

        assert state.status == "error"
        assert state.error is not None and "boom" in state.error
        assert state.active_node_ids == set()

    def test_continuation_preserves_and_accumulates_history(self):
        agent = _StubMultiturnAgent()

        # Turn 1: fresh conversation, no reset.
        q1: queue.Queue = queue.Queue()
        s1 = LiveLifecycleState()
        start_run(
            agent=agent,
            question="Who directed Inception?",
            q=q1,
            is_continuation=False,
        )
        _wait_until_terminal(q1, s1)

        assert s1.status == "done"
        rec_after_t1 = agent.recorder
        # Live trace_id was emitted to the main thread via the queue.
        assert s1.trace_id == rec_after_t1.trace_id
        assert agent.reset_calls == []  # first turn never resets
        assert sum(1 for m in agent._messages if m["content"] == "CATALOG") == 1

        # Turn 2: continuation on the SAME instance.
        q2: queue.Queue = queue.Queue()
        s2 = LiveLifecycleState()
        start_run(
            agent=agent,
            question="Where was he born?",
            q=q2,
            is_continuation=True,
        )
        _wait_until_terminal(q2, s2)

        assert s2.status == "done"
        # Reset ran once, preserving history and not closing MCP across loops.
        assert agent.reset_calls == [{"keep_mcp_open": True, "keep_history": True}]
        # MCP was orphaned then rebuilt.
        assert agent.mcp is not None
        # Fresh recorder + trace_id propagated to the live panel.
        assert agent.recorder is not rec_after_t1
        assert s2.trace_id == agent.recorder.trace_id
        assert s2.trace_id != s1.trace_id
        # History ACCUMULATED across turns.
        contents = [m["content"] for m in agent._messages]
        assert "Who directed Inception?" in contents
        assert "Where was he born?" in contents
        # Catalog was NOT duplicated onto the preserved stack.
        assert sum(1 for m in agent._messages if m["content"] == "CATALOG") == 1

    def test_drain_into_is_idempotent_after_terminal(self):
        agent = _StubAgent()
        q: queue.Queue = queue.Queue()
        state = LiveLifecycleState()
        start_run(agent=agent, question="hi", q=q)

        _wait_until_terminal(q, state)
        snapshot_active = set(state.active_node_ids)
        snapshot_visited = set(state.visited_node_ids)

        # Subsequent drains on an empty queue must not mutate state.
        assert drain_into(q, state) is True
        assert state.active_node_ids == snapshot_active
        assert state.visited_node_ids == snapshot_visited
