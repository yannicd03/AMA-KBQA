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
