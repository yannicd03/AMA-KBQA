"""The `cancel_token` seam on ``lifecycle_runner.start_run``.

The HTTP layer creates the token and triggers it; the runner only has to carry
it to ``agent.ask`` without disturbing callers that pass none (the Streamlit
page). The unwinding rule is the load-bearing part: a cancelled run must finish
through the agent's own code on the worker thread, so the worker's ``finally``
closes MCP on the loop that opened it.
"""

from __future__ import annotations

import inspect
import queue
import threading
import time

from ama_kbqa.framework.cancellation import CANCELLED_ANSWER, CancellationToken
from ama_kbqa.framework.trace import TraceRecorder
from ama_kbqa.frontend.utils.lifecycle_runner import (
    LiveLifecycleState,
    drain_into,
    start_run,
)


def _wait(q: queue.Queue, state: LiveLifecycleState, timeout: float = 5.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if drain_into(q, state):
            return
        time.sleep(0.02)
    raise AssertionError(f"runner did not terminate; state={state.status}")


class _TokenlessAgent:
    """Pre-cancellation agent surface: ask() takes only the question."""

    def __init__(self) -> None:
        self.recorder = TraceRecorder()

    async def ask(self, question: str) -> str:
        async with self.recorder.span("agent_run", "ask"):
            pass
        return f"answer to: {question}"


class _CancellableAgent:
    """Waits for the token, then unwinds through its own code like a real agent."""

    def __init__(self) -> None:
        self.recorder = TraceRecorder()
        self.seen_token = None
        self.teardown_thread = None

    async def ask(self, question: str, cancel_token=None) -> str:
        self.seen_token = cancel_token
        try:
            async with self.recorder.span("agent_run", "ask"):
                # Stand-in for the tool loop's per-iteration checkpoint.
                while not (cancel_token is not None and cancel_token.cancelled):
                    await _sleep(0.01)
            return CANCELLED_ANSWER
        finally:
            # A real agent closes MCP here; record where that happened.
            self.teardown_thread = threading.current_thread().name


async def _sleep(seconds: float) -> None:
    import asyncio

    await asyncio.sleep(seconds)


def test_cancel_token_is_keyword_only_and_optional():
    param = inspect.signature(start_run).parameters["cancel_token"]
    assert param.default is None
    assert param.kind is inspect.Parameter.KEYWORD_ONLY


def test_without_a_token_ask_is_called_with_the_question_alone():
    agent = _TokenlessAgent()
    q: queue.Queue = queue.Queue()
    state = LiveLifecycleState()

    start_run(agent=agent, question="hi", q=q)
    _wait(q, state)

    assert state.status == "done"
    assert state.answer == "answer to: hi"


def test_token_is_forwarded_and_the_run_unwinds_on_the_worker_thread():
    agent = _CancellableAgent()
    token = CancellationToken()
    q: queue.Queue = queue.Queue()
    state = LiveLifecycleState()

    start_run(agent=agent, question="hi", q=q, cancel_token=token)
    # Let the worker reach its checkpoint loop, then stop it from THIS thread.
    time.sleep(0.05)
    token.cancel("user pressed stop")
    _wait(q, state)

    assert agent.seen_token is token
    assert state.status == "done"
    assert state.answer == CANCELLED_ANSWER
    # The agent's own teardown ran on the worker thread, not the canceller's.
    assert agent.teardown_thread == "lifecycle-runner"
