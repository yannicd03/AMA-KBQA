"""The optional ``on_thread_start`` hook on ``lifecycle_runner.start_run``.

The Streamlit page never passes it, so the first tests pin the old behaviour
(stdout captured through ``redirect_stdout``); the rest cover the hook the API
uses to bind the worker thread to its run's log.
"""

from __future__ import annotations

import inspect
import io
import queue
import threading
import time

from ama_kbqa.api.stdout_router import RunLog, install, route_current_context, uninstall
from ama_kbqa.framework.trace import TraceRecorder
from ama_kbqa.frontend.utils.lifecycle_runner import LiveLifecycleState, drain_into, start_run


class _PrintingAgent:
    def __init__(self, calls: list) -> None:
        self.recorder = TraceRecorder()
        self.calls = calls

    async def ask(self, question: str) -> str:
        self.calls.append(("ask", threading.current_thread().name))
        async with self.recorder.span("agent_run", "ask"):
            print(f"working on {question}")
        return f"answer to: {question}"


def _wait(q: queue.Queue, state: LiveLifecycleState, timeout: float = 5.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if drain_into(q, state):
            return
        time.sleep(0.02)
    raise AssertionError(f"runner did not terminate; state={state.status}")


def test_hook_is_optional_and_defaults_to_none():
    param = inspect.signature(start_run).parameters["on_thread_start"]
    assert param.default is None
    assert param.kind is inspect.Parameter.KEYWORD_ONLY


def test_without_hook_capture_io_still_captures_stdout():
    calls: list = []
    capture = io.StringIO()
    q: queue.Queue = queue.Queue()
    state = LiveLifecycleState()
    t = start_run(agent=_PrintingAgent(calls), question="hi", q=q, capture_io=capture)
    _wait(q, state)

    assert t.name == "lifecycle-runner" and t.daemon
    assert state.status == "done" and state.answer == "answer to: hi"
    assert capture.getvalue() == "working on hi\n"
    assert calls == [("ask", "lifecycle-runner")]


def test_hook_runs_first_inside_the_worker_thread():
    calls: list = []
    q: queue.Queue = queue.Queue()
    state = LiveLifecycleState()

    def hook():
        calls.append(("hook", threading.current_thread().name))

    start_run(agent=_PrintingAgent(calls), question="hi", q=q, on_thread_start=hook)
    _wait(q, state)
    assert calls == [("hook", "lifecycle-runner"), ("ask", "lifecycle-runner")]
    assert state.status == "done"


def test_failing_hook_ends_the_run_with_an_error():
    calls: list = []
    q: queue.Queue = queue.Queue()
    state = LiveLifecycleState()

    def hook():
        raise RuntimeError("no log")

    start_run(agent=_PrintingAgent(calls), question="hi", q=q, on_thread_start=hook)
    _wait(q, state)
    assert state.status == "error"
    assert state.error == "RuntimeError: no log"
    assert calls == []  # ask() never ran


def test_hook_binds_the_worker_to_a_run_log(capsys):
    proxy = install()
    try:
        calls: list = []
        log = RunLog()
        q: queue.Queue = queue.Queue()
        state = LiveLifecycleState()
        start_run(
            agent=_PrintingAgent(calls), question="hi", q=q, capture_io=None,
            on_thread_start=lambda: route_current_context(log),
        )
        print("main thread output")
        _wait(q, state)
    finally:
        uninstall(proxy)
    assert log.text() == "working on hi\n"
    out = capsys.readouterr().out
    assert "main thread output" in out and "working on hi" not in out
