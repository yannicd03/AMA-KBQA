"""Per-run stdout capture that survives concurrent runs.

The agents report progress with plain ``print()`` (ANSI-coloured), and the
live log in the UI is that output. The Streamlit page captures it with
``contextlib.redirect_stdout``, which swaps the *process-global*
``sys.stdout``: with two runs in flight, whichever started last steals the
other's output. That is tolerable for one browser tab, not for a server.

This module installs one permanent proxy as ``sys.stdout`` at API startup and
routes each ``write()`` by a :class:`~contextvars.ContextVar`:

* The run's worker thread binds its :class:`RunLog` through
  ``start_run(..., on_thread_start=...)``, before its event loop exists.
* Tasks on that thread's loop copy the thread's context when they are
  created, so every coroutine of the run sees the binding. So do calls the
  run makes through ``asyncio.to_thread`` (it copies the context too).
* Every other thread (uvicorn's loop, the threadpool, unrelated workers)
  starts with an empty context and writes straight through to the real
  stdout.

A ContextVar rather than a ``threading.local`` or a thread-id map because the
binding then dies with the worker's context: a later thread that happens to
reuse the same thread id cannot inherit a stale run's log.
"""

from __future__ import annotations

import sys
import threading
from contextvars import ContextVar
from typing import Any, Optional, TextIO

# Upper bound on what one run keeps in memory. The UI only ever shows the
# last ~100 KB (see runs.LOG_TAIL_CHARS); the rest is headroom so the final
# log still reads well. Older output is dropped from the head.
DEFAULT_MAX_CHARS = 2_000_000

_CURRENT_LOG: ContextVar[Optional["RunLog"]] = ContextVar(
    "ama_kbqa_api_run_log", default=None
)


class RunLog:
    """Thread-safe, append-only text buffer holding one run's stdout."""

    def __init__(self, max_chars: int = DEFAULT_MAX_CHARS) -> None:
        self._lock = threading.Lock()
        self._chunks: list[str] = []
        self._size = 0
        self._max_chars = max_chars
        # Bumped on every non-empty write; lets readers skip re-rendering an
        # unchanged log without comparing strings.
        self.version = 0
        # True once the head of the log was dropped to respect ``max_chars``.
        self.truncated = False

    def write(self, text: Any) -> int:
        if not isinstance(text, str):
            text = str(text)
        if not text:
            return 0
        with self._lock:
            self._chunks.append(text)
            self._size += len(text)
            self.version += 1
            if self._size > self._max_chars:
                keep = "".join(self._chunks)[-(self._max_chars // 2):]
                self._chunks = [keep]
                self._size = len(keep)
                self.truncated = True
        return len(text)

    def text(self) -> str:
        with self._lock:
            if len(self._chunks) > 1:
                self._chunks = ["".join(self._chunks)]
            return self._chunks[0] if self._chunks else ""


class RoutingStdout:
    """``sys.stdout`` stand-in that sends each write to the bound run's log,
    or to the real stream when the calling context has no binding."""

    def __init__(self, real: TextIO) -> None:
        self._real = real

    @property
    def real(self) -> TextIO:
        return self._real

    def write(self, text: str) -> int:
        log = _CURRENT_LOG.get()
        if log is not None:
            return log.write(text)
        return self._real.write(text)

    def writelines(self, lines) -> None:
        for line in lines:
            self.write(line)

    def flush(self) -> None:
        if _CURRENT_LOG.get() is None:
            self._real.flush()

    def isatty(self) -> bool:
        # A routed write lands in a buffer, like the StringIO the Streamlit
        # page captures into, so report what that would report.
        if _CURRENT_LOG.get() is not None:
            return False
        return bool(getattr(self._real, "isatty", lambda: False)())

    def __getattr__(self, name: str) -> Any:
        # encoding, errors, fileno, buffer, ... come from the real stream.
        return getattr(self._real, name)


def install() -> RoutingStdout:
    """Install the proxy as ``sys.stdout`` (idempotent) and return it."""
    current = sys.stdout
    if isinstance(current, RoutingStdout):
        return current
    proxy = RoutingStdout(current)
    sys.stdout = proxy
    return proxy


def uninstall(proxy: RoutingStdout) -> None:
    """Restore the real stream, but only if nobody replaced the proxy since."""
    if sys.stdout is proxy:
        sys.stdout = proxy.real


def route_current_context(log: RunLog) -> None:
    """Send this context's stdout (and that of everything it spawns through
    the event loop or ``asyncio.to_thread``) to ``log``.

    Meant to be passed to ``start_run`` as the ``on_thread_start`` hook,
    wrapped in a closure that supplies the run's log.
    """
    _CURRENT_LOG.set(log)


def current_log() -> Optional[RunLog]:
    """The log bound to the calling context, if any (used by tests)."""
    return _CURRENT_LOG.get()
