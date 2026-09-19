"""Cooperative cancellation for agent runs.

A run executes on a worker thread that owns its own event loop
(``frontend/utils/lifecycle_runner.start_run``), and its MCP connection is
bound to *that* loop: closing it from any other thread or task trips anyio's
"Attempted to exit cancel scope in a different task" (the reason
``lifecycle_runner`` orphans a stale connection instead of closing it, and the
reason ``Orchestrator._run_specialist`` closes each specialist inside its own
task). A run therefore cannot be stopped from the outside by cancelling a task
or killing the thread.

So cancellation is cooperative: the caller flips a flag, the agent notices it
at one of its own checkpoints and returns normally through its own code. Every
``finally`` on the way out — including the MCP teardown — then runs on the
worker's own loop, exactly as it does for a run that finished by itself.

The token is a thin :class:`threading.Event` wrapper because the setter (an
HTTP handler on the API's event loop, a Streamlit button on the main thread)
and the reader (the agent, on the worker thread) live on different threads.

Everything here is optional. Every entry point keeps ``cancel_token=None`` as
its default and :func:`is_cancelled` is False for ``None``, so callers that
pass no token — the Streamlit page, the benchmark runners — are unaffected.
"""

from __future__ import annotations

import threading
from typing import Optional

# The answer a cancelled run returns in place of a synthesized one. Callers
# that need to distinguish "cancelled" from a real answer should look at their
# own token rather than string-matching this.
CANCELLED_ANSWER = "Run cancelled."


class CancellationToken:
    """A thread-safe stop flag for one run, with an optional reason.

    Create one per run, hand it to the agent entry point (``ask``,
    ``start_run``), and call :meth:`cancel` from wherever the stop request
    arrives. Cancellation is one-way: a cancelled token stays cancelled.
    """

    __slots__ = ("_event", "_reason")

    def __init__(self, reason: str = "") -> None:
        self._event = threading.Event()
        self._reason = reason

    def cancel(self, reason: str = "") -> None:
        """Request cancellation. Safe to call from any thread, repeatedly."""
        if reason:
            self._reason = reason
        self._event.set()

    @property
    def cancelled(self) -> bool:
        """True once :meth:`cancel` has been called."""
        return self._event.is_set()

    @property
    def reason(self) -> str:
        """Why the run was cancelled; empty when no reason was given."""
        return self._reason

    def wait(self, timeout: Optional[float] = None) -> bool:
        """Block until cancelled (or ``timeout`` elapses). Mostly for tests."""
        return self._event.wait(timeout)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        state = "cancelled" if self.cancelled else "active"
        return f"<CancellationToken {state} reason={self._reason!r}>"


def is_cancelled(token: Optional[CancellationToken]) -> bool:
    """True only when a token was supplied AND it has been cancelled.

    The None-safe form every checkpoint uses, so "no token" costs one identity
    comparison and never changes behaviour.
    """
    return token is not None and token.cancelled


def cancelled_answer(token: Optional[CancellationToken] = None) -> str:
    """The user-facing text a cancelled run returns, with the reason if any."""
    reason = token.reason if token is not None else ""
    return f"{CANCELLED_ANSWER} ({reason})" if reason else CANCELLED_ANSWER
