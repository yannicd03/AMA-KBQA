"""Background-thread driver for the live agent-lifecycle visualisation.

The Streamlit Chat page kicks off ``start_run(...)`` which spawns a daemon
thread. The thread:

  * Attaches a listener to ``agent.recorder`` that pushes open/close/event
    notifications to a thread-safe ``queue.Queue``.
  * Spins up its own ``asyncio`` loop and runs ``agent.ask(question)`` to
    completion (or error).
  * Pushes a sentinel ``("__done__", {...})`` / ``("__error__", {...})`` when
    finished.

The main Streamlit thread polls the queue via ``drain_into(state, ...)``
inside an ``@st.fragment(run_every=N)`` block. All ``LiveLifecycleState``
mutations happen on the main thread; the queue is the only cross-thread
channel.
"""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
import time
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from ama_kbqa.frontend.utils.lifecycle_mapping import span_to_node_ids


_LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# State container (lives in st.session_state on the main thread)
# ---------------------------------------------------------------------------

@dataclass
class LiveLifecycleState:
    trace_id: Optional[str] = None
    status: str = "idle"  # "idle" | "running" | "done" | "error"
    active_node_ids: set[str] = field(default_factory=set)
    visited_node_ids: set[str] = field(default_factory=set)
    current_label: str = ""
    span_count: int = 0
    answer: Optional[str] = None
    error: Optional[str] = None
    started_at: Optional[float] = None
    finished_at: Optional[float] = None

    # Active interval-span stack: list of (span_id, node_ids). When a span
    # closes we pop it and recompute ``active_node_ids`` as the union of the
    # remaining entries. Point-in-time events (``is_event=True``) don't push
    # onto the stack; they get a brief flash via ``transient_event_nodes``.
    _active_stack: list[tuple[str, list[str]]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Background worker
# ---------------------------------------------------------------------------

def start_run(
    *,
    agent: Any,
    question: str,
    q: queue.Queue,
    capture_io=None,
) -> threading.Thread:
    """Spawn a daemon thread that runs the agent and streams notifications.

    ``capture_io`` is an optional stdout capture (e.g. ``StreamlitHTMLCapture``
    from ``styling.py``). When provided, the worker wraps ``ask()`` in
    ``redirect_stdout(capture_io)`` so the existing live-log strip keeps
    working.
    """
    recorder = getattr(agent, "recorder", None)

    def listener(phase: str, info: dict) -> None:
        # Only the bits we need on the consumer side. Strip large payloads.
        try:
            q.put_nowait((
                phase,
                {
                    "kind": info.get("kind"),
                    "name": info.get("name"),
                    "span_id": info.get("span_id"),
                    "parent_span_id": info.get("parent_span_id"),
                    "status": info.get("status", "ok"),
                    "is_event": info.get("is_event", False),
                    "attributes": info.get("attributes") or {},
                },
            ))
        except Exception:
            _LOG.exception("Failed to enqueue trace notification")

    def worker() -> None:
        if recorder is not None:
            recorder.add_listener(listener)
        loop = asyncio.new_event_loop()
        try:
            if capture_io is not None:
                with redirect_stdout(capture_io):
                    answer = loop.run_until_complete(agent.ask(question))
            else:
                answer = loop.run_until_complete(agent.ask(question))
            q.put(("__done__", {"answer": answer}))
        except BaseException as exc:  # noqa: BLE001
            _LOG.exception("Lifecycle agent run failed")
            try:
                q.put(("__error__", {"message": f"{type(exc).__name__}: {exc}"}))
            except Exception:
                pass
        finally:
            if recorder is not None:
                recorder.remove_listener(listener)
            try:
                loop.close()
            except Exception:
                pass

    t = threading.Thread(target=worker, daemon=True, name="lifecycle-runner")
    t.start()
    return t


# ---------------------------------------------------------------------------
# Drain — called by the Streamlit fragment on every tick
# ---------------------------------------------------------------------------

def _recompute_active(state: LiveLifecycleState) -> None:
    active: set[str] = set()
    for _sid, node_ids in state._active_stack:
        active.update(node_ids)
    state.active_node_ids = active


def _describe(kind: str, name: str) -> str:
    if not name or name == kind:
        return kind
    return f"{kind} · {name}"


def drain_into(
    q: queue.Queue,
    state: LiveLifecycleState,
    mapping_fn: Callable[..., list[str]] = span_to_node_ids,
) -> bool:
    """Drain queued notifications into ``state``. Returns True when terminal.

    Idempotent and safe to call repeatedly from the main thread.
    """
    if state.status == "idle":
        state.status = "running"
        state.started_at = state.started_at or time.time()

    terminal = state.status in ("done", "error")

    while True:
        try:
            phase, info = q.get_nowait()
        except queue.Empty:
            break

        if phase == "__done__":
            state.status = "done"
            state.answer = info.get("answer")
            state.finished_at = time.time()
            # Settle: drop active highlights, keep visited.
            state._active_stack.clear()
            _recompute_active(state)
            terminal = True
            continue

        if phase == "__error__":
            state.status = "error"
            state.error = info.get("message")
            state.finished_at = time.time()
            state._active_stack.clear()
            _recompute_active(state)
            terminal = True
            continue

        kind = info.get("kind") or ""
        name = info.get("name") or ""
        attrs = info.get("attributes") or {}

        node_ids = mapping_fn(kind, name, phase=phase, attributes=attrs)
        if node_ids:
            state.visited_node_ids.update(node_ids)

        state.span_count += 1

        if phase == "open" and not info.get("is_event", False):
            span_id = info.get("span_id") or ""
            state._active_stack.append((span_id, node_ids))
            state.current_label = _describe(kind, name)
            _recompute_active(state)
        elif phase == "close":
            span_id = info.get("span_id") or ""
            # Pop the matching span (search from end — should be the top in
            # nearly all cases since spans nest cleanly).
            for i in range(len(state._active_stack) - 1, -1, -1):
                if state._active_stack[i][0] == span_id:
                    state._active_stack.pop(i)
                    break
            # Close-time mapping may light NEW nodes (e.g. classify-close
            # lights Strategy Inject). Treat them as visited; not active.
            _recompute_active(state)
            if not state._active_stack:
                state.current_label = ""
        elif phase == "event":
            # Point-in-time event: brief label change, no stack push.
            state.current_label = _describe(kind, name)

    return terminal
