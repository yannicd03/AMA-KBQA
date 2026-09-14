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

Besides the single-agent lifecycle figure, the state also tracks the
**orchestrator** view: when the run is the multi-agent Orchestrator, each
``delegate`` span lights a sub-agent container in the orchestrator figure and
spins up a per-sub-agent :class:`SubAgentLifecycle` (driven by the sub-agent's
own nested spans) so its detailed Pre→Main→Post figure can be shown in a pane
underneath. Single-agent runs simply leave those structures empty.
"""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
import time
from collections import OrderedDict
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional

from ama_kbqa.frontend.utils.lifecycle_mapping import (
    ORCH_SUBAGENTS,
    lifecycle_node_phase,
    orchestrator_span_to_node_ids,
    span_to_edge_ids,
    span_to_node_ids,
)


_LOG = logging.getLogger(__name__)


# Minimum time (seconds) a node/edge stays lit once it becomes active. Spans
# can open and close faster than the UI polls (every ~0.4s), so without a floor
# a stage would flash for a single frame — or be skipped entirely when its
# whole span lands between two ticks. Holding each activation for at least this
# long makes the figure step cleanly through every stage.
MIN_LIGHTUP_SECONDS = 0.5


def _no_edge_ids(*_args: Any, **_kwargs: Any) -> list[str]:
    """Edge mapping for figures that have no live-highlighted edges."""
    return []


# ---------------------------------------------------------------------------
# Figure state (shared by the main lifecycle figure, the orchestrator figure,
# and each sub-agent's detail figure)
# ---------------------------------------------------------------------------

@dataclass
class _FigureState:
    """Active/visited bookkeeping for one lit figure.

    A figure tracks an *active span stack* (interval spans currently open),
    the set of *visited* nodes (ever lit), and per-node/edge minimum-lightup
    deadlines so fast spans stay visible for at least ``MIN_LIGHTUP_SECONDS``.
    """

    active_node_ids: set[str] = field(default_factory=set)
    visited_node_ids: set[str] = field(default_factory=set)
    current_label: str = ""
    span_count: int = 0

    # Active interval-span stack: list of (span_id, node_ids). When a span
    # closes we pop it and recompute ``active_node_ids`` as the union of the
    # remaining entries. Point-in-time events don't push onto the stack.
    _active_stack: list[tuple[str, list[str]]] = field(default_factory=list)

    # Minimum-lightup bookkeeping: node-id / edge-id → wall-clock time until
    # which it must keep rendering as *active*, even after its span has closed.
    _node_lit_until: dict[str, float] = field(default_factory=dict)
    _edge_lit_until: dict[str, float] = field(default_factory=dict)

    def render_active_node_ids(self, now: Optional[float] = None) -> set[str]:
        """Node ids to render as *active* right now.

        The union of nodes currently on the live span stack and nodes whose
        minimum-lightup window has not yet elapsed. The hold lets fast spans
        (which open and close between UI ticks) stay visibly lit for at least
        ``MIN_LIGHTUP_SECONDS`` so the figure walks cleanly through each stage.
        """
        if now is None:
            now = time.time()
        held = {nid for nid, until in self._node_lit_until.items() if until > now}
        return self.active_node_ids | held

    def render_active_edge_ids(self, now: Optional[float] = None) -> set[str]:
        """Edge ids to render as *active* right now (held for the same floor)."""
        if now is None:
            now = time.time()
        return {eid for eid, until in self._edge_lit_until.items() if until > now}


@dataclass
class SubAgentLifecycle(_FigureState):
    """One dispatched sub-agent's detailed lifecycle (Fig. 1) state.

    Built lazily when the orchestrator opens a ``delegate`` span. The
    sub-agent's own nested spans (attributed via ``parent_span_id``) drive this
    figure exactly like a standalone single-agent run.
    """

    agent_id: str = ""        # e.g. "kqapro_agent"
    display: str = ""         # e.g. "KQAPro"
    container_id: str = ""    # orchestrator-figure container node, e.g. "sub_kqapro"
    status: str = "running"   # "running" | "done" | "error"

    def current_phase(self, now: Optional[float] = None) -> Optional[str]:
        """Coarse phase (``pre``/``main``/``post``) currently in flight.

        Drives the Pre/Main/Post mini-boxes inside this sub-agent's container
        in the orchestrator figure. Prefers the most-advanced active phase.
        """
        phases = {
            lifecycle_node_phase(nid)
            for nid in self.render_active_node_ids(now)
        }
        for ph in ("post", "main", "pre"):
            if ph in phases:
                return ph
        return None


def orchestrator_visited_node_ids(
    orch_visited: set[str],
    subagents: "OrderedDict[str, SubAgentLifecycle]",
) -> set[str]:
    """Full visited set for the orchestrator figure: orchestrator-level nodes
    plus each dispatched sub-agent's container and the Pre/Main/Post minis its
    lifecycle touched. Shared by the live state method and the frozen view."""
    ids = set(orch_visited)
    for sub in subagents.values():
        ids.add(sub.container_id)
        for nid in sub.visited_node_ids:
            ph = lifecycle_node_phase(nid)
            if ph:
                ids.add(f"{sub.container_id}_{ph}")
    return ids


# ---------------------------------------------------------------------------
# State container (lives in st.session_state on the main thread)
# ---------------------------------------------------------------------------

@dataclass
class LiveLifecycleState(_FigureState):
    """Live state for one run.

    Inherits the single-agent lifecycle figure bookkeeping from
    :class:`_FigureState` (``active_node_ids`` / ``visited_node_ids`` /
    ``render_active_node_ids`` / ...). Adds run status + answer, the
    orchestrator figure (``orch``), and per-sub-agent detail figures
    (``subagents``) used only in Orchestrator mode.
    """

    trace_id: Optional[str] = None
    status: str = "idle"  # "idle" | "running" | "done" | "error"
    answer: Optional[str] = None
    error: Optional[str] = None
    started_at: Optional[float] = None
    finished_at: Optional[float] = None

    # Orchestrator figure (User Query / Probing / Dispatch / Combination +
    # sub-agent containers). Empty for single-agent runs.
    orch: _FigureState = field(default_factory=_FigureState)
    # Per-sub-agent detail lifecycles, keyed by agent id, in dispatch order.
    subagents: "OrderedDict[str, SubAgentLifecycle]" = field(default_factory=OrderedDict)

    # Span attribution: delegate span_id → sub-agent id; span_id → owning
    # sub-agent id (or None for orchestrator-level spans).
    _delegate_owner: dict[str, str] = field(default_factory=dict)
    _span_owner: dict[str, Optional[str]] = field(default_factory=dict)

    def render_orchestrator_active_node_ids(self, now: Optional[float] = None) -> set[str]:
        """Active node ids for the orchestrator figure right now.

        Orchestrator-level active/held nodes, plus — for each *running*
        sub-agent — the Pre/Main/Post mini matching its current phase.
        """
        if now is None:
            now = time.time()
        ids = self.orch.render_active_node_ids(now)
        for sub in self.subagents.values():
            if sub.status == "running":
                phase = sub.current_phase(now)
                if phase:
                    ids.add(f"{sub.container_id}_{phase}")
        return ids

    def render_orchestrator_visited_node_ids(self) -> set[str]:
        """Visited node ids for the orchestrator figure (containers + minis)."""
        return orchestrator_visited_node_ids(self.orch.visited_node_ids, self.subagents)


# ---------------------------------------------------------------------------
# Background worker
# ---------------------------------------------------------------------------

def start_run(
    *,
    agent: Any,
    question: str,
    q: queue.Queue,
    capture_io=None,
    is_continuation: bool = False,
) -> threading.Thread:
    """Spawn a daemon thread that runs the agent and streams notifications.

    ``capture_io`` is an optional stdout capture (e.g. ``StreamlitHTMLCapture``
    from ``styling.py``). When provided, the worker wraps ``ask()`` in
    ``redirect_stdout(capture_io)`` so the existing live-log strip keeps
    working.

    ``is_continuation`` marks a follow-up turn on a reused agent instance
    (multiturn). When set, the worker first runs ``reset(keep_history=True)``
    so the agent keeps its accumulated message stack but gets a fresh trace,
    fresh token counts, and a freshly re-initialised MCP connection on *this*
    worker's event loop. The reset runs before the recorder is captured so the
    live-trace listener attaches to the new turn's recorder, not the previous
    turn's.
    """

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
        loop = asyncio.new_event_loop()
        recorder = None
        try:
            # Follow-up turn: reset per-turn state (fresh trace/tokens) while
            # preserving the conversation stack. The previous turn's MCP
            # connection is bound to a now-closed event loop, so we orphan it
            # (its loop is dead; awaiting close() on it from this new loop can
            # error) and reset with keep_mcp_open=True. agent.ask() then calls
            # _init_mcp, which rebuilds a fresh MCP connection on THIS loop.
            if is_continuation:
                agent.mcp = None
                loop.run_until_complete(
                    agent.reset(keep_history=True, keep_mcp_open=True)
                )

            # Capture the recorder AFTER any reset so the listener binds to the
            # live recorder. Tell the main thread the new trace_id so the live
            # panel does not point at the previous turn's trace.
            recorder = getattr(agent, "recorder", None)
            if recorder is not None:
                recorder.add_listener(listener)
                tid = getattr(recorder, "trace_id", None)
                if tid is not None:
                    try:
                        q.put_nowait(("__trace_id__", {"trace_id": tid}))
                    except Exception:
                        pass

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

def _recompute_active_into(target: _FigureState) -> None:
    active: set[str] = set()
    for _sid, node_ids in target._active_stack:
        active.update(node_ids)
    target.active_node_ids = active


def _recompute_active(state: LiveLifecycleState) -> None:
    _recompute_active_into(state)


def _hold(store: dict[str, float], ids: Iterable[str], now: float) -> None:
    """Extend the minimum-lightup window for ``ids`` to at least ``now + floor``."""
    until = now + MIN_LIGHTUP_SECONDS
    for i in ids:
        if until > store.get(i, 0.0):
            store[i] = until


def _describe(kind: str, name: str) -> str:
    if not name or name == kind:
        return kind
    return f"{kind} · {name}"


def _apply_span(
    target: _FigureState,
    kind: str,
    name: str,
    phase: str,
    attrs: dict,
    span_id: str,
    is_event: bool,
    now: float,
    mapping_fn: Callable[..., list[str]],
    edge_mapping_fn: Callable[..., list[str]],
) -> None:
    """Apply one span/event notification to a single figure's state.

    Shared by the main lifecycle figure, the orchestrator figure, and each
    sub-agent detail figure — only the mapping functions differ.
    """
    node_ids = mapping_fn(kind, name, phase=phase, attributes=attrs)
    if node_ids:
        target.visited_node_ids.update(node_ids)
        # Hold every activated node lit for the minimum window, even the
        # close-time / event-time ones that never join the live stack (e.g.
        # classify-close → Strategy Injection, a loop-iter event → LLM
        # Reasoning). Without this they would only ever show as the dimmer
        # "visited" colour.
        _hold(target._node_lit_until, node_ids, now)

    edge_ids = edge_mapping_fn(kind, name, phase=phase, attributes=attrs)
    if edge_ids:
        _hold(target._edge_lit_until, edge_ids, now)

    target.span_count += 1

    if phase == "open" and not is_event:
        target._active_stack.append((span_id, node_ids))
        target.current_label = _describe(kind, name)
        _recompute_active_into(target)
    elif phase == "close":
        # Pop the matching span (search from end — should be the top in nearly
        # all cases since spans nest cleanly).
        for i in range(len(target._active_stack) - 1, -1, -1):
            if target._active_stack[i][0] == span_id:
                target._active_stack.pop(i)
                break
        # Close-time mapping may light NEW nodes (e.g. classify-close lights
        # Strategy Inject). Treat them as visited; not active.
        _recompute_active_into(target)
        if not target._active_stack:
            target.current_label = ""
    elif phase == "event":
        # Point-in-time event: brief label change, no stack push.
        target.current_label = _describe(kind, name)


def _owner_for(
    state: LiveLifecycleState,
    kind: str,
    phase: str,
    span_id: str,
    parent: Optional[str],
) -> Optional[str]:
    """Resolve which sub-agent (if any) a span belongs to.

    A ``delegate`` span is itself orchestrator-level (returns None) but marks
    its descendants as belonging to its sub-agent. Any other span inherits the
    owner of its parent: directly if the parent is a delegate, otherwise the
    parent's recorded owner.
    """
    if kind == "delegate":
        return None
    if phase == "close":
        return state._span_owner.get(span_id)
    if parent in state._delegate_owner:
        return state._delegate_owner[parent]
    return state._span_owner.get(parent)


def drain_into(
    q: queue.Queue,
    state: LiveLifecycleState,
    mapping_fn: Callable[..., list[str]] = span_to_node_ids,
    edge_mapping_fn: Callable[..., list[str]] = span_to_edge_ids,
) -> bool:
    """Drain queued notifications into ``state``. Returns True when terminal.

    Idempotent and safe to call repeatedly from the main thread. Each
    notification updates the single-agent lifecycle figure (always) and, when
    the run is the Orchestrator, the orchestrator figure + the relevant
    sub-agent's detail figure.
    """
    if state.status == "idle":
        state.status = "running"
        state.started_at = state.started_at or time.time()

    terminal = state.status in ("done", "error")

    # One timestamp for the whole drain: every activation seen this tick holds
    # for at least MIN_LIGHTUP_SECONDS from now, so nothing flashes sub-frame.
    now = time.time()

    while True:
        try:
            phase, info = q.get_nowait()
        except queue.Empty:
            break

        if phase == "__trace_id__":
            # The worker (re)created the recorder for this turn; adopt its
            # trace_id so the live panel tracks the correct trace.
            tid = info.get("trace_id")
            if tid:
                state.trace_id = tid
            continue

        if phase in ("__done__", "__error__"):
            if phase == "__done__":
                state.status = "done"
                state.answer = info.get("answer")
            else:
                state.status = "error"
                state.error = info.get("message")
            state.finished_at = time.time()
            # Settle: drop active highlights, keep visited — on every figure.
            state._active_stack.clear()
            _recompute_active(state)
            state.orch._active_stack.clear()
            _recompute_active_into(state.orch)
            for sub in state.subagents.values():
                if sub.status == "running":
                    sub.status = "error" if phase == "__error__" else "done"
                sub._active_stack.clear()
                _recompute_active_into(sub)
            terminal = True
            continue

        kind = info.get("kind") or ""
        name = info.get("name") or ""
        attrs = info.get("attributes") or {}
        span_id = info.get("span_id") or ""
        parent = info.get("parent_span_id")
        is_event = bool(info.get("is_event", False))

        # 1) Single-agent lifecycle figure: every span feeds it, exactly as
        #    before. In Orchestrator mode this figure is simply not shown.
        _apply_span(
            state, kind, name, phase, attrs, span_id, is_event, now,
            mapping_fn, edge_mapping_fn,
        )

        # 2) Orchestrator figure + per-sub-agent detail figures (additive).
        if kind == "delegate" and phase == "open":
            sub_id = attrs.get("sub_agent") or ""
            state._delegate_owner[span_id] = sub_id
            state._span_owner[span_id] = None
            if sub_id in ORCH_SUBAGENTS and sub_id not in state.subagents:
                display, container_id = ORCH_SUBAGENTS[sub_id]
                state.subagents[sub_id] = SubAgentLifecycle(
                    agent_id=sub_id, display=display, container_id=container_id,
                )
            owner: Optional[str] = None
        else:
            owner = _owner_for(state, kind, phase, span_id, parent)
            if phase == "open":
                state._span_owner[span_id] = owner

        if owner is None:
            _apply_span(
                state.orch, kind, name, phase, attrs, span_id, is_event, now,
                orchestrator_span_to_node_ids, _no_edge_ids,
            )
        else:
            sub = state.subagents.get(owner)
            if sub is not None:
                _apply_span(
                    sub, kind, name, phase, attrs, span_id, is_event, now,
                    span_to_node_ids, span_to_edge_ids,
                )

        if kind == "delegate" and phase == "close":
            sub_id = state._delegate_owner.get(span_id)
            sub = state.subagents.get(sub_id) if sub_id else None
            if sub is not None:
                sub.status = "error" if info.get("status") == "error" else "done"
                sub._active_stack.clear()
                _recompute_active_into(sub)

    return terminal


# ---------------------------------------------------------------------------
# Frozen-view reconstruction (from a completed trace's event dicts)
# ---------------------------------------------------------------------------

def reconstruct_orchestrator(
    events: Iterable[dict],
) -> tuple[set[str], "OrderedDict[str, SubAgentLifecycle]"]:
    """Rebuild the orchestrator figure's visited set + per-sub-agent figures
    from a completed trace's event dicts (``TraceRecorder.to_dicts()``).

    Used by the frozen "completed" inspector view, where the live ``state`` is
    gone. Returns ``(orch_visited_node_ids, subagents_in_dispatch_order)``;
    each sub-agent's ``visited_node_ids`` is the union of every node its spans
    touched (open + close + event), so its detail figure reads as fully run.
    """
    events = list(events)
    span_parent: dict[str, Optional[str]] = {}
    delegate_owner: dict[str, str] = {}
    for e in events:
        sid = e.get("span_id") or ""
        span_parent[sid] = e.get("parent_span_id")
        if e.get("kind") == "delegate":
            delegate_owner[sid] = (e.get("attributes") or {}).get("sub_agent") or ""

    def owner_of(sid: Optional[str]) -> Optional[str]:
        cur, seen = sid, 0
        while cur is not None and seen < 256:
            if cur in delegate_owner:
                return delegate_owner[cur]
            cur = span_parent.get(cur)
            seen += 1
        return None

    orch_visited: set[str] = set()
    subs: "OrderedDict[str, SubAgentLifecycle]" = OrderedDict()

    def ensure_sub(agent_id: str, status: str = "done") -> Optional[SubAgentLifecycle]:
        if agent_id not in ORCH_SUBAGENTS:
            return None
        sub = subs.get(agent_id)
        if sub is None:
            display, container_id = ORCH_SUBAGENTS[agent_id]
            sub = SubAgentLifecycle(
                agent_id=agent_id, display=display,
                container_id=container_id, status=status,
            )
            subs[agent_id] = sub
        return sub

    for e in events:
        kind = e.get("kind") or ""
        name = e.get("name") or ""
        attrs = e.get("attributes") or {}
        sid = e.get("span_id") or ""
        is_event = bool(e.get("is_event", False))
        phases = ("event",) if is_event else ("open", "close")

        if kind == "delegate":
            agent_id = attrs.get("sub_agent") or ""
            ensure_sub(agent_id, "error" if e.get("status") == "error" else "done")
            for nid in orchestrator_span_to_node_ids(kind, name, phase="open", attributes=attrs):
                orch_visited.add(nid)
            continue

        owner = owner_of(sid)
        if owner is None:
            for ph in phases:
                for nid in orchestrator_span_to_node_ids(kind, name, phase=ph, attributes=attrs):
                    orch_visited.add(nid)
        else:
            sub = ensure_sub(owner)
            if sub is not None:
                for ph in phases:
                    for nid in span_to_node_ids(kind, name, phase=ph, attributes=attrs):
                        sub.visited_node_ids.add(nid)

    return orch_visited, subs
