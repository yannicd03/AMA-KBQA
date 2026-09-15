"""Run and session bookkeeping for the API: the server-side twin of the
Streamlit chat page's ``live_run`` / ``persistent_agent`` session state.

One :class:`Run` per question. Its agent runs in the lifecycle worker thread
(``lifecycle_runner.start_run``), exactly as on the Streamlit page, and a
single asyncio consumer task per run drains the worker's queue with
``drain_into`` every ``TICK_SECONDS`` and *publishes* immutable snapshots.
SSE handlers only ever read those snapshots; nothing but the consumer touches
the queue or the ``LiveLifecycleState`` (a queue drained by two readers would
lose notifications at random, the same rule ``live_graph_panel`` follows).

Everything lives in process memory, hence the single-worker requirement.
"""

from __future__ import annotations

import asyncio
import json
import logging
import queue
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from ama_kbqa.api import meta, stdout_router
from ama_kbqa.api.stdout_router import RunLog, route_current_context
from ama_kbqa.config import get_live_graph_enabled
from ama_kbqa.frontend.utils.agent_factory import (
    AGENT_INFO,
    ORCHESTRATOR_MODES,
    create_agent,
    is_orchestrator,
)
from ama_kbqa.frontend.utils.chat_controls import apply_chat_settings
from ama_kbqa.frontend.utils.lifecycle_runner import (
    LiveLifecycleState,
    drain_into,
    live_graph_snapshot,
    orchestrator_visited_node_ids,
    reconstruct_orchestrator,
    start_run,
)
from ama_kbqa.frontend.utils.lifecycle_svg import LIFECYCLE_NODES, render_lifecycle_svg
from ama_kbqa.frontend.utils.live_graph_data import (
    GraphData,
    answer_node_ids,
    apply_caps,
    graph_from_trace,
    source_for_agent,
    to_vis_payload,
)
from ama_kbqa.frontend.utils.orchestrator_svg import render_orchestrator_svg
from ama_kbqa.frontend.utils.styling import ansi_to_html
from ama_kbqa.frontend.utils.trace_render import summarise
from ama_kbqa.pricing import estimate_cost_usd, format_cost_usd

_LOG = logging.getLogger(__name__)

# Consumer cadence, same as the Streamlit fragment (run_every=0.4).
TICK_SECONDS = 0.4
# A tick that changed nothing but the clock still republishes this often, so
# the "elapsed" counter keeps moving while the agent waits on an LLM call.
HEARTBEAT_SECONDS = 1.0

MAX_RUNS = 200
SESSION_IDLE_SECONDS = 2 * 60 * 60

# The live log shown in the UI is the tail of the captured stdout.
LOG_TAIL_CHARS = 100_000
_LOG_TRUNCATED_MARKER = (
    '<span style="color:#6e7681">… earlier output truncated …</span>\n'
)

# /trace payload strings are cut to this many characters. A SPARQL tool
# result can be megabytes; the span tree only needs enough to read it.
TRACE_STRING_LIMIT = 20_000
TRUNCATION_SUFFIX = "…[truncated]"

# Same cap as the Streamlit panel's "Nodes in the answer" list.
MAX_ANSWER_NODES = 20


class ApiError(Exception):
    """An error the routes turn into ``{"detail": ...}`` with this status."""

    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


# ---------------------------------------------------------------------------
# Pure helpers (render / payload shaping)
# ---------------------------------------------------------------------------

def orchestrator_mode(agent_name: str) -> str:
    return "federated" if ORCHESTRATOR_MODES.get(agent_name) else "router"


def log_tail_html(raw: str, *, head_dropped: bool = False) -> str:
    """``ansi_to_html`` of the last ~``LOG_TAIL_CHARS`` of ``raw``.

    The cut lands on a line boundary, so it never splits an ANSI escape
    sequence (those never span a newline). An unmatched ``</span>`` or
    ``<span>`` at the edge of the cut is harmless once the browser parses it.
    """
    prefix = _LOG_TRUNCATED_MARKER if head_dropped else ""
    if len(raw) > LOG_TAIL_CHARS:
        tail = raw[-LOG_TAIL_CHARS:]
        newline = tail.find("\n")
        if newline != -1:
            tail = tail[newline + 1:]
        raw = tail
        prefix = _LOG_TRUNCATED_MARKER
    return prefix + ansi_to_html(raw)


def graph_stats(graph: GraphData) -> dict[str, int]:
    counts = graph.stats()
    return {
        "nodes": len(graph.nodes),
        "edges": len(graph.edges),
        "truncated": graph.truncated_nodes,
        "entities": counts["entities"],
        "literals": counts["literals"],
        "candidates": counts["candidates"],
    }


def _truncate_strings(value: Any, limit: int) -> Any:
    if isinstance(value, str):
        return value if len(value) <= limit else value[:limit] + TRUNCATION_SUFFIX
    if isinstance(value, dict):
        return {k: _truncate_strings(v, limit) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_truncate_strings(v, limit) for v in value]
    return value


def truncate_trace_events(events: list[dict], limit: int = TRACE_STRING_LIMIT) -> list[dict]:
    """Recorder event dicts with every string inside ``payload`` capped at
    ``limit`` characters, made JSON-safe (unknown objects become ``str``)."""
    out = []
    for event in events:
        event = dict(event)
        if "payload" in event:
            event["payload"] = _truncate_strings(event["payload"], limit)
        out.append(event)
    return json.loads(json.dumps(out, default=str, skipkeys=True))


def _live_subagent(sub: Any, now: float) -> dict[str, Any]:
    return {
        "id": sub.agent_id,
        "display": sub.display,
        "status": sub.status,
        "svg": render_lifecycle_svg(
            sub.render_active_node_ids(now),
            sub.visited_node_ids,
            current_label=sub.current_label or None,
            active_edge_ids=sub.render_active_edge_ids(now),
        ),
    }


def live_body(agent_name: str, state: LiveLifecycleState, now: float) -> dict[str, Any]:
    """The non-diffed part of a live snapshot (mirrors chat.py's _live_tick)."""
    if is_orchestrator(agent_name):
        svg = render_orchestrator_svg(
            state.render_orchestrator_active_node_ids(now),
            state.render_orchestrator_visited_node_ids(),
            current_label=state.current_label or "starting…",
            active_edge_ids=state.orch.render_active_edge_ids(now),
            mode=orchestrator_mode(agent_name),
        )
        figure = {"kind": "orchestrator", "svg": svg}
        subagents = [_live_subagent(sub, now) for sub in state.subagents.values()]
    else:
        svg = render_lifecycle_svg(
            state.render_active_node_ids(now),
            state.visited_node_ids,
            current_label=state.current_label or "starting…",
            active_edge_ids=state.render_active_edge_ids(now),
        )
        figure = {"kind": "lifecycle", "svg": svg}
        subagents = []
    started = state.started_at or now
    return {
        # A run is "running" from the moment it is accepted; the state only
        # leaves "idle" on its first drain.
        "status": "running" if state.status == "idle" else state.status,
        "stage": state.current_label or "",
        "elapsed_s": round(max(0.0, now - started), 1),
        "span_count": state.span_count,
        "figure": figure,
        "subagents": subagents,
    }


def _body_signature(body: dict[str, Any]) -> tuple:
    """Everything in a body except the clock, for change detection."""
    return (
        body["status"],
        body["stage"],
        body["span_count"],
        body["figure"]["svg"],
        tuple((s["id"], s["status"], s["svg"]) for s in body["subagents"]),
    )


def frozen_figure(agent_name: str, events: list[dict], duration: float) -> tuple[dict, list]:
    """The completed-run figure, exactly like chat.py's frozen inspector view."""
    caption = f"completed in {duration:.1f}s"
    if is_orchestrator(agent_name):
        orch_visited, subs = reconstruct_orchestrator(events)
        svg = render_orchestrator_svg(
            set(),
            orchestrator_visited_node_ids(orch_visited, subs),
            current_label=caption,
            mode=orchestrator_mode(agent_name),
        )
        subagents = [
            {
                "id": sub.agent_id,
                "display": sub.display,
                "status": sub.status,
                "svg": render_lifecycle_svg(set(), sub.visited_node_ids),
            }
            for sub in subs.values()
        ]
        return {"kind": "orchestrator", "svg": svg}, subagents
    visited = {n.id for n in LIFECYCLE_NODES}
    svg = render_lifecycle_svg(set(), visited, current_label=caption)
    return {"kind": "lifecycle", "svg": svg}, []


def frozen_graph(trace: dict) -> Optional[dict[str, Any]]:
    """The completed-run subgraph (live_graph_panel's frozen view), or None
    when the trace yields nothing to draw."""
    try:
        graph = apply_caps(graph_from_trace(trace))
    except Exception:  # noqa: BLE001 - a broken trace must not break the run
        _LOG.debug("Frozen graph rebuild failed", exc_info=True)
        return None
    if graph.is_empty():
        return None
    highlight = answer_node_ids(graph, trace.get("answer"))
    payload = to_vis_payload(graph, highlight=highlight, new_ids=set())
    payload["stats"] = graph_stats(graph)
    payload["highlight"] = sorted(highlight)
    payload["answer_nodes"] = sorted(
        graph.nodes[node_id].label for node_id in highlight if node_id in graph.nodes
    )[:MAX_ANSWER_NODES]
    return payload


def _json_frame(event: str, seq: int, data: dict) -> str:
    body = json.dumps(data, ensure_ascii=False, separators=(",", ":"), default=str)
    return f"id: {seq}\nevent: {event}\ndata: {body}\n\n"


# ---------------------------------------------------------------------------
# State containers
# ---------------------------------------------------------------------------

@dataclass
class StoredAgent:
    """A session's persisted multiturn agent (specialists only)."""

    agent_name: str
    model: str
    agent: Any


@dataclass
class Session:
    session_id: str
    stored: Optional[StoredAgent] = None
    active_run_id: Optional[str] = None
    last_seen: float = field(default_factory=time.time)
    demo_query_count: int = 0
    demo_last_query_time: float = 0.0


@dataclass(frozen=True)
class Snapshot:
    """One published live state. Immutable: SSE handlers share it."""

    seq: int
    body: dict
    log_version: int
    log_html: str
    graph: Optional[GraphData]
    graph_key: Any


class _Connection:
    """Per-SSE-connection memory, so unchanged log/graph are sent as null."""

    _UNSET = object()

    def __init__(self) -> None:
        self.log_version: Any = self._UNSET
        self.graph_key: Any = self._UNSET
        self.graph_ids: set[str] = set()

    def snapshot_data(self, snap: Snapshot) -> dict[str, Any]:
        data = dict(snap.body)
        data["seq"] = snap.seq
        if snap.log_version != self.log_version:
            data["log_html"] = snap.log_html
            self.log_version = snap.log_version
        else:
            data["log_html"] = None
        if snap.graph is not None and snap.graph_key != self.graph_key:
            ids = set(snap.graph.nodes)
            new_ids = ids - self.graph_ids
            payload = to_vis_payload(snap.graph, highlight=set(), new_ids=new_ids)
            payload["new_ids"] = [nid for nid in snap.graph.nodes if nid in new_ids]
            payload["stats"] = graph_stats(snap.graph)
            data["graph"] = payload
            self.graph_key = snap.graph_key
            self.graph_ids = ids
        else:
            data["graph"] = None
        return data


class Run:
    """One question, from acceptance to its ``done``/``error`` event."""

    def __init__(
        self,
        *,
        session_id: str,
        agent_name: str,
        question: str,
        model: str,
        temperature: float,
        live_graph: bool,
    ) -> None:
        self.run_id = uuid.uuid4().hex
        self.session_id = session_id
        self.agent_name = agent_name
        self.question = question
        self.model = model
        self.temperature = temperature
        self.live_graph = live_graph
        self.continuation = False
        self.status = "running"
        self.started_at = time.time()
        self.started_at_iso = datetime.now().isoformat(timespec="seconds")

        self.agent: Any = None
        self.queue: Optional[queue.Queue] = queue.Queue()
        self.state: Optional[LiveLifecycleState] = None
        self.log = RunLog()
        self.task: Optional[asyncio.Task] = None

        self.snapshot: Optional[Snapshot] = None
        self.final_event: Optional[tuple[str, dict]] = None
        self.final_seq = 0
        self.trace_id: Optional[str] = None
        self.trace_events: Optional[list[dict]] = None

        self._seq = 0
        self._log_html = ""
        self._log_html_version = -1
        self._changed = asyncio.Event()

    # -- publication (event-loop thread only) --------------------------------

    def _notify(self) -> None:
        # Swap before setting so a waiter that re-arms after waking always
        # gets a fresh, unset event.
        old, self._changed = self._changed, asyncio.Event()
        old.set()

    def change_event(self) -> asyncio.Event:
        return self._changed

    def publish(self, body: dict, graph: Optional[GraphData], graph_key: Any,
                log_version: int, log_html: str) -> None:
        self._seq += 1
        self.snapshot = Snapshot(
            seq=self._seq,
            body=body,
            log_version=log_version,
            log_html=log_html,
            graph=graph,
            graph_key=graph_key,
        )
        self._notify()

    def finish(self, kind: str, payload: dict) -> None:
        self.status = kind
        self.final_seq = self._seq + 1
        self.final_event = (kind, payload)
        # The frozen payload holds everything the UI still needs; drop the
        # heavy live objects (the session keeps a specialist agent itself).
        self.agent = None
        self.queue = None
        self.state = None
        self._notify()

    @property
    def finished(self) -> bool:
        return self.final_event is not None

    # -- views ----------------------------------------------------------------

    def record(self) -> dict[str, Any]:
        rec = {
            "run_id": self.run_id,
            "session_id": self.session_id,
            "agent": self.agent_name,
            "question": self.question,
            "model": self.model,
            "continuation": self.continuation,
            "status": self.status,
            "started_at": self.started_at_iso,
        }
        if self.final_event is not None:
            rec.update(self.final_event[1])
        return rec

    def final_frame(self) -> str:
        assert self.final_event is not None
        kind, payload = self.final_event
        return _json_frame(kind, self.final_seq, payload)

    def trace_payload(self) -> dict[str, Any]:
        events = self.trace_events or []
        return {
            "trace_id": self.trace_id,
            "summary": summarise(events),
            "events": events,
        }


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------

class RunManager:
    """Owns every session and run of this process."""

    def __init__(self) -> None:
        self.sessions: dict[str, Session] = {}
        self.runs: "OrderedDict[str, Run]" = OrderedDict()
        # apply_chat_settings mutates process-global config/env and
        # create_agent reads it, so the pair must not interleave between two
        # requests. Known limitation (same as Streamlit): the override stays
        # process-global while a run is in flight.
        self._build_lock = threading.Lock()

    # -- lookups ----------------------------------------------------------------

    def get_run(self, run_id: str) -> Run:
        run = self.runs.get(run_id)
        if run is None:
            raise ApiError(404, "Unknown run (it may have been evicted).")
        return run

    # -- eviction ---------------------------------------------------------------

    def evict_idle_sessions(self, now: Optional[float] = None) -> None:
        now = time.time() if now is None else now
        for session_id, session in list(self.sessions.items()):
            if session.active_run_id is None and now - session.last_seen > SESSION_IDLE_SECONDS:
                del self.sessions[session_id]

    def _evict_runs(self) -> None:
        while len(self.runs) > MAX_RUNS:
            oldest_finished = next(
                (rid for rid, run in self.runs.items() if run.finished), None
            )
            if oldest_finished is None:
                return
            del self.runs[oldest_finished]

    # -- sessions ---------------------------------------------------------------

    def reset_session(self, session_id: str) -> None:
        self.evict_idle_sessions()
        session = self.sessions.get(session_id)
        if session is None:
            return
        if session.active_run_id is not None:
            raise ApiError(409, "This session has a run in flight; wait for it to finish.")
        session.stored = None
        session.last_seen = time.time()

    @staticmethod
    def _check_demo_limits(session: Session) -> None:
        demo = meta.demo_settings()
        if not demo["enabled"]:
            return
        max_queries = demo["max_queries_per_session"]
        min_gap = demo["min_seconds_between_queries"]
        if session.demo_query_count >= max_queries:
            raise ApiError(
                429,
                f"Demo limit reached: {max_queries} queries per session. "
                "Refresh the page to start a new session.",
            )
        elapsed = time.time() - session.demo_last_query_time
        if elapsed < min_gap:
            raise ApiError(429, f"Please wait {min_gap - elapsed:.1f}s before the next query.")
        session.demo_query_count += 1
        session.demo_last_query_time = time.time()

    # -- runs -------------------------------------------------------------------

    async def create_run(
        self,
        *,
        session_id: str,
        question: str,
        agent_name: str,
        model: str,
        temperature: float,
    ) -> Run:
        """Validate, build (or reuse) the agent, start it, return the run."""
        if agent_name not in AGENT_INFO:
            raise ApiError(400, f"Unknown agent: {agent_name}")
        if not 0.0 <= temperature <= 2.0:
            raise ApiError(400, "Temperature must be between 0 and 2.")
        if not question.strip():
            raise ApiError(400, "The question must not be empty.")
        model_ids = await asyncio.to_thread(meta.get_model_ids)
        if model not in model_ids:
            raise ApiError(400, f"Unknown model: {model}")

        self.evict_idle_sessions()
        session = self.sessions.get(session_id)
        if session is None:
            session = self.sessions[session_id] = Session(session_id=session_id)
        session.last_seen = time.time()
        if session.active_run_id is not None:
            raise ApiError(409, "This session already has a run in flight.")
        self._check_demo_limits(session)

        run = Run(
            session_id=session_id,
            agent_name=agent_name,
            question=question,
            model=model,
            temperature=temperature,
            live_graph=bool(get_live_graph_enabled()),
        )
        # Reserve the session before the first await below: the event loop is
        # single-threaded, so check-and-set cannot interleave with another POST.
        session.active_run_id = run.run_id
        self.runs[run.run_id] = run
        self._evict_runs()

        try:
            agent, continuation = await asyncio.to_thread(
                self._build_agent, session, agent_name, model, temperature
            )
        except Exception as exc:  # noqa: BLE001 - surfaces as the run's error event
            _LOG.exception("Agent creation failed for %s", agent_name)
            self._release(session, run)
            run.finish("error", {
                "run_id": run.run_id,
                "status": "error",
                "message": f"Could not create the agent: {type(exc).__name__}: {exc}",
                "log_html": "",
            })
            return run

        run.agent = agent
        run.continuation = continuation
        state = LiveLifecycleState(
            trace_id=getattr(getattr(agent, "recorder", None), "trace_id", None),
            started_at=run.started_at,
        )
        # Which graph paints owner-less nodes; "" for an orchestrator pick,
        # where each delegate span's owner decides (same as chat.py).
        state.graph_source_default = source_for_agent(agent)
        run.state = state

        log = run.log
        # The app installs the proxy at startup; re-assert it here (idempotent)
        # in case something replaced sys.stdout since, which would otherwise
        # send this run's log to the replacement instead of its buffer.
        stdout_router.install()
        start_run(
            agent=agent,
            question=question,
            q=run.queue,
            capture_io=None,
            is_continuation=continuation,
            on_thread_start=lambda: route_current_context(log),
        )
        run.publish(*self._render(run))
        run.task = asyncio.create_task(self._consume(run, session))
        return run

    def _build_agent(
        self, session: Session, agent_name: str, model: str, temperature: float
    ) -> tuple[Any, bool]:
        """Mirror chat.py's multiturn rules. Runs in a worker thread.

        Specialists keep ONE agent per session, keyed by (agent, model): the
        same pair again is a follow-up turn on that instance. The Orchestrator
        is stateless, so it is always fresh and drops any stored agent.
        """
        with self._build_lock:
            apply_chat_settings(model, temperature)
            multiturn = not is_orchestrator(agent_name)
            stored = session.stored
            if (
                multiturn
                and stored is not None
                and stored.agent_name == agent_name
                and stored.model == model
            ):
                return stored.agent, True
            agent = create_agent(agent_name)
            session.stored = StoredAgent(agent_name, model, agent) if multiturn else None
            return agent, False

    def _render(self, run: Run) -> tuple[dict, Optional[GraphData], Any, int, str]:
        """Live snapshot parts. Only the consumer (and create_run, before the
        consumer exists) calls this, so the state is never read mid-drain."""
        now = time.time()
        body = live_body(run.agent_name, run.state, now)
        graph: Optional[GraphData] = None
        graph_key: Any = None
        if run.live_graph:
            graph, graph_key = live_graph_snapshot(run.state, run.agent)
        version = run.log.version
        if version != run._log_html_version:
            run._log_html = log_tail_html(run.log.text(), head_dropped=run.log.truncated)
            run._log_html_version = version
        return body, graph, graph_key, version, run._log_html

    def _tick(self, run: Run) -> Optional[tuple]:
        """Drain once; None when terminal, else the snapshot parts."""
        if drain_into(run.queue, run.state):
            return None
        return self._render(run)

    async def _consume(self, run: Run, session: Session) -> None:
        """The run's single queue consumer (one asyncio task per run)."""
        last_signature: Any = None
        last_publish = time.monotonic()
        try:
            while True:
                parts = await asyncio.to_thread(self._tick, run)
                if parts is None:
                    break
                body, graph, graph_key, log_version, log_html = parts
                signature = (_body_signature(body), log_version, graph_key)
                now = time.monotonic()
                if signature != last_signature or now - last_publish >= HEARTBEAT_SECONDS:
                    run.publish(body, graph, graph_key, log_version, log_html)
                    last_signature = signature
                    last_publish = now
                await asyncio.sleep(TICK_SECONDS)
            kind, payload = await asyncio.to_thread(self._final_event, run)
        except asyncio.CancelledError:
            self._release(session, run)
            raise
        except Exception as exc:  # noqa: BLE001
            _LOG.exception("Run consumer failed for %s", run.run_id)
            kind, payload = "error", {
                "run_id": run.run_id,
                "status": "error",
                "message": f"Internal error while following the run: {type(exc).__name__}: {exc}",
                "log_html": log_tail_html(run.log.text(), head_dropped=run.log.truncated),
            }
        # No await between these two: a client that reacts to "done" with a
        # follow-up POST always finds the session free.
        self._release(session, run)
        run.finish(kind, payload)

    @staticmethod
    def _release(session: Session, run: Run) -> None:
        if session.active_run_id == run.run_id:
            session.active_run_id = None
        session.last_seen = time.time()

    def _final_event(self, run: Run) -> tuple[str, dict]:
        """Build the done/error payload (worker thread: graph_from_trace
        parses every tool result of the run, which can be megabytes)."""
        state = run.state
        agent = run.agent
        events: list[dict] = []
        journal: list = []
        trace_id = state.trace_id
        recorder = getattr(agent, "recorder", None)
        if recorder is not None:
            try:
                events = recorder.to_dicts()
                trace_id = recorder.trace_id
            except AttributeError:
                pass
        try:
            journal = list(agent.journal_snapshots)
        except (AttributeError, TypeError):
            journal = []
        # Capture now: a follow-up turn resets the (reused) agent's recorder.
        run.trace_id = trace_id
        run.trace_events = truncate_trace_events(events)

        log_html = log_tail_html(run.log.text(), head_dropped=run.log.truncated)
        if state.status != "done":
            return "error", {
                "run_id": run.run_id,
                "status": "error",
                "message": state.error or "Unknown error",
                "log_html": log_html,
            }

        duration = (state.finished_at or time.time()) - (state.started_at or time.time())
        answer = state.answer if isinstance(state.answer, str) else str(state.answer or "")
        agent_model = getattr(agent, "model", None)
        model = agent_model if isinstance(agent_model, str) and agent_model else run.model

        tokens = None
        cost = None
        usage = getattr(agent, "token_usage", None)
        if isinstance(usage, dict) and usage.get("total_tokens", 0) > 0:
            tokens = {
                "prompt": usage.get("prompt_tokens", 0),
                "completion": usage.get("completion_tokens", 0),
                "total": usage["total_tokens"],
            }
            cost = format_cost_usd(
                estimate_cost_usd(model, tokens["prompt"], tokens["completion"])
            )

        figure, subagents = frozen_figure(run.agent_name, events, duration)
        graph = None
        if run.live_graph:
            graph = frozen_graph({
                "trace_id": trace_id,
                "agent": run.agent_name,
                "events": events,
                "journal_snapshots": journal,
                "answer": answer,
            })
        return "done", {
            "run_id": run.run_id,
            "status": "done",
            "agent": run.agent_name,
            "question": run.question,
            "answer": answer,
            "duration_s": round(duration, 2),
            "tokens": tokens,
            "cost": cost,
            "model": model,
            "trace_id": trace_id,
            "log_html": log_html,
            "figure": figure,
            "subagents": subagents,
            "graph": graph,
        }

    # -- lifecycle ----------------------------------------------------------------

    async def shutdown(self) -> None:
        tasks = [run.task for run in self.runs.values() if run.task and not run.task.done()]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)


# ---------------------------------------------------------------------------
# SSE
# ---------------------------------------------------------------------------

PING_SECONDS = 15.0


async def sse_stream(run: Run):
    """Server-Sent Events for one run: snapshots, then done/error, then close."""
    if run.finished:
        yield run.final_frame()
        return
    conn = _Connection()
    sent_seq: Optional[int] = None
    loop = asyncio.get_running_loop()
    last_sent = loop.time()
    while True:
        # Grab the event before inspecting state, so a publish that lands
        # between the checks below and the wait is never missed.
        changed = run.change_event()
        if run.finished:
            yield run.final_frame()
            return
        snap = run.snapshot
        if snap is not None and snap.seq != sent_seq:
            yield _json_frame("snapshot", snap.seq, conn.snapshot_data(snap))
            sent_seq = snap.seq
            last_sent = loop.time()
        timeout = max(0.0, PING_SECONDS - (loop.time() - last_sent))
        try:
            await asyncio.wait_for(changed.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            yield ": ping\n\n"
            last_sent = loop.time()
