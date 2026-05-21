"""
Structured trace recorder for KBQA agents.

Emits OpenTelemetry-compatible span events as the agent runs. Spans nest via
contextvars (auto-propagating across `await` boundaries), so the recorder
produces a correctly-parented tree without threading parent ids through every
function signature.

Public API:
    recorder = TraceRecorder()
    async with recorder.span("agent_run", "ask", attributes={...}) as span:
        # nested awaits create child spans automatically
        async with recorder.span("llm_call", "gpt-4o"):
            ...
    recorder.event("intervention", "zero-tool-call retry")
    recorder.events  # -> list[TraceEvent]
"""

from __future__ import annotations

import contextlib
import json
import logging
import time
import uuid
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from typing import Any, AsyncIterator, Callable, Optional


_LOG = logging.getLogger(__name__)


# Listener signature: fn(phase, info_dict). `phase` is "open", "close", or
# "event". `info_dict` for "open" is a synthetic dict (the real TraceEvent
# doesn't exist yet); for "close" and "event" it is the full TraceEvent dict.
Listener = Callable[[str, dict[str, Any]], None]


SpanKind = str  # one of the literals listed in trace.py docs; kept loose for forward-compat


_current_span_id: ContextVar[Optional[str]] = ContextVar("ama_kbqa_current_span_id", default=None)


@dataclass
class TraceEvent:
    """A single span event in OpenTelemetry-compatible shape.

    Both interval spans (with `end_time_unix_nano > start_time_unix_nano`)
    and point-in-time events (where the two timestamps are equal and
    `duration_ms == 0`) use this same shape, distinguished by the `kind`
    convention plus the `is_event` flag.
    """

    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    kind: SpanKind
    name: str
    start_time_unix_nano: int
    end_time_unix_nano: int
    duration_ms: float
    status: str  # "ok" or "error"
    is_event: bool  # True for instantaneous events, False for interval spans
    attributes: dict[str, Any] = field(default_factory=dict)
    payload: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TraceRecorder:
    """Collects TraceEvents for one agent run (or a delegated sub-run).

    Sub-agents can share a parent recorder by passing it (and the current
    span_id) through. A single recorder produces a single trace_id; merging
    across recorders is intentionally not supported.
    """

    def __init__(self, trace_id: Optional[str] = None) -> None:
        self.trace_id: str = trace_id or uuid.uuid4().hex
        self.events: list[TraceEvent] = []
        self._listeners: list[Listener] = []

    def add_listener(self, listener: Listener) -> None:
        self._listeners.append(listener)

    def remove_listener(self, listener: Listener) -> None:
        try:
            self._listeners.remove(listener)
        except ValueError:
            pass

    def _notify(self, phase: str, info: dict[str, Any]) -> None:
        # Listener failures must never break agent execution.
        for fn in list(self._listeners):
            try:
                fn(phase, info)
            except Exception:
                _LOG.exception("TraceRecorder listener raised; ignoring")

    @staticmethod
    def _now_ns() -> int:
        return time.time_ns()

    def _new_span_id(self) -> str:
        return uuid.uuid4().hex[:16]

    def current_span_id(self) -> Optional[str]:
        return _current_span_id.get()

    def _open_span(
        self,
        kind: SpanKind,
        name: str,
        attributes: Optional[dict[str, Any]],
        payload: Optional[dict[str, Any]],
    ) -> tuple["_SpanHandle", Any, Optional[str], int, dict[str, Any], dict[str, Any]]:
        """Shared open-side logic for both sync and async span CMs.

        Returns (handle, contextvar_token, parent_id, start_ns, evt_attrs, evt_payload).
        """
        span_id = self._new_span_id()
        parent_id = _current_span_id.get()
        start_ns = self._now_ns()
        evt_attrs: dict[str, Any] = dict(attributes or {})
        evt_payload: dict[str, Any] = dict(payload or {})
        token = _current_span_id.set(span_id)
        handle = _SpanHandle(span_id=span_id, attributes=evt_attrs, payload=evt_payload)
        self._notify(
            "open",
            {
                "trace_id": self.trace_id,
                "span_id": span_id,
                "parent_span_id": parent_id,
                "kind": kind,
                "name": name,
                "start_time_unix_nano": start_ns,
                "attributes": dict(evt_attrs),
                "is_event": False,
            },
        )
        return handle, token, parent_id, start_ns, evt_attrs, evt_payload

    def _close_span(
        self,
        kind: SpanKind,
        name: str,
        span_id: str,
        token: Any,
        parent_id: Optional[str],
        start_ns: int,
        evt_attrs: dict[str, Any],
        evt_payload: dict[str, Any],
        status: str,
        error_msg: Optional[str],
    ) -> None:
        _current_span_id.reset(token)
        end_ns = self._now_ns()
        evt = TraceEvent(
            trace_id=self.trace_id,
            span_id=span_id,
            parent_span_id=parent_id,
            kind=kind,
            name=name,
            start_time_unix_nano=start_ns,
            end_time_unix_nano=end_ns,
            duration_ms=(end_ns - start_ns) / 1e6,
            status=status,
            is_event=False,
            attributes=evt_attrs,
            payload=evt_payload,
            error=error_msg,
        )
        self.events.append(evt)
        self._notify("close", evt.to_dict())

    @contextlib.asynccontextmanager
    async def span(
        self,
        kind: SpanKind,
        name: str,
        attributes: Optional[dict[str, Any]] = None,
        payload: Optional[dict[str, Any]] = None,
    ) -> AsyncIterator["_SpanHandle"]:
        """Async context manager that records an interval span.

        Sets the new span as the current one via the contextvar, so nested
        `async with recorder.span(...)` calls auto-parent.
        """
        handle, token, parent_id, start_ns, evt_attrs, evt_payload = self._open_span(
            kind, name, attributes, payload
        )
        status = "ok"
        error_msg: Optional[str] = None
        try:
            yield handle
        except BaseException as e:
            status = "error"
            error_msg = f"{type(e).__name__}: {e}"
            raise
        finally:
            self._close_span(
                kind, name, handle.span_id, token, parent_id, start_ns,
                evt_attrs, evt_payload, status, error_msg,
            )

    @contextlib.contextmanager
    def span_sync(
        self,
        kind: SpanKind,
        name: str,
        attributes: Optional[dict[str, Any]] = None,
        payload: Optional[dict[str, Any]] = None,
    ):
        """Synchronous counterpart to `span()`. Same contextvar semantics."""
        handle, token, parent_id, start_ns, evt_attrs, evt_payload = self._open_span(
            kind, name, attributes, payload
        )
        status = "ok"
        error_msg: Optional[str] = None
        try:
            yield handle
        except BaseException as e:
            status = "error"
            error_msg = f"{type(e).__name__}: {e}"
            raise
        finally:
            self._close_span(
                kind, name, handle.span_id, token, parent_id, start_ns,
                evt_attrs, evt_payload, status, error_msg,
            )

    def event(
        self,
        kind: SpanKind,
        name: str,
        attributes: Optional[dict[str, Any]] = None,
        payload: Optional[dict[str, Any]] = None,
    ) -> None:
        """Record a point-in-time event under the current span."""
        now = self._now_ns()
        parent_id = _current_span_id.get()
        evt = TraceEvent(
            trace_id=self.trace_id,
            span_id=self._new_span_id(),
            parent_span_id=parent_id,
            kind=kind,
            name=name,
            start_time_unix_nano=now,
            end_time_unix_nano=now,
            duration_ms=0.0,
            status="ok",
            is_event=True,
            attributes=dict(attributes or {}),
            payload=dict(payload or {}),
        )
        self.events.append(evt)
        self._notify("event", evt.to_dict())

    def to_jsonl(self) -> str:
        return "\n".join(json.dumps(e.to_dict(), default=str) for e in self.events)

    def to_dicts(self) -> list[dict[str, Any]]:
        return [e.to_dict() for e in self.events]

    def reset(self) -> None:
        """Clear all events but keep the trace_id."""
        self.events = []


class _SpanHandle:
    """Handle yielded by `recorder.span(...)`.

    Lets the caller stash extra attributes/payload onto the span as the work
    inside it progresses (e.g. token counts known only after the LLM call
    returns).
    """

    __slots__ = ("span_id", "attributes", "payload")

    def __init__(self, span_id: str, attributes: dict[str, Any], payload: dict[str, Any]) -> None:
        self.span_id = span_id
        self.attributes = attributes
        self.payload = payload

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value

    def update_attributes(self, mapping: dict[str, Any]) -> None:
        self.attributes.update(mapping)

    def set_payload(self, key: str, value: Any) -> None:
        self.payload[key] = value

    def update_payload(self, mapping: dict[str, Any]) -> None:
        self.payload.update(mapping)


# ---------------------------------------------------------------------------
# Mutating-tool registry
# ---------------------------------------------------------------------------
#
# Tool names whose execution writes to the server-side `session_journal`. The
# agent uses this set to decide when to fetch a fresh structured journal
# snapshot for the live graph view (instead of fetching after every iteration,
# which adds measurable latency).
#
# Cross-checked by grep against `session_journal.` writes in
# ama_kbqa/server/kqapro_server.py and ama_kbqa/server/sciqa_server.py.

JOURNAL_MUTATING_TOOLS: frozenset[str] = frozenset(
    {
        # KQAPro server (Wikidata-style)
        "FindNode",
        "BatchGetNodeLabels",
        "GetEdgeQualifiers",
        "GetQualifiersByPredicate",
        "GetQualifierValue",
        "FindByAttribute",
        "ManageJournal",
        "VerifyFact",
        # SciQA server (ORKG)
        "FindResource",
        "FindPredicate",
        "GetResourceDetails",
        "ExploreNeighborhood",
        "FindEntitiesByRelationPath",
    }
)
