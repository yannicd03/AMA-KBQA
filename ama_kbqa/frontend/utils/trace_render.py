"""Helpers for rendering a flat list of TraceEvent dicts as a hierarchical tree.

Pure-Python — no Streamlit imports — so the logic is unit-testable.
"""

from __future__ import annotations

import html
from typing import Any, Optional


# Kinds we treat as instantaneous events (rendered as a thin marker row, not a
# nested branch) when the recorder didn't already flag them via `is_event`.
_POINT_EVENT_KINDS = {
    "intervention",
    "loop_detected",
    "context_trim",
    "journal_refresh",
    "tool_loop_iter",
}


def build_tree(events: list[dict]) -> dict[str, Any]:
    """Build a tree of events by parent_span_id chain.

    Returns a dict with:
        roots: list of root events (parent_span_id is None or unknown)
        children: dict[span_id, list[child events]]
        by_id: dict[span_id, event]

    Order within each parent is preserved by start_time_unix_nano.
    """
    by_id = {e["span_id"]: e for e in events}
    children: dict[Optional[str], list[dict]] = {}
    for e in events:
        parent = e.get("parent_span_id")
        # Treat references to unknown parents as orphans rooted at the trace
        # top level (they shouldn't normally happen but we don't want to
        # silently drop events).
        if parent is not None and parent not in by_id:
            parent = None
        children.setdefault(parent, []).append(e)
    for siblings in children.values():
        siblings.sort(key=lambda e: e.get("start_time_unix_nano", 0))
    return {
        "roots": children.get(None, []),
        "children": children,
        "by_id": by_id,
    }


def summarise(events: list[dict]) -> dict[str, Any]:
    """Compute a top-level summary for the whole trace."""
    if not events:
        return {
            "total_duration_ms": 0.0,
            "total_tokens": 0,
            "n_llm_calls": 0,
            "n_tool_calls": 0,
            "n_errors": 0,
            "kind_counts": {},
        }

    spans = [e for e in events if not e.get("is_event")]
    if spans:
        starts = [e["start_time_unix_nano"] for e in spans]
        ends = [e["end_time_unix_nano"] for e in spans]
        total_duration_ms = (max(ends) - min(starts)) / 1e6
    else:
        total_duration_ms = 0.0

    total_tokens = 0
    n_llm = 0
    n_tool = 0
    n_errors = 0
    kind_counts: dict[str, int] = {}
    for e in events:
        kind = e.get("kind", "?")
        kind_counts[kind] = kind_counts.get(kind, 0) + 1
        if kind == "llm_call" or kind == "classify" or kind == "synthesis":
            n_llm += 1
            attrs = e.get("attributes", {}) or {}
            total_tokens += int(attrs.get("total_tokens") or
                                ((attrs.get("prompt_tokens") or 0) +
                                 (attrs.get("completion_tokens") or 0)))
        elif kind == "tool_call":
            n_tool += 1
        if e.get("status") == "error":
            n_errors += 1

    return {
        "total_duration_ms": total_duration_ms,
        "total_tokens": total_tokens,
        "n_llm_calls": n_llm,
        "n_tool_calls": n_tool,
        "n_errors": n_errors,
        "kind_counts": kind_counts,
    }


def format_duration(ms: float) -> str:
    if ms < 1:
        return f"{ms*1000:.0f}µs"
    if ms < 1000:
        return f"{ms:.0f}ms"
    if ms < 60_000:
        return f"{ms/1000:.2f}s"
    m = int(ms // 60_000)
    s = (ms - m * 60_000) / 1000
    return f"{m}m{s:.0f}s"


def format_token_badge(attrs: dict) -> str:
    pt = attrs.get("prompt_tokens")
    ct = attrs.get("completion_tokens")
    if pt is None and ct is None:
        return ""
    parts = []
    if pt is not None:
        parts.append(f"↑{pt}")
    if ct is not None:
        parts.append(f"↓{ct}")
    return " ".join(parts)


def render_span_row_html(
    event: dict,
    depth: int,
    selected_span_id: Optional[str],
) -> str:
    """Produce one row of HTML for the trace tree."""
    span_id = event["span_id"]
    kind = event.get("kind", "?")
    name = event.get("name", "")
    is_event = bool(event.get("is_event"))
    status = event.get("status", "ok")
    attrs = event.get("attributes", {}) or {}

    indent_px = depth * 16
    classes = ["span-row"]
    if selected_span_id == span_id:
        classes.append("selected")
    if is_event:
        classes.append("span-event-marker")

    pill_class = f"span-pill kind-{html.escape(kind)}"
    duration_html = (
        f'<span class="span-duration">{format_duration(event.get("duration_ms", 0))}</span>'
        if not is_event
        else ""
    )

    token_badge = format_token_badge(attrs)
    token_html = (
        f'<span class="span-tokens">{html.escape(token_badge)}</span>'
        if token_badge
        else ""
    )

    status_icon = ""
    if status == "error":
        status_icon = '<span class="span-status-error">●</span>'

    name_label = html.escape(str(name))[:90]
    return (
        f'<div class="{" ".join(classes)}" style="padding-left: {indent_px}px"'
        f' data-span-id="{html.escape(span_id)}">'
        f'<span class="{pill_class}">{html.escape(kind)}</span>'
        f'<span class="span-name">{name_label}</span>'
        f"{status_icon}"
        f"{token_html}"
        f"{duration_html}"
        f"</div>"
    )


def render_tree_html(
    events: list[dict],
    selected_span_id: Optional[str] = None,
) -> str:
    """Render the entire tree as an HTML string. Layout-only — no JS.

    Selection is delivered separately (Streamlit-side via radio/buttons) since
    the embedded HTML doesn't natively talk back to Python without a custom
    component. The renderer just highlights the currently-selected row.
    """
    tree = build_tree(events)
    rows: list[str] = []

    def walk(node: dict, depth: int) -> None:
        rows.append(render_span_row_html(node, depth, selected_span_id))
        for child in tree["children"].get(node["span_id"], []):
            walk(child, depth + 1)

    for root in tree["roots"]:
        walk(root, 0)

    return f'<div class="trace-tree">{"".join(rows)}</div>'


def render_summary_html(events: list[dict]) -> str:
    s = summarise(events)
    return (
        f'<div class="trace-summary">'
        f'<span class="trace-summary-stat"><strong>{format_duration(s["total_duration_ms"])}</strong>total</span>'
        f'<span class="trace-summary-stat"><strong>{s["n_llm_calls"]}</strong>LLM calls</span>'
        f'<span class="trace-summary-stat"><strong>{s["n_tool_calls"]}</strong>tool calls</span>'
        f'<span class="trace-summary-stat"><strong>{s["total_tokens"]:,}</strong>tokens</span>'
        f'<span class="trace-summary-stat"><strong>{s["n_errors"]}</strong>errors</span>'
        f"</div>"
    )
