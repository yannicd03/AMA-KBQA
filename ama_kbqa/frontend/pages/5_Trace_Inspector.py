"""Trace Inspector — Langfuse-style hierarchical view of one agent run.

Reads from `st.session_state["traces"]` (populated by the Chat page) and
renders a two-pane layout:

- Left (~45%): hierarchical span tree, click to select.
- Right (~55%): span detail (preview, JSON, messages, raw payload).
"""

from __future__ import annotations

import json
from typing import Optional

import streamlit as st

from ama_kbqa.frontend.utils.styling import inject_css
from ama_kbqa.frontend.utils.trace_render import (
    build_tree,
    format_duration,
    render_summary_html,
    render_tree_html,
)


st.set_page_config(page_title="Trace Inspector — AMA KBQA", page_icon="🔍", layout="wide")
inject_css()

st.markdown("# 🔍 Trace Inspector")
st.caption("Hierarchical view of an agent run — spans, LLM calls, tool calls, interventions.")

traces: dict = st.session_state.get("traces", {})

if not traces:
    st.info(
        "No traces yet. Ask a question on the **Chat** page first; the run will "
        "appear here automatically."
    )
    st.stop()

# ── Trace selector ──────────────────────────────────────────────────────────
trace_ids = sorted(
    traces.keys(),
    key=lambda tid: traces[tid].get("timestamp", ""),
    reverse=True,
)

# Pre-select if the Chat page or another page handed us a trace via
# session_state.
default_idx = 0
pinned = st.session_state.get("pinned_trace_id")
if pinned in trace_ids:
    default_idx = trace_ids.index(pinned)

def _label(tid: str) -> str:
    t = traces[tid]
    q = (t.get("query") or "").replace("\n", " ")
    if len(q) > 60:
        q = q[:60] + "…"
    return f"{t.get('timestamp', '?')} · {t.get('agent', '?')} · {q or tid[:8]}"

selected_trace_id = st.selectbox(
    "Trace",
    options=trace_ids,
    index=default_idx,
    format_func=_label,
    key="trace_selector",
)
trace = traces[selected_trace_id]
events: list[dict] = trace.get("events", [])

# ── Top summary ─────────────────────────────────────────────────────────────
st.markdown(render_summary_html(events), unsafe_allow_html=True)

with st.expander("Question & answer", expanded=False):
    st.markdown(f"**Question:** {trace.get('query', '?')}")
    st.markdown("**Answer:**")
    st.markdown(trace.get("answer", "_(empty)_"))

if not events:
    st.warning("This trace has no recorded events.")
    st.stop()

# ── Two-pane layout ─────────────────────────────────────────────────────────
left, right = st.columns([0.45, 0.55], gap="medium")

# Default-select the root span, or whatever the user last picked for this trace.
selected_key = f"selected_span:{selected_trace_id}"
if selected_key not in st.session_state:
    tree = build_tree(events)
    if tree["roots"]:
        st.session_state[selected_key] = tree["roots"][0]["span_id"]
    else:
        st.session_state[selected_key] = events[0]["span_id"]

selected_span_id: Optional[str] = st.session_state.get(selected_key)

with left:
    st.markdown("##### Spans")
    # Render the visual tree (read-only HTML for highlighting + indentation).
    st.markdown(
        render_tree_html(events, selected_span_id=selected_span_id),
        unsafe_allow_html=True,
    )

    # Selection input — Streamlit widgets can't read clicks inside arbitrary
    # HTML, so we render a parallel selector. A radio is the cleanest mapping
    # to "pick one span"; we feed it the same depth-ordered list as the tree.
    tree = build_tree(events)
    ordered: list[dict] = []
    def _walk(node: dict, depth: int) -> None:
        ordered.append({**node, "_depth": depth})
        for c in tree["children"].get(node["span_id"], []):
            _walk(c, depth + 1)
    for r in tree["roots"]:
        _walk(r, 0)

    def _radio_label(e: dict) -> str:
        d = e["_depth"]
        prefix = "  " * d + ("• " if e.get("is_event") else "")
        kind = e.get("kind", "?")
        name = e.get("name", "")[:50]
        if e.get("is_event"):
            tail = ""
        else:
            tail = f"  ({format_duration(e.get('duration_ms', 0))})"
        return f"{prefix}[{kind}] {name}{tail}"

    span_ids = [e["span_id"] for e in ordered]
    try:
        idx = span_ids.index(selected_span_id) if selected_span_id else 0
    except ValueError:
        idx = 0
    picked = st.radio(
        "Select span",
        options=span_ids,
        index=idx,
        format_func=lambda sid: _radio_label(tree["by_id"][sid]),
        label_visibility="collapsed",
        key=f"radio:{selected_trace_id}",
    )
    if picked != selected_span_id:
        st.session_state[selected_key] = picked
        st.rerun()

# ── Right pane: detail of selected span ─────────────────────────────────────
with right:
    if selected_span_id and selected_span_id in tree["by_id"]:
        evt = tree["by_id"][selected_span_id]
    else:
        evt = ordered[0] if ordered else None

    if evt is None:
        st.info("Select a span to inspect.")
    else:
        kind = evt.get("kind", "?")
        name = evt.get("name", "")
        st.markdown(f"##### `{kind}` — {name}")
        meta_cols = st.columns(4)
        with meta_cols[0]:
            st.caption("Duration")
            st.markdown(f"**{format_duration(evt.get('duration_ms', 0))}**")
        with meta_cols[1]:
            st.caption("Status")
            st.markdown(
                "**OK**" if evt.get("status") == "ok" else f"**{evt.get('status', '?')}**"
            )
        with meta_cols[2]:
            attrs = evt.get("attributes", {}) or {}
            st.caption("Tokens")
            pt = attrs.get("prompt_tokens")
            ct = attrs.get("completion_tokens")
            if pt is not None or ct is not None:
                st.markdown(f"**↑{pt or 0} / ↓{ct or 0}**")
            else:
                st.markdown("—")
        with meta_cols[3]:
            st.caption("Span ID")
            st.code(evt.get("span_id", "")[:12], language=None)

        # Tabs: pick a sensible default per kind.
        if kind in ("llm_call", "classify", "synthesis"):
            tabs = st.tabs(["Messages", "Attributes", "Payload", "Raw"])
            with tabs[0]:
                payload = evt.get("payload", {}) or {}
                msgs = payload.get("messages")
                if msgs:
                    for m in msgs:
                        role = m.get("role", "?")
                        content = m.get("content")
                        with st.expander(f"**{role}**", expanded=(role in ("user", "assistant"))):
                            if isinstance(content, str):
                                st.markdown(f"```\n{content}\n```")
                            else:
                                st.json(content)
                            if "tool_calls" in m:
                                st.caption("tool_calls")
                                st.json(m["tool_calls"])
                else:
                    if "assistant_content" in payload:
                        st.markdown("**Assistant content:**")
                        st.markdown(f"```\n{payload['assistant_content']}\n```")
                    else:
                        st.info("No messages captured for this span.")
            with tabs[1]:
                st.json(evt.get("attributes", {}) or {})
            with tabs[2]:
                st.json(evt.get("payload", {}) or {})
            with tabs[3]:
                st.code(json.dumps(evt, indent=2, default=str), language="json")
        elif kind == "tool_call":
            tabs = st.tabs(["Args / Result", "Attributes", "Raw"])
            with tabs[0]:
                payload = evt.get("payload", {}) or {}
                st.markdown("**Arguments:**")
                st.json(payload.get("arguments", {}))
                st.markdown("**Result:**")
                result = payload.get("result", "")
                if isinstance(result, str) and len(result) > 4000:
                    st.markdown("_(showing first 4000 chars)_")
                    st.code(result[:4000], language="json")
                else:
                    st.code(str(result), language="json")
            with tabs[1]:
                st.json(evt.get("attributes", {}) or {})
            with tabs[2]:
                st.code(json.dumps(evt, indent=2, default=str), language="json")
        else:
            tabs = st.tabs(["Attributes", "Payload", "Raw"])
            with tabs[0]:
                st.json(evt.get("attributes", {}) or {})
            with tabs[1]:
                st.json(evt.get("payload", {}) or {})
            with tabs[2]:
                st.code(json.dumps(evt, indent=2, default=str), language="json")

        if evt.get("error"):
            st.error(evt["error"])

# ── Sidebar utilities ───────────────────────────────────────────────────────
with st.sidebar:
    st.caption(f"{len(traces)} trace(s) in this session")
    if st.button("Clear all traces", help="Remove all stored traces from this session."):
        st.session_state["traces"] = {}
        st.session_state.pop("pinned_trace_id", None)
        st.rerun()
    st.download_button(
        "Download this trace (JSONL)",
        data="\n".join(json.dumps(e, default=str) for e in events),
        file_name=f"trace_{selected_trace_id[:8]}.jsonl",
        mime="application/x-ndjson",
        use_container_width=True,
    )
