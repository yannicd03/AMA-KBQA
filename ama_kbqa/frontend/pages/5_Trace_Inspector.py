"""Trace Inspector — Langfuse-style hierarchical view of one agent run.

This page is a thin wrapper around ``utils/trace_panel.render_trace_panel``.
The same panel is embedded in the Chat page; the page-level view here adds a
trace selector, sidebar utilities, and a download button.
"""

from __future__ import annotations

import json

import streamlit as st

from ama_kbqa.frontend.utils.styling import inject_css
from ama_kbqa.frontend.utils.trace_panel import render_trace_panel


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

render_trace_panel(trace, key_prefix="page5")

# ── Sidebar utilities ───────────────────────────────────────────────────────
events: list[dict] = trace.get("events", []) or []
with st.sidebar:
    st.caption(f"{len(traces)} trace(s) in this session")
    if st.button("Clear all traces", help="Remove all stored traces from this session."):
        st.session_state["traces"] = {}
        st.session_state.pop("pinned_trace_id", None)
        st.rerun()
    if events:
        st.download_button(
            "Download this trace (JSONL)",
            data="\n".join(json.dumps(e, default=str) for e in events),
            file_name=f"trace_{selected_trace_id[:8]}.jsonl",
            mime="application/x-ndjson",
            use_container_width=True,
        )
