"""Graph View — visualise the subgraph the agent has discovered.

Thin wrapper around ``utils/graph_panel.render_graph_panel``. The same panel
is embedded in the Chat page.
"""

from __future__ import annotations

import json

import streamlit as st

from ama_kbqa.frontend.utils.graph_panel import render_graph_panel
from ama_kbqa.frontend.utils.styling import inject_css


st.set_page_config(page_title="Graph View — AMA KBQA", page_icon="🕸️", layout="wide")
inject_css()

st.markdown("# 🕸️ Graph View")
st.caption(
    "The subgraph the agent has discovered while investigating. Use the "
    "scrubber to replay how it grew over the course of the run."
)

traces: dict = st.session_state.get("traces", {})

if not traces:
    st.info(
        "No traces yet. Ask a question on the **Chat** page first; the "
        "discovered subgraph will appear here."
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
    key="graph_trace_selector",
)
trace = traces[selected_trace_id]

render_graph_panel(trace, key_prefix="page6", height_px=620)

# ── Sidebar utilities ───────────────────────────────────────────────────────
snapshots: list[dict] = trace.get("journal_snapshots", []) or []
with st.sidebar:
    st.caption(f"{len(snapshots)} snapshot(s) for this trace")
    if snapshots:
        st.download_button(
            "Download snapshots (JSON)",
            data=json.dumps(snapshots, indent=2, default=str),
            file_name=f"journal_snapshots_{selected_trace_id[:8]}.json",
            mime="application/json",
            use_container_width=True,
        )
