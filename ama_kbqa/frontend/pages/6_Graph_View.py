"""Graph View — visualise the subgraph the agent has discovered.

Reads `journal_snapshots` from `st.session_state["traces"][trace_id]` and
renders the latest (or scrubbed-to) journal state as a node-link diagram via
vis-network embedded in an `st.components.v1.html` iframe.
"""

from __future__ import annotations

import json
from typing import Any

import streamlit as st
import streamlit.components.v1 as components

from ama_kbqa.frontend.utils.graph_html import build_graph_html, journal_to_graph
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
snapshots: list[dict] = trace.get("journal_snapshots", []) or []

if not snapshots:
    st.warning(
        "This trace has no journal snapshots. Either the agent didn't visit "
        "any KG nodes, or the MCP server is missing the `GetJournalStateJSON` "
        "tool — restart the server after pulling the latest code."
    )
    st.stop()

# ── Scrubber ────────────────────────────────────────────────────────────────
ctrl_cols = st.columns([0.7, 0.15, 0.15])
with ctrl_cols[0]:
    snapshot_idx = st.slider(
        "Step",
        min_value=0,
        max_value=len(snapshots) - 1,
        value=len(snapshots) - 1,  # default: most recent
        key=f"graph_step:{selected_trace_id}",
        help="Replay the agent's exploration step by step.",
    )
with ctrl_cols[1]:
    show_literals = st.toggle("Show literals", value=True, help="Include literal-value nodes.")
with ctrl_cols[2]:
    highlight_new = st.toggle("Highlight new", value=True, help="Highlight nodes added in the most recent step.")

snap = snapshots[snapshot_idx]
state: dict[str, Any] = snap.get("state", {})

# ── Compute new-since-previous for highlighting ─────────────────────────────
prev_node_ids: set[str] = set()
if highlight_new and snapshot_idx > 0:
    prev_state = snapshots[snapshot_idx - 1].get("state", {})
    prev_node_ids = set((prev_state.get("visited_nodes") or {}).keys())

nodes, edges = journal_to_graph(state)

if not show_literals:
    nodes = [n for n in nodes if n.get("group") != "literal"]
    keep_ids = {n["id"] for n in nodes}
    edges = [e for e in edges if e["from"] in keep_ids and e["to"] in keep_ids]

current_node_ids = {n["id"] for n in nodes}
highlight_ids = sorted(current_node_ids - prev_node_ids) if highlight_new else []

# ── Stats line ──────────────────────────────────────────────────────────────
stats = st.columns(5)
stats[0].metric("Nodes", len(nodes))
stats[1].metric("Edges", len(edges))
stats[2].metric("Visited", len(state.get("visited_nodes") or {}))
stats[3].metric("Verified facts", len(state.get("verified_facts") or []))
stats[4].metric("Step", f"{snapshot_idx + 1}/{len(snapshots)}")

st.caption(f"Trigger: `{snap.get('trigger', '?')}`")

# ── Graph (in a fragment so unrelated reruns don't remount the iframe) ───────
@st.fragment
def render_graph(_nodes, _edges, _highlight, _view_key):
    html_doc = build_graph_html(
        _nodes,
        _edges,
        view_state_key=_view_key,
        highlight_node_ids=_highlight,
        height_px=620,
    )
    components.html(html_doc, height=640, scrolling=False)


# Stable view-state key per trace so pan/zoom is per-trace, not global.
render_graph(nodes, edges, highlight_ids, f"ama_kbqa_graph::{selected_trace_id}")

# ── Side panels (under the graph) ───────────────────────────────────────────
panel_cols = st.columns([0.5, 0.5])
with panel_cols[0]:
    with st.expander("Visited nodes", expanded=False):
        st.json(state.get("visited_nodes") or {})
    with st.expander("Found values", expanded=False):
        st.json(state.get("found_values") or {})
with panel_cols[1]:
    with st.expander("Verified facts", expanded=False):
        st.json(state.get("verified_facts") or [])
    with st.expander("Plan / progress", expanded=False):
        st.write({
            "current_plan": state.get("current_plan") or [],
            "completed_steps": state.get("completed_steps") or [],
            "failed_attempts": state.get("failed_attempts") or [],
            "partial_answer": state.get("partial_answer") or "",
        })

# ── Sidebar utilities ───────────────────────────────────────────────────────
with st.sidebar:
    st.caption(f"{len(snapshots)} snapshot(s) for this trace")
    st.download_button(
        "Download snapshots (JSON)",
        data=json.dumps(snapshots, indent=2, default=str),
        file_name=f"journal_snapshots_{selected_trace_id[:8]}.json",
        mime="application/json",
        use_container_width=True,
    )
