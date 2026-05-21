"""Embeddable graph-view panel.

Mirrors ``utils/trace_panel.py`` for the discovered-subgraph visualisation.
Both ``pages/6_Graph_View.py`` and the Chat page's tabbed panel call into
``render_graph_panel(trace, key_prefix=…)``.
"""

from __future__ import annotations

from typing import Any

import streamlit as st
import streamlit.components.v1 as components

from ama_kbqa.frontend.utils.graph_html import build_graph_html, journal_to_graph


def render_graph_panel(
    trace: dict,
    *,
    key_prefix: str = "graph",
    height_px: int = 540,
) -> None:
    """Render the snapshot scrubber + vis-network graph + side panels.

    ``height_px`` is the inner graph height; the iframe gets ~20 px more to
    leave room for the vis-network toolbar.
    """
    trace_id: str = trace.get("trace_id", "unknown")
    snapshots: list[dict] = trace.get("journal_snapshots", []) or []

    if not snapshots:
        st.warning(
            "This trace has no journal snapshots. Either the agent didn't "
            "visit any KG nodes, or the MCP server is missing the "
            "`GetJournalStateJSON` tool — restart the server after pulling "
            "the latest code."
        )
        return

    # ── Scrubber + toggles ──────────────────────────────────────────────────
    ctrl_cols = st.columns([0.7, 0.15, 0.15])
    if len(snapshots) == 1:
        snapshot_idx = 0
        with ctrl_cols[0]:
            st.caption("Only one snapshot for this trace.")
    else:
        with ctrl_cols[0]:
            snapshot_idx = st.slider(
                "Step",
                min_value=0,
                max_value=len(snapshots) - 1,
                value=len(snapshots) - 1,
                key=f"{key_prefix}:graph_step:{trace_id}",
                help="Replay the agent's exploration step by step.",
            )
    with ctrl_cols[1]:
        show_literals = st.toggle(
            "Show literals",
            value=True,
            help="Include literal-value nodes.",
            key=f"{key_prefix}:graph_lit:{trace_id}",
        )
    with ctrl_cols[2]:
        highlight_new = st.toggle(
            "Highlight new",
            value=True,
            help="Highlight nodes added in the most recent step.",
            key=f"{key_prefix}:graph_hl:{trace_id}",
        )

    snap = snapshots[snapshot_idx]
    state: dict[str, Any] = snap.get("state", {})

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
    highlight_ids = (
        sorted(current_node_ids - prev_node_ids) if highlight_new else []
    )

    # ── Stats ───────────────────────────────────────────────────────────────
    stats = st.columns(5)
    stats[0].metric("Nodes", len(nodes))
    stats[1].metric("Edges", len(edges))
    stats[2].metric("Visited", len(state.get("visited_nodes") or {}))
    stats[3].metric("Verified facts", len(state.get("verified_facts") or []))
    stats[4].metric("Step", f"{snapshot_idx + 1}/{len(snapshots)}")

    st.caption(f"Trigger: `{snap.get('trigger', '?')}`")

    # ── Graph ───────────────────────────────────────────────────────────────
    @st.fragment
    def _render(_nodes, _edges, _highlight, _view_key, _height):
        html_doc = build_graph_html(
            _nodes,
            _edges,
            view_state_key=_view_key,
            highlight_node_ids=_highlight,
            height_px=_height,
        )
        components.html(html_doc, height=_height + 20, scrolling=False)

    _render(
        nodes,
        edges,
        highlight_ids,
        f"ama_kbqa_graph::{key_prefix}::{trace_id}",
        height_px,
    )

    # ── Side panels ─────────────────────────────────────────────────────────
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
