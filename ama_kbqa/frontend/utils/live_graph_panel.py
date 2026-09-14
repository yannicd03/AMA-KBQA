"""The "Explored subgraph" side panel of the chat page.

Why this is its own module and not a few lines in ``chat.py``: the panel has
two quite different lifetimes that must produce the same picture.

* **Live**, while the agent is running: a 1 Hz fragment re-reads the graph the
  worker thread has accumulated and rewrites an ``st.empty()`` placeholder that
  was created *outside* the fragment. Writing into a pre-existing placeholder
  (rather than emitting the iframe from the fragment's own body) is what keeps
  the vis-network canvas from being torn down and remounted on every tick, and
  the version-key check means an unchanged second does not even redraw.
* **Frozen**, once a run has completed: the same normalisers rebuild the graph
  from the recorded trace and highlight the nodes the final answer mentions.

The panel is strictly a reader. It never touches ``live_run["queue"]``: the
lifecycle fragment in ``chat.py`` stays the single consumer, because a queue
drained by two fragments would lose notifications at random.
"""

from __future__ import annotations

from typing import Any, Optional

import streamlit as st
import streamlit.components.v1 as components

from ama_kbqa.frontend.utils.graph_html import build_graph_html
from ama_kbqa.frontend.utils.lifecycle_runner import live_graph_snapshot
from ama_kbqa.frontend.utils.live_graph_data import (
    GraphData,
    answer_node_ids,
    apply_caps,
    graph_from_trace,
    to_vis_payload,
)

# The iframe is a little taller than the canvas so vis-network's own chrome
# (and the legend overlay) is not clipped by the component boundary.
GRAPH_HEIGHT_PX = 560
IFRAME_HEIGHT_PX = 580

# The answer expander is a reading aid, not an export: a run that lights up
# fifty nodes does not need fifty bullet points.
MAX_ANSWER_NODES_LISTED = 20

REFRESH_SECONDS = 1.0


def render_live_graph_panel(
    *,
    live_run: Optional[dict] = None,
    trace: Optional[dict] = None,
    key_prefix: str = "chat",
) -> None:
    """Render the panel into the current container.

    ``live_run`` is the session's in-flight run dict (``{"state", "agent",
    "question", ...}``) or None; ``trace`` is the completed trace to freeze on
    when nothing is running. A live run always wins: while the agent works, the
    panel shows what it is finding, not the previous question's result.
    """
    st.markdown("**Explored subgraph**")
    caption_ph = st.empty()
    stats_ph = st.empty()
    graph_ph = st.empty()

    if live_run is not None:
        _render_live(
            live_run=live_run,
            key_prefix=key_prefix,
            caption_ph=caption_ph,
            stats_ph=stats_ph,
            graph_ph=graph_ph,
        )
        return
    _render_frozen(
        trace=trace,
        key_prefix=key_prefix,
        caption_ph=caption_ph,
        stats_ph=stats_ph,
        graph_ph=graph_ph,
    )


# ---------------------------------------------------------------------------
# Live mode
# ---------------------------------------------------------------------------

def _live_trace_key(live_run: dict, state: Any) -> str:
    """Identity of the run being drawn, for the reset / localStorage key.

    ``state.trace_id`` is the right key (the frozen view uses the same one, so
    the layout survives the run completing), but it only arrives once the
    worker has enqueued ``__trace_id__``, which can be a tick after the panel
    first renders. Until then the question plus the start timestamp identifies
    the run uniquely, and the one-off switch to the real trace id is just an
    extra reset while the canvas is still empty.
    """
    trace_id = getattr(state, "trace_id", None)
    if trace_id:
        return str(trace_id)
    return f"pending:{live_run.get('question', '')}:{getattr(state, 'started_at', None)}"


def _render_live(
    *,
    live_run: dict,
    key_prefix: str,
    caption_ph: Any,
    stats_ph: Any,
    graph_ph: Any,
) -> None:
    state = live_run.get("state")
    agent = live_run.get("agent")
    if state is None:
        caption_ph.caption("no graph data for this run")
        return

    version_key = f"{key_prefix}:live_graph_version"
    ids_key = f"{key_prefix}:live_graph_ids"
    trace_state_key = f"{key_prefix}:live_graph_trace"

    # Reset bookkeeping is done once per script run, outside the fragment: the
    # fragment may tick many times before the next rerun, and a reset on every
    # tick would wipe the node positions the canvas just saved.
    trace_key = _live_trace_key(live_run, state)
    first_render_of_trace = st.session_state.get(trace_state_key) != trace_key
    if first_render_of_trace:
        st.session_state[trace_state_key] = trace_key
        st.session_state[ids_key] = set()
        st.session_state[version_key] = None

    # The version check below would otherwise skip the first tick after a
    # Streamlit rerun, when session_state still holds the previous script run's
    # key but the placeholders are empty again.
    pending_reset = {"value": first_render_of_trace, "drawn": False}

    @st.fragment(run_every=REFRESH_SECONDS)
    def _graph_tick() -> None:
        graph, current_key = live_graph_snapshot(state, agent)
        if pending_reset["drawn"] and st.session_state.get(version_key) == current_key:
            return
        st.session_state[version_key] = current_key

        previous_ids = st.session_state.get(ids_key) or set()
        current_ids = set(graph.nodes)
        st.session_state[ids_key] = current_ids

        _write_caption(caption_ph, "building…", graph)
        _write_stats(stats_ph, graph)
        if graph.is_empty():
            graph_ph.empty()
        else:
            _write_graph(
                graph_ph,
                graph,
                highlight=set(),
                new_ids=current_ids - previous_ids,
                view_state_key=_view_state_key(key_prefix, trace_key),
                reset=pending_reset["value"],
            )
            pending_reset["value"] = False
        pending_reset["drawn"] = True

    _graph_tick()


# ---------------------------------------------------------------------------
# Frozen mode
# ---------------------------------------------------------------------------

def _render_frozen(
    *,
    trace: Optional[dict],
    key_prefix: str,
    caption_ph: Any,
    stats_ph: Any,
    graph_ph: Any,
) -> None:
    if not isinstance(trace, dict):
        caption_ph.caption("no graph data for this run")
        return

    trace_key = str(trace.get("trace_id") or "unknown")
    try:
        graph = apply_caps(graph_from_trace(trace))
    except Exception:  # noqa: BLE001 - a broken trace must not break the page
        graph = GraphData()

    if graph.is_empty():
        caption_ph.caption("no graph data for this run")
        return

    highlight = answer_node_ids(graph, trace.get("answer"))
    _write_caption(
        caption_ph,
        f"final · {len(graph.nodes)} nodes · {len(graph.edges)} edges",
        graph,
    )
    _write_stats(stats_ph, graph)

    trace_state_key = f"{key_prefix}:live_graph_trace"
    reset = st.session_state.get(trace_state_key) != trace_key
    st.session_state[trace_state_key] = trace_key
    _write_graph(
        graph_ph,
        graph,
        highlight=highlight,
        new_ids=set(),
        view_state_key=_view_state_key(key_prefix, trace_key),
        reset=reset,
    )

    with st.expander("Nodes in the answer", expanded=False):
        if highlight:
            labels = [
                graph.nodes[node_id].label
                for node_id in list(highlight)[:MAX_ANSWER_NODES_LISTED]
                if node_id in graph.nodes
            ]
            st.write("\n".join(f"- {label}" for label in sorted(labels)))
            if len(highlight) > MAX_ANSWER_NODES_LISTED:
                st.caption(
                    f"and {len(highlight) - MAX_ANSWER_NODES_LISTED} more"
                )
        else:
            st.caption("No node of the graph matched the answer text.")

    with st.expander("Raw journal", expanded=False):
        st.json(_last_journal_states(trace), expanded=False)


def _last_journal_states(trace: dict) -> dict:
    """The newest journal state per source, the debugging hatch of §3.2.

    Only the last snapshot per source is kept: snapshots are cumulative, so
    dumping all of them would push megabytes of duplicate JSON into the page.
    """
    snapshots = trace.get("journal_snapshots") or []
    if not isinstance(snapshots, list):
        return {}
    latest: dict[str, Any] = {}
    for snapshot in snapshots:
        if isinstance(snapshot, dict):
            tag = str(snapshot.get("source_agent") or trace.get("agent") or "agent")
            latest[tag] = snapshot.get("state") or {}
    return latest


# ---------------------------------------------------------------------------
# Shared drawing helpers
# ---------------------------------------------------------------------------

def _view_state_key(key_prefix: str, trace_key: str) -> str:
    return f"ama_kbqa_live_graph::{key_prefix}::{trace_key}"


def _write_caption(placeholder: Any, text: str, graph: GraphData) -> None:
    if graph.truncated_nodes:
        shown = len(graph.nodes)
        text = f"{text} · showing {shown} of {shown + graph.truncated_nodes}"
    placeholder.caption(text)


def _write_stats(placeholder: Any, graph: GraphData) -> None:
    stats = graph.stats()
    with placeholder.container():
        cols = st.columns(3)
        cols[0].metric(
            "Entities",
            stats["entities"],
            help=(
                f"{stats['candidates']} further search candidate(s) the agent "
                "has seen but not visited."
            ),
        )
        cols[1].metric("Literals", stats["literals"])
        cols[2].metric("Edges", stats["edges"])


def _write_graph(
    placeholder: Any,
    graph: GraphData,
    *,
    highlight: set,
    new_ids: set,
    view_state_key: str,
    reset: bool,
) -> None:
    payload = to_vis_payload(graph, highlight=highlight, new_ids=new_ids)
    html_doc = build_graph_html(
        payload["nodes"],
        payload["edges"],
        view_state_key=view_state_key,
        height_px=GRAPH_HEIGHT_PX,
        highlight_node_ids=sorted(highlight),
        new_node_ids=sorted(new_ids),
        reset=reset,
    )
    with placeholder.container():
        components.html(html_doc, height=IFRAME_HEIGHT_PX, scrolling=False)
