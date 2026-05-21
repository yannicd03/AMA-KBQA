"""Embeddable trace-inspector panel.

The full-page version lives at ``pages/5_Trace_Inspector.py``; the same
content is also embedded in the Chat page's tabbed panel. Both call
``render_trace_panel(trace, key_prefix=…)`` so the logic stays in one place.
"""

from __future__ import annotations

import json
from typing import Optional

import streamlit as st

from ama_kbqa.frontend.utils.trace_render import (
    build_tree,
    format_duration,
    render_summary_html,
    render_tree_html,
)


def render_trace_panel(trace: dict, *, key_prefix: str = "trace") -> None:
    """Render the two-pane span tree + detail view for a single trace.

    ``key_prefix`` is used to scope all Streamlit widget keys so multiple
    instances on the same page (or with the same trace_id) don't collide.
    """
    trace_id: str = trace.get("trace_id", "unknown")
    events: list[dict] = trace.get("events", []) or []

    # ── Top summary ─────────────────────────────────────────────────────────
    st.markdown(render_summary_html(events), unsafe_allow_html=True)

    with st.expander("Question & answer", expanded=False):
        st.markdown(f"**Question:** {trace.get('query', '?')}")
        st.markdown("**Answer:**")
        st.markdown(trace.get("answer", "_(empty)_"))

    if not events:
        st.warning("This trace has no recorded events.")
        return

    # ── Two-pane layout ─────────────────────────────────────────────────────
    left, right = st.columns([0.45, 0.55], gap="medium")

    selected_key = f"{key_prefix}:selected_span:{trace_id}"
    tree = build_tree(events)
    if selected_key not in st.session_state:
        if tree["roots"]:
            st.session_state[selected_key] = tree["roots"][0]["span_id"]
        else:
            st.session_state[selected_key] = events[0]["span_id"]

    selected_span_id: Optional[str] = st.session_state.get(selected_key)

    with left:
        st.markdown("##### Spans")
        st.markdown(
            render_tree_html(events, selected_span_id=selected_span_id),
            unsafe_allow_html=True,
        )

        # Depth-ordered list (drives the radio selector).
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
            tail = (
                ""
                if e.get("is_event")
                else f"  ({format_duration(e.get('duration_ms', 0))})"
            )
            return f"{prefix}[{kind}] {name}{tail}"

        span_ids = [e["span_id"] for e in ordered]
        ordered_by_id = {e["span_id"]: e for e in ordered}
        try:
            idx = span_ids.index(selected_span_id) if selected_span_id else 0
        except ValueError:
            idx = 0
        picked = st.radio(
            "Select span",
            options=span_ids,
            index=idx,
            format_func=lambda sid: _radio_label(ordered_by_id[sid]),
            label_visibility="collapsed",
            key=f"{key_prefix}:radio:{trace_id}",
        )
        if picked != selected_span_id:
            st.session_state[selected_key] = picked
            st.rerun()

    # ── Right pane ──────────────────────────────────────────────────────────
    with right:
        if selected_span_id and selected_span_id in tree["by_id"]:
            evt = tree["by_id"][selected_span_id]
        else:
            evt = ordered[0] if ordered else None

        if evt is None:
            st.info("Select a span to inspect.")
            return

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
                "**OK**"
                if evt.get("status") == "ok"
                else f"**{evt.get('status', '?')}**"
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

        if kind in ("llm_call", "classify", "synthesis"):
            tabs = st.tabs(["Messages", "Attributes", "Payload", "Raw"])
            with tabs[0]:
                payload = evt.get("payload", {}) or {}
                msgs = payload.get("messages")
                if msgs:
                    for m in msgs:
                        role = m.get("role", "?")
                        content = m.get("content")
                        with st.expander(
                            f"**{role}**",
                            expanded=(role in ("user", "assistant")),
                        ):
                            if isinstance(content, str):
                                st.markdown(f"```\n{content}\n```")
                            else:
                                st.json(content)
                            if "tool_calls" in m:
                                st.caption("tool_calls")
                                st.json(m["tool_calls"])
                elif "assistant_content" in payload:
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
