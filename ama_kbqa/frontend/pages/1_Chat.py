"""Chat page — interactive Q&A with knowledge graph agents.

While the agent is running, an inline SVG of the paper's agent-lifecycle
figure (`fig:agent_flow`) is rendered live and highlights the stage the agent
is currently in. Below the conversation, a tabbed panel hosts the same
content as the dedicated Trace Inspector page so the user can inspect spans
without leaving the chat.

A sidebar **Simplified view** toggle hides all of the inspector chrome,
leaving only the chat bubbles and input.
"""

import os
import queue
import time
from datetime import datetime
from typing import Any

import streamlit as st

DEMO_MODE = os.environ.get("DEMO_MODE", "0") == "1"
DEMO_MAX_QUERIES_PER_SESSION = int(os.environ.get("DEMO_MAX_QUERIES_PER_SESSION", "20"))
DEMO_MIN_SECONDS_BETWEEN_QUERIES = float(os.environ.get("DEMO_MIN_SECONDS_BETWEEN_QUERIES", "3"))
from htbuilder import div, styles
from htbuilder.units import rem

from ama_kbqa.frontend.utils.agent_factory import (
    AGENT_INFO,
    AGENT_SUGGESTIONS,
    create_agent,
)
from ama_kbqa.frontend.utils.lifecycle_runner import (
    LiveLifecycleState,
    drain_into,
    start_run,
)
from ama_kbqa.frontend.utils.lifecycle_svg import LIFECYCLE_NODES, render_lifecycle_svg
from ama_kbqa.frontend.utils.styling import (
    BOT_AVATAR,
    USER_AVATAR,
    StreamlitHTMLCapture,
    ansi_to_html,
    inject_css,
)
from ama_kbqa.frontend.utils.trace_panel import render_trace_panel

inject_css()

# ── Sidebar: agent selector + view toggle ────────────────────────────────────
with st.sidebar:
    selected_agent = st.radio(
        "Agent",
        options=list(AGENT_INFO.keys()),
        index=0,
        key="agent_selection",
    )
    info = AGENT_INFO[selected_agent]
    st.info(
        f"**{selected_agent}**\n\n"
        f"{info['description']}\n\n"
        f"Databases: {info['databases']}\n\n"
        f"Tools: {info['tools']}"
    )
    simplified = st.toggle(
        "Simplified view",
        value=False,
        help=(
            "Hide the lifecycle figure, reasoning traces, token counters, "
            "and the inspector panel — just chat."
        ),
        key="simplified_view",
    )

suggestions = AGENT_SUGGESTIONS.get(selected_agent, {})

# ── Header ───────────────────────────────────────────────────────────────────
if not simplified:
    st.html(div(style=styles(font_size=rem(5), line_height=1))["❉"])
title_row = st.container(horizontal=True, vertical_alignment="bottom")
with title_row:
    st.title("AMA KBQA Assistant", anchor=False)

# ── Session state ────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

user_first_interaction = (
    ("initial_question" in st.session_state and st.session_state.initial_question)
    or ("selected_suggestion" in st.session_state and st.session_state.selected_suggestion)
)
has_message_history = len(st.session_state.messages) > 0
live_run = st.session_state.get("live_run")

# ── Initial view (no messages yet, no live run) ─────────────────────────────
if not user_first_interaction and not has_message_history and not live_run:
    if not simplified:
        st.markdown("#### :gray[Explore the Knowledge Graph.]")
    with st.container():
        st.chat_input("Ask a question...", key="initial_question")
        if not simplified:
            st.pills(
                "Examples",
                options=suggestions.keys(),
                key="selected_suggestion",
                label_visibility="collapsed",
            )
    st.stop()

# ── Chat interface ───────────────────────────────────────────────────────────
user_message = st.chat_input("Follow up...", disabled=live_run is not None)

if not user_message:
    if "initial_question" in st.session_state and st.session_state.initial_question:
        user_message = st.session_state.initial_question
    if "selected_suggestion" in st.session_state and st.session_state.selected_suggestion:
        user_message = suggestions[st.session_state.selected_suggestion]

with title_row:
    if st.button("Restart", icon=":material/refresh:", disabled=live_run is not None):
        st.session_state.messages = []
        st.session_state.initial_question = None
        st.session_state.selected_suggestion = None
        st.rerun()


# ── Helpers ──────────────────────────────────────────────────────────────────
def _render_message_footer(message: dict) -> None:
    parts: list[str] = []
    if "duration" in message:
        parts.append(
            f'<span style="vertical-align: middle;">⏱️</span> '
            f'{message["duration"]:.2f}s'
        )
    if message.get("tokens"):
        t = message["tokens"]
        parts.append(
            f'<span style="vertical-align: middle;">🔤</span> '
            f'{t["prompt"]:,} prompt + {t["completion"]:,} completion = '
            f'{t["total"]:,} tokens'
        )
    if parts:
        st.markdown(
            f'<div class="execution-time">{" &nbsp;|&nbsp; ".join(parts)}</div>',
            unsafe_allow_html=True,
        )


def _persist_completed_run(
    *,
    state: LiveLifecycleState,
    agent: Any,
    capture_io: StreamlitHTMLCapture,
    selected_agent: str,
    user_message: str,
) -> None:
    duration = (state.finished_at or time.time()) - (state.started_at or time.time())
    answer = state.answer or ""

    token_info = getattr(agent, "token_usage", None)
    tokens_dict = None
    if token_info and token_info.get("total_tokens", 0) > 0:
        tokens_dict = {
            "prompt": token_info["prompt_tokens"],
            "completion": token_info["completion_tokens"],
            "total": token_info["total_tokens"],
        }

    trace_events: list = []
    journal_snapshots: list = []
    trace_id = None
    try:
        trace_events = agent.recorder.to_dicts()
        trace_id = agent.recorder.trace_id
        journal_snapshots = list(agent.journal_snapshots)
    except AttributeError:
        pass

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "trace": ansi_to_html(capture_io.raw_buffer),
        "duration": duration,
        "tokens": tokens_dict,
        "trace_events": trace_events,
        "journal_snapshots": journal_snapshots,
        "trace_id": trace_id,
    })

    if trace_id:
        traces_registry = st.session_state.setdefault("traces", {})
        traces_registry[trace_id] = {
            "trace_id": trace_id,
            "agent": selected_agent,
            "query": user_message,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "duration_s": duration,
            "tokens": tokens_dict,
            "events": trace_events,
            "journal_snapshots": journal_snapshots,
            "answer": answer,
        }
        MAX_TRACES = 30
        if len(traces_registry) > MAX_TRACES:
            oldest = sorted(
                traces_registry.items(), key=lambda kv: kv[1]["timestamp"]
            )[: len(traces_registry) - MAX_TRACES]
            for k, _ in oldest:
                traces_registry.pop(k, None)
        st.session_state["latest_trace_id"] = trace_id


# ── Render past history ──────────────────────────────────────────────────────
for message in st.session_state.messages:
    avatar = BOT_AVATAR if message["role"] == "assistant" else USER_AVATAR

    with st.chat_message(message["role"], avatar=avatar):
        if not simplified and "trace" in message and message["trace"]:
            with st.status("View reasoning trace", state="complete", expanded=False):
                st.markdown(
                    f'<div class="console-container">{message["trace"]}</div>',
                    unsafe_allow_html=True,
                )

        st.markdown(message["content"])

        if not simplified:
            _render_message_footer(message)


# ── Kick off a new live run ──────────────────────────────────────────────────
class _SilentPlaceholder:
    """No-op stand-in: lets the worker accumulate into ``raw_buffer`` without
    making Streamlit calls from a non-main thread."""

    def markdown(self, *args, **kwargs) -> None:
        return None


if user_message and live_run is None:
    if DEMO_MODE:
        st.session_state.setdefault("demo_query_count", 0)
        if st.session_state.demo_query_count >= DEMO_MAX_QUERIES_PER_SESSION:
            st.error(
                f"Demo limit reached: {DEMO_MAX_QUERIES_PER_SESSION} queries per session. "
                "Refresh the page to start a new session."
            )
            st.stop()
        elapsed = time.time() - st.session_state.get("demo_last_query_time", 0.0)
        if elapsed < DEMO_MIN_SECONDS_BETWEEN_QUERIES:
            st.warning(
                f"Please wait {DEMO_MIN_SECONDS_BETWEEN_QUERIES - elapsed:.1f}s before the next query."
            )
            st.stop()
        st.session_state.demo_query_count += 1
        st.session_state.demo_last_query_time = time.time()

    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(user_message)
    st.session_state.messages.append({"role": "user", "content": user_message})

    agent = create_agent(selected_agent)
    capture_io = StreamlitHTMLCapture(_SilentPlaceholder())
    q: queue.Queue = queue.Queue()
    state = LiveLifecycleState(
        trace_id=getattr(getattr(agent, "recorder", None), "trace_id", None),
        started_at=time.time(),
    )
    start_run(agent=agent, question=user_message, q=q, capture_io=capture_io)

    st.session_state["live_run"] = {
        "agent": agent,
        "capture_io": capture_io,
        "queue": q,
        "state": state,
        "question": user_message,
        "agent_name": selected_agent,
    }
    st.session_state.initial_question = None
    st.session_state.selected_suggestion = None
    st.rerun()


# ── Render the live-run "thinking" bubble (chat-only chrome) ─────────────────
live_run = st.session_state.get("live_run")
if live_run is not None:
    with st.chat_message("assistant", avatar=BOT_AVATAR):
        st.markdown(":gray[Agent is thinking…]")


# ── Inspector panel below the conversation ──────────────────────────────────
traces_registry: dict = st.session_state.get("traces", {})


def _label(tid: str, traces: dict) -> str:
    t = traces[tid]
    q = (t.get("query") or "").replace("\n", " ")
    if len(q) > 60:
        q = q[:60] + "…"
    return f"{t.get('timestamp', '?')} · {t.get('agent', '?')} · {q or tid[:8]}"


if not simplified and (live_run is not None or traces_registry):
    st.divider()

    # Choose which trace the Trace + Graph tabs are scoped to. Default to the
    # latest, but expose a selector when there's more than one.
    trace_ids = sorted(
        traces_registry.keys(),
        key=lambda tid: traces_registry[tid].get("timestamp", ""),
        reverse=True,
    )
    panel_trace_id = None
    if trace_ids:
        if len(trace_ids) == 1:
            panel_trace_id = trace_ids[0]
        else:
            panel_trace_id = st.selectbox(
                "Inspect trace",
                options=trace_ids,
                index=0,
                format_func=lambda tid: _label(tid, traces_registry),
                key="chat_panel_trace_selector",
            )

    lifecycle_tab, trace_tab = st.tabs(["🔁 Lifecycle", "🔍 Trace"])

    with lifecycle_tab:
        if live_run is not None:
            state: LiveLifecycleState = live_run["state"]
            q: queue.Queue = live_run["queue"]
            agent = live_run["agent"]
            capture_io: StreamlitHTMLCapture = live_run["capture_io"]

            log_status = st.status("Live log", expanded=False)
            console_placeholder = log_status.empty()

            @st.fragment(run_every=0.4)
            def _live_tick() -> None:
                terminal = drain_into(q, state)

                elapsed = time.time() - (state.started_at or time.time())
                svg = render_lifecycle_svg(
                    state.active_node_ids,
                    state.visited_node_ids,
                    current_label=state.current_label or "starting…",
                )
                st.markdown(
                    f'<div class="lifecycle-wrap">{svg}'
                    f'<div class="lifecycle-status">'
                    f'<span><span class="label">stage:</span> '
                    f'{state.current_label or "—"}</span>'
                    f'<span><span class="label">elapsed:</span> '
                    f'{elapsed:.1f}s</span>'
                    f'<span><span class="label">spans:</span> '
                    f'{state.span_count}</span>'
                    f"</div></div>",
                    unsafe_allow_html=True,
                )
                if capture_io.raw_buffer:
                    console_placeholder.markdown(
                        f'<div class="console-container">'
                        f'{ansi_to_html(capture_io.raw_buffer)}</div>',
                        unsafe_allow_html=True,
                    )

                if terminal:
                    if state.status == "done":
                        log_status.update(
                            label="Response generated!",
                            state="complete",
                            expanded=False,
                        )
                        _persist_completed_run(
                            state=state,
                            agent=agent,
                            capture_io=capture_io,
                            selected_agent=live_run["agent_name"],
                            user_message=live_run["question"],
                        )
                    else:
                        log_status.update(
                            label="Error!", state="error", expanded=True
                        )
                        err = state.error or "Unknown error"
                        st.error(f"Execution error: {err}")
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": f"⚠️ {err}",
                            "trace": ansi_to_html(capture_io.raw_buffer),
                        })
                    st.session_state["live_run"] = None
                    st.rerun()

            _live_tick()
        elif panel_trace_id:
            # Frozen "all visited" view of the most recent completed trace.
            visited = {n.id for n in LIFECYCLE_NODES}
            t = traces_registry[panel_trace_id]
            svg = render_lifecycle_svg(
                set(),
                visited,
                current_label=f"completed in {t.get('duration_s', 0):.1f}s",
            )
            st.markdown(
                f'<div class="lifecycle-wrap">{svg}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.caption("No trace selected.")

    with trace_tab:
        if live_run is not None:
            st.caption(
                ":gray[Trace details will appear here once the run completes.]"
            )
        elif panel_trace_id:
            render_trace_panel(
                traces_registry[panel_trace_id], key_prefix="chat"
            )
        else:
            st.caption("No trace selected.")

elif simplified and live_run is not None:
    # In simplified mode we still need to drain the queue and detect
    # completion so the answer renders. Use a hidden fragment with no
    # visible output.
    state: LiveLifecycleState = live_run["state"]
    q: queue.Queue = live_run["queue"]
    agent = live_run["agent"]
    capture_io: StreamlitHTMLCapture = live_run["capture_io"]

    @st.fragment(run_every=0.4)
    def _live_tick_silent() -> None:
        terminal = drain_into(q, state)
        if terminal:
            if state.status == "done":
                _persist_completed_run(
                    state=state,
                    agent=agent,
                    capture_io=capture_io,
                    selected_agent=live_run["agent_name"],
                    user_message=live_run["question"],
                )
            else:
                err = state.error or "Unknown error"
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"⚠️ {err}",
                    "trace": ansi_to_html(capture_io.raw_buffer),
                })
            st.session_state["live_run"] = None
            st.rerun()

    _live_tick_silent()
