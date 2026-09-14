"""Chat page — interactive Q&A with knowledge graph agents.

While the agent is running, an inline SVG of the paper's agent-lifecycle
figure (`fig:agent_flow`) is rendered live and highlights the stage the agent
is currently in. Below the conversation, a tabbed panel hosts the same
content as the dedicated Trace Inspector page so the user can inspect spans
without leaving the chat.

A sidebar **Simplified view** toggle hides all of the inspector chrome,
leaving only the chat bubbles and input.
"""

import contextlib
import os
import queue
import time
from datetime import datetime
from typing import Any

import streamlit as st

from ama_kbqa.frontend.utils.agent_factory import (
    AGENT_INFO,
    AGENT_SUGGESTIONS,
    ORCHESTRATOR_MODES,
    create_agent,
    is_orchestrator,
)
from ama_kbqa.frontend.utils.lifecycle_runner import (
    LiveLifecycleState,
    drain_into,
    orchestrator_visited_node_ids,
    reconstruct_orchestrator,
    start_run,
)
from ama_kbqa.frontend.utils.lifecycle_svg import LIFECYCLE_NODES, render_lifecycle_svg
from ama_kbqa.frontend.utils.live_graph_data import source_for_agent
from ama_kbqa.frontend.utils.live_graph_panel import render_live_graph_panel
from ama_kbqa.frontend.utils.orchestrator_svg import render_orchestrator_svg
from ama_kbqa.frontend.utils.styling import (
    BOT_AVATAR,
    USER_AVATAR,
    StreamlitHTMLCapture,
    ansi_to_html,
    inject_css,
)
from ama_kbqa.frontend.utils.trace_panel import render_trace_panel
from ama_kbqa.frontend.utils.chat_controls import (
    apply_chat_settings,
    available_models,
    default_model,
    display_model_name,
    price_caption,
)
from ama_kbqa.config import get_chat_temperature, get_live_graph_enabled
from ama_kbqa.pricing import estimate_cost_usd, format_cost_usd

DEMO_MODE = os.environ.get("DEMO_MODE", "0") == "1"
DEMO_MAX_QUERIES_PER_SESSION = int(os.environ.get("DEMO_MAX_QUERIES_PER_SESSION", "20"))
DEMO_MIN_SECONDS_BETWEEN_QUERIES = float(os.environ.get("DEMO_MIN_SECONDS_BETWEEN_QUERIES", "3"))

inject_css()

# Default picker entry: the paper's single-dispatch router. Federated is
# opt-in and experimental (see AGENT_INFO taglines).
DEFAULT_AGENT = "Orchestrator (Router)"


def _current_agent_selection() -> str:
    """Return the persisted agent selection, normalizing a stale value.

    A session value that no longer names a live agent (e.g. the old single
    "Orchestrator" entry from before the Router/Federated split) falls back
    to the default Router entry instead of crashing the picker.
    """
    current = st.session_state.get("agent_selection", DEFAULT_AGENT)
    if current not in AGENT_INFO:
        current = DEFAULT_AGENT
        st.session_state["agent_selection"] = current
    return current


def render_agent_picker(*, disabled: bool = False) -> str:
    """Agent selector shown as a pill next to the chat bar, like a model picker.

    Lists each agent with a one-line description of what it does and when to use
    it. On change it updates the selection in session_state, drops any persisted
    multiturn conversation (a new agent handles the next turn), and reruns.
    """
    current = _current_agent_selection()
    with st.popover(current, disabled=disabled, use_container_width=False):
        st.caption("Choose an agent")
        for name, meta in AGENT_INFO.items():
            is_sel = name == current
            if st.button(
                name,
                key=f"agentpick_{name}",
                use_container_width=True,
                type="secondary" if is_sel else "tertiary",
            ):
                if name != current:
                    st.session_state["agent_selection"] = name
                    st.session_state.pop("persistent_agent", None)
                    st.rerun()
            st.caption(meta.get("tagline", meta.get("description", "")))
    return _current_agent_selection()


# Current agent selection. The picker itself is rendered next to the chat bar
# (below); this just reads the persisted choice for the rest of the page logic.
selected_agent = _current_agent_selection()

# ── Sidebar: model, temperature, view toggle ─────────────────────────────────
with st.sidebar:
    # ── Model & temperature (demo build: KIT endpoint only) ──────────────────
    _models = available_models()
    # Default to the pinned demo model (gemma4-31b), not the server config's
    # chat_model — keeps the default stable regardless of config drift.
    _default = default_model(_models)
    _model_idx = _models.index(_default) if _default in _models else 0
    selected_model = st.selectbox(
        "Model",
        options=_models,
        index=_model_idx,
        format_func=display_model_name,
        key="chat_model_select",
        help="KIT-hosted chat model. The demo always uses the KIT endpoint.",
    )
    _cap = price_caption(selected_model)
    if _cap:
        st.caption(_cap)
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=2.0,
        value=float(get_chat_temperature()),
        step=0.05,
        key="chat_temperature_slider",
        help="Sampling temperature for the agent's LLM calls. Defaults to 1.0.",
    )
    # Apply to the in-memory config so the next run uses them (session only,
    # never written to disk). Switching models starts a fresh conversation,
    # since a persisted multiturn agent is bound to the model it was built with.
    apply_chat_settings(selected_model, temperature)
    if st.session_state.get("_active_chat_model") not in (None, selected_model):
        st.session_state.pop("persistent_agent", None)
    st.session_state["_active_chat_model"] = selected_model

    simplified = st.toggle(
        "Simplified view",
        value=False,
        help=(
            "Hide the lifecycle figure, reasoning traces, token counters, "
            "and the inspector panel — just chat."
        ),
        key="simplified_view",
    )

    # The toggle only exists where the feature does: with [frontend].live_graph
    # off the page must look exactly like the pre-feature build, down to the
    # sidebar.
    live_graph_enabled = get_live_graph_enabled()
    live_graph_on = live_graph_enabled and st.toggle(
        "Live graph",
        value=True,
        key="live_graph_view",
        help="Show the subgraph the agent gathers while it answers.",
    )

suggestions = AGENT_SUGGESTIONS.get(selected_agent, {})

# ── Header ───────────────────────────────────────────────────────────────────
title_row = st.container(horizontal=True, vertical_alignment="bottom")
with title_row:
    st.title("AMA-KBQA Assistant", anchor=False)

# ── Session state ────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

user_first_interaction = (
    ("initial_question" in st.session_state and st.session_state.initial_question)
    or ("selected_suggestion" in st.session_state and st.session_state.selected_suggestion)
)
has_message_history = len(st.session_state.messages) > 0
live_run = st.session_state.get("live_run")
traces_registry: dict = st.session_state.get("traces", {})

# The live graph panel needs a second column, so the decision has to be made
# before any page body is emitted. It shows once there is something to show:
# a run in flight or at least one completed trace. The landing view stays
# single-column (nothing has been explored yet).
panel_active = bool(
    live_graph_on
    and not simplified
    and (live_run is not None or traces_registry)
)

# ── Initial view (no messages yet, no live run) ─────────────────────────────
if not user_first_interaction and not has_message_history and not live_run:
    if not simplified:
        st.markdown("#### :gray[Explore the Knowledge Graph.]")
    with st.container():
        render_agent_picker()
        st.chat_input("Ask a question...", key="initial_question")
        if not simplified:
            st.pills(
                "Examples",
                options=suggestions.keys(),
                key="selected_suggestion",
                label_visibility="collapsed",
            )
    st.stop()

# ── Pending question carried over from the initial view ─────────────────────
# The composer (agent picker + input) is rendered lower down, attached to the
# conversation. Here we only resolve a question carried over from the landing
# page (typed or an example pill) so the run can kick off below.
pending_question = None
if st.session_state.get("initial_question"):
    pending_question = st.session_state.initial_question
if st.session_state.get("selected_suggestion"):
    pending_question = suggestions[st.session_state.selected_suggestion]

with title_row:
    if st.button("Restart", icon=":material/refresh:", disabled=live_run is not None):
        st.session_state.messages = []
        st.session_state.initial_question = None
        st.session_state.selected_suggestion = None
        # Drop the persisted multiturn agent so the next question starts a
        # brand-new conversation with no carried-over message stack.
        st.session_state.pop("persistent_agent", None)
        st.rerun()


# ── Helpers ──────────────────────────────────────────────────────────────────
def _render_message_footer(message: dict) -> None:
    parts: list[str] = []
    if "duration" in message:
        parts.append(f'{message["duration"]:.2f}s')
    if message.get("tokens"):
        t = message["tokens"]
        parts.append(
            f'{t["prompt"]:,} prompt + {t["completion"]:,} completion = '
            f'{t["total"]:,} tokens'
        )
        cost_str = format_cost_usd(
            estimate_cost_usd(message.get("model"), t["prompt"], t["completion"])
        )
        if cost_str:
            parts.append(f'~{cost_str} est.')
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
        "model": getattr(agent, "model", None),
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


def _render_subagent_pane(sub: Any, *, live: bool, now: float) -> None:
    """Render one dispatched sub-agent's detailed lifecycle figure in a pane.

    While the sub-agent is running the pane is expanded and live; once it
    finishes (or in the frozen completed view) it collapses into an expander.
    """
    with st.expander(f"{sub.display} — {sub.status}", expanded=live and sub.status == "running"):
        if live:
            svg = render_lifecycle_svg(
                sub.render_active_node_ids(now),
                sub.visited_node_ids,
                current_label=sub.current_label or None,
                active_edge_ids=sub.render_active_edge_ids(now),
            )
        else:
            svg = render_lifecycle_svg(set(), sub.visited_node_ids)
        st.markdown(f'<div class="lifecycle-wrap">{svg}</div>', unsafe_allow_html=True)


# ── Page body ────────────────────────────────────────────────────────────────
# Everything below (history, composer, thinking bubble, inspector) belongs to
# the chat column. When the graph panel is inactive the body must render
# exactly as it did before the feature existed, so the "column" is a
# nullcontext rather than a second code path.
if panel_active:
    chat_col, graph_col = st.columns([0.58, 0.42], gap="medium")
    body_container = chat_col
else:
    chat_col = graph_col = None
    body_container = contextlib.nullcontext()

with body_container:
    # ── Render past history ──────────────────────────────────────────────────
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


    # ── Composer: agent picker + input, attached like a model picker ─────────
    # Multiturn follow-ups are only implemented for directly-selected sub-agents.
    # The Orchestrator is stateless, so a "Follow up..." box would wrongly imply it
    # remembers context: hide the input for it and point to the picker instead.
    with st.container():
        render_agent_picker(disabled=live_run is not None)
        if is_orchestrator(selected_agent):
            typed = None
            if not simplified:
                st.caption(
                    "The Orchestrator answers one question at a time. Use "
                    ":material/refresh: Restart for a new question, or pick KQAPro / "
                    "SciQA above for a follow-up conversation."
                )
        else:
            typed = st.chat_input("Follow up...", disabled=live_run is not None)

    user_message = typed or pending_question


    # ── Kick off a new live run ──────────────────────────────────────────────
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

        # Multiturn: for a directly-selected sub-agent we keep ONE agent instance
        # alive across the conversation so its message stack accumulates and
        # follow-ups can resolve against prior turns (the agent self-resolves
        # coreference from the preserved tool results + answers). The Orchestrator
        # path stays stateless (fresh agent per turn), so the router is unaffected.
        multiturn_enabled = not is_orchestrator(selected_agent)
        stored = st.session_state.get("persistent_agent")
        if (
            multiturn_enabled
            and stored is not None
            and stored.get("agent_name") == selected_agent
        ):
            # Reuse the existing instance: this is a follow-up turn.
            agent = stored["agent"]
            is_continuation = True
        else:
            # First turn of a conversation, an agent switch, or the Orchestrator:
            # start fresh. Persist the new instance only for direct sub-agents.
            agent = create_agent(selected_agent)
            is_continuation = False
            if multiturn_enabled:
                st.session_state["persistent_agent"] = {
                    "agent": agent,
                    "agent_name": selected_agent,
                }
            else:
                st.session_state.pop("persistent_agent", None)

        capture_io = StreamlitHTMLCapture(_SilentPlaceholder())
        q: queue.Queue = queue.Queue()
        state = LiveLifecycleState(
            trace_id=getattr(getattr(agent, "recorder", None), "trace_id", None),
            started_at=time.time(),
        )
        # Which knowledge graph paints the nodes that carry no owner. A
        # directly-selected specialist owns every node of its run; an
        # orchestrator pick resolves to "" on purpose, because there the owner
        # of each delegate span decides the colour.
        state.graph_source_default = source_for_agent(agent)
        start_run(
            agent=agent,
            question=user_message,
            q=q,
            capture_io=capture_io,
            is_continuation=is_continuation,
        )

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


    # ── Render the live-run "thinking" bubble (chat-only chrome) ─────────────
    live_run = st.session_state.get("live_run")
    if live_run is not None:
        with st.chat_message("assistant", avatar=BOT_AVATAR):
            st.markdown(":gray[Agent is thinking…]")


    # ── Inspector panel below the conversation ───────────────────────────────
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

        lifecycle_tab, trace_tab = st.tabs(["Lifecycle", "Trace"])

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
                    # Use the held active sets so each stage stays lit for at least
                    # MIN_LIGHTUP_SECONDS even when its span closed between ticks.
                    now = time.time()
                    status_html = (
                        f'<div class="lifecycle-status">'
                        f'<span><span class="label">stage:</span> '
                        f'{state.current_label or "—"}</span>'
                        f'<span><span class="label">elapsed:</span> '
                        f'{elapsed:.1f}s</span>'
                        f'<span><span class="label">spans:</span> '
                        f'{state.span_count}</span>'
                        f"</div>"
                    )
                    if is_orchestrator(live_run["agent_name"]):
                        # Multi-agent view: orchestrator figure + a live pane per
                        # dispatched specialist showing its own Fig.1 lifecycle.
                        osvg = render_orchestrator_svg(
                            state.render_orchestrator_active_node_ids(now),
                            state.render_orchestrator_visited_node_ids(),
                            current_label=state.current_label or "starting…",
                            active_edge_ids=state.orch.render_active_edge_ids(now),
                            mode="federated" if ORCHESTRATOR_MODES.get(live_run["agent_name"]) else "router",
                        )
                        st.markdown(
                            f'<div class="lifecycle-wrap">{osvg}{status_html}</div>',
                            unsafe_allow_html=True,
                        )
                        for sub in state.subagents.values():
                            _render_subagent_pane(sub, live=True, now=now)
                    else:
                        svg = render_lifecycle_svg(
                            state.render_active_node_ids(now),
                            state.visited_node_ids,
                            current_label=state.current_label or "starting…",
                            active_edge_ids=state.render_active_edge_ids(now),
                        )
                        st.markdown(
                            f'<div class="lifecycle-wrap">{svg}{status_html}</div>',
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
                                "content": err,
                                "trace": ansi_to_html(capture_io.raw_buffer),
                            })
                        st.session_state["live_run"] = None
                        st.rerun()

                _live_tick()
            elif panel_trace_id:
                # Frozen view of a completed trace. Orchestrator runs get the
                # multi-agent figure (reconstructed from the trace) + collapsed
                # per-specialist panes; single-agent runs get the lifecycle figure.
                t = traces_registry[panel_trace_id]
                caption = f"completed in {t.get('duration_s', 0):.1f}s"
                if is_orchestrator(t.get("agent")):
                    orch_visited, subs = reconstruct_orchestrator(t.get("events") or [])
                    osvg = render_orchestrator_svg(
                        set(),
                        orchestrator_visited_node_ids(orch_visited, subs),
                        current_label=caption,
                        mode="federated" if ORCHESTRATOR_MODES.get(t.get("agent")) else "router",
                    )
                    st.markdown(
                        f'<div class="lifecycle-wrap">{osvg}</div>',
                        unsafe_allow_html=True,
                    )
                    for sub in subs.values():
                        _render_subagent_pane(sub, live=False, now=0.0)
                else:
                    visited = {n.id for n in LIFECYCLE_NODES}
                    svg = render_lifecycle_svg(
                        set(),
                        visited,
                        current_label=caption,
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
                        "content": err,
                        "trace": ansi_to_html(capture_io.raw_buffer),
                    })
                st.session_state["live_run"] = None
                st.rerun()

        _live_tick_silent()


# ── Live graph panel (right column) ─────────────────────────────────────────
if panel_active and graph_col is not None:
    with graph_col:
        # The panel follows the inspector's trace selector when the user has
        # one (it only exists once a second trace is recorded), otherwise the
        # newest completed run.
        _panel_tid = (
            st.session_state.get("chat_panel_trace_selector")
            or st.session_state.get("latest_trace_id")
        )
        render_live_graph_panel(
            live_run=st.session_state.get("live_run"),
            trace=traces_registry.get(_panel_tid) if _panel_tid else None,
            key_prefix="chat",
        )
