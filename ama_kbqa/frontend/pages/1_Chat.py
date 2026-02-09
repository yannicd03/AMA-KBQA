"""Chat page - interactive Q&A with knowledge graph agents."""

import time
from contextlib import redirect_stdout

import streamlit as st
from htbuilder import div, styles
from htbuilder.units import rem

from ama_kbqa.frontend.utils.styling import (
    BOT_AVATAR,
    USER_AVATAR,
    StreamlitHTMLCapture,
    ansi_to_html,
    inject_css,
)
from ama_kbqa.frontend.utils.async_helpers import run_async
from ama_kbqa.frontend.utils.agent_factory import (
    AGENT_INFO,
    AGENT_SUGGESTIONS,
    create_agent,
)

inject_css()

# ── Sidebar: agent selector ─────────────────────────────────────────────────
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

# ── Resolve suggestions for the selected agent ──────────────────────────────
suggestions = AGENT_SUGGESTIONS.get(selected_agent, {})

# ── Header ───────────────────────────────────────────────────────────────────
st.html(div(style=styles(font_size=rem(5), line_height=1))["❉"])

title_row = st.container(horizontal=True, vertical_alignment="bottom")
with title_row:
    st.title("AMA KBQA Assistant", anchor=False)

# ── Session state ────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# Detect first interaction
user_first_interaction = (
    ("initial_question" in st.session_state and st.session_state.initial_question)
    or ("selected_suggestion" in st.session_state and st.session_state.selected_suggestion)
)
has_message_history = len(st.session_state.messages) > 0

# ── Initial view (no messages yet) ──────────────────────────────────────────
if not user_first_interaction and not has_message_history:
    st.markdown("#### :gray[Explore the Knowledge Graph.]")
    with st.container():
        st.chat_input("Ask a question...", key="initial_question")
        st.pills(
            "Examples",
            options=suggestions.keys(),
            key="selected_suggestion",
            label_visibility="collapsed",
        )
    st.stop()

# ── Chat interface ───────────────────────────────────────────────────────────
user_message = st.chat_input("Follow up...")

if not user_message:
    if "initial_question" in st.session_state and st.session_state.initial_question:
        user_message = st.session_state.initial_question
    if "selected_suggestion" in st.session_state and st.session_state.selected_suggestion:
        user_message = suggestions[st.session_state.selected_suggestion]

# Restart button
with title_row:
    if st.button("Restart", icon=":material/refresh:"):
        st.session_state.messages = []
        st.session_state.initial_question = None
        st.session_state.selected_suggestion = None
        st.rerun()

# ── Render history ───────────────────────────────────────────────────────────
for message in st.session_state.messages:
    avatar = BOT_AVATAR if message["role"] == "assistant" else USER_AVATAR

    with st.chat_message(message["role"], avatar=avatar):
        if "trace" in message and message["trace"]:
            with st.status("View reasoning trace", state="complete", expanded=False):
                st.markdown(
                    f'<div class="console-container">{message["trace"]}</div>',
                    unsafe_allow_html=True,
                )

        st.markdown(message["content"])

        # Footer: execution time + tokens
        footer_parts = []
        if "duration" in message:
            footer_parts.append(f'<span style="vertical-align: middle;">⏱️</span> {message["duration"]:.2f}s')
        if "tokens" in message:
            t = message["tokens"]
            footer_parts.append(
                f'<span style="vertical-align: middle;">🔤</span> '
                f'{t["prompt"]:,} prompt + {t["completion"]:,} completion = {t["total"]:,} tokens'
            )
        if footer_parts:
            st.markdown(
                f'<div class="execution-time">{" &nbsp;|&nbsp; ".join(footer_parts)}</div>',
                unsafe_allow_html=True,
            )

# ── Handle new interaction ───────────────────────────────────────────────────
if user_message:
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(user_message)
    st.session_state.messages.append({"role": "user", "content": user_message})

    with st.chat_message("assistant", avatar=BOT_AVATAR):
        start_time = time.time()

        status = st.status("Agent is thinking...", expanded=False)
        with status:
            st.write(":gray[Live Log Output:]")
            console_placeholder = st.empty()

        capture_io = StreamlitHTMLCapture(console_placeholder)
        agent = create_agent(selected_agent)

        try:
            with redirect_stdout(capture_io):
                print(f"\033[1;36m[Frontend]\033[0m -> Starting analysis for: '{user_message}'")
                result_text = run_async(agent.ask(user_message))
                print(f"\033[1;32m[Frontend]\033[0m -> Processing complete.")

            end_time = time.time()
            duration = end_time - start_time

            status.update(label="Response generated!", state="complete", expanded=False)
            st.markdown(result_text)

            # Token usage
            token_info = getattr(agent, "token_usage", None)
            tokens_dict = None
            if token_info and token_info.get("total_tokens", 0) > 0:
                tokens_dict = {
                    "prompt": token_info["prompt_tokens"],
                    "completion": token_info["completion_tokens"],
                    "total": token_info["total_tokens"],
                }

            # Footer
            footer_parts = [f'<span style="vertical-align: middle;">⏱️</span> {duration:.2f}s']
            if tokens_dict:
                footer_parts.append(
                    f'<span style="vertical-align: middle;">🔤</span> '
                    f'{tokens_dict["prompt"]:,} prompt + {tokens_dict["completion"]:,} completion = {tokens_dict["total"]:,} tokens'
                )
            st.markdown(
                f'<div class="execution-time">{" &nbsp;|&nbsp; ".join(footer_parts)}</div>',
                unsafe_allow_html=True,
            )

            st.session_state.messages.append({
                "role": "assistant",
                "content": result_text,
                "trace": ansi_to_html(capture_io.raw_buffer),
                "duration": duration,
                "tokens": tokens_dict,
            })

        except Exception as e:
            status.update(label="Error!", state="error")
            st.error(f"Execution error: {e}")
            st.markdown(
                f'<div class="console-container">{ansi_to_html(capture_io.raw_buffer)}</div>',
                unsafe_allow_html=True,
            )

    # State cleanup
    st.session_state.initial_question = None
    st.session_state.selected_suggestion = None
