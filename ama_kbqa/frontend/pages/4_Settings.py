"""Settings page - edit LLM and search configuration."""

import streamlit as st

from ama_kbqa.frontend.utils.styling import inject_css
from ama_kbqa.frontend.utils.config_editor import (
    apply_to_session,
    load_config_raw,
    save_config,
)

inject_css()

st.title("Settings")

# ── Load current config ──────────────────────────────────────────────────────
try:
    config = load_config_raw()
except Exception as e:
    st.error(f"Failed to load config.toml: {e}")
    st.stop()

# We work on a deep copy so edits don't affect the loaded dict until save
import copy
edited = copy.deepcopy(config)

PROVIDERS = ["openrouter", "kit", "llamacpp"]

# ── LLM Configuration ───────────────────────────────────────────────────────
st.markdown("### LLM Configuration")

col1, col2 = st.columns(2)

with col1:
    current_provider = edited["llm"].get("chat_provider", "openrouter")
    provider_idx = PROVIDERS.index(current_provider) if current_provider in PROVIDERS else 0
    chat_provider = st.selectbox("Chat Provider", PROVIDERS, index=provider_idx)
    edited["llm"]["chat_provider"] = chat_provider

    # Pre-fill model from the selected provider section
    provider_section = edited.get(chat_provider, {})
    chat_model = st.text_input(
        "Chat Model",
        value=provider_section.get("chat_model", ""),
    )
    if chat_provider in edited:
        edited[chat_provider]["chat_model"] = chat_model

with col2:
    chat_temp = st.slider(
        "Chat Temperature",
        min_value=0.0,
        max_value=2.0,
        value=min(float(edited["llm"].get("chat_temperature", 1.0)), 2.0),
        step=0.05,
    )
    edited["llm"]["chat_temperature"] = chat_temp

    chat_max_tokens = st.number_input(
        "Chat Max Tokens",
        min_value=100,
        max_value=100000,
        value=int(edited["llm"].get("chat_max_tokens", 16000)),
        step=500,
    )
    edited["llm"]["chat_max_tokens"] = chat_max_tokens

# ── Agent Configuration ──────────────────────────────────────────────────────
st.markdown("### Agent Configuration")

agent_cfg = edited.get("agent", {})

auto_inject = st.toggle(
    "Auto-inject journal into context",
    value=bool(agent_cfg.get("auto_inject_journal", True)),
    help=(
        "When enabled, the journal summary is periodically injected into the "
        "agent's context and an answer prompt is appended after the agent "
        "calls GetJournalSummary. Disable to let the agent rely solely on its "
        "own tool-result history (GetJournalSummary is still available as a "
        "tool it can call explicitly)."
    ),
    key="agent_auto_inject_journal",
)
agent_cfg["auto_inject_journal"] = auto_inject
edited["agent"] = agent_cfg

# ── Synthesis Configuration ──────────────────────────────────────────────────
st.markdown("### Synthesis Configuration")

synthesis = edited.get("synthesis", {})

syn_enabled = st.toggle(
    "Run synthesis step",
    value=bool(synthesis.get("synthesis_enabled", True)),
    help=(
        "When enabled, a dedicated synthesis LLM call shapes the final answer "
        "from the journal. When disabled, the agent's own final message is "
        "returned directly (skips one LLM call, but answer shape is less "
        "deterministic)."
    ),
    key="syn_enabled",
)
synthesis["synthesis_enabled"] = syn_enabled

SYNTHESIS_MODES = ["benchmark", "conversational"]
current_mode = synthesis.get("synthesis_mode", "benchmark")
if current_mode not in SYNTHESIS_MODES:
    current_mode = "benchmark"
mode_labels = {
    "benchmark": "Benchmark — short exact-match answers (for evaluation)",
    "conversational": "Conversational — verbose, human-friendly answers",
}
syn_mode = st.radio(
    "Answer style",
    options=SYNTHESIS_MODES,
    index=SYNTHESIS_MODES.index(current_mode),
    format_func=lambda m: mode_labels[m],
    horizontal=True,
    key="syn_mode",
)
synthesis["synthesis_mode"] = syn_mode

syn_col1, syn_col2, syn_col3 = st.columns(3)

with syn_col1:
    current_syn_provider = synthesis.get("synthesis_provider", "openrouter")
    syn_provider_idx = (
        PROVIDERS.index(current_syn_provider)
        if current_syn_provider in PROVIDERS
        else 0
    )
    syn_provider = st.selectbox(
        "Synthesis Provider", PROVIDERS, index=syn_provider_idx, key="syn_provider"
    )
    synthesis["synthesis_provider"] = syn_provider

with syn_col2:
    syn_model = st.text_input(
        "Synthesis Model",
        value=synthesis.get("synthesis_model", ""),
    )
    synthesis["synthesis_model"] = syn_model

with syn_col3:
    syn_temp = st.slider(
        "Synthesis Temperature",
        min_value=0.0,
        max_value=2.0,
        value=min(float(synthesis.get("synthesis_temperature", 0.3)), 2.0),
        step=0.05,
    )
    synthesis["synthesis_temperature"] = syn_temp

edited["synthesis"] = synthesis

# ── Judge Configuration ──────────────────────────────────────────────────────
st.markdown("### Judge Configuration")

postproc = edited.get("postprocessing", {})
judge_col1, judge_col2, judge_col3 = st.columns(3)

with judge_col1:
    current_judge_provider = postproc.get("judge_provider", "openrouter")
    judge_provider_idx = (
        PROVIDERS.index(current_judge_provider)
        if current_judge_provider in PROVIDERS
        else 0
    )
    judge_provider = st.selectbox(
        "Judge Provider", PROVIDERS, index=judge_provider_idx, key="judge_provider"
    )
    postproc["judge_provider"] = judge_provider

with judge_col2:
    judge_model = st.text_input(
        "Judge Model",
        value=postproc.get("judge_model", ""),
    )
    postproc["judge_model"] = judge_model

with judge_col3:
    judge_temp = st.slider(
        "Judge Temperature",
        min_value=0.0,
        max_value=2.0,
        value=min(float(postproc.get("judge_temperature", 0.3)), 2.0),
        step=0.05,
    )
    postproc["judge_temperature"] = judge_temp

edited["postprocessing"] = postproc

# ── Search Configuration ─────────────────────────────────────────────────────
st.markdown("### Search Configuration")

search = edited.get("search", {})
score_threshold = st.slider(
    "Vector Search Score Threshold",
    min_value=0.0,
    max_value=1.0,
    value=float(search.get("score_threshold", 0.7)),
    step=0.05,
)
search["score_threshold"] = score_threshold
edited["search"] = search

# ── Save buttons ─────────────────────────────────────────────────────────────
st.markdown("---")
save_col1, save_col2 = st.columns(2)

with save_col1:
    if st.button("Apply to Session Only", type="secondary", use_container_width=True):
        apply_to_session(edited)
        st.success("Configuration applied to current session (not saved to disk).")

with save_col2:
    if st.button("Save to config.toml", type="primary", use_container_width=True):
        try:
            save_config(edited, backup=True)
            st.success("Configuration saved to config.toml (backup created as config.toml.bak).")
        except Exception as e:
            st.error(f"Failed to save: {e}")

# ── Current config preview ───────────────────────────────────────────────────
with st.expander("View current config.toml (raw)", expanded=False):
    import toml
    st.code(toml.dumps(edited), language="toml")
