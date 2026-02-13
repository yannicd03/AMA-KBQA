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

PROVIDERS = ["openrouter", "kit"]

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
        max_value=1.0,
        value=float(edited["llm"].get("chat_temperature", 0.2)),
        step=0.05,
    )
    edited["llm"]["chat_temperature"] = chat_temp

    chat_max_tokens = st.number_input(
        "Chat Max Tokens",
        min_value=100,
        max_value=100000,
        value=int(edited["llm"].get("chat_max_tokens", 15000)),
        step=500,
    )
    edited["llm"]["chat_max_tokens"] = chat_max_tokens

# ── Synthesis Configuration ──────────────────────────────────────────────────
st.markdown("### Synthesis Configuration")

synthesis = edited.get("synthesis", {})
syn_col1, syn_col2 = st.columns(2)

with syn_col1:
    syn_model = st.text_input(
        "Synthesis Model",
        value=synthesis.get("synthesis_model", ""),
    )
    synthesis["synthesis_model"] = syn_model

with syn_col2:
    syn_temp = st.slider(
        "Synthesis Temperature",
        min_value=0.0,
        max_value=1.0,
        value=float(synthesis.get("synthesis_temperature", 0.3)),
        step=0.05,
    )
    synthesis["synthesis_temperature"] = syn_temp

edited["synthesis"] = synthesis

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
