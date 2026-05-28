"""AMA KBQA Assistant - Multi-page Streamlit Application.

This is the main entry point. It configures the page and renders shared sidebar
branding. The actual pages live in the pages/ directory.
"""

import streamlit as st
from ama_kbqa.frontend.utils.styling import inject_css

st.set_page_config(
    page_title="AMA KBQA Assistant",
    page_icon="✨",
    layout="centered",
)

inject_css()

# Shared sidebar branding
with st.sidebar:
    st.markdown("### AMA KBQA")
    try:
        from ama_kbqa.config import get_chat_model_name, load_config
        config = load_config()
        provider = config["llm"]["chat_provider"]
        model = get_chat_model_name()
        st.caption(f"Provider: **{provider}**")
        st.caption(f"Model: **{model}**")
    except Exception:
        st.caption("Config not loaded")

# Landing page content (shown when navigating to the root URL)
st.markdown("# Welcome to AMA KBQA Assistant")
st.markdown(
    "Use the sidebar to navigate between pages:\n"
    "- **Chat** - Ask questions to the knowledge graph agents\n"
    "- **Evaluation** - View batch run results and charts\n"
    "- **Settings** - Configure LLM providers and parameters"
)
