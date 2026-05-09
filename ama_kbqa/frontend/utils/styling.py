"""Shared CSS, ANSI conversion, and avatar constants for the Streamlit frontend."""

import re
import io
import streamlit as st


BOT_AVATAR = "https://api.iconify.design/streamline:ai-technology-spark-solid.svg?color=%2360a5fa"
USER_AVATAR = "https://api.iconify.design/solar:user-circle-bold.svg?color=%23555555"

SHARED_CSS = """
<style>
    @keyframes slideInUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .stChatMessage {
        animation: slideInUp 0.5s ease-out forwards;
    }

    .console-container {
        background-color: #0d1117;
        color: #c9d1d9;
        font-family: 'JetBrains Mono', 'Fira Code', 'Consolas', monospace;
        font-size: 0.80rem;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #30363d;
        white-space: pre-wrap;
        line-height: 1.5;
        max-height: 500px;
        overflow-y: auto;
        box-shadow: inset 0 0 10px rgba(0,0,0,0.5);
    }

    .console-container::-webkit-scrollbar {
        width: 8px;
    }
    .console-container::-webkit-scrollbar-track {
        background: #0d1117;
    }
    .console-container::-webkit-scrollbar-thumb {
        background: #30363d;
        border-radius: 4px;
    }
    .console-container::-webkit-scrollbar-thumb:hover {
        background: #58a6ff;
    }

    div[data-testid="stStatusWidget"] {
        background-color: #f0f2f6;
        border-radius: 10px;
    }

    .execution-time {
        font-size: 0.7rem;
        color: #888;
        margin-top: 0.2rem;
        display: flex;
        align-items: center;
        gap: 4px;
        opacity: 0.8;
    }

    .token-info {
        font-size: 0.7rem;
        color: #888;
        display: flex;
        align-items: center;
        gap: 4px;
        opacity: 0.8;
    }

    /* === Trace Inspector ====================================================
       Langfuse-style hierarchical span tree. Each row carries a kind-coloured
       pill + name + duration + token badge. Selected row highlights via
       data-attr applied by the renderer.
    */
    .trace-tree {
        background-color: #0d1117;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 0.75rem 0.5rem;
        font-family: 'JetBrains Mono', 'Fira Code', monospace;
        font-size: 0.78rem;
        max-height: 70vh;
        overflow-y: auto;
        line-height: 1.5;
    }
    .span-row {
        display: flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.18rem 0.3rem;
        border-radius: 4px;
        white-space: nowrap;
        color: #c9d1d9;
    }
    .span-row:hover { background: #161b22; }
    .span-row.selected { background: #1f2933; outline: 1px solid #58a6ff; }
    .span-pill {
        display: inline-block;
        font-size: 0.65rem;
        font-weight: 600;
        padding: 1px 6px;
        border-radius: 4px;
        text-transform: uppercase;
        letter-spacing: 0.4px;
        flex-shrink: 0;
    }
    .kind-agent_run    { background: #4c1d95; color: #ddd6fe; }
    .kind-classify     { background: #1e3a8a; color: #bfdbfe; }
    .kind-fast_path    { background: #064e3b; color: #a7f3d0; }
    .kind-tool_loop_iter { background: #374151; color: #d1d5db; }
    .kind-llm_call     { background: #831843; color: #fbcfe8; }
    .kind-tool_call    { background: #14532d; color: #bbf7d0; }
    .kind-synthesis    { background: #7c2d12; color: #fed7aa; }
    .kind-journal_refresh { background: #1e40af; color: #bfdbfe; }
    .kind-loop_detected   { background: #7f1d1d; color: #fecaca; }
    .kind-context_trim    { background: #78350f; color: #fde68a; }
    .kind-intervention    { background: #9f1239; color: #fecdd3; }
    .kind-delegate        { background: #4c1d95; color: #ddd6fe; }

    .span-name { font-weight: 600; color: #e6edf3; }
    .span-duration {
        font-size: 0.7rem;
        color: #8b949e;
        margin-left: auto;
        padding-left: 0.5rem;
    }
    .span-tokens {
        font-size: 0.65rem;
        color: #8b949e;
        background: #161b22;
        padding: 0 5px;
        border-radius: 3px;
    }
    .span-status-error { color: #f85149; }
    .span-status-ok    { color: #3fb950; }
    .span-event-marker {
        opacity: 0.55;
        font-style: italic;
    }
    .trace-summary {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 0.6rem 1rem;
        margin-bottom: 0.75rem;
        font-size: 0.85rem;
        color: #c9d1d9;
    }
    .trace-summary-stat { color: #8b949e; }
    .trace-summary-stat strong { color: #e6edf3; margin-right: 1.2rem; }
</style>
"""


def inject_css():
    """Inject shared CSS into the Streamlit page."""
    st.markdown(SHARED_CSS, unsafe_allow_html=True)


def ansi_to_html(text):
    """Convert ANSI terminal color codes to HTML spans."""
    html_text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    ansi_codes = {
        r'\x1b\[0m': '</span>',
        r'\x1b\[1m': '<span style="font-weight:bold; color: #fff;">',
        r'\x1b\[30m': '<span style="color:#484f58">',
        r'\x1b\[31m': '<span style="color:#ff7b72">',
        r'\x1b\[32m': '<span style="color:#3fb950">',
        r'\x1b\[33m': '<span style="color:#d29922">',
        r'\x1b\[34m': '<span style="color:#58a6ff">',
        r'\x1b\[35m': '<span style="color:#bc8cff">',
        r'\x1b\[36m': '<span style="color:#39c5cf">',
        r'\x1b\[37m': '<span style="color:#b1bac4">',
        r'\x1b\[90m': '<span style="color:#6e7681">',
        r'\x1b\[91m': '<span style="color:#ff7b72">',
        r'\x1b\[92m': '<span style="color:#3fb950">',
        r'\x1b\[93m': '<span style="color:#d29922">',
        r'\x1b\[94m': '<span style="color:#58a6ff">',
        r'\x1b\[95m': '<span style="color:#bc8cff">',
        r'\x1b\[96m': '<span style="color:#39c5cf">',
        r'\x1b\[97m': '<span style="color:#f0f6fc">',
    }

    for pattern, replacement in ansi_codes.items():
        html_text = re.sub(pattern, replacement, html_text, flags=re.IGNORECASE)

    html_text = re.sub(r'\x1b\[[0-9;]*[a-zA-Z]', '', html_text)
    return html_text


class StreamlitHTMLCapture(io.StringIO):
    """Captures stdout, converts ANSI colors to HTML, and updates a Streamlit container."""

    def __init__(self, placeholder):
        super().__init__()
        self.placeholder = placeholder
        self.raw_buffer = ""

    def write(self, string):
        self.raw_buffer += string
        colored_html = ansi_to_html(self.raw_buffer)
        self.placeholder.markdown(
            f'<div class="console-container">{colored_html}</div>',
            unsafe_allow_html=True,
        )

    def flush(self):
        pass
