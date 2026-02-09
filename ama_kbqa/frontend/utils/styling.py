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
