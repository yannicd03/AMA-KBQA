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

    /* Clickable span tree: native Streamlit buttons styled as dark rows so a
       click reruns over the websocket (no page reload). Scoped to the keyed
       container st.container(key="tracetree-<prefix>") in trace_panel.py. */
    [class*="st-key-tracetree-"] {
        background-color: #0d1117;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 0.5rem 0.4rem;
        max-height: 70vh;
        overflow-y: auto;
    }
    [class*="st-key-tracetree-"] [data-testid="stVerticalBlock"] { gap: 0.08rem; }
    [class*="st-key-tracetree-"] .stButton { margin: 0; }
    [class*="st-key-tracetree-"] .stButton > button {
        background: transparent;
        border: none;
        box-shadow: none;
        color: #c9d1d9;
        font-family: 'JetBrains Mono', 'Fira Code', monospace;
        font-size: 0.78rem;
        text-align: left;
        justify-content: flex-start;
        padding: 0.14rem 0.4rem;
        min-height: 0;
        line-height: 1.5;
        border-radius: 4px;
        width: 100%;
    }
    [class*="st-key-tracetree-"] .stButton > button p {
        margin: 0;
        white-space: nowrap;
    }
    [class*="st-key-tracetree-"] .stButton > button:hover {
        background: #161b22;
        color: #ffffff;
        border: none;
    }
    [class*="st-key-tracetree-"] .stButton > button:focus:not(:active) {
        color: #c9d1d9;
        border: none;
        box-shadow: none;
    }
    /* Selected row = type="primary" (covers old `kind` attr + new testid). */
    [class*="st-key-tracetree-"] .stButton > button[kind="primary"],
    [class*="st-key-tracetree-"] .stButton > button[data-testid="stBaseButton-primary"] {
        background: #1f2933;
        outline: 1px solid #58a6ff;
        color: #ffffff;
    }
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

    /* ── Lifecycle figure (pages/1_Chat.py live view) ─────────────────── */
    .lifecycle-wrap {
        background: #fafbfc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 0.6rem 0.8rem 0.4rem;
        margin: 0.25rem 0 0.5rem;
    }
    .lifecycle-status {
        font-size: 0.78rem;
        color: #475569;
        font-family: 'JetBrains Mono', 'Fira Code', 'Consolas', monospace;
        display: flex;
        gap: 1rem;
        margin-top: 0.35rem;
    }
    .lifecycle-status .label {
        color: #94a3b8;
    }
    .lifecycle-svg {
        width: 100%;
        height: auto;
        max-height: 360px;
        display: block;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    .lifecycle-svg .phase-bg.phase-pre  { fill: #dbe7ff; opacity: 0.55; }
    .lifecycle-svg .phase-bg.phase-main { fill: #fff1c7; opacity: 0.55; }
    .lifecycle-svg .phase-bg.phase-post { fill: #fde0ec; opacity: 0.55; }
    .lifecycle-svg .phase-title {
        font-size: 13px;
        font-weight: 700;
        fill: #475569;
        letter-spacing: 0.02em;
    }
    .lifecycle-svg .node {
        fill: white;
        stroke: #94a3b8;
        stroke-width: 1.4;
        transition: stroke 0.2s, stroke-width 0.2s, filter 0.2s;
    }
    .lifecycle-svg [data-state="visited"] .node {
        stroke: #475569;
        stroke-width: 1.7;
    }
    .lifecycle-svg [data-state="active"] .node {
        stroke: #2563eb;
        stroke-width: 2.6;
        filter: drop-shadow(0 0 6px rgba(37, 99, 235, 0.55));
        animation: lifecycle-pulse 1.4s ease-in-out infinite;
    }
    .lifecycle-svg [data-state="active"] .node-label {
        fill: #1d4ed8;
    }
    .lifecycle-svg .node-label {
        font-size: 12px;
        font-weight: 600;
        fill: #1e293b;
        pointer-events: none;
    }
    .lifecycle-svg .node-label-small {
        font-size: 11px;
    }
    .lifecycle-svg .textlabel-text {
        font-size: 11px;
        fill: #334155;
    }
    .lifecycle-svg .textlabel-head {
        font-weight: 700;
        fill: #1e293b;
    }
    .lifecycle-svg .textlabel-tail {
        font-weight: 500;
        fill: #475569;
    }
    .lifecycle-svg .edge {
        stroke: #64748b;
        stroke-width: 1.5;
        fill: none;
    }
    .lifecycle-svg .edge.dashed {
        stroke-dasharray: 5 4;
        stroke: #94a3b8;
    }
    .lifecycle-svg .edge-arrow {
        fill: #64748b;
    }
    .lifecycle-svg .edge-label {
        font-size: 10.5px;
        fill: #475569;
        font-style: italic;
    }
    .lifecycle-svg .lifecycle-caption {
        font-size: 11px;
        fill: #475569;
        font-style: italic;
    }
    @keyframes lifecycle-pulse {
        0%, 100% { opacity: 1; }
        50%      { opacity: 0.78; }
    }

    /* ── Floating "About" help button (demo) ──────────────────────────── */
    .st-key-about_help_btn {
        position: fixed;
        bottom: 1.25rem;
        right: 1.25rem;
        z-index: 1000;
        width: auto;
    }
    .st-key-about_help_btn button {
        border-radius: 50%;
        width: 2.5rem;
        min-width: 2.5rem;
        height: 2.5rem;
        padding: 0;
        font-size: 1.15rem;
        font-weight: 700;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.18);
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
