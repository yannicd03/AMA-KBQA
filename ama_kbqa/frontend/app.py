# Copyright 2025 Snowflake Inc. (Adapted for AMA KBQA)
import streamlit as st
from htbuilder.units import rem
from htbuilder import div, styles
import datetime
import time
import asyncio
import sys
import io
import re
from contextlib import redirect_stdout

# Importiere deinen Orchestrator
try:
    from ama_kbqa.agents.orchestrator_agent.agent import Orchestrator
except ImportError:
    # Fallback für Development
    sys.path.append("../..")
    from ama_kbqa.agents.orchestrator_agent.agent import Orchestrator

st.set_page_config(page_title="AMA KBQA Assistant", page_icon="✨", layout="centered")

# -----------------------------------------------------------------------------
# CSS Styling & Animations
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    /* Keyframes für Fade-In Animation */
    @keyframes slideInUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Animation auf Chat-Nachrichten anwenden */
    .stChatMessage {
        animation: slideInUp 0.5s ease-out forwards;
    }

    /* Terminal/Console Output Style */
    .console-container {
        background-color: #0d1117; /* Sehr dunkles Grau/Blau wie Github Dark */
        color: #c9d1d9;
        font-family: 'JetBrains Mono', 'Fira Code', 'Consolas', monospace;
        font-size: 0.80rem;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #30363d;
        white-space: pre-wrap; /* Wichtig für Zeilenumbrüche */
        line-height: 1.5;
        max-height: 500px;
        overflow-y: auto;
        box-shadow: inset 0 0 10px rgba(0,0,0,0.5);
    }
    
    /* Scrollbar im Terminal hübscher machen */
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

    /* Status Container Styling */
    div[data-testid="stStatusWidget"] {
        background-color: #f0f2f6;
        border-radius: 10px;
    }

    /* Footer für Zeitangabe */
    .execution-time {
        font-size: 0.7rem;
        color: #888;
        margin-top: 0.2rem;
        display: flex;
        align-items: center;
        gap: 4px;
        opacity: 0.8;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Icons (SVG Data URIs)
# -----------------------------------------------------------------------------

BOT_AVATAR = "https://api.iconify.design/streamline:ai-technology-spark-solid.svg?color=%2360a5fa"
USER_AVATAR = "https://api.iconify.design/solar:user-circle-bold.svg?color=%23555555"

# -----------------------------------------------------------------------------
# Helper: ANSI to HTML Converter for Colored Logs
# -----------------------------------------------------------------------------

def ansi_to_html(text):
    """
    Wandelt ANSI-Terminal-Farbcodes (inkl. High Intensity) in HTML-Spans um
    und entfernt unbekannte Sequenzen.
    """
    # 1. HTML Escaping (wichtig, damit <foo> nicht als Tag interpretiert wird)
    html_text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    # 2. ANSI Mapping (Standard + High Intensity)
    ansi_codes = {
        # Resets und Styles
        r'\x1b\[0m': '</span>',
        r'\x1b\[1m': '<span style="font-weight:bold; color: #fff;">', # Bold (oft auch heller)
        
        # Standard Colors (30-37)
        r'\x1b\[30m': '<span style="color:#484f58">', # Black/Gray
        r'\x1b\[31m': '<span style="color:#ff7b72">', # Red
        r'\x1b\[32m': '<span style="color:#3fb950">', # Green
        r'\x1b\[33m': '<span style="color:#d29922">', # Yellow
        r'\x1b\[34m': '<span style="color:#58a6ff">', # Blue
        r'\x1b\[35m': '<span style="color:#bc8cff">', # Magenta
        r'\x1b\[36m': '<span style="color:#39c5cf">', # Cyan
        r'\x1b\[37m': '<span style="color:#b1bac4">', # White

        # High Intensity Colors (90-97) - Das hat gefehlt!
        r'\x1b\[90m': '<span style="color:#6e7681">', # Dark Gray
        r'\x1b\[91m': '<span style="color:#ff7b72">', # Bright Red
        r'\x1b\[92m': '<span style="color:#3fb950">', # Bright Green (in deinem Log verwendet)
        r'\x1b\[93m': '<span style="color:#d29922">', # Bright Yellow
        r'\x1b\[94m': '<span style="color:#58a6ff">', # Bright Blue
        r'\x1b\[95m': '<span style="color:#bc8cff">', # Bright Magenta
        r'\x1b\[96m': '<span style="color:#39c5cf">', # Bright Cyan
        r'\x1b\[97m': '<span style="color:#f0f6fc">', # Bright White
    }
    
    # 3. Ersetzen der bekannten Codes
    for pattern, replacement in ansi_codes.items():
        # Wir nutzen re.IGNORECASE, falls mal klein/groß gemischt wird
        html_text = re.sub(pattern, replacement, html_text, flags=re.IGNORECASE)

    # 4. Cleanup: Alle übrigen ANSI-Sequenzen entfernen (verhindert "Boxen")
    # Das Pattern fängt alles beginnend mit ESC [ ... bis zum nächsten Buchstaben ab
    ansi_cleanup_pattern = r'\x1b\[[0-9;]*[a-zA-Z]'
    html_text = re.sub(ansi_cleanup_pattern, '', html_text)

    return html_text

class StreamlitHTMLCapture(io.StringIO):
    """
    Fängt stdout ab, wandelt Farben um und updated einen Streamlit Container.
    """
    def __init__(self, placeholder):
        super().__init__()
        self.placeholder = placeholder
        self.raw_buffer = ""

    def write(self, string):
        self.raw_buffer += string
        # Umwandlung zu HTML inkl. Farben
        colored_html = ansi_to_html(self.raw_buffer)
        
        # Rendern im Console-Look
        self.placeholder.markdown(
            f'<div class="console-container">{colored_html}</div>', 
            unsafe_allow_html=True
        )

    def flush(self):
        pass

# -----------------------------------------------------------------------------
# UI Logic
# -----------------------------------------------------------------------------

SUGGESTIONS = {
    ":blue[:material/movie:] Wer ist der Regisseur von Inception?": "Wer ist der Regisseur von Inception?",
    ":green[:material/music_note:] Heavy Metal Bands wie Queen": "How many heavy metal groups are in the genre of Queen?",
    ":orange[:material/location_on:] Wo wurde Einstein geboren?": "In welcher Stadt wurde Albert Einstein geboren?",
}

# Header (Snowflake Style)
st.html(div(style=styles(font_size=rem(5), line_height=1))["❉"])

title_row = st.container(horizontal=True, vertical_alignment="bottom")
with title_row:
    st.title("AMA KBQA Assistant", anchor=False)

# Init Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Logic for Initial View vs Chat View ---
user_first_interaction = ("initial_question" in st.session_state and st.session_state.initial_question) or \
                         ("selected_suggestion" in st.session_state and st.session_state.selected_suggestion)
has_message_history = len(st.session_state.messages) > 0

if not user_first_interaction and not has_message_history:
    st.markdown("#### :gray[Erkunde den Wissensgraphen.]")
    with st.container():
        st.chat_input("Stelle eine Frage...", key="initial_question")
        st.pills("Beispiele", options=SUGGESTIONS.keys(), key="selected_suggestion", label_visibility="collapsed")
    st.stop()

# --- Chat Interface ---
user_message = st.chat_input("Nachhaken...")

if not user_message:
    if "initial_question" in st.session_state and st.session_state.initial_question:
        user_message = st.session_state.initial_question
    if "selected_suggestion" in st.session_state and st.session_state.selected_suggestion:
        user_message = SUGGESTIONS[st.session_state.selected_suggestion]

# Restart Button
with title_row:
    if st.button("Neustart", icon=":material/refresh:"):
        st.session_state.messages = []
        st.session_state.initial_question = None
        st.session_state.selected_suggestion = None
        st.rerun()

# --- Render History ---
for message in st.session_state.messages:
    avatar = BOT_AVATAR if message["role"] == "assistant" else USER_AVATAR
    
    with st.chat_message(message["role"], avatar=avatar):
        # 1. Trace Log (Expander) falls vorhanden
        if "trace" in message and message["trace"]:
            # State 'complete' macht einen grünen Rand/Haken
            with st.status("Gedankenprozess ansehen", state="complete", expanded=False):
                st.markdown(f'<div class="console-container">{message["trace"]}</div>', unsafe_allow_html=True)
        
        # 2. Content
        st.markdown(message["content"])
        
        # 3. Execution Time Footer
        if "duration" in message:
            st.markdown(f"""
            <div class="execution-time">
                <span style="vertical-align: middle;">⏱️</span> 
                {message['duration']:.2f}s
            </div>
            """, unsafe_allow_html=True)

# --- Handle New Interaction ---
if user_message:
    # User Message
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(user_message)
    st.session_state.messages.append({"role": "user", "content": user_message})

    # Assistant Response
    with st.chat_message("assistant", avatar=BOT_AVATAR):
        
        start_time = time.time()
        
        # Der Status-Container: Standardmäßig geschlossen (expanded=False)
        status = st.status("Agent denkt nach...", expanded=False)
        
        # Placeholder für die Konsole INNERHALB des Expanders
        with status:
            st.write(":gray[Live Log Output:]")
            console_placeholder = st.empty()
        
        # Log Capture Setup
        capture_io = StreamlitHTMLCapture(console_placeholder)
        orchestrator = Orchestrator()
        
        # Async Execution
        try:
            # Event Loop Handling
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            with redirect_stdout(capture_io):
                # Wir printen ein Start-Event, damit man sieht, dass es losgeht
                print(f"\033[1;36m[Frontend]\033[0m -> Starte Analyse für: '{user_message}'")
                
                # DIE AGENTEN LOGIK
                result_text = loop.run_until_complete(orchestrator.ask(user_message))
                
                print(f"\033[1;32m[Frontend]\033[0m -> Vorgang abgeschlossen.")

            end_time = time.time()
            duration = end_time - start_time
            
            # Status Update: Fertig
            status.update(label="Antwort generiert!", state="complete", expanded=False)
            
            # Finale Antwort rendern
            st.markdown(result_text)
            
            # Zeitangabe
            st.markdown(f"""
            <div class="execution-time">
                <span style="vertical-align: middle;">⏱️</span> 
                {duration:.2f}s
            </div>
            """, unsafe_allow_html=True)
            
            # In History speichern
            st.session_state.messages.append({
                "role": "assistant", 
                "content": result_text,
                "trace": ansi_to_html(capture_io.raw_buffer), # Trace HTML speichern
                "duration": duration
            })

        except Exception as e:
            status.update(label="Fehler!", state="error")
            st.error(f"Fehler bei der Ausführung: {str(e)}")
            # Bei Fehler auch den Trace zeigen
            st.error("Trace Log:")
            st.markdown(f'<div class="console-container">{ansi_to_html(capture_io.raw_buffer)}</div>', unsafe_allow_html=True)

    # State Cleanup
    st.session_state.initial_question = None
    st.session_state.selected_suggestion = None