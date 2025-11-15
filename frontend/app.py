import streamlit as st
import time
from datetime import datetime

# --- AGENT IMPORT ---
from ama_kbqa.orchestrator_agent.agent import OrchestratorAgent

# --- KONFIGURATION ---
st.set_page_config(
    layout="wide", 
    page_title="KI-Wissensassistent",
    page_icon="🧠",
    initial_sidebar_state="collapsed" # Sidebar ist weg, aber dies ist ein Fallback
)

# --- CUSTOM CSS FÜR MODERNES DARK MODE STYLING ---
st.markdown("""
<style>
    /* Global & Body */
    body {
        color: #e0e0e0;
    }
    
    /* Hauptcontainer */
    .main {
        background: linear-gradient(135deg, #1f1c2c 0%, #0c0c0e 100%);
        padding: 2rem;
        color: #e0e0e0;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff;
    }
    
    /* Chat Container */
    .chat-container {
        background: rgba(44, 44, 46, 0.6); /* Semi-transparentes dunkles Glas */
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 10px 40px rgba(0,0,0,0.4);
        margin-bottom: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
    }
    
    /* User Message */
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.2rem 1.5rem;
        border-radius: 18px;
        margin: 1rem 0;
        margin-left: 20%;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        animation: slideInRight 0.3s ease-out;
    }
    
    /* Assistant Message */
    .assistant-message {
        background: #2C2C2E; /* Sauberer, dunkler Hintergrund */
        color: #f0f0f0;
        padding: 1.2rem 1.5rem;
        border-radius: 18px;
        margin: 1rem 0;
        margin-right: 20%;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        border: 1px solid #444;
        animation: slideInLeft 0.3s ease-out;
    }
    
    /* === Die "Nachdenken..." Animation === */
    .typing-animation {
        display: flex;
        align-items: center;
        padding-top: 10px;
        padding-left: 5px;
    }
    .typing-animation span {
        width: 8px;
        height: 8px;
        background-color: #aaa;
        border-radius: 50%;
        margin: 0 4px;
        opacity: 0;
        animation: typing-bounce 1.4s infinite;
    }
    .typing-animation span:nth-child(1) { animation-delay: 0.2s; }
    .typing-animation span:nth-child(2) { animation-delay: 0.4s; }
    .typing-animation span:nth-child(3) { animation-delay: 0.6s; }
    
    @keyframes typing-bounce {
        0% {
            opacity: 0.2;
            transform: translateY(0);
        }
        50% {
            opacity: 1;
            transform: translateY(-6px);
        }
        100% {
            opacity: 0.2;
            transform: translateY(0);
        }
    }
    /* === Ende Animation === */

    
    /* Agent Process Steps */
    .agent-step {
        background: #2a2a2d;
        color: #f0f0f0;
        border-left: 4px solid #667eea;
        padding: 1rem 1.5rem;
        margin: 0.5rem 0;
        border-radius: 8px;
        font-family: 'Monaco', monospace;
        font-size: 0.9rem;
        transition: all 0.3s ease;
    }
    
    .agent-step:hover {
        background: #3a3a3d;
        transform: translateX(5px);
    }
    
    .step-orchestrator { border-left-color: #667eea; }
    .step-agent { border-left-color: #f5576c; }
    .step-system { border-left-color: #4ecdc4; }
    .step-evaluation { border-left-color: #95e1d3; }
    
    /* Header Styling */
    .main-header {
        text-align: center;
        color: white;
        margin-bottom: 2rem;
        animation: fadeIn 0.6s ease-out;
    }
    
    .main-header h1 {
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    .main-header p {
        font-size: 1.2rem;
        opacity: 0.9;
    }
    
    /* Input Field */
    .stTextInput > div > div > input {
        background-color: #2C2C2E;
        color: white;
        border: 1px solid #555;
        border-radius: 25px;
        padding: 1rem 1.5rem;
        font-size: 1rem;
        transition: all 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.3);
    }
    
    /* Sidebar entfernen */
    .sidebar .sidebar-content {
        display: none;
    }
    
    /* Animations */
    @keyframes slideInRight {
        from { opacity: 0; transform: translateX(50px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    @keyframes slideInLeft {
        from { opacity: 0; transform: translateX(-50px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
</style>
""", unsafe_allow_html=True)

# --- SESSION STATE INITIALISIERUNG ---
# (Wird für die Statistikberechnung benötigt, auch wenn sie nicht angezeigt wird)
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'query_count' not in st.session_state:
    st.session_state.query_count = 0
if 'processing_time' not in st.session_state:
    st.session_state.processing_time = []

# --- AGENT INITIALISIERUNG ---
@st.cache_resource
def get_orchestrator_agent():
    """Initialisiert den Orchestrator Agenten nur einmal."""
    return OrchestratorAgent("orch", "session-123")

orchestrator = get_orchestrator_agent()

# --- SIDEBAR WURDE ENTFERNT ---

# --- HILFSFUNKTIONEN ---
def parse_agent_response(response_string):
    """
    Parst die Agent-Antwort und extrahiert strukturierte Informationen.
    Fokussiert sich auf das saubere Extrahieren der 'final_answer'.
    """
    lines = response_string.split('\n')
    
    parsed = {
        'user_query': '',
        'routing': '',
        'agent_used': '',
        'iterations': [],
        'evaluation': '',
        'final_answer': ''
    }
    
    current_iteration = None
    capture_final = False
    
    for line in lines:
        line = line.strip()

        # --- Status 1: Finale Antwort wird erfasst ---
        if capture_final:
            if line.startswith('['): # Stoppen, wenn ein neuer Tag beginnt
                capture_final = False
            elif line: # Leere Zeilen ignorieren, aber Text hinzufügen
                parsed['final_answer'] += line + '\n'
                continue # Sofort zur nächsten Zeile
        
        # --- Status 2: Tags parsen ---
        if '[ORCHESTRATOR → USER] Finale Antwort:' in line:
            capture_final = True
            parts = line.split('Finale Antwort:', 1)
            if len(parts) > 1 and parts[1].strip():
                # Wenn Text auf derselben Zeile ist, nimm ihn
                parsed['final_answer'] += parts[1].strip() + '\n'
            continue # Zur nächsten Zeile, um mit dem Erfassen zu beginnen
        
        # Andere Tags (nur parsen, wenn WIR NICHT die finale Antwort erfassen)
        elif '[USER]' in line:
            parsed['user_query'] = line.split('[USER]', 1)[1].strip()
        elif '[ORCHESTRATOR] Routing-Entscheidung:' in line:
            parsed['routing'] = line.split(':', 1)[1].strip()
            parsed['agent_used'] = parsed['routing']
        elif '[SYSTEM]' in line:
            parsed['system_info'] = line.split('[SYSTEM]', 1)[1].strip()
        elif '[ORCHESTRATOR → ' in line and 'Iteration' in line:
            current_iteration = {'query': '', 'response': ''}
            parsed['iterations'].append(current_iteration)
        elif current_iteration and 'Query:' in line:
            current_iteration['query'] = line.split('Query:', 1)[1].strip()
        elif '→ ORCHESTRATOR]' in line and current_iteration:
            current_iteration['response'] = line.split(']', 1)[1].strip()
        elif '[ORCHESTRATOR] Evaluation:' in line:
            parsed['evaluation'] = line.split(':', 1)[1].strip()

    parsed['final_answer'] = parsed['final_answer'].strip()
    return parsed

def handle_agent_call(question: str):
    """Ruft den Orchestrator Agenten auf."""
    try:
        start_time = time.time()
        agent_response_string = orchestrator.ask(question)
        processing_time = time.time() - start_time
        
        st.session_state.processing_time.append(processing_time)
        
        return {
            "status": "success",
            "answer_text": agent_response_string,
            "processing_time": processing_time
        }
    except Exception as e:
        return {
            "status": "error",
            "answer_text": f"Fehler bei der Verarbeitung: {str(e)}",
            "error_message": str(e)
        }

# --- HAUPTBEREICH ---
st.markdown("""
<div class="main-header">
    <h1>🧠 KI-Wissensassistent</h1>
    <p>Stellen Sie Ihre Fragen – Künstliche Intelligenz findet die Antworten</p>
</div>
""", unsafe_allow_html=True)

# Chat-Container
chat_container = st.container()

with chat_container:
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    # Eingabefeld
    user_input = st.text_input(
        label="Ihre Frage",
        placeholder="💬 Was möchten Sie wissen?",
        key="user_input_key",
        label_visibility="collapsed"
    )
    
    # Verarbeitung der Benutzereingabe
    if user_input:
        st.session_state.query_count += 1
        
        # User Message anzeigen
        st.markdown(f"""
        <div class="user-message">
            <strong>👤 Sie:</strong><br>
            {user_input}
        </div>
        """, unsafe_allow_html=True)
        
        # 1. Platzhalter für "Nachdenken"-Animation
        thinking_placeholder = st.empty()
        
        # 2. Animation anzeigen
        thinking_placeholder.markdown(f"""
        <div class="assistant-message">
            <strong>🤖 Assistent:</strong>
            <div class="typing-animation">
                <span></span><span></span><span></span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # 3. Agent aufrufen
        backend_response = handle_agent_call(user_input)
        
        # 4. Platzhalter LEEREN
        thinking_placeholder.empty()

        # 5. Antwort verarbeiten und anzeigen
        if backend_response["status"] == "success":
            parsed = parse_agent_response(backend_response["answer_text"])
            
            # Fallback, falls das Parsen der finalen Antwort fehlschlägt
            if parsed['final_answer']:
                final_answer_display = parsed['final_answer'].replace(chr(10), '<br>')
            else:
                final_answer_display = f"""
                <i>[Fehler beim Parsen der Antwort.]</i>
                <br><br>
                <strong>Rohe Antwort:</strong>
                <pre>{backend_response['answer_text']}</pre>
                """
            
            # --- ANTWORT-BLOCK ---
            with st.container():
                # 5a. Finale Antwort
                st.markdown(f"""
                <div class="assistant-message">
                    <strong>🤖 Assistent:</strong><br><br>
                    {final_answer_display}
                </div>
                """, unsafe_allow_html=True)
                
                # 5b. Metadaten
                st.caption(f"⏱️ Antwortzeit: {backend_response['processing_time']:.2f}s | Routing: {parsed['agent_used'] or 'Unbekannt'}")
                
                # 5c. Verarbeitungsschritte (IMMER ANZEIGEN, aufgeklappt)
                with st.expander("🔍 Verarbeitungsschritte", expanded=True):
                    if not parsed['routing'] and not parsed['iterations'] and not parsed['evaluation']:
                        st.info("Keine detaillierten Verarbeitungsschritte in der Antwort gefunden.")
                    
                    if parsed['routing']:
                        st.markdown(f"""
                        <div class="agent-step step-orchestrator">
                            <strong>🎯 Routing:</strong> {parsed['routing']}
                        </div>
                        """, unsafe_allow_html=True)
                    
                    for i, iteration in enumerate(parsed['iterations'], 1):
                        if iteration['query']:
                            st.markdown(f"""
                            <div class="agent-step step-agent">
                                <strong>🔄 Iteration {i}:</strong> {iteration['query']}
                            </div>
                            """, unsafe_allow_html=True)
                        if iteration['response']:
                            st.markdown(f"""
                            <div class="agent-step step-agent">
                                <strong>💬 Antwort:</strong> {iteration['response'][:200]}...
                            </div>
                            """, unsafe_allow_html=True)
                    
                    if parsed['evaluation']:
                        st.markdown(f"""
                        <div class="agent-step step-evaluation">
                            <strong>✅ Evaluation:</strong> {parsed['evaluation']}
                        </div>
                        """, unsafe_allow_html=True)
                
                # 5d. Debug-Info (Immer verfügbar, eingeklappt)
                with st.expander("🔧 Debug-Informationen (Rohe Agenten-Antwort)"):
                    st.code(backend_response["answer_text"], language="text")
        
        else:
            # Bei Fehler (z.B. Agent-Absturz)
            st.error(f"❌ {backend_response['answer_text']}")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Footer (ohne den grauen Balken darüber)
st.markdown("""
<div style='text-align: center; color: #aaa; padding: 1rem;'>
    <p>Powered by Multi-Agent Orchestrator | Made with ❤️ and Streamlit</p>
</div>
""", unsafe_allow_html=True)