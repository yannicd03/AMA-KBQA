import streamlit as st

import time # Importiere time, um eine Verzögerung für die Simulation einzufügen

# --- WICHTIG: AGENTEN-IMPORT IST AUSKOMMENTIERT ---
from ama_kbqa.orchestrator_agent.agent import OrchestratorAgent

# --- 1. KONFIGURATION UND INITIALISIERUNG ---
st.set_page_config(layout="wide", page_title="Chat-Frontend mit Platzhalter-Agent")

# Initialisiere den Chat-Verlauf (wichtig für den Session State)
if 'messages' not in st.session_state:
    st.session_state.messages = []

@st.cache_resource
def get_orchestrator_agent():
    """Initialisiert den Orchestrator Agenten nur einmal pro Sitzung."""
    # Verwenden Sie die gewünschten Parameter für die Initialisierung
    return OrchestratorAgent("orch", "session-123")
orchestrator = get_orchestrator_agent()
# --- 2. BACKEND-AGENT-SIMULATION (PLATZHALTER) ---

def simulate_agent_call(question: str):
    """
    Simuliert den Backend-Agenten-Aufruf und gibt eine feste Platzhalter-Antwort zurück.
    Die Struktur bleibt ein Dictionary, um das Frontend für die finale Integration bereit zu halten.
    """
    # Simuliere eine kurze Wartezeit, um den Ladevorgang darzustellen
    time.sleep(1.5) 
    
    # Rückgabe der festen Platzhalter-Antwort (Die Struktur ist wie erwartet)
    return {
        "status": "success",
        "answer_text": (
            f"**Dies ist die Platzhalter-Antwort.**\n\n"
            f"Ihre gestellte Frage: **'{question}'** wurde erfolgreich an den simulierten Agenten übergeben. "
            f"Sobald `ask_agent` implementiert ist, wird die Antwort des Agenten hier erscheinen."
        ),
        "source_documents": [
            {"title": "Platzhalter Quelle A", "url": "https://deine-datenbank.de/doc1"},
            {"title": "Platzhalter Quelle B", "url": "https://deine-datenbank.de/doc2"}
        ]
    }
def handle_agent_call(question: str):
    """
    Ruft den Orchestrator Agenten auf und formatiert die String-Antwort 
    in die vom Frontend erwartete Dictionary-Struktur um.
    """
    try:
        # [GEÄNDERT] ECHTER AGENTEN-AUFRUF
        agent_response_string = orchestrator.ask(question)
        
        # WICHTIG: Die reine String-Antwort in die erwartete Struktur umwandeln
        return {
            "status": "success",
            "answer_text": agent_response_string,
            "source_documents": [] # Quellen können später hier eingefügt werden
        }
    except Exception as e:
        # Fehlerbehandlung, falls der Agenten-Aufruf fehlschlägt
        st.error(f"Fehler bei der Kommunikation mit dem Agenten: {e}")
        return {
            "status": "error",
            "answer_text": "Es ist ein technischer Fehler bei der Verarbeitung Ihrer Frage aufgetreten.",
            "error_message": str(e)
        }

# --- 3. HAUPT-UI UND CHAT-LOGIK ---

st.header("Chat-Assistent für Wissensdatenbank (KBQA)")
st.markdown("---")

# Text-Eingabefenster und Logik zur Verarbeitung der Benutzerfrage
user_input = st.text_input(
    label="Hello, what can I help you with today?", 
    placeholder="Geben Sie hier Ihre Frage ein...", 
    key="user_input_key"
)

# Führe die Backend-Logik nur aus, wenn der Benutzer etwas eingegeben hat
if user_input:
    # 1. Benutzerfrage zum Verlauf hinzufügen und anzeigen
    st.chat_message("user").write(user_input)

    # 2. Agenten-Aufruf (jetzt simuliert)
    with st.spinner("Antwort wird generiert... (Simulation)"):
        backend_response = handle_agent_call(user_input)
    
    # 3. Antwort verarbeiten und anzeigen
    if backend_response["status"] == "success":
        # Die eigentliche Antwort anzeigen
        with st.chat_message("assistant"):
            st.markdown(backend_response["answer_text"])
            
            # Quellen in einem Expander (ausklappbarer Bereich) anzeigen
            if backend_response.get("source_documents"):
                with st.expander("Quellen & Belege (Platzhalter)"):
                    for source in backend_response["source_documents"]:
                        st.markdown(f"[{source['title']}]({source['url']})")
    
    else:
        # Platzhalter-Fehlermeldung, falls wir den Status auf 'error' setzen würden
        st.error(backend_response["answer_text"])

st.divider()
st.caption("Frontend läuft im Simulationsmodus. Bereit für die Agent-Integration.")