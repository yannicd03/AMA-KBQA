from __future__ import annotations
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "openai/gpt-4o")
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", "Du bist ein Orchestrator. Du delegierst Aufgaben an spezialisierte Agenten.")
REQUEST_TIMEOUT_SECONDS = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "60"))
ENABLE_REASONING = os.getenv("ENABLE_REASONING", "true").lower() in ("true", "1", "yes")
HTTP_REFERER = os.getenv("OFFICIAL_SITE_URL")
X_TITLE = os.getenv("APP_NAME_FOR_REFERER")


class OrchestratorAgent:
    def __init__(self, name: str, session_id: str):
        if not OPENROUTER_API_KEY:
            raise RuntimeError("OPENROUTER_API_KEY fehlt in .env")
        
        self.name = name
        self.session_id = session_id
        self.dialogue_log: List[str] = []
        
        default_headers = {}
        if HTTP_REFERER:
            default_headers["HTTP-Referer"] = HTTP_REFERER
        if X_TITLE:
            default_headers["X-Title"] = X_TITLE
        
        self.client = OpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=OPENROUTER_API_KEY,
            default_headers=default_headers if default_headers else None
        )
        
        self.model = MODEL_NAME
        self.request_timeout = REQUEST_TIMEOUT_SECONDS
        self.enable_reasoning = ENABLE_REASONING
        self._messages: List[Dict[str, Any]] = []
        
        if SYSTEM_PROMPT:
            self._messages.append({"role": "system", "content": SYSTEM_PROMPT})
        
        # Cache für lazy-loaded Agenten
        self._agent_cache: Dict[str, Any] = {}
        
        # Agent-Konfiguration (welche Agenten verfügbar sind)
        self._available_agents = {
            "kqapro_agent": {
                "module": "ama_kbqa.kqapro_agent.agent",
                "class": "KQAProAgent",
                "capabilities": "Knowledge Graph Question Answering, Fakten über Entities und deren Relationen"
            },
            "code_agent": {
                "module": "ama_kbqa.placeholder_agent.agent",
                "class": "PlaceholderAgent",
                "init_kwargs": {
                    "domain": "Programmierung",
                    "capabilities": "Python, JavaScript, Code-Erklärungen, Debugging"
                },
                "capabilities": "Python, JavaScript, Code-Erklärungen, Debugging"
            },
            "math_agent": {
                "module": "ama_kbqa.placeholder_agent.agent",
                "class": "PlaceholderAgent",
                "init_kwargs": {
                    "domain": "Mathematik",
                    "capabilities": "Berechnungen, Algebra, Statistik"
                },
                "capabilities": "Berechnungen, Algebra, Statistik"
            }
        }
    
    def _load_agent(self, agent_name: str) -> Any:
        """
        Lazy Loading: Erstelle Agent nur wenn er gebraucht wird.
        
        Args:
            agent_name: Name des Agenten (z.B. 'kqapro_agent')
            
        Returns:
            Agent-Instanz
        """
        # Prüfe ob Agent bereits im Cache
        if agent_name in self._agent_cache:
            self.dialogue_log.append(f"[SYSTEM] Agent '{agent_name}' aus Cache geladen")
            return self._agent_cache[agent_name]
        
        # Prüfe ob Agent konfiguriert ist
        if agent_name not in self._available_agents:
            raise ValueError(f"Agent '{agent_name}' nicht verfügbar")
        
        config = self._available_agents[agent_name]
        
        try:
            # Dynamischer Import
            module = __import__(config["module"], fromlist=[config["class"]])
            agent_class = getattr(module, config["class"])
            
            # Erstelle Agent-Instanz
            init_kwargs = config.get("init_kwargs", {})
            agent = agent_class(
                name=agent_name,
                session_id=self.session_id,
                **init_kwargs
            )
            
            # Cache Agent
            self._agent_cache[agent_name] = agent
            self.dialogue_log.append(f"[SYSTEM] Agent '{agent_name}' erstellt und gecacht")
            
            return agent
            
        except Exception as e:
            raise RuntimeError(f"Fehler beim Laden von Agent '{agent_name}': {e}")
    
    def _call_llm(self, messages: List[Dict[str, Any]]) -> str:
        """Interne Methode für LLM-Aufrufe."""
        extra_body = None
        if self.enable_reasoning:
            extra_body = {"reasoning": {"enabled": True}}
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            timeout=self.request_timeout,
            extra_body=extra_body,
        )
        
        return response.choices[0].message.content or ""
    
    def _decide_routing(self, user_query: str) -> Optional[str]:
        """
        Entscheide, welcher Agent die Anfrage bearbeiten soll.
        
        Der Orchestrator analysiert die User-Frage und wählt den
        passendsten Agenten basierend auf deren Capabilities.
        
        Workflow:
        1. Liste alle verfügbaren Agenten mit ihren Fähigkeiten auf
        2. Frage das LLM: "Welcher Agent passt am besten?"
        3. LLM antwortet mit Agent-Namen oder 'none'
        4. Validiere dass der Agent existiert
        """
        if not self._available_agents:
            return None
        
        # Erstelle Liste der verfügbaren Agenten (ohne sie zu laden!)
        agent_list = "\n".join(
            f"- {name}: {config['capabilities']}" 
            for name, config in self._available_agents.items()
        )
        
        routing_prompt = f"""Du bist ein Routing-System. Analysiere die Benutzeranfrage und wähle den BESTEN spezialisierten Agenten.

VERFÜGBARE AGENTEN:
{agent_list}

BENUTZERANFRAGE: {user_query}

REGELN:
- Antworte NUR mit dem exakten Agent-Namen (z.B. 'kqapro_agent')
- Wenn KEIN Agent passt, antworte mit 'none'
- Wähle den spezialisiertesten Agenten für die Aufgabe
- Bei Wissensfragen über Fakten/Entities → kqapro_agent
- Bei Code/Programmierung → code_agent  
- Bei Mathematik/Berechnungen → math_agent

ANTWORT (nur Agent-Name):"""
        
        decision_messages = [
            {"role": "system", "content": "Du bist ein präziser Routing-Experte. Antworte nur mit Agent-Namen."},
            {"role": "user", "content": routing_prompt}
        ]
        
        decision = self._call_llm(decision_messages).strip().lower()
        self.dialogue_log.append(f"[ORCHESTRATOR] Routing-Entscheidung: {decision}")
        
        # Validiere dass Agent verfügbar ist
        if decision in self._available_agents:
            return decision
        
        # Fallback: Suche nach Teilstring-Match
        for agent_name in self._available_agents.keys():
            if agent_name in decision or decision in agent_name:
                self.dialogue_log.append(f"[ORCHESTRATOR] Fuzzy-Match gefunden: {agent_name}")
                return agent_name
        
        return None
    
    def ask(self, user_text: str, max_iterations: int = 3) -> str:
        """
        Hauptmethode: Beantworte die Anfrage durch Delegation an Sub-Agenten.
        
        Args:
            user_text: Die Benutzeranfrage
            max_iterations: Maximale Anzahl von Rückfragen an Agenten
        
        Returns:
            Der komplette Dialog-Log als String
        """
        if not user_text.strip():
            raise ValueError("Leerer Prompt nicht erlaubt.")
        
        self.dialogue_log.append(f"\n{'='*60}")
        self.dialogue_log.append(f"[USER] {user_text}")
        self.dialogue_log.append(f"{'='*60}\n")
        
        # Routing-Entscheidung
        selected_agent_name = self._decide_routing(user_text)
        
        if selected_agent_name and selected_agent_name in self._available_agents:
            # Lade Agent JETZT (Lazy Loading!)
            agent = self._load_agent(selected_agent_name)
            
            # Delegiere an Sub-Agent
            iterations = 0
            current_query = user_text
            agent_responses = []
            
            while iterations < max_iterations:
                iterations += 1
                self.dialogue_log.append(f"\n[ORCHESTRATOR → {selected_agent_name.upper()}] Iteration {iterations}")
                self.dialogue_log.append(f"Query: {current_query}\n")
                
                # Hole Antwort vom Sub-Agent
                agent_answer = agent.ask(current_query)
                agent_responses.append(agent_answer)
                
                self.dialogue_log.append(f"[{selected_agent_name.upper()} → ORCHESTRATOR]")
                self.dialogue_log.append(f"{agent_answer}\n")
                
                # Bewerte ob weitere Iteration nötig ist
                evaluation_prompt = f"""Ursprüngliche Frage: {user_text}

Agent-Antwort: {agent_answer}

Ist diese Antwort ausreichend um die ursprüngliche Frage zu beantworten?
Antworte nur mit 'JA' oder 'NEIN' gefolgt von einer kurzen Begründung."""
                
                eval_messages = [
                    {"role": "system", "content": "Du bewertest Antwortqualität."},
                    {"role": "user", "content": evaluation_prompt}
                ]
                
                evaluation = self._call_llm(eval_messages)
                self.dialogue_log.append(f"[ORCHESTRATOR] Evaluation: {evaluation}\n")
                
                if evaluation.strip().upper().startswith("JA"):
                    break
                
                # Formuliere Nachfrage
                if iterations < max_iterations:
                    followup_prompt = f"""Die bisherige Antwort war unvollständig.

Ursprüngliche Frage: {user_text}
Bisherige Antwort: {agent_answer}

Formuliere eine präzise Nachfrage um die fehlenden Informationen zu erhalten."""
                    
                    followup_messages = [
                        {"role": "system", "content": "Du formulierst präzise Nachfragen."},
                        {"role": "user", "content": followup_prompt}
                    ]
                    
                    current_query = self._call_llm(followup_messages)
                    self.dialogue_log.append(f"[ORCHESTRATOR] Nachfrage: {current_query}\n")
            
            # Finale Zusammenfassung
            self.dialogue_log.append(f"\n{'='*60}")
            self.dialogue_log.append("[ORCHESTRATOR] Erstelle finale Antwort...")
            self.dialogue_log.append(f"{'='*60}\n")
            
            summary_prompt = f"""Ursprüngliche Benutzeranfrage: {user_text}

Gesammelte Informationen von {selected_agent_name}:
{chr(10).join(f"- {resp}" for resp in agent_responses)}

Erstelle eine präzise, vollständige Antwort auf die ursprüngliche Frage basierend auf diesen Informationen."""
            
            self._messages.append({"role": "user", "content": summary_prompt})
            final_answer = self._call_llm(self._messages)
            
            self.dialogue_log.append(f"[ORCHESTRATOR → USER] Finale Antwort:")
            self.dialogue_log.append(f"{final_answer}")
            
        else:
            # Kein passender Agent, beantworte selbst
            self.dialogue_log.append("[ORCHESTRATOR] Kein spezialisierter Agent verfügbar, beantworte selbst...\n")
            self._messages.append({"role": "user", "content": user_text})
            final_answer = self._call_llm(self._messages)
            self.dialogue_log.append(f"[ORCHESTRATOR → USER] Antwort:")
            self.dialogue_log.append(f"{final_answer}")
        
        # Gib kompletten Dialog-Log zurück
        return "\n".join(self.dialogue_log)
    
    def reset(self):
        """Setze die Konversation zurück."""
        system = next((m for m in self._messages if m.get("role") == "system"), None)
        self._messages = []
        if system:
            self._messages.append(system)
        self.dialogue_log = []
        # Agenten bleiben im Cache!


if __name__ == "__main__":
    # Teste Orchestrator - Agenten werden automatisch bei Bedarf geladen
    agent = OrchestratorAgent("orchestrator", "test-session")
    result = agent.ask("Wer ist der Regisseur von Inception?")
    print(result)