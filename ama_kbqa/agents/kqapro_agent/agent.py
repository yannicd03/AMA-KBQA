from __future__ import annotations
import os
from typing import Any, Dict, List
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "openai/gpt-4o")
REQUEST_TIMEOUT_SECONDS = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "60"))
HTTP_REFERER = os.getenv("OFFICIAL_SITE_URL")
X_TITLE = os.getenv("APP_NAME_FOR_REFERER")


class KQAProAgent:
    """
    Spezialisierter Agent für Knowledge Graph Question Answering.
    """
    
    def __init__(self, name: str = "kqapro_agent", session_id: str = "default"):
        if not OPENROUTER_API_KEY:
            raise RuntimeError("OPENROUTER_API_KEY fehlt in .env")
        
        self.name = name
        self.session_id = session_id
        
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
        
        self.system_prompt = """Du bist ein Experte für Knowledge Graph Question Answering (KGQA).
Du analysierst Fragen über strukturierte Wissensgraphen und beantwortest sie präzise.

Deine Fähigkeiten:
- Verständnis von Entity-Relationen in Wissensgraphen
- Beantwortung komplexer Fragen über Fakten und Zusammenhänge
- Erkennung von mehrschrittigen Reasoning-Aufgaben
- Umgang mit unvollständigen oder mehrdeutigen Anfragen

Antworte immer faktisch und präzise."""
        
        self._messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt}
        ]
    
    def get_capabilities(self) -> str:
        """Beschreibe die Fähigkeiten dieses Agenten."""
        return "Knowledge Graph Question Answering (KGQA), Fakten über Entities und deren Relationen, strukturiertes Wissen"
    
    def ask(self, query: str) -> str:
        """
        Beantworte eine Frage im Kontext von Knowledge Graphs.
        
        Args:
            query: Die Frage des Orchestrators
            
        Returns:
            Die Antwort als String
        """
        if not query.strip():
            raise ValueError("Leere Anfrage nicht erlaubt.")
        
        self._messages.append({"role": "user", "content": query})
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self._messages,
                timeout=self.request_timeout,
            )
        except Exception as e:
            if self._messages and self._messages[-1]["role"] == "user":
                self._messages.pop()
            raise RuntimeError(f"KQAProAgent API Fehler: {e}")
        
        assistant_text = response.choices[0].message.content or ""
        
        self._messages.append({
            "role": "assistant",
            "content": assistant_text
        })
        
        return assistant_text.strip()
    
    def reset(self):
        """Setze die Konversation zurück (behält System Prompt)."""
        self._messages = [
            {"role": "system", "content": self.system_prompt}
        ]


if __name__ == "__main__":
    agent = KQAProAgent()
    answer = agent.ask("Wer ist der Regisseur von Inception?")
    print(f"KQAPro Agent: {answer}")