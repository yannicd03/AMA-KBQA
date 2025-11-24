
# app.py — Orchestrator mit MCP-STDIO-Integration und Toolliste für das LLM
from __future__ import annotations
import os
import sys
import asyncio
from contextlib import AsyncExitStack
from typing import Any, Dict, List, Optional, Tuple

from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

# MCP-Client SDK (offiziell)
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

load_dotenv(override=True)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "openai/gpt-4o")
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", "Du bist ein Orchestrator. Du delegierst Aufgaben an spezialisierte Agenten.")
REQUEST_TIMEOUT_SECONDS = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "60"))
ENABLE_REASONING = os.getenv("ENABLE_REASONING", "true").lower() in ("true", "1", "yes")
HTTP_REFERER = os.getenv("OFFICIAL_SITE_URL")
X_TITLE = os.getenv("APP_NAME_FOR_REFERER")

# Pfad zu deinem MCP-Server-Script (STDIO)
MCP_SERVER_PATH = "/Users/Admin/DEV_UNI/AMAKBQA/ama_kbqa/server/orchestrator_server.py"

# ---------- MCP STDIO Client-Manager ----------

class MCPServerManager:
    """
    Startet den MCP-Server via STDIO als Subprozess, listet Tools und
    führt Toolaufrufe aus. Der Prozess wird nach Nutzung sauber geschlossen.
    (Siehe offizielle MCP-Client-Guides.)  # Referenzen unten
    """

    def __init__(self, server_script_path: str):
        self.server_script_path = server_script_path
        self._session: Optional[ClientSession] = None
        self._stack: Optional[AsyncExitStack] = None

    async def __aenter__(self) -> "MCPServerManager":
        self._stack = AsyncExitStack()
        # Wichtig: sys.executable statt "python", damit venv korrekt ist
        transport = await self._stack.enter_async_context(stdio_client(
            StdioServerParameters(
                command=sys.executable,
                args=[self.server_script_path],
                env=None
            )
        ))
        read, write = transport
        self._session = await self._stack.enter_async_context(ClientSession(read, write))
        await self._session.initialize()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if self._stack:
            await self._stack.aclose()

    async def list_tools(self):
        assert self._session is not None
        # tools/list → liefert Tool-Metadaten (Name, Desc, InputSchema, Meta)
        resp = await self._session.list_tools()
        return resp.tools

    async def call_tool(self, name: str, arguments: Optional[Dict[str, Any]] = None):
        assert self._session is not None
        call_res = await self._session.call_tool(name, arguments or {})
        # Ergebnis robust extrahieren:
        if hasattr(call_res, "data") and call_res.data is not None:
            return call_res.data
        if getattr(call_res, "content", None):
            for block in call_res.content:
                if getattr(block, "type", "") == "text" and getattr(block, "text", None):
                    return block.text
        return None


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

        # Basis-Systemprompt
        if SYSTEM_PROMPT:
            self._messages.append({"role": "system", "content": SYSTEM_PROMPT})

        # Lazy-Cache für Sub-Agenten
        self._agent_cache: Dict[str, Any] = {}

        # Verfügbare Sub-Agenten
        self._available_agents = {
            "kqapro_agent": {
                "module": "ama_kbqa.agents.kqapro_agent.agent",
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

        # Pfad zum MCP-Server
        self.mcp_server_path = MCP_SERVER_PATH

    # ---------- Sub-Agent Lazy Loader ----------

    def _load_agent(self, agent_name: str) -> Any:
        if agent_name in self._agent_cache:
            self.dialogue_log.append(f"[SYSTEM] Agent '{agent_name}' aus Cache geladen")
            return self._agent_cache[agent_name]

        if agent_name not in self._available_agents:
            raise ValueError(f"Agent '{agent_name}' nicht verfügbar")

        config = self._available_agents[agent_name]
        try:
            module = __import__(config["module"], fromlist=[config["class"]])
            agent_class = getattr(module, config["class"])
            init_kwargs = config.get("init_kwargs", {})
            agent = agent_class(
                name=agent_name,
                session_id=self.session_id,
                **init_kwargs
            )
            self._agent_cache[agent_name] = agent
            self.dialogue_log.append(f"[SYSTEM] Agent '{agent_name}' erstellt und gecacht")
            return agent
        except Exception as e:
            raise RuntimeError(f"Fehler beim Laden von Agent '{agent_name}': {e}")

    # ---------- LLM Call ----------

    def _call_llm(self, messages: List[Dict[str, Any]]) -> str:
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

    # ---------- Routing ----------

    def _decide_routing(self, user_query: str) -> Optional[str]:
        if not self._available_agents:
            return None

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

        if decision in self._available_agents:
            return decision

        for agent_name in self._available_agents.keys():
            if agent_name in decision or decision in agent_name:
                self.dialogue_log.append(f"[ORCHESTRATOR] Fuzzy-Match gefunden: {agent_name}")
                return agent_name

        return None

    # ---------- Utility: MCP-Hinweis → Agent ----------

    def _map_mcp_db_hint_to_agent(self, hint: str) -> Optional[str]:
        if not hint:
            return None
        h = hint.lower()
        if "kqapro" in h or "knowledge" in h or "entity" in h or "relation" in h:
            return "kqapro_agent"
        if "code" in h or "python" in h or "programm" in h or "debug" in h:
            return "code_agent"
        if "math" in h or "algebra" in h or "statistik" in h or "berechn" in h:
            return "math_agent"
        for name in self._available_agents.keys():
            if name in h:
                return name
        return None

    # ---------- Hauptmethode ----------

    def ask(self, user_text: str, max_iterations: int = 3) -> str:
        if not user_text.strip():
            raise ValueError("Leerer Prompt nicht erlaubt.")

        self.dialogue_log.append(f"\n{'='*60}")
        self.dialogue_log.append(f"[USER] {user_text}")
        self.dialogue_log.append(f"{'='*60}\n")

        # 1) MCP-Server starten (STDIO), TOOLS LISTEN und databaseSearch AUSFÜHREN
        #    → Die Toolliste wird dem LLM als System-Kontext gegeben (sichtbar).
        tools = []
        db_hint_str = ""
        try:
            self.dialogue_log.append("[MCP] Starte MCP-Server (STDIO)...")
            async def mcp_phase():
                async with MCPServerManager(self.mcp_server_path) as mcp:
                    listed = await mcp.list_tools()  # tools/list
                    db = await mcp.call_tool("databaseSearch", {"question": user_text})  # tools/call
                    return listed, db

            tools, db_hint_raw = asyncio.run(mcp_phase())
            db_hint_str = str(db_hint_raw) if db_hint_raw is not None else ""
            tool_lines = []
            for t in tools:
                name = getattr(t, "name", "unknown")
                desc = getattr(t, "description", "") or ""
                # InputSchema könnte groß sein — hier optional
                tool_lines.append(f"- {name}: {desc}")

            tools_for_prompt = "\n".join(tool_lines) if tool_lines else "(keine Tools gefunden)"
            self.dialogue_log.append(f"[MCP] Verfügbare Tools: {[getattr(t, 'name', 'unknown') for t in tools]}")
            self.dialogue_log.append(f"[MCP] databaseSearch-Ergebnis: {db_hint_str}")

            # → LLM soll Toolliste explizit sehen und die Regel kennen (databaseSearch zuerst)
            self._messages.insert(0, {
                "role": "system",
                "content": (
                    "MCP-Tools sind verfügbar und wurden bereits inspiziert.\n"
                    "VERFÜGBARE MCP-TOOLS:\n"
                    f"{tools_for_prompt}\n\n"
                    "REGELN:\n"
                    "1) Das Tool 'databaseSearch(question)' MUSS immer zuerst laufen.\n"
                    "2) Dessen Textantwort bestimmt, auf welcher Knowledge-Base/mit welchem Agenten gesucht wird.\n"
                    "3) Danach wird der passende Sub-Agent delegiert.\n\n"
                    f"databaseSearch(question='{user_text}') → Antwort: {db_hint_str}\n"
                    "Hinweis: Die Ausführung von 'databaseSearch' wurde bereits serverseitig vorgenommen."
                )
            })

        except Exception as e:
            self.dialogue_log.append(f"[MCP] Fehler beim MCP-Vorprozess: {e}")
            # Minimum: trotzdem eine Regel ins Prompt
            self._messages.insert(0, {
                "role": "system",
                "content": (
                    "MCP-Tools sollten verfügbar sein, jedoch gab es einen Fehler beim Vorprozess.\n"
                    "REGELN: Versuche dennoch, nach der passenden Knowledge-Base zu entscheiden."
                )
            })

        # 2) Routing-Entscheidung durch MCP-Hinweis bevorzugen, sonst LLM
        selected_agent_name = self._map_mcp_db_hint_to_agent(db_hint_str)
        if not selected_agent_name:
            self.dialogue_log.append("[ORCHESTRATOR] Kein klarer MCP-Hinweis, nutze LLM-Routing...")
            selected_agent_name = self._decide_routing(user_text)

        if selected_agent_name and selected_agent_name in self._available_agents:
            agent = self._load_agent(selected_agent_name)
            iterations = 0
            current_query = user_text
            agent_responses = []

            while iterations < max_iterations:
                iterations += 1
                self.dialogue_log.append(f"\n[ORCHESTRATOR → {selected_agent_name.upper()}] Iteration {iterations}")
                self.dialogue_log.append(f"Query: {current_query}\n")

                agent_answer = agent.ask(current_query)
                agent_responses.append(agent_answer)

                self.dialogue_log.append(f"[{selected_agent_name.upper()} → ORCHESTRATOR]")
                self.dialogue_log.append(f"{agent_answer}\n")

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
            self.dialogue_log.append("[ORCHESTRATOR] Kein spezialisierter Agent verfügbar, beantworte selbst...\n")
            self._messages.append({"role": "user", "content": user_text})
            final_answer = self._call_llm(self._messages)
            self.dialogue_log.append(f"[ORCHESTRATOR → USER] Antwort:")
            self.dialogue_log.append(f"{final_answer}")

        return "\n".join(self.dialogue_log)

    def reset(self):
        system = next((m for m in self._messages if m.get("role") == "system"), None)
        self._messages = []
        if system:
            self._messages.append(system)
        self.dialogue_log = []
        # Agenten bleiben im Cache!


# ---- Optionaler Direktstart für schnellen Test ----
if __name__ == "__main__":
    agent = OrchestratorAgent("orchestrator", "test-session")
    result = agent.ask("Wer ist der Regisseur von Inception?")
    print(result)
