import os
import sys
import asyncio
import json
import inspect
from pathlib import Path
from typing import Any, Dict, List, Optional
from contextlib import AsyncExitStack
from datetime import datetime
import traceback
import signal 

from dotenv import load_dotenv
from openai import OpenAI
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import Tool as McpTool
from asyncio.exceptions import CancelledError

load_dotenv()

# --- KONFIGURATION & PFAD-LOGIK ---
current_file = Path(__file__).resolve()
# Gehe zwei Ebenen hoch zur Wurzel des Projekts (ama_kbqa)
ama_kbqa_root = current_file.parents[2]
default_server_path = ama_kbqa_root / "server" / "orchestrator_server.py"

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "openai/gpt-4o")
# Nutze den Orchestrator-Serverpfad
MCP_SERVER_PATH = os.getenv("ORCHESTRATOR_SERVER_PATH", str(default_server_path))
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", "Du bist ein intelligenter Orchestrator.")

# --- TRACING & FARBEN ---

# ANSI Farben
COLOR_BLUE = '\033[94m' # Orchestrator Standard
COLOR_GREEN = '\033[92m' # Agenten-Standard / User-Query / Finale Antwort
COLOR_RED = '\033[91m' # Fehler
COLOR_YELLOW = '\033[93m' # Warnung / Info
COLOR_END = '\033[0m'

def trace(agent_name: str, msg: str, color: str = COLOR_BLUE):
    """Standardisiertes Tracing mit Zeitstempel und Agenten-Präfix."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    # Stelle sicher, dass der Agentenname die richtige Farbe hat, um Verwirrung zu vermeiden
    print(f"[{color}{timestamp}{COLOR_END}] {color}[{agent_name}]{COLOR_END} -> {msg}")


# --- MCP CLIENT (für Orchestrator und Sub-Agenten) ---

class MCPClient:
    def __init__(self, server_path: str, agent_name: str):
        self.server_path = Path(server_path)
        self.agent_name = agent_name
        self.exit_stack = AsyncExitStack()
        self.session: Optional[ClientSession] = None
        self._connected = False
        # self._process entfernt, da wir keinen direkten Zugriff mehr benötigen

    async def start(self):
        if self._connected: return
        if not self.server_path.exists():
            trace(self.agent_name, f"{COLOR_RED}MCP-Server nicht gefunden: {self.server_path}{COLOR_END}", COLOR_RED)
            raise FileNotFoundError(f"MCP-Server nicht gefunden: {self.server_path}")
        
        # Erzeuge eine `stdio_client` Instanz (ein async generator)
        client_gen = stdio_client(StdioServerParameters(command=sys.executable, args=[str(self.server_path)], env=None))
        
        # FIX: Nur read und write entpacken (Library Change: process wird nicht mehr zurückgegeben)
        read, write = await self.exit_stack.enter_async_context(client_gen)
        
        self.session = await self.exit_stack.enter_async_context(ClientSession(read, write))
        await self.session.initialize()
        self._connected = True

    async def list_tools(self) -> List[McpTool]:
        if not self.session: raise RuntimeError("Not connected")
        result = await self.session.list_tools()
        return result.tools

    async def call_tool(self, name: str, args: Dict) -> str:
        if not self.session: raise RuntimeError("Not connected")
        result = await self.session.call_tool(name, arguments=args)
        if hasattr(result, "content") and result.content:
            return result.content[0].text
        return str(result)

    async def close(self):
        if self._connected:
            try:
                # FIX: Nur noch aclose aufrufen. Der ExitStack beendet den Prozess automatisch.
                await self.exit_stack.aclose()
                # Kleine Pause zur Unterstützung des asynchronen Cleanups
                await asyncio.sleep(0.05) 
            except (CancelledError, RuntimeError) as e:
                trace(self.agent_name, f"{COLOR_YELLOW}WARNUNG: MCP-Close Fehler ({type(e).__name__}).{COLOR_END}", COLOR_YELLOW)
            except Exception as e:
                 trace(self.agent_name, f"{COLOR_RED}Fehler beim Schließen des MCP-Clients: {e}{COLOR_END}", COLOR_RED)
            finally:
                self._connected = False


# --- Orchestrator ---

class Orchestrator:
    
    def __init__(self, session_id: str = "default"):
        self.name = "ORCHESTRATOR"
        self.session_id = session_id
        # Der OpenAI-Client wird mit der OpenRouter-Basis-URL und dem Key initialisiert
        self.client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)
        self.model = MODEL_NAME
        self.mcp: Optional[MCPClient] = None
        self._agents = {}
        
        self._agent_config = {
            "kqapro_agent": {
                "module": "ama_kbqa.agents.kqapro_agent.agent",
                "class": "KQAProAgent",
                "description": "Faktenwissen, Knowledge Graph, Beziehungen"
            },
            "code_agent": {
                "module": "ama_kbqa.placeholder_agent.agent",
                "class": "PlaceholderAgent",
                "init_kwargs": {"domain": "Coding", "capabilities": "Python, Algorithmen"},
                "description": "Programmierung, Python"
            },
            "math_agent": {
                "module": "ama_kbqa.placeholder_agent.agent",
                "class": "PlaceholderAgent",
                "init_kwargs": {"domain": "Math", "capabilities": "Gleichungen, Algebra"},
                "description": "Rechnen, Mathematik"
            }
        }

    def _trace(self, msg: str, color: str = COLOR_BLUE):
        trace(self.name, msg, color)

    def _mcp_tool_to_openai(self, mcp_tool: McpTool) -> Dict:
        """Konvertiert MCP Tool-Schema in OpenAI/OpenRouter Tool-Format"""
        return {
            "type": "function",
            "function": {
                "name": mcp_tool.name,
                "description": mcp_tool.description,
                "parameters": mcp_tool.inputSchema
            }
        }

    async def _init_mcp(self):
        """Initialisiert und startet den Orchestrator MCP-Client."""
        if self.mcp: return
        try:
            self._trace(f"Starte MCP Server: {MCP_SERVER_PATH}")
            self.mcp = MCPClient(MCP_SERVER_PATH, self.name)
            await self.mcp.start()
            self._trace("MCP verbunden")
        except Exception as e:
            self._trace(f"{COLOR_RED}MCP Fehler: {e}{COLOR_END}", COLOR_RED)
            self.mcp = None

    async def _route_autonomously(self, query: str) -> Optional[str]:
        """Nutzt LLM und MCP Tools zur Agenten-Auswahl."""
        if not self.mcp: return None

        self._trace("Bereite Routing vor: Hole Tool-Definitionen vom Server (ListTools)...")

        try:
            mcp_tools = await self.mcp.list_tools()
            openai_tools = [self._mcp_tool_to_openai(t) for t in mcp_tools]
            self._trace(f"Habe {len(openai_tools)} Tools geladen. Frage nun das LLM...")
        except Exception as e:
            self._trace(f"{COLOR_RED}Fehler beim Listen der Tools: {e}{COLOR_END}", COLOR_RED)
            return None

        if not openai_tools:
            self._trace(f"{COLOR_YELLOW}Keine Tools verfügbar.{COLOR_END}", COLOR_YELLOW)
            return None

        messages = [
            {"role": "system", "content": "Du bist ein Router. Wähle das passende Tool um herauszufinden, welcher Agent zuständig ist."},
            {"role": "user", "content": f"Query: {query}"}
        ]

        try:
            completion = self.client.chat.completions.create(
                model=self.model, messages=messages, tools=openai_tools, tool_choice="required"
            )
            
            message = completion.choices[0].message
            
            if message.tool_calls:
                tool_call = message.tool_calls[0]
                func_name = tool_call.function.name
                func_args = json.loads(tool_call.function.arguments)
                
                self._trace(f"LLM hat entschieden: Rufe Tool '{func_name}' auf mit {func_args}")
                
                tool_result = await self.mcp.call_tool(func_name, func_args)
                self._trace(f"Tool Ergebnis: {tool_result}")
                
                # Mapping des Tool-Ergebnisses auf den Agenten-Namen
                res_lower = tool_result.lower()
                if "kqapro" in res_lower: return "kqapro_agent"
                if "code" in res_lower: return "code_agent"
                if "math" in res_lower: return "math_agent"
                
                self._trace(f"{COLOR_YELLOW}Tool-Ergebnis konnte keinem Agenten zugeordnet werden.{COLOR_END}", COLOR_YELLOW)
                return None
            else:
                self._trace(f"{COLOR_YELLOW}LLM hat kein Tool aufgerufen.{COLOR_END}", COLOR_YELLOW)
                return None

        except Exception as e:
            self._trace(f"{COLOR_RED}Fehler im Routing-Prozess: {e}{COLOR_END}", COLOR_RED)
            return None

    def _load_agent(self, agent_name: str):
        """Lädt und initialisiert einen Sub-Agenten."""
        if agent_name in self._agents: return self._agents[agent_name]
        if agent_name not in self._agent_config: return None
        
        try:
            cfg = self._agent_config[agent_name]
            # Importiere das Modul und die Klasse dynamisch
            mod = __import__(cfg["module"], fromlist=[cfg["class"]]) 
            cls = getattr(mod, cfg["class"])
            kwargs = cfg.get("init_kwargs", {})
            # Instanziierung des Agenten
            agent = cls(name=agent_name, session_id=self.session_id, **kwargs)
            self._agents[agent_name] = agent
            return agent
        except Exception as e:
            self._trace(f"{COLOR_RED}Ladefehler {agent_name}: {e}{COLOR_END}", COLOR_RED)
            # print(traceback.format_exc()) # Für detaillierte Debugging-Ausgabe
            return None

    async def ask(self, query: str) -> str:
        """Hauptmethode: Route die Anfrage und hole die Antwort."""
        self._trace(f"USER: {query}", COLOR_GREEN)
        
        try:
            await self._init_mcp()
            selected_agent_name = await self._route_autonomously(query)
            answer = ""
            
            print("-" * 50) # Visuelle Trennlinie

            if selected_agent_name:
                self._trace(f"Routing erfolgreich -> {selected_agent_name}", COLOR_GREEN)
                agent = self._load_agent(selected_agent_name)
                
                if agent:
                    try:
                        # Wichtig: Call muss ASYNC sein, da KQAProAgent.ask async ist
                        if inspect.iscoroutinefunction(agent.ask):
                            answer = await agent.ask(query)
                        else:
                            answer = agent.ask(query)
                    except Exception as e:
                        self._trace(f"{COLOR_RED}Agent Error: {e}{COLOR_END}", COLOR_RED)
                        self._trace("Führe LLM Fallback durch.")
                        answer = self._fallback_llm(query)
                else:
                    self._trace("Agent konnte nicht geladen werden. Führe LLM Fallback durch.")
                    answer = self._fallback_llm(query)
            else:
                self._trace("Routing fehlgeschlagen. Führe LLM Fallback durch.")
                answer = self._fallback_llm(query)
                
            return answer

        finally:
            if self.mcp:
                await self.mcp.close()
                self._trace("Orchestrator MCP-Server sauber beendet")

    def _fallback_llm(self, query: str) -> str:
        """Direkter LLM-Call, wenn kein Agent zuständig ist oder Fehler auftreten."""
        return self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": query}]
        ).choices[0].message.content


async def main():
    orchestrator = Orchestrator()
    result = await orchestrator.ask("Wer ist der Regisseur von Inception?")
    print("-" * 50) 
    print(f"\n[{COLOR_GREEN}FINALE ANTWORT{COLOR_END}]\n{result}")


if __name__ == "__main__":
    try:
        # Führe die asynchrone Hauptfunktion aus
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
    except Exception:
        # Fange unhandled exceptions außerhalb von asyncio.run()
        print(f"\n{COLOR_RED}--- KRITISCHER FEHLER IM HAUPTLAUF ---{COLOR_END}")
        traceback.print_exc()