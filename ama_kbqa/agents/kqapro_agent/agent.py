from __future__ import annotations
import os
import sys
import asyncio
import json
import inspect
from typing import Any, Dict, List, Optional
from pathlib import Path
from contextlib import AsyncExitStack
from datetime import datetime

from dotenv import load_dotenv
from openai import OpenAI
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import Tool as McpTool
from asyncio.exceptions import CancelledError

load_dotenv(override=True)

# --- TRACING & FARBEN ---

COLOR_BLUE = '\033[94m'
COLOR_GREEN = '\033[92m'
COLOR_RED = '\033[91m'
COLOR_YELLOW = '\033[93m'
COLOR_CYAN = '\033[96m'  # Added Cyan for better distinction
COLOR_END = '\033[0m'


def trace(agent_name: str, msg: str, color: str = COLOR_BLUE):
    """Standardisiertes Tracing mit Zeitstempel und Agenten-Präfix."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    # Handle multi-line messages nicely by indenting subsequent lines
    lines = msg.split('\n')
    header = f"[{color}{timestamp}{COLOR_END}] {color}[{agent_name}]{COLOR_END} -> {lines[0]}"
    print(header)
    for line in lines[1:]:
        print(f"{' ' * (len(timestamp) + 2 + len(agent_name) + 6)} {color}{line}{COLOR_END}")


# Dynamische Pfad-Ermittlung für den Sub-Agent Server
current_file = Path(__file__).resolve()
ama_kbqa_root = current_file.parents[2]
default_subagent_server_path = ama_kbqa_root / "server" / "subagent_server.py"

# --- KONFIGURATION ---
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "openai/gpt-4o")
REQUEST_TIMEOUT_SECONDS = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "60"))

MCP_SERVER_PATH = os.getenv("SUBAGENT_SERVER_PATH", str(default_subagent_server_path))


# --- MCP CLIENT FÜR SUB-AGENTEN (Angepasst) ---

class MCPClient:
    def __init__(self, server_path: str, agent_name: str):
        self.server_path = Path(server_path)
        self.agent_name = agent_name
        self.exit_stack = AsyncExitStack()
        self.session: Optional[ClientSession] = None
        self._connected = False

    async def start(self):
        if self._connected:
            return
        if not self.server_path.exists():
            trace(self.agent_name, f"{COLOR_RED}MCP-Server nicht gefunden: {self.server_path}{COLOR_END}", COLOR_RED)
            raise FileNotFoundError(f"MCP-Server nicht gefunden: {self.server_path}")

        client_gen = stdio_client(StdioServerParameters(command=sys.executable, args=[str(self.server_path)], env=None))

        read, write = await self.exit_stack.enter_async_context(client_gen)

        self.session = await self.exit_stack.enter_async_context(ClientSession(read, write))
        await self.session.initialize()
        self._connected = True

    async def list_tools(self) -> List[McpTool]:
        if not self.session:
            raise RuntimeError("Not connected")
        result = await self.session.list_tools()
        return result.tools

    async def call_tool(self, name: str, args: Dict) -> str:
        if not self.session:
            raise RuntimeError("Not connected")
        result = await self.session.call_tool(name, arguments=args)
        if hasattr(result, "content") and result.content:
            return result.content[0].text
        return str(result)

    async def close(self):
        if self._connected:
            try:
                await self.exit_stack.aclose()
                await asyncio.sleep(0.05)
            except (CancelledError, RuntimeError) as e:
                trace(
                    self.agent_name, f"{COLOR_YELLOW}WARNUNG: MCP-Close Fehler beim Beenden ignoriert ({type(e).__name__}).{COLOR_END}", COLOR_YELLOW)
            except Exception as e:
                trace(self.agent_name, f"{COLOR_RED}Fehler beim Schließen des MCP-Clients: {e}{COLOR_END}", COLOR_RED)
            finally:
                self._connected = False


# --- KQAProAgent ---

class KQAProAgent:

    def __init__(self, name: str = "kqapro_agent", session_id: str = "default"):
        if not OPENROUTER_API_KEY:
            raise RuntimeError("OPENROUTER_API_KEY fehlt in .env")

        self.name = name
        self.session_id = session_id
        self.mcp: Optional[MCPClient] = None

        self.client = OpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=OPENROUTER_API_KEY
        )

        self.model = MODEL_NAME
        self.request_timeout = REQUEST_TIMEOUT_SECONDS

        self.system_prompt = """Du bist ein Experte für Knowledge Graph Question Answering (KGQA).
Du analysierst Fragen über strukturierte Wissensgraphen und beantwortest sie präzise.

Deine Hauptaufgabe ist es, die verfügbaren Tools zu nutzen, um Fakten zu recherchieren, bevor du die finale Antwort gibst.
Wenn du eine Frage beantworten kannst, nachdem du die nötigen Tools aufgerufen hast, antworte direkt und präzise.
"""

        self._messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt}
        ]

    def _trace(self, msg: str, color: str = COLOR_GREEN):
        trace(self.name, msg, color)

    def _mcp_tool_to_openai(self, mcp_tool: McpTool) -> Dict:
        return {
            "type": "function",
            "function": {
                "name": mcp_tool.name,
                "description": mcp_tool.description,
                "parameters": mcp_tool.inputSchema
            }
        }

    async def _init_mcp(self):
        if self.mcp:
            return
        try:
            self._trace(f"Starte eigenen MCP Server: {MCP_SERVER_PATH}")
            self.mcp = MCPClient(MCP_SERVER_PATH, self.name)
            await self.mcp.start()
            self._trace("Eigener MCP verbunden")
        except Exception as e:
            self._trace(f"{COLOR_RED}Eigener MCP Fehler: {e}{COLOR_END}", COLOR_RED)
            self.mcp = None
            raise

    async def ask(self, query: str) -> str:
        self._trace(f"Eingehende Query: '{query}'", COLOR_GREEN)

        try:
            # 1. MCP starten und Tools laden
            await self._init_mcp()

            if not self.mcp:
                self._trace(f"{COLOR_YELLOW}Tool-Server nicht verfügbar. Antworte ohne Tools.{COLOR_END}", COLOR_YELLOW)
                self._messages.append({"role": "user", "content": query})
                return self._llm_call_text_only()

            mcp_tools = await self.mcp.list_tools()
            openai_tools = [self._mcp_tool_to_openai(t) for t in mcp_tools]
            self._trace(f"Habe {len(openai_tools)} interne Tools geladen.")

            # 2. Iterativer Tool-Call Loop
            self._messages.append({"role": "user", "content": query})

            while True:
                response = self._llm_call(tools=openai_tools)
                message = response.choices[0].message

                # -------------------------------------------------------
                # VERBESSERTES LOGGING: Zwischengedanken (Thoughts/Text)
                # -------------------------------------------------------
                if message.content:
                    self._trace(f"🧠 Gedanke/Text: {message.content}", COLOR_BLUE)

                # Finale Antwort (Keine Tools mehr)
                if not message.tool_calls:
                    assistant_text = message.content or ""
                    self._messages.append({"role": "assistant", "content": assistant_text})
                    self._trace("🏁 Finale Antwort vom LLM generiert.")
                    return assistant_text.strip()

                # Tool-Call(s) ausführen
                for tool_call in message.tool_calls:
                    func_name = tool_call.function.name
                    try:
                        func_args = json.loads(tool_call.function.arguments)
                    except json.JSONDecodeError:
                        func_args = {}

                    # -------------------------------------------------------
                    # VERBESSERTES LOGGING: Tool Call & Parameter
                    # -------------------------------------------------------
                    args_pretty = json.dumps(func_args, indent=2, ensure_ascii=False)
                    self._trace(f"🛠️  Tool Call: {func_name}\n   Params: {args_pretty}", COLOR_YELLOW)

                    tool_result = await self.mcp.call_tool(func_name, func_args)

                    # -------------------------------------------------------
                    # VERBESSERTES LOGGING: Tool Ergebnisse
                    # -------------------------------------------------------
                    # Kürze Ergebnis für Logs, falls es riesig ist, um Konsole nicht zu fluten
                    log_result = tool_result
                    if len(log_result) > 500:
                        log_result = log_result[:500] + f"... [truncated, total len: {len(tool_result)}]"

                    self._trace(f"🔙 Result ({func_name}): {log_result}", COLOR_CYAN)

                    self._messages.append(message)
                    self._messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": func_name,
                        "content": tool_result
                    })

        except Exception as e:
            self._trace(f"{COLOR_RED}Fehler im Agenten-Loop: {e}{COLOR_END}", COLOR_RED)
            raise

        finally:
            if self.mcp:
                await self.mcp.close()
                self._trace("Eigener MCP-Server sauber beendet")

    def _llm_call(self, tools: Optional[List[Dict[str, Any]]] = None):
        """Führt den eigentlichen API-Call aus (mit Tools)."""
        return self.client.chat.completions.create(
            model=self.model,
            messages=self._messages,
            timeout=self.request_timeout,
            tools=tools
        )

    def _llm_call_text_only(self):
        """Führt den eigentlichen API-Call aus (ohne Tools)."""
        return self.client.chat.completions.create(
            model=self.model,
            messages=self._messages,
            timeout=self.request_timeout,
        ).choices[0].message.content

    def reset(self):
        self._messages = [
            {"role": "system", "content": self.system_prompt}
        ]
        if self.mcp:
            asyncio.run(self.mcp.close())
            self.mcp = None


if __name__ == "__main__":
    async def run_test():
        agent = KQAProAgent()
        try:
            answer = await agent.ask("Wer ist der Regisseur von Inception?")
            print(f"\n[KQAPro Agent Antwort]\n{answer}")
        except Exception as e:
            print(f"Fehler im Testlauf: {e}")

    asyncio.run(run_test())
