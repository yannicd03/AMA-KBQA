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

from dotenv import load_dotenv, find_dotenv
from openai import OpenAI
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import Tool as McpTool
from asyncio.exceptions import CancelledError

load_dotenv(find_dotenv())

# --- CONFIGURATION & PATH LOGIC ---
current_file = Path(__file__).resolve()
ama_kbqa_root = current_file.parents[2]
default_server_path = ama_kbqa_root / "server" / "orchestrator_server.py"

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "minimax/minimax-m2")
MCP_SERVER_PATH = os.getenv("ORCHESTRATOR_SERVER_PATH", str(default_server_path))
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", "You are an intelligent orchestrator.")

# --- TRACING & COLORS ---

COLOR_BLUE = '\033[94m'      # Orchestrator Info
COLOR_GREEN = '\033[92m'     # Success / Final Answer
COLOR_RED = '\033[91m'       # Error
COLOR_YELLOW = '\033[93m'    # Tool Calls / Warning
COLOR_CYAN = '\033[96m'      # Reasoning / Thoughts
COLOR_MAGENTA = '\033[95m'   # Tool Inputs/Outputs
COLOR_END = '\033[0m'


def trace(agent_name: str, msg: str, color: str = COLOR_BLUE):
    """Standardized tracing with timestamp and agent prefix."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    try:
        print(f"[{color}{timestamp}{COLOR_END}] {color}[{agent_name}]{COLOR_END} -> {msg}")
    except UnicodeEncodeError:
        # Fallback to ASCII-safe output on Windows
        safe_msg = msg.encode('ascii', 'replace').decode('ascii')
        print(f"[{color}{timestamp}{COLOR_END}] {color}[{agent_name}]{COLOR_END} -> {safe_msg}")

# --- MCP CLIENT ---


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
            trace(self.agent_name, f"{COLOR_RED}MCP server not found: {self.server_path}{COLOR_END}", COLOR_RED)
            raise FileNotFoundError(f"MCP server not found: {self.server_path}")

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
                trace(self.agent_name,
                      f"{COLOR_YELLOW}WARNING: MCP-Close error ({type(e).__name__}).{COLOR_END}", COLOR_YELLOW)
            except Exception as e:
                trace(self.agent_name, f"{COLOR_RED}Error closing MCP client: {e}{COLOR_END}", COLOR_RED)
            finally:
                self._connected = False

# --- Orchestrator ---


class Orchestrator:

    def __init__(self, session_id: str = "default"):
        self.name = "ORCHESTRATOR"
        self.session_id = session_id
        self.client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)
        self.model = MODEL_NAME
        self.mcp: Optional[MCPClient] = None
        self._agents = {}

        self._agent_config = {
            "kqapro_agent": {
                "module": "ama_kbqa.agents.kqapro_agent.agent",
                "class": "KQAProAgent",
                "description": "Factual knowledge, Knowledge Graph, Relationships"
            },
            "code_agent": {
                "module": "ama_kbqa.placeholder_agent.agent",
                "class": "PlaceholderAgent",
                "init_kwargs": {"domain": "Coding", "capabilities": "Python, Algorithms"},
                "description": "Programming, Python"
            },
            "math_agent": {
                "module": "ama_kbqa.placeholder_agent.agent",
                "class": "PlaceholderAgent",
                "init_kwargs": {"domain": "Math", "capabilities": "Equations, Algebra"},
                "description": "Computation, Mathematics"
            }
        }

    def _trace(self, msg: str, color: str = COLOR_BLUE):
        trace(self.name, msg, color)

    def _log_pretty(self, label: str, data: Any, color: str = COLOR_MAGENTA):
        """Helper function for pretty-printing JSON data."""
        try:
            if isinstance(data, str):
                # Try to parse string as JSON for better display
                try:
                    data = json.loads(data)
                except:
                    pass  # Just a normal string

            # Use ensure_ascii=True to avoid Unicode encoding issues on Windows
            pretty_json = json.dumps(data, indent=2, ensure_ascii=True)
            # Indent the JSON so it appears cleanly under the label
            indented_json = "\n".join([f"    {line}" for line in pretty_json.splitlines()])
            print(f"{color}    {label}:{COLOR_END}\n{color}{indented_json}{COLOR_END}")
        except Exception as e:
            # Fallback with ASCII-safe output
            print(f"{color}    {label}: {str(data).encode('ascii', 'replace').decode('ascii')}{COLOR_END}")

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
            self._trace(f"Starting MCP server: {MCP_SERVER_PATH}")
            self.mcp = MCPClient(MCP_SERVER_PATH, self.name)
            await self.mcp.start()
            self._trace("MCP connected")
        except Exception as e:
            self._trace(f"{COLOR_RED}MCP error: {e}{COLOR_END}", COLOR_RED)
            self.mcp = None

    async def _route_autonomously(self, query: str) -> Optional[str]:
        """Uses LLM and MCP Tools for agent selection."""
        if not self.mcp:
            return None

        self._trace("Preparing routing: fetching tool definitions...")

        try:
            mcp_tools = await self.mcp.list_tools()
            openai_tools = [self._mcp_tool_to_openai(t) for t in mcp_tools]
        except Exception as e:
            self._trace(f"{COLOR_RED}Error listing tools: {e}{COLOR_END}", COLOR_RED)
            return None

        if not openai_tools:
            self._trace(f"{COLOR_YELLOW}No tools available.{COLOR_END}", COLOR_YELLOW)
            return None

        # Updated system prompt to encourage reasoning process
        messages = [
            {"role": "system", "content": (
                "You are a router. Your task is to find the right sub-agent for a user request. "
                "Think step by step. First analyze the request, then decide which tool to call. "
                "Briefly state your reasoning in text before using the tool."
            )},
            {"role": "user", "content": f"Query: {query}"}
        ]

        try:
            completion = self.client.chat.completions.create(
                model=self.model, messages=messages, tools=openai_tools, tool_choice="required"
            )

            message = completion.choices[0].message

            # 1. Logging: Reasoning process
            if message.content:
                self._trace(f"🤔 {message.content}", color=COLOR_CYAN)

            if message.tool_calls:
                tool_call = message.tool_calls[0]
                func_name = tool_call.function.name
                func_args_str = tool_call.function.arguments
                func_args = json.loads(func_args_str)

                # 2. Logging: Tool call and parameters
                self._trace(f"🛠️ Calling tool: {func_name}", color=COLOR_YELLOW)
                self._log_pretty("Arguments", func_args, COLOR_YELLOW)

                tool_result = await self.mcp.call_tool(func_name, func_args)

                # 3. Logging: Result
                self._log_pretty("Result", tool_result, COLOR_MAGENTA)

                # Mapping
                res_lower = tool_result.lower()
                if "kqapro" in res_lower:
                    return "kqapro_agent"
                if "code" in res_lower:
                    return "code_agent"
                if "math" in res_lower:
                    return "math_agent"

                self._trace(f"{COLOR_YELLOW}Tool result unclear: {tool_result}{COLOR_END}", COLOR_YELLOW)
                return None
            else:
                self._trace(f"{COLOR_YELLOW}LLM did not call any tool.{COLOR_END}", COLOR_YELLOW)
                return None

        except Exception as e:
            self._trace(f"{COLOR_RED}Error in routing process: {e}{COLOR_END}", COLOR_RED)
            return None

    def _load_agent(self, agent_name: str):
        if agent_name in self._agents:
            return self._agents[agent_name]
        if agent_name not in self._agent_config:
            return None

        try:
            cfg = self._agent_config[agent_name]
            mod = __import__(cfg["module"], fromlist=[cfg["class"]])
            cls = getattr(mod, cfg["class"])
            kwargs = cfg.get("init_kwargs", {})
            agent = cls(name=agent_name, session_id=self.session_id, **kwargs)
            self._agents[agent_name] = agent
            return agent
        except Exception as e:
            self._trace(f"{COLOR_RED}Loading error {agent_name}: {e}{COLOR_END}", COLOR_RED)
            return None

    async def ask(self, query: str) -> str:
        """Main method: Route the request and get the answer."""
        self._trace(f"USER: {query}", COLOR_GREEN)

        try:
            await self._init_mcp()
            selected_agent_name = await self._route_autonomously(query)
            answer = ""

            print("-" * 50)

            if selected_agent_name:
                self._trace(f"Routing successful -> {selected_agent_name}", COLOR_GREEN)
                agent = self._load_agent(selected_agent_name)

                if agent:
                    try:
                        if inspect.iscoroutinefunction(agent.ask):
                            answer = await agent.ask(query)
                        else:
                            answer = agent.ask(query)
                    except Exception as e:
                        self._trace(f"{COLOR_RED}Agent Error: {e}{COLOR_END}", COLOR_RED)
                        self._trace("Executing KQAPro agent fallback.", COLOR_YELLOW)
                        answer = await self._fallback_kqapro(query)
                else:
                    self._trace("Agent could not be loaded. Fallback to KQAPro.", COLOR_YELLOW)
                    answer = await self._fallback_kqapro(query)
            else:
                self._trace("Routing failed. Fallback to KQAPro agent.", COLOR_YELLOW)
                answer = await self._fallback_kqapro(query)

            return answer

        finally:
            if self.mcp:
                await self.mcp.close()
                self._trace("Orchestrator MCP server cleanly terminated")

    async def _fallback_kqapro(self, query: str) -> str:
        """Fallback to KQAPro agent for knowledge base queries."""
        try:
            self._trace("Loading KQAPro agent as fallback...", COLOR_CYAN)
            agent = self._load_agent("kqapro_agent")

            if agent:
                if inspect.iscoroutinefunction(agent.ask):
                    return await agent.ask(query)
                else:
                    return agent.ask(query)
            else:
                self._trace(f"{COLOR_RED}KQAPro agent failed to load. Using LLM fallback.{COLOR_END}", COLOR_RED)
                return self._fallback_llm(query)
        except Exception as e:
            self._trace(f"{COLOR_RED}KQAPro fallback error: {e}. Using LLM fallback.{COLOR_END}", COLOR_RED)
            return self._fallback_llm(query)

    def _fallback_llm(self, query: str) -> str:
        """Last resort: Use LLM directly without knowledge base."""
        return self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": query}]
        ).choices[0].message.content


async def main():
    orchestrator = Orchestrator()
    # Test question adapted to trigger routing
    result = await orchestrator.ask("How many heavy metal groups are in the genre of Queen (the one famous for heavy metal) ?")
    print("-" * 50)
    # Use safe encoding for Windows console
    try:
        print(f"\n[{COLOR_GREEN}FINALE ANTWORT{COLOR_END}]\n{result}")
    except UnicodeEncodeError:
        # Fallback to ASCII-safe output
        safe_result = result.encode('ascii', 'replace').decode('ascii') if result else ""
        print(f"\n[{COLOR_GREEN}FINAL ANSWER{COLOR_END}]\n{safe_result}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
    except Exception:
        print(f"\n{COLOR_RED}--- CRITICAL ERROR IN MAIN RUN ---{COLOR_END}")
        traceback.print_exc()
