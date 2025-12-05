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

# --- TRACING & COLORS ---

COLOR_BLUE = '\033[94m'
COLOR_GREEN = '\033[92m'
COLOR_RED = '\033[91m'
COLOR_YELLOW = '\033[93m'
COLOR_CYAN = '\033[96m'  # Added Cyan for better distinction
COLOR_END = '\033[0m'


def trace(agent_name: str, msg: str, color: str = COLOR_BLUE):
    """Standardized tracing with timestamp and agent prefix."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    # Handle multi-line messages nicely by indenting subsequent lines
    lines = msg.split('\n')
    header = f"[{color}{timestamp}{COLOR_END}] {color}[{agent_name}]{COLOR_END} -> {lines[0]}"
    print(header)
    for line in lines[1:]:
        print(f"{' ' * (len(timestamp) + 2 + len(agent_name) + 6)} {color}{line}{COLOR_END}")


# Dynamic path resolution for the Sub-Agent Server
current_file = Path(__file__).resolve()
ama_kbqa_root = current_file.parents[2]
default_subagent_server_path = ama_kbqa_root / "server" / "kqapro_server.py"

# --- CONFIGURATION ---
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "openai/gpt-4o")
REQUEST_TIMEOUT_SECONDS = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "60"))

MCP_SERVER_PATH = os.getenv("SUBAGENT_SERVER_PATH", str(default_subagent_server_path))


# --- MCP CLIENT FOR SUB-AGENTS (Adapted) ---

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
                trace(
                    self.agent_name, f"{COLOR_YELLOW}WARNING: MCP-Close error on shutdown ignored ({type(e).__name__}).{COLOR_END}", COLOR_YELLOW)
            except Exception as e:
                trace(self.agent_name, f"{COLOR_RED}Error closing MCP client: {e}{COLOR_END}", COLOR_RED)
            finally:
                self._connected = False


# --- KQAProAgent ---

class KQAProAgent:

    def __init__(self, name: str = "kqapro_agent", session_id: str = "default"):
        if not OPENROUTER_API_KEY:
            raise RuntimeError("OPENROUTER_API_KEY missing in .env")

        self.name = name
        self.session_id = session_id
        self.mcp: Optional[MCPClient] = None

        self.client = OpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=OPENROUTER_API_KEY
        )

        self.model = MODEL_NAME
        self.request_timeout = REQUEST_TIMEOUT_SECONDS

        # NEW: Token Tracking
        self.token_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }

        self.system_prompt = """You are an expert in Knowledge Graph Question Answering (KGQA).
            You analyze questions about structured knowledge graphs and answer them precisely.

            ### YOUR WORKING MEMORY (SCRATCHPAD)
            You have access to the tool `ManageJournal`. This is your most important tool.
            You MUST use it at every step to:
            1. **Log visited nodes**: To ensure you do not run in circles (Loop Avoidance).
            2. **Save facts**: When you have verified a triple, write it down here.
            3. **Planning**: Update your plan whenever you find new information.

            ### PROCESS
            1. Analyze the question.
            2. Search for start nodes (`FindNode`).
            3. Write your plan into the journal (`ManageJournal`).
            4. Explore neighborhoods (`ExploreNeighborhood`).
            5. Write found facts into the journal (`ManageJournal`).
            6. When enough facts are gathered -> Answer.
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
            self._trace(f"Starting own MCP server: {MCP_SERVER_PATH}")
            self.mcp = MCPClient(MCP_SERVER_PATH, self.name)
            await self.mcp.start()
            self._trace("Own MCP connected")
        except Exception as e:
            self._trace(f"{COLOR_RED}Own MCP error: {e}{COLOR_END}", COLOR_RED)
            self.mcp = None
            raise

    async def ask(self, query: str) -> str:
        self._trace(f"Incoming query: '{query}'", COLOR_GREEN)

        try:
            # 1. Start MCP and load tools
            await self._init_mcp()

            if not self.mcp:
                self._trace(f"{COLOR_YELLOW}Tool server not available. Answering without tools.{COLOR_END}", COLOR_YELLOW)
                self._messages.append({"role": "user", "content": query})
                return self._llm_call_text_only()

            mcp_tools = await self.mcp.list_tools()
            openai_tools = [self._mcp_tool_to_openai(t) for t in mcp_tools]
            self._trace(f"Found {len(openai_tools)} tools.")

            # 2. Iterative tool-call loop
            self._messages.append({"role": "user", "content": query})

            while True:
                response = self._llm_call(tools=openai_tools)
                message = response.choices[0].message
                
                # Update Token Usage
                if response.usage:
                    self.token_usage["prompt_tokens"] += response.usage.prompt_tokens
                    self.token_usage["completion_tokens"] += response.usage.completion_tokens
                    self.token_usage["total_tokens"] += response.usage.total_tokens

                # Logging intermediate thoughts (Thoughts/Text)
                if message.content:
                    self._trace(f"🧠 Thought/Text: {message.content}", COLOR_BLUE)

                # Final answer (no more tools)
                if not message.tool_calls:
                    assistant_text = message.content or ""
                    self._messages.append({"role": "assistant", "content": assistant_text})
                    self._trace("🏁 Final answer generated by LLM.")
                    return assistant_text.strip()

                # --- CORRECTION START ---
                # The assistant message MUST be added to the history ONCE
                # BEFORE processing the tools.
                self._messages.append(message)
                # --- CORRECTION END ---

                # Execute tool call(s)
                for tool_call in message.tool_calls:
                    func_name = tool_call.function.name
                    try:
                        func_args = json.loads(tool_call.function.arguments)
                    except json.JSONDecodeError:
                        func_args = {}

                    # Logging
                    args_pretty = json.dumps(func_args, indent=2, ensure_ascii=False)
                    self._trace(f"🛠️  Tool Call: {func_name}\n   Params: {args_pretty}", COLOR_YELLOW)

                    # Execution
                    tool_result = await self.mcp.call_tool(func_name, func_args)

                    # Logging result
                    log_result = tool_result
                    if len(log_result) > 500:
                        log_result = log_result[:500] + f"... [truncated, total len: {len(tool_result)}]"
                    self._trace(f"🔙 Result ({func_name}): {log_result}", COLOR_CYAN)

                    # --- BUG WAS HERE (Removed: self._messages.append(message)) ---

                    # Only the tool result is appended in the loop
                    self._messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": func_name,
                        "content": tool_result
                    })

        except Exception as e:
            self._trace(f"{COLOR_RED}Error in agent loop: {e}{COLOR_END}", COLOR_RED)
            raise

        finally:
            if self.mcp:
                # --- NEW: TRACEABILITY BLOCK ---
                try:
                    # Explicitly fetch the final state of the scratchpad
                    # We send 'content="Final"' just to satisfy the schema, though 'read' ignores it.
                    final_state = await self.mcp.call_tool("ManageJournal", {"action": "read", "content": "Final Trace"})

                    # Print it using the agent's distinct color scheme
                    self._trace(f"🛑 FINAL SCRATCHPAD STATE:\n{final_state}", COLOR_CYAN)
                except Exception:
                    # Fails silently if the server doesn't have the ManageJournal tool yet
                    pass
                # -------------------------------

                await self.mcp.close()
                self._trace("Own MCP server cleanly terminated")

    def _llm_call(self, tools: Optional[List[Dict[str, Any]]] = None):
        """Executes the actual API call (with tools)."""
        return self.client.chat.completions.create(
            model=self.model,
            messages=self._messages,
            timeout=self.request_timeout,
            tools=tools
        )

    def _llm_call_text_only(self):
        """Executes the actual API call (without tools)."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self._messages,
            timeout=self.request_timeout,
        )
        if response.usage:
            self.token_usage["prompt_tokens"] += response.usage.prompt_tokens
            self.token_usage["completion_tokens"] += response.usage.completion_tokens
            self.token_usage["total_tokens"] += response.usage.total_tokens
            
        return response.choices[0].message.content

    def reset(self):
        self._messages = [
            {"role": "system", "content": self.system_prompt}
        ]
        self.token_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }
        if self.mcp:
            asyncio.run(self.mcp.close())
            self.mcp = None


if __name__ == "__main__":
    async def run_test():
        agent = KQAProAgent()
        try:
            answer = await agent.ask("Who is the director of Inception?")
            print(f"\n[KQAPro Agent Answer]\n{answer}")
        except Exception as e:
            print(f"Error in test run: {e}")

    asyncio.run(run_test())