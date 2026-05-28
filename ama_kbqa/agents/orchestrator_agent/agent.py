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

from ama_kbqa.config import (
    assert_provider_api_key_present,
    get_chat_client,
    get_chat_model_name,
)

load_dotenv(find_dotenv())

# --- CONFIGURATION & PATH LOGIC ---
current_file = Path(__file__).resolve()
ama_kbqa_root = current_file.parents[2]
default_server_path = ama_kbqa_root / "server" / "orchestrator_server.py"

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

        try:
            # Pass the parent environment through to the MCP subprocess. With
            # env=None the MCP SDK hands the child only a minimal default
            # environment, dropping API keys (OPENROUTER_API_KEY / KIT_API_KEY)
            # that are injected into the container at runtime (docker-compose
            # env_file) rather than present in an on-disk .env. Without them the
            # server's startup get_chat_client() raises and the child exits,
            # surfacing as an opaque "Connection closed" on the client side.
            client_gen = stdio_client(StdioServerParameters(command=sys.executable, args=[str(self.server_path)], env=os.environ.copy()))
            read, write = await self.exit_stack.enter_async_context(client_gen)

            self.session = await self.exit_stack.enter_async_context(ClientSession(read, write))
            await self.session.initialize()
            self._connected = True
        except Exception:
            # Clean up partially entered async contexts to avoid orphaned anyio tasks
            try:
                await self.exit_stack.aclose()
            except Exception:
                pass
            self.exit_stack = AsyncExitStack()
            self.session = None
            raise

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
        if not self._connected:
            return
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
            self.session = None
            self.exit_stack = AsyncExitStack()

# --- Orchestrator ---


class Orchestrator:

    def __init__(self, session_id: str = "default"):
        self.name = "ORCHESTRATOR"
        self.session_id = session_id
        # Fail fast with a clear message if no provider key is configured at
        # all, rather than deep inside client construction.
        assert_provider_api_key_present()
        self.client = get_chat_client()
        self.model = get_chat_model_name()
        self.mcp: Optional[MCPClient] = None
        self._agents = {}

        # Frontend reads these — Orchestrator mirrors the BaseKBQAAgent
        # surface so the Chat page can capture trace events + journal
        # snapshots regardless of which agent answered.
        from ama_kbqa.framework.trace import TraceRecorder
        self.recorder: TraceRecorder = TraceRecorder()
        self.journal_snapshots: list = []
        self.token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        # The descriptions are injected into the routing system prompt so the
        # router LLM can judge a question's domain even when the probing
        # tool's entity-linking evidence is weak or missing.
        self._agent_config = {
            "kqapro_agent": {
                "module": "ama_kbqa.agents.kqapro_agent.agent",
                "class": "KQAProAgent",
                "description": (
                    "General world knowledge over a Wikidata-style knowledge "
                    "graph (KQA Pro): people, places, films, organizations, "
                    "dates, quantities, comparisons."
                )
            },
            "sciqa_agent": {
                "module": "ama_kbqa.agents.sciqa_agent.agent",
                "class": "SciQAAgent",
                "description": (
                    "Scholarly knowledge over the Open Research Knowledge "
                    "Graph (SciQA/ORKG): research papers, contributions, "
                    "benchmarks, datasets, evaluation metrics, models and "
                    "methods from the scientific literature."
                )
            },
        }

        # Set by _route_autonomously: the router LLM's one-sentence reason
        # for its last decision (surfaced on the classify trace span).
        self.last_routing_reason: Optional[str] = None

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
        mcp = None
        try:
            self._trace(f"Starting MCP server: {MCP_SERVER_PATH}")
            mcp = MCPClient(MCP_SERVER_PATH, self.name)
            await mcp.start()
            self.mcp = mcp
            self._trace("MCP connected")
        except Exception as e:
            self._trace(f"{COLOR_RED}MCP error: {e}{COLOR_END}", COLOR_RED)
            if mcp:
                try:
                    await mcp.close()
                except Exception:
                    pass
            self.mcp = None

    def _routing_system_prompt(self) -> str:
        agent_lines = "\n".join(
            f"- {name}: {cfg['description']}" for name, cfg in self._agent_config.items()
        )
        return (
            "You route user questions to one of these specialist agents:\n"
            f"{agent_lines}\n\n"
            "First call the probing tool with the user's question VERBATIM to "
            "gather entity-linking evidence from both knowledge graphs. Strong, "
            "on-topic matches in one graph are a good signal, but judge the "
            "question's domain yourself: generic terms can match spuriously in "
            "either graph, so check the matched labels, not just the scores. "
            "If the evidence is weak, missing, or degraded, decide from the "
            "question's domain alone. Prefer kqapro_agent only when the "
            "question is genuinely ambiguous between the two."
        )

    def _select_agent_tool(self) -> Dict:
        """Forced final tool call: an enum keeps the decision exact (no
        substring matching on free text)."""
        return {
            "type": "function",
            "function": {
                "name": "select_agent",
                "description": (
                    "Commit to the specialist agent that should answer the "
                    "user's question."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "agent": {
                            "type": "string",
                            "enum": list(self._agent_config.keys()),
                            "description": "The agent to route the question to."
                        },
                        "reason": {
                            "type": "string",
                            "description": "One sentence explaining the routing decision."
                        },
                    },
                    "required": ["agent", "reason"],
                },
            },
        }

    async def _route_autonomously(self, query: str) -> Optional[str]:
        """Two-step LLM routing: probe both KGs, then decide on the evidence.

        Step 1 forces the LLM to call the MCP probing tool, which returns
        raw entity-linking evidence for BOTH knowledge graphs (no collapsed
        verdict). Step 2 feeds that evidence back and forces a structured
        `select_agent` call, so the decision weighs the evidence against the
        agents' domain descriptions instead of a hardcoded threshold gate.
        Any failure returns None and the caller falls back to KQAPro.
        """
        self.last_routing_reason = None
        if not self.mcp:
            return None

        self._trace("Preparing routing: fetching tool definitions...")

        try:
            mcp_tools = await self.mcp.list_tools()
            probe_tools = [self._mcp_tool_to_openai(t) for t in mcp_tools]
        except Exception as e:
            self._trace(f"{COLOR_RED}Error listing tools: {e}{COLOR_END}", COLOR_RED)
            return None

        if not probe_tools:
            self._trace(f"{COLOR_YELLOW}No tools available.{COLOR_END}", COLOR_YELLOW)
            return None

        messages = [
            {"role": "system", "content": self._routing_system_prompt()},
            {"role": "user", "content": f"Query: {query}"}
        ]

        try:
            # Step 1: gather evidence (forced probe call).
            completion = self.client.chat.completions.create(
                model=self.model, messages=messages, tools=probe_tools, tool_choice="required"
            )

            message = completion.choices[0].message

            if message.content:
                self._trace(message.content, color=COLOR_CYAN)

            if not message.tool_calls:
                self._trace(f"{COLOR_YELLOW}LLM did not call the probing tool.{COLOR_END}", COLOR_YELLOW)
                return None

            tool_call = message.tool_calls[0]
            func_name = tool_call.function.name
            func_args_str = tool_call.function.arguments
            func_args = json.loads(func_args_str)

            self._trace(f"Calling tool: {func_name}", color=COLOR_YELLOW)
            self._log_pretty("Arguments", func_args, COLOR_YELLOW)

            tool_result = await self.mcp.call_tool(func_name, func_args)

            self._log_pretty("Evidence", tool_result, COLOR_MAGENTA)

            # Step 2: decide (forced structured select_agent call).
            messages.append({
                "role": "assistant",
                "content": message.content,
                "tool_calls": [{
                    "id": tool_call.id,
                    "type": "function",
                    "function": {"name": func_name, "arguments": func_args_str},
                }],
            })
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": tool_result,
            })

            decision = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=[self._select_agent_tool()],
                tool_choice={"type": "function", "function": {"name": "select_agent"}},
            )

            decision_msg = decision.choices[0].message
            if not decision_msg.tool_calls:
                self._trace(f"{COLOR_YELLOW}LLM did not commit to an agent.{COLOR_END}", COLOR_YELLOW)
                return None

            decision_args = json.loads(decision_msg.tool_calls[0].function.arguments)
            agent_name = decision_args.get("agent")
            reason = decision_args.get("reason", "")

            if agent_name not in self._agent_config:
                self._trace(f"{COLOR_YELLOW}Unknown agent '{agent_name}' selected.{COLOR_END}", COLOR_YELLOW)
                return None

            self.last_routing_reason = reason
            self._trace(f"Decision: {agent_name} ({reason})", COLOR_CYAN)
            return agent_name

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

        async with self.recorder.span(
            "agent_run",
            self.name,
            attributes={"agent": self.name, "model": self.model, "query": query[:500]},
            payload={"query": query},
        ):
            try:
                await self._init_mcp()

                async with self.recorder.span(
                    "classify",
                    "route",
                    attributes={"model": self.model},
                ) as _route_span:
                    selected_agent_name = await self._route_autonomously(query)
                    _route_span.set_attribute("selected_agent", selected_agent_name or "<none>")
                    if self.last_routing_reason:
                        _route_span.set_attribute("route_reason", self.last_routing_reason)

                answer = ""
                print("-" * 50)

                if selected_agent_name:
                    self._trace(f"Routing successful -> {selected_agent_name}", COLOR_GREEN)
                    answer = await self._delegate(selected_agent_name, query)
                else:
                    self._trace("Routing failed. Fallback to KQAPro agent.", COLOR_YELLOW)
                    answer = await self._fallback_kqapro(query)

                return answer
            finally:
                if self.mcp:
                    await self.mcp.close()
                    self._trace("Orchestrator MCP server cleanly terminated")

    async def _delegate(self, agent_name: str, query: str) -> str:
        """Run a sub-agent under a `delegate` span so its trace nests cleanly.

        We hand the sub-agent our recorder + the current span id; its
        `agent_run` root span will then parent under our delegate span
        instead of starting a new trace.
        """
        async with self.recorder.span(
            "delegate",
            agent_name,
            attributes={"sub_agent": agent_name},
        ) as _delegate_span:
            current_span_id = self.recorder.current_span_id()
            agent = self._load_agent(agent_name)
            if not agent:
                self._trace(
                    "Agent could not be loaded. Fallback to KQAPro.", COLOR_YELLOW
                )
                _delegate_span.set_attribute("loaded", False)
                return await self._fallback_kqapro(query)

            # Re-target the sub-agent's recorder at ours for this call. Sub-
            # agent instances are cached, so we reset these on every call.
            try:
                agent.recorder = self.recorder
                agent._parent_span_id_override = current_span_id
            except AttributeError:
                pass

            try:
                if inspect.iscoroutinefunction(agent.ask):
                    answer = await agent.ask(query)
                else:
                    answer = agent.ask(query)
            except Exception as e:
                self._trace(f"{COLOR_RED}Agent Error: {e}{COLOR_END}", COLOR_RED)
                self._trace("Executing KQAPro agent fallback.", COLOR_YELLOW)
                _delegate_span.set_attribute("error", str(e))
                return await self._fallback_kqapro(query)

            # Hoist the sub-agent's journal snapshots up so the Graph View on
            # the orchestrator's trace shows what the delegate discovered.
            try:
                self.journal_snapshots.extend(agent.journal_snapshots)
            except AttributeError:
                pass
            try:
                tu = agent.token_usage
                for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
                    self.token_usage[k] = self.token_usage.get(k, 0) + tu.get(k, 0)
            except AttributeError:
                pass

            return answer

    async def _fallback_kqapro(self, query: str) -> str:
        """Route to the KQAPro agent for knowledge base queries.

        There is no further LLM-only fallback: if the KQAPro agent cannot be
        loaded, or raises while answering, the error propagates to the caller.
        A degraded, knowledge-base-free LLM answer would silently mask internal
        issues (missing API key, unreachable MCP server, ...) behind a
        plausible-looking response, so we surface the failure instead.
        """
        self._trace("Loading KQAPro agent...", COLOR_CYAN)
        agent = self._load_agent("kqapro_agent")
        if not agent:
            raise RuntimeError("KQAPro agent could not be loaded.")

        if inspect.iscoroutinefunction(agent.ask):
            return await agent.ask(query)
        return agent.ask(query)


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
