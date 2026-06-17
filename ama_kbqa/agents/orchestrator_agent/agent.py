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
    get_federation_enabled,
    get_federation_max_specialists,
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

    # Class-level defaults so instances built without __init__ (hermetic
    # tests use __new__) get the safe single-dispatch behaviour.
    _federation_enabled: bool = False
    _federation_max_specialists: int = 2

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
        # for its last decision (surfaced on the classify trace span), and
        # the raw probe evidence JSON (reused as a fusion prior when a
        # federated dispatch produces conflicting answers).
        self.last_routing_reason: Optional[str] = None
        self.last_routing_evidence: Optional[str] = None

        # Federated dispatch: when enabled, the router may select multiple
        # specialists for one question; their answers are fused afterwards.
        self._federation_enabled = get_federation_enabled()
        self._federation_max_specialists = get_federation_max_specialists()

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
        prompt = (
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
        if self._federation_enabled:
            prompt += (
                "\n\nYou may select MULTIPLE agents (federated dispatch): they "
                "run concurrently and their answers are fused. Federation "
                "costs every selected agent a full run, so select multiple "
                "agents only when (a) the evidence shows strong, on-topic "
                "matches in MORE THAN ONE graph, or (b) the question "
                "genuinely spans both domains (e.g. scholarly work about a "
                "general-world entity). For a clearly single-domain question, "
                "select exactly one agent."
            )
        return prompt

    def _select_agents_tool(self) -> Dict:
        """Forced final tool call: an enum array keeps the decision exact (no
        substring matching on free text).

        With federation disabled the array is capped at one entry, so the
        contract degenerates to the original single-agent selection.
        """
        max_items = (
            max(1, self._federation_max_specialists)
            if self._federation_enabled else 1
        )
        return {
            "type": "function",
            "function": {
                "name": "select_agents",
                "description": (
                    "Commit to the specialist agent(s) that should answer the "
                    "user's question."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "agents": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": list(self._agent_config.keys()),
                            },
                            "minItems": 1,
                            "maxItems": max_items,
                            "description": (
                                "The agent(s) to route the question to. "
                                "Multiple agents run concurrently and their "
                                "answers are fused."
                            ),
                        },
                        "reason": {
                            "type": "string",
                            "description": "One sentence explaining the routing decision."
                        },
                    },
                    "required": ["agents", "reason"],
                },
            },
        }

    async def _route_autonomously(self, query: str) -> Optional[List[str]]:
        """Two-step LLM routing: probe both KGs, then decide on the evidence.

        Step 1 forces the LLM to call the MCP probing tool, which returns
        raw entity-linking evidence for BOTH knowledge graphs (no collapsed
        verdict). Step 2 feeds that evidence back and forces a structured
        `select_agents` call, so the decision weighs the evidence against the
        agents' domain descriptions instead of a hardcoded threshold gate.

        Returns the selected agent names in router order: a single-element
        list for normal dispatch, multiple elements for a federated dispatch
        (only possible when federation is enabled). Any failure returns None
        and the caller falls back to KQAPro.
        """
        self.last_routing_reason = None
        self.last_routing_evidence = None
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
            self.last_routing_evidence = tool_result

            self._log_pretty("Evidence", tool_result, COLOR_MAGENTA)

            # Step 2: decide (forced structured select_agents call).
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
                tools=[self._select_agents_tool()],
                tool_choice={"type": "function", "function": {"name": "select_agents"}},
            )

            decision_msg = decision.choices[0].message
            if not decision_msg.tool_calls:
                self._trace(f"{COLOR_YELLOW}LLM did not commit to an agent.{COLOR_END}", COLOR_YELLOW)
                return None

            decision_args = json.loads(decision_msg.tool_calls[0].function.arguments)
            raw_agents = decision_args.get("agents") or []
            reason = decision_args.get("reason", "")

            # Dedupe in router order; any unknown name invalidates the whole
            # decision (same strictness as the previous single-agent enum).
            agent_names: List[str] = []
            for name in raw_agents:
                if name not in self._agent_config:
                    self._trace(f"{COLOR_YELLOW}Unknown agent '{name}' selected.{COLOR_END}", COLOR_YELLOW)
                    return None
                if name not in agent_names:
                    agent_names.append(name)

            if not agent_names:
                self._trace(f"{COLOR_YELLOW}LLM selected no agents.{COLOR_END}", COLOR_YELLOW)
                return None

            # Enforce the configured fan-out cap defensively: the schema
            # already caps maxItems, but not every provider validates it.
            max_agents = (
                max(1, self._federation_max_specialists)
                if self._federation_enabled else 1
            )
            if len(agent_names) > max_agents:
                self._trace(
                    f"{COLOR_YELLOW}Capping selection {agent_names} to first "
                    f"{max_agents} (federation limit).{COLOR_END}",
                    COLOR_YELLOW,
                )
                agent_names = agent_names[:max_agents]

            self.last_routing_reason = reason
            self._trace(f"Decision: {', '.join(agent_names)} ({reason})", COLOR_CYAN)
            return agent_names

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
                    selected_agent_names = await self._route_autonomously(query)
                    _route_span.set_attribute(
                        "selected_agent",
                        ", ".join(selected_agent_names) if selected_agent_names else "<none>",
                    )
                    _route_span.set_attribute(
                        "route_mode",
                        "federated" if selected_agent_names and len(selected_agent_names) > 1 else "single",
                    )
                    if self.last_routing_reason:
                        _route_span.set_attribute("route_reason", self.last_routing_reason)

                answer = ""
                print("-" * 50)

                if not selected_agent_names:
                    self._trace("Routing failed. Fallback to KQAPro agent.", COLOR_YELLOW)
                    answer = await self._fallback_kqapro(query)
                elif len(selected_agent_names) == 1:
                    self._trace(f"Routing successful -> {selected_agent_names[0]}", COLOR_GREEN)
                    answer = await self._delegate(selected_agent_names[0], query)
                else:
                    self._trace(
                        f"Routing successful -> federated: {', '.join(selected_agent_names)}",
                        COLOR_GREEN,
                    )
                    answer = await self._federate(selected_agent_names, query)

                return answer
            finally:
                if self.mcp:
                    await self.mcp.close()
                    self._trace("Orchestrator MCP server cleanly terminated")

    def _extract_scratchpad(self, agent) -> Optional[str]:
        """Render a sub-agent's final journal as an LLM-legible scratchpad.

        Reads the last journal snapshot the sub-agent captured during its run
        (a `JournalState` dump) and renders it via `to_summary_str()` so the
        orchestrator receives the specialist's working notes — entities
        visited, values found, verified facts — alongside its prose answer.

        Returns None when the sub-agent captured no usable journal (e.g. its
        MCP server lacks the GetJournalStateJSON tool, or the run never
        mutated the journal), so callers can omit the block cleanly rather
        than forwarding an empty placeholder.
        """
        try:
            snapshots = agent.journal_snapshots
        except AttributeError:
            return None
        if not snapshots:
            return None
        state = snapshots[-1].get("state")
        if not isinstance(state, dict) or not state:
            return None
        try:
            from ama_kbqa.framework.state import JournalState
            rendered = JournalState.from_dict(state).to_summary_str().strip()
        except Exception:
            return None
        return rendered or None

    async def _run_specialist(self, agent_name: str, query: str, fallback: bool = False) -> Dict[str, Any]:
        """Run one sub-agent under its own `delegate` span; raises on failure.

        We hand the sub-agent our recorder + the current span id; its
        `agent_run` root span will then parent under our delegate span
        instead of starting a new trace. This is the shared primitive under
        both single dispatch (`_delegate`, which adds the fallback policy)
        and federated dispatch (`_federate`, which captures failures per
        specialist). It is safe to run concurrently for DIFFERENT agent
        names: the recorder's contextvar is task-local under asyncio.gather,
        so each delegate span nests independently.

        Returns a handoff record `{"agent", "answer", "scratchpad"}`: the
        specialist's prose answer plus a rendering of its final journal
        (None if it captured none). Single dispatch uses only the answer;
        federated fusion also consumes the scratchpad to ground and
        adjudicate the combined answer.

        Args:
            agent_name: Key into self._agent_config.
            query: The user's question, verbatim.
            fallback: Mark this run as a fallback on the delegate span.

        Raises:
            RuntimeError: If the agent cannot be loaded.
            Exception: Whatever the sub-agent's ask() raises.
        """
        attributes: Dict[str, Any] = {"sub_agent": agent_name}
        if fallback:
            attributes["fallback"] = True
        async with self.recorder.span(
            "delegate",
            agent_name,
            attributes=attributes,
        ) as _delegate_span:
            current_span_id = self.recorder.current_span_id()
            agent = self._load_agent(agent_name)
            if not agent:
                _delegate_span.set_attribute("loaded", False)
                raise RuntimeError(f"Agent '{agent_name}' could not be loaded.")

            # Re-target the sub-agent's recorder at ours for this call. Sub-
            # agent instances are cached, so we reset these on every call.
            try:
                agent.recorder = self.recorder
                agent._parent_span_id_override = current_span_id
            except AttributeError:
                pass

            if inspect.iscoroutinefunction(agent.ask):
                answer = await agent.ask(query)
            else:
                answer = agent.ask(query)

            # Hoist the sub-agent's journal snapshots up so the Graph View on
            # the orchestrator's trace shows what the delegate discovered.
            # Tag each snapshot with its source so a federated run's merged
            # snapshots stay attributable.
            try:
                self.journal_snapshots.extend(
                    {**snap, "source_agent": agent_name}
                    for snap in agent.journal_snapshots
                )
            except AttributeError:
                pass
            try:
                tu = agent.token_usage
                for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
                    self.token_usage[k] = self.token_usage.get(k, 0) + tu.get(k, 0)
            except AttributeError:
                pass

            scratchpad = self._extract_scratchpad(agent)
            _delegate_span.set_attribute("scratchpad_captured", scratchpad is not None)
            return {"agent": agent_name, "answer": answer, "scratchpad": scratchpad}

    async def _delegate(self, agent_name: str, query: str) -> str:
        """Single dispatch: run one specialist, falling back to KQAPro on error.

        Single dispatch returns the specialist's answer verbatim; the
        scratchpad in the handoff record is only consumed when answers are
        fused (`_federate`), since there is nothing to combine here.
        """
        try:
            result = await self._run_specialist(agent_name, query)
            return result["answer"]
        except Exception as e:
            self._trace(f"{COLOR_RED}Agent Error: {e}{COLOR_END}", COLOR_RED)
            self._trace("Executing KQAPro agent fallback.", COLOR_YELLOW)
            return await self._fallback_kqapro(query)

    async def _federate(self, agent_names: List[str], query: str) -> str:
        """Federated dispatch: run several specialists concurrently, fuse answers.

        Each specialist runs under its own `delegate` span via
        asyncio.gather. A failing specialist does not abort the dispatch:
        we degrade to the surviving answers and only fall back to KQAPro
        when ALL specialists failed (mirroring `_delegate`'s policy).
        """
        results = await asyncio.gather(
            *(self._run_specialist(name, query) for name in agent_names),
            return_exceptions=True,
        )

        answers: List[Dict[str, Any]] = []
        failures: List[str] = []
        for name, result in zip(agent_names, results):
            if isinstance(result, BaseException):
                self._trace(f"{COLOR_RED}{name} failed: {result}{COLOR_END}", COLOR_RED)
                failures.append(f"{name}: {result}")
            else:
                # result is the handoff record {"agent", "answer", "scratchpad"}.
                answers.append(result)

        if not answers:
            self._trace(
                "All federated specialists failed. Executing KQAPro agent fallback.",
                COLOR_YELLOW,
            )
            return await self._fallback_kqapro(query)

        if len(answers) == 1:
            self._trace(
                f"Federation degraded to single answer from {answers[0]['agent']}.",
                COLOR_YELLOW,
            )
            return answers[0]["answer"]

        return await self._fuse_answers(query, answers)

    def _fusion_system_prompt(self) -> str:
        return (
            "You combine answers from multiple knowledge-graph specialist "
            "agents into ONE final answer. Each specialist consulted a "
            "different knowledge graph and answered independently.\n"
            "Besides each specialist's prose answer you may be given its "
            "working notes (scratchpad): the entities it visited, the values "
            "it found, and the facts it verified against its graph. Treat the "
            "notes as grounding evidence, not as additional answers.\n"
            "Rules:\n"
            "- If the answers agree, state the answer once; you may note it "
            "is confirmed by both sources.\n"
            "- If the answers complement each other (different facets), merge "
            "them into one coherent answer, attributing each facet to its "
            "source graph.\n"
            "- If the answers CONFLICT, do not silently pick one: weigh them "
            "by the working notes and routing evidence (an answer backed by "
            "concrete verified facts outweighs an unsupported assertion), "
            "then present both, name the source of each, and say which is "
            "better supported and why.\n"
            "- If a specialist clearly found nothing (empty / 'unknown' "
            "answer, or empty working notes), rely on the other and briefly "
            "say so.\n"
            "- Never introduce facts that appear in neither the answers nor "
            "the working notes."
        )

    async def _fuse_answers(self, query: str, answers: List[Dict[str, str]]) -> str:
        """Fuse the answers of a federated dispatch with one LLM call.

        The routing probe's raw entity-linking evidence is forwarded as a
        fusion prior so conflicting answers can be weighed by how well each
        graph actually grounded the question. If fusion itself fails we
        degrade to the first answer (router order) rather than discarding
        two successful specialist runs.
        """
        agent_list = ", ".join(a["agent"] for a in answers)
        async with self.recorder.span(
            "synthesis",
            "fuse",
            attributes={"model": self.model, "agents": agent_list},
            payload={"query": query, "answers": answers},
        ) as _fuse_span:
            blocks = []
            for a in answers:
                desc = self._agent_config[a["agent"]]["description"]
                block = f"### {a['agent']} ({desc})\nAnswer: {a['answer']}"
                scratchpad = a.get("scratchpad")
                if scratchpad:
                    block += (
                        "\n\nWorking notes (this specialist's scratchpad — "
                        "entities visited, values found, verified facts):\n"
                        f"{scratchpad}"
                    )
                blocks.append(block)
            user_content = f"Question: {query}\n\n" + "\n\n".join(blocks)
            if self.last_routing_evidence:
                user_content += (
                    "\n\nRouting evidence (entity-linking probe over both "
                    f"knowledge graphs):\n{self.last_routing_evidence}"
                )

            try:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self._fusion_system_prompt()},
                        {"role": "user", "content": user_content},
                    ],
                )
                fused = (completion.choices[0].message.content or "").strip()
                usage = getattr(completion, "usage", None)
                if usage is not None:
                    for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
                        v = getattr(usage, k, 0) or 0
                        self.token_usage[k] = self.token_usage.get(k, 0) + v
            except Exception as e:
                self._trace(f"{COLOR_RED}Fusion error: {e}{COLOR_END}", COLOR_RED)
                _fuse_span.set_attribute("error", str(e))
                fused = ""

            if not fused:
                _fuse_span.set_attribute("degraded", True)
                self._trace(
                    f"Fusion produced no answer; degrading to {answers[0]['agent']}.",
                    COLOR_YELLOW,
                )
                return answers[0]["answer"]

            self._trace(f"Fused answer from [{agent_list}]", COLOR_GREEN)
            return fused

    async def _fallback_kqapro(self, query: str) -> str:
        """Route to the KQAPro agent for knowledge base queries.

        There is no further LLM-only fallback: if the KQAPro agent cannot be
        loaded, or raises while answering, the error propagates to the caller.
        A degraded, knowledge-base-free LLM answer would silently mask internal
        issues (missing API key, unreachable MCP server, ...) behind a
        plausible-looking response, so we surface the failure instead.

        Runs through `_run_specialist` so the fallback's trace nests under a
        `delegate` span like any other dispatch (previously its spans landed
        in a separate, invisible trace).
        """
        self._trace("Loading KQAPro agent...", COLOR_CYAN)
        result = await self._run_specialist("kqapro_agent", query, fallback=True)
        return result["answer"]


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
