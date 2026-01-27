"""
SciQA Agent for ORKG Knowledge Graph Question Answering.

This agent uses the Open Research Knowledge Graph (ORKG) loaded from the
SciQA dataset to answer scientific research questions.
"""

from __future__ import annotations
from ama_kbqa.config import (
    get_chat_client,
    get_chat_model_name,
    get_chat_temperature,
    get_chat_max_tokens,
    get_provider_preferences,
    get_synthesis_client,
    get_synthesis_model_name,
    get_synthesis_temperature,
    get_synthesis_max_tokens,
    get_synthesis_provider_preferences,
)
import os
import sys
import asyncio
import json
import time
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

from ama_kbqa.agents.sciqa_agent.prompts import (
    QTYPE_STRATEGIES,
    SYSTEM_PROMPT,
    CLASSIFICATION_PROMPT_TEMPLATE,
    ENTITY_EXTRACTION_PROMPT,
    ANALYSIS_CONTEXT_TEMPLATE,
    FEWSHOT_EXAMPLES_TEMPLATE,
    ANALYSIS_CONTEXT_SUFFIX,
    JOURNAL_REFRESH_TEMPLATE,
    NO_PROGRESS_TEMPLATE,
    SYNTHESIS_PROMPT_TEMPLATE,
    JOURNAL_SUMMARY_ANSWER_PROMPT,
    TOOL_LOOP_GUIDANCE,
    GENERIC_LOOP_GUIDANCE,
    LOOP_INTERVENTION_TEMPLATE,
)

# Configure stdout to handle Unicode on Windows
if sys.platform == 'win32':
    import codecs
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    else:
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, errors='replace')

load_dotenv(override=True)

# Path to the SciQA MCP server
current_file = Path(__file__).resolve()
ama_kbqa_root = current_file.parents[2]
default_sciqa_server_path = ama_kbqa_root / "server" / "sciqa_server.py"

# Configuration
REQUEST_TIMEOUT_SECONDS = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "60"))
MCP_SERVER_PATH = os.getenv("SCIQA_SERVER_PATH", str(default_sciqa_server_path))

# Colors for terminal output
COLOR_BLUE = '\033[94m'
COLOR_GREEN = '\033[92m'
COLOR_RED = '\033[91m'
COLOR_YELLOW = '\033[93m'
COLOR_CYAN = '\033[96m'
COLOR_END = '\033[0m'


def trace(agent_name: str, msg: str, color: str = COLOR_BLUE):
    """Standardized tracing with timestamp and agent prefix."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    lines = msg.split('\n')
    header = f"[{color}{timestamp}{COLOR_END}] {color}[{agent_name}]{COLOR_END} -> {lines[0]}"
    print(header)
    for line in lines[1:]:
        print(f"{' ' * (len(timestamp) + 2 + len(agent_name) + 6)} {color}{line}{COLOR_END}")


# ==============================================================================
# MCP Client
# ==============================================================================

class MCPClient:
    """Client for communicating with the SciQA MCP server."""

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

        client_gen = stdio_client(StdioServerParameters(
            command=sys.executable,
            args=[str(self.server_path)],
            env=None
        ))

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
                await asyncio.sleep(0.5)
            except (CancelledError, RuntimeError) as e:
                trace(self.agent_name, f"{COLOR_YELLOW}WARNING: MCP-Close error ignored ({type(e).__name__}).{COLOR_END}", COLOR_YELLOW)
            except Exception as e:
                trace(self.agent_name, f"{COLOR_RED}Error closing MCP client: {e}{COLOR_END}", COLOR_RED)
            finally:
                self._connected = False
                await asyncio.sleep(0.2)


# ==============================================================================
# SciQA Agent
# ==============================================================================

class SciQAAgent:
    """
    SciQA Agent for ORKG Knowledge Graph Question Answering.

    Uses the SciQA MCP server to query the Open Research Knowledge Graph
    and answer scientific research questions.
    """

    def __init__(self, name: str = "sciqa_agent", session_id: str = "default"):
        self.name = name
        self.session_id = session_id
        self.mcp: Optional[MCPClient] = None

        # Get chat client and model from config
        try:
            self.client = get_chat_client()
            self.model = get_chat_model_name()
        except (FileNotFoundError, ValueError, KeyError) as e:
            raise RuntimeError(f"Failed to initialize LLM client: {e}")

        # Get synthesis client and model
        try:
            self.synthesis_client = get_synthesis_client()
            self.synthesis_model = get_synthesis_model_name()
        except (FileNotFoundError, ValueError, KeyError) as e:
            raise RuntimeError(f"Failed to initialize synthesis client: {e}")

        self.request_timeout = REQUEST_TIMEOUT_SECONDS

        # Token tracking
        self.token_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }

        # Tool call tracking
        self.tool_call_counts: Dict[str, int] = {}
        self.tool_call_durations: List[Dict[str, Any]] = []

        # Loop detection
        self.tool_call_history = []
        self.tool_sequence = []
        self.empty_result_count = 0
        self.last_journal_state = None

        self.system_prompt = SYSTEM_PROMPT

        self._messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt}
        ]

    def _trace(self, msg: str, color: str = COLOR_GREEN):
        trace(self.name, msg, color)

    def _classify_question(self, question: str) -> Dict[str, str]:
        """Classify question into SciQA taxonomy."""
        prompt = CLASSIFICATION_PROMPT_TEMPLATE.format(question=question)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": prompt}],
                temperature=get_chat_temperature(),
                response_format={"type": "json_object"},
                timeout=30.0
            )

            if response.usage:
                self.token_usage["prompt_tokens"] += response.usage.prompt_tokens
                self.token_usage["completion_tokens"] += response.usage.completion_tokens
                self.token_usage["total_tokens"] += response.usage.total_tokens

            json_content = response.choices[0].message.content
            result = json.loads(json_content)
            question_type = result.get("question_type", "General")

            return {
                "question_type": question_type,
                "fewshot_examples": ""
            }

        except Exception as e:
            self._trace(f"Question classification failed: {e}", COLOR_YELLOW)
            return {
                "question_type": "General",
                "fewshot_examples": ""
            }

    def _extract_entities(self, question: str) -> Dict[str, List[str]]:
        """Extract entities and relations from query."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": ENTITY_EXTRACTION_PROMPT},
                    {"role": "user", "content": question}
                ],
                temperature=get_chat_temperature(),
                response_format={"type": "json_object"},
                timeout=30.0
            )

            if response.usage:
                self.token_usage["prompt_tokens"] += response.usage.prompt_tokens
                self.token_usage["completion_tokens"] += response.usage.completion_tokens
                self.token_usage["total_tokens"] += response.usage.total_tokens

            json_content = response.choices[0].message.content
            result = json.loads(json_content)

            return {
                "entities": result.get("entities", []),
                "relations": result.get("relations", [])
            }

        except Exception as e:
            self._trace(f"Entity extraction failed: {e}", COLOR_YELLOW)
            return {"entities": [], "relations": []}

    def _detect_loops(self, func_name: str, func_args: dict) -> tuple[bool, str]:
        """Multi-layered loop detection."""
        args_str = json.dumps(func_args, sort_keys=True)
        current_call = (func_name, args_str)

        self.tool_call_history.append(current_call)
        self.tool_sequence.append(func_name)

        if len(self.tool_call_history) > 12:
            self.tool_call_history.pop(0)
        if len(self.tool_sequence) > 12:
            self.tool_sequence.pop(0)

        # Detection 1: Identical repeated calls
        if len(self.tool_call_history) >= 3:
            recent_calls = self.tool_call_history[-3:]
            if all(call == current_call for call in recent_calls):
                return True, f"Identical call repeated 3 times: {func_name}"

        # Detection 2: Oscillating pattern
        if len(self.tool_sequence) >= 6:
            last_6 = self.tool_sequence[-6:]
            if last_6[0] == last_6[2] == last_6[4] and last_6[1] == last_6[3] == last_6[5] and last_6[0] != last_6[1]:
                return True, f"Oscillating between {last_6[0]} and {last_6[1]}"

        # Detection 3: Same tool called too often
        if len(self.tool_sequence) >= 6:
            last_6 = self.tool_sequence[-6:]
            tool_counts = {}
            for t in last_6:
                tool_counts[t] = tool_counts.get(t, 0) + 1
            for tool, count in tool_counts.items():
                if count >= 5:
                    return True, f"Tool '{tool}' called {count} times in last 6 iterations"

        return False, ""

    def _get_tool_specific_loop_guidance(self, func_name: str) -> str:
        """Get tool-specific recovery guidance."""
        return TOOL_LOOP_GUIDANCE.get(func_name, GENERIC_LOOP_GUIDANCE)

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
            self._trace(f"Starting SciQA MCP server: {MCP_SERVER_PATH}")
            self.mcp = MCPClient(MCP_SERVER_PATH, self.name)
            await self.mcp.start()
            self._trace("SciQA MCP connected")
        except Exception as e:
            self._trace(f"{COLOR_RED}MCP error: {e}{COLOR_END}", COLOR_RED)
            self.mcp = None
            raise

    async def ask(self, query: str) -> str:
        """Answer a scientific research question using ORKG."""
        self._trace(f"Incoming query: '{query}'", COLOR_GREEN)

        try:
            # 1. Start MCP and load tools
            await self._init_mcp()

            if not self.mcp:
                self._trace(f"{COLOR_YELLOW}Tool server not available.{COLOR_END}", COLOR_YELLOW)
                self._messages.append({"role": "user", "content": query})
                return self._llm_call_text_only()

            mcp_tools = await self.mcp.list_tools()
            openai_tools = [self._mcp_tool_to_openai(t) for t in mcp_tools]
            self._trace(f"Found {len(openai_tools)} tools.")

            # 2. Add user query
            self._messages.append({"role": "user", "content": query})

            # 3. Pre-agent hook: Classification & Entity Extraction
            self._trace("Starting pre-agent classification hook", COLOR_CYAN)

            qtype_data = self._classify_question(query)
            self._trace(f"Question classification: {qtype_data['question_type']}", COLOR_GREEN)

            entities_data = self._extract_entities(query)
            self._trace(
                f"Entity extraction: {len(entities_data['entities'])} entities, "
                f"{len(entities_data['relations'])} relations",
                COLOR_GREEN
            )

            qtype = qtype_data.get("question_type", "General")
            fewshot_examples = qtype_data.get("fewshot_examples", "")
            entities = entities_data.get("entities", [])
            relations = entities_data.get("relations", [])

            formatted_entities = "\n".join([f"  - {e}" for e in entities]) if entities else "  (none identified)"
            formatted_relations = "\n".join([f"  - {r}" for r in relations]) if relations else "  (none identified)"

            qtype_strategy = QTYPE_STRATEGIES.get(qtype, QTYPE_STRATEGIES["General"])

            analysis_context = ANALYSIS_CONTEXT_TEMPLATE.format(
                qtype=qtype,
                formatted_entities=formatted_entities,
                formatted_relations=formatted_relations,
                qtype_strategy=qtype_strategy
            )

            if fewshot_examples and fewshot_examples.strip():
                analysis_context += FEWSHOT_EXAMPLES_TEMPLATE.format(
                    qtype=qtype,
                    fewshot_examples=fewshot_examples
                )

            analysis_context += ANALYSIS_CONTEXT_SUFFIX

            self._messages.append({"role": "user", "content": analysis_context})
            self._trace(f"Pre-agent hook complete - Type: {qtype}", COLOR_GREEN)

            # 4. Iterative tool-call loop
            iteration_count = 0
            max_iterations = 50

            while True:
                iteration_count += 1
                self._trace(f"Starting iteration {iteration_count}/{max_iterations}", COLOR_CYAN)

                if iteration_count > max_iterations:
                    self._trace(f"WARNING: Reached max iterations ({max_iterations})", COLOR_RED)
                    return "Error: Agent reached maximum iteration limit."

                # Periodic journal refresh
                if iteration_count % 5 == 0 and iteration_count > 0:
                    self._trace("Injecting journal refresh", COLOR_CYAN)
                    try:
                        journal_refresh = await self.mcp.call_tool("GetJournalSummary", {})

                        if self.last_journal_state is not None and journal_refresh == self.last_journal_state:
                            self._trace("WARNING: No progress in last 5 iterations!", COLOR_YELLOW)
                            self._messages.append({
                                "role": "user",
                                "content": NO_PROGRESS_TEMPLATE.format(
                                    iteration_count=iteration_count,
                                    journal_refresh=journal_refresh
                                )
                            })
                        else:
                            self._messages.append({
                                "role": "user",
                                "content": JOURNAL_REFRESH_TEMPLATE.format(
                                    iteration_count=iteration_count,
                                    journal_refresh=journal_refresh
                                )
                            })

                        self.last_journal_state = journal_refresh

                    except Exception as e:
                        self._trace(f"Failed to inject journal refresh: {e}", COLOR_YELLOW)

                self._trace(f"Calling LLM with {len(self._messages)} messages...", COLOR_YELLOW)
                response = self._llm_call(tools=openai_tools)
                message = response.choices[0].message
                finish_reason = response.choices[0].finish_reason

                self._trace(f"LLM response (finish_reason: {finish_reason})", COLOR_CYAN)

                if response.usage:
                    self.token_usage["prompt_tokens"] += response.usage.prompt_tokens
                    self.token_usage["completion_tokens"] += response.usage.completion_tokens
                    self.token_usage["total_tokens"] += response.usage.total_tokens

                if message.content:
                    self._trace(f"Thought: {message.content}", COLOR_BLUE)

                # No more tool calls - break for synthesis
                if not message.tool_calls:
                    self._trace("No more tool calls - breaking to synthesis", COLOR_GREEN)
                    if message.content:
                        self._messages.append({"role": "assistant", "content": message.content})
                    break

                # Add assistant message to history
                msg_dict = {"role": message.role, "content": message.content}
                if message.tool_calls:
                    msg_dict["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": tc.type,
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments
                            }
                        }
                        for tc in message.tool_calls
                    ]
                self._messages.append(msg_dict)

                # Execute tool calls
                called_get_journal_summary = False

                for tool_call in message.tool_calls:
                    if not tool_call.function or not tool_call.function.name:
                        self._messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": "invalid_tool",
                            "content": "Error: Invalid tool call."
                        })
                        continue

                    func_name = tool_call.function.name

                    if func_name == "GetJournalSummary":
                        called_get_journal_summary = True

                    try:
                        args_str = tool_call.function.arguments
                        func_args = json.loads(args_str) if args_str else {}
                    except json.JSONDecodeError:
                        func_args = {}

                    args_pretty = json.dumps(func_args, indent=2, ensure_ascii=False)
                    self._trace(f"Tool Call: {func_name}\n   Params: {args_pretty}", COLOR_YELLOW)

                    # Loop detection
                    loop_detected, loop_reason = self._detect_loops(func_name, func_args)

                    if loop_detected:
                        self._trace(f"LOOP DETECTED: {loop_reason}", COLOR_RED)

                        try:
                            journal_state = await self.mcp.call_tool("GetJournalSummary", {})
                        except Exception:
                            journal_state = "(Journal unavailable)"

                        tool_specific_guidance = self._get_tool_specific_loop_guidance(func_name)

                        tool_result = LOOP_INTERVENTION_TEMPLATE.format(
                            loop_reason=loop_reason,
                            func_name=func_name,
                            tool_specific_guidance=tool_specific_guidance,
                            journal_state=journal_state
                        )

                        self.tool_call_history = []
                        self.tool_sequence = []
                    else:
                        # Normal execution
                        tool_start_time = time.time()
                        try:
                            tool_result = await self.mcp.call_tool(func_name, func_args)
                            tool_duration = time.time() - tool_start_time

                            self.tool_call_counts[func_name] = self.tool_call_counts.get(func_name, 0) + 1
                            self.tool_call_durations.append({
                                "tool_name": func_name,
                                "duration_seconds": round(tool_duration, 3),
                                "success": True,
                                "timestamp": datetime.now().isoformat()
                            })

                            log_result = tool_result
                            if len(log_result) > 500:
                                log_result = log_result[:500] + f"... [truncated]"
                            self._trace(f"Result ({func_name}) [{tool_duration:.3f}s]: {log_result}", COLOR_CYAN)

                        except Exception as tool_error:
                            tool_duration = time.time() - tool_start_time
                            self.tool_call_counts[func_name] = self.tool_call_counts.get(func_name, 0) + 1
                            self.tool_call_durations.append({
                                "tool_name": func_name,
                                "duration_seconds": round(tool_duration, 3),
                                "success": False,
                                "error": str(tool_error),
                                "timestamp": datetime.now().isoformat()
                            })

                            self._trace(f"Tool {func_name} failed: {tool_error}", COLOR_RED)
                            tool_result = f"Error executing {func_name}: {tool_error}. Try a different approach."

                    self._messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": func_name,
                        "content": tool_result
                    })

                if called_get_journal_summary:
                    self._trace("GetJournalSummary called - injecting answer prompt", COLOR_CYAN)
                    self._messages.append({
                        "role": "user",
                        "content": JOURNAL_SUMMARY_ANSWER_PROMPT
                    })

            # 5. Deterministic synthesis
            self._trace("Starting synthesis step", COLOR_CYAN)

            journal_summary = await self.mcp.call_tool("GetJournalSummary", {})
            self._trace(f"Journal fetched ({len(journal_summary)} chars)", COLOR_GREEN)

            synthesis_prompt = SYNTHESIS_PROMPT_TEMPLATE.format(
                journal_summary=journal_summary,
                query=query
            )

            self._messages.append({"role": "user", "content": synthesis_prompt})

            self._trace("Making final synthesis LLM call...", COLOR_YELLOW)
            final_answer = self._llm_call_synthesis()

            if final_answer and final_answer.strip():
                self._trace(f"Synthesis complete ({len(final_answer)} chars)", COLOR_GREEN)
                self._messages.append({"role": "assistant", "content": final_answer})
                return final_answer.strip()
            else:
                self._trace("WARNING: Synthesis returned empty", COLOR_RED)
                fallback = "Unable to generate answer. The investigation completed but synthesis failed."
                self._messages.append({"role": "assistant", "content": fallback})
                return fallback

        except Exception as e:
            self._trace(f"{COLOR_RED}Error in agent loop: {e}{COLOR_END}", COLOR_RED)
            raise

        finally:
            if self.mcp:
                try:
                    final_state = await self.mcp.call_tool("ManageJournal", {"action": "read", "content": "Final"})
                    self._trace(f"FINAL SCRATCHPAD STATE:\n{final_state}", COLOR_CYAN)
                except Exception:
                    pass

                tool_summary = self.get_tool_call_summary()
                if tool_summary["total_calls"] > 0:
                    self._trace(
                        f"TOOL SUMMARY: {tool_summary['total_calls']} calls, "
                        f"total {tool_summary['total_duration_seconds']}s",
                        COLOR_CYAN
                    )

                self._trace("Question complete (MCP preserved)", COLOR_GREEN)

    def _llm_call(self, tools: Optional[List[Dict[str, Any]]] = None):
        """Execute API call with tools."""
        call_params = {
            "model": self.model,
            "messages": self._messages,
            "timeout": self.request_timeout,
            "temperature": get_chat_temperature(),
            "max_tokens": get_chat_max_tokens(),
            "tools": tools
        }

        provider_prefs = get_provider_preferences()
        if provider_prefs:
            call_params["extra_body"] = {"provider": provider_prefs}

        self._trace(f"Calling {self.model} (timeout: {self.request_timeout}s)", COLOR_CYAN)

        try:
            response = self.client.chat.completions.create(**call_params)
            self._trace("LLM call completed", COLOR_GREEN)
            return response
        except Exception as e:
            self._trace(f"LLM call failed: {e}", COLOR_RED)
            raise

    def _llm_call_text_only(self):
        """Execute API call without tools."""
        call_params = {
            "model": self.model,
            "messages": self._messages,
            "timeout": self.request_timeout,
            "temperature": get_chat_temperature(),
            "max_tokens": get_chat_max_tokens(),
        }

        provider_prefs = get_provider_preferences()
        if provider_prefs:
            call_params["extra_body"] = {"provider": provider_prefs}

        try:
            response = self.client.chat.completions.create(**call_params)

            if response.usage:
                self.token_usage["prompt_tokens"] += response.usage.prompt_tokens
                self.token_usage["completion_tokens"] += response.usage.completion_tokens
                self.token_usage["total_tokens"] += response.usage.total_tokens

            return response.choices[0].message.content
        except Exception as e:
            self._trace(f"LLM text-only call failed: {e}", COLOR_RED)
            raise

    def _llm_call_synthesis(self):
        """Execute synthesis API call."""
        call_params = {
            "model": self.synthesis_model,
            "messages": self._messages,
            "timeout": self.request_timeout,
            "temperature": get_synthesis_temperature(),
            "max_tokens": get_synthesis_max_tokens(),
        }

        provider_prefs = get_synthesis_provider_preferences()
        if provider_prefs:
            call_params["extra_body"] = {"provider": provider_prefs}

        self._trace(f"Calling {self.synthesis_model} for synthesis", COLOR_CYAN)

        try:
            response = self.synthesis_client.chat.completions.create(**call_params)
            self._trace("Synthesis LLM call completed", COLOR_GREEN)

            if response.usage:
                self.token_usage["prompt_tokens"] += response.usage.prompt_tokens
                self.token_usage["completion_tokens"] += response.usage.completion_tokens
                self.token_usage["total_tokens"] += response.usage.total_tokens

            return response.choices[0].message.content
        except Exception as e:
            self._trace(f"Synthesis LLM call failed: {e}", COLOR_RED)
            raise

    async def reset(self, keep_mcp_open: bool = False):
        """Reset agent state for a new question."""
        self._messages = [{"role": "system", "content": self.system_prompt}]
        self.token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self.tool_call_counts = {}
        self.tool_call_durations = []
        self.tool_call_history = []
        self.tool_sequence = []
        self.empty_result_count = 0
        self.last_journal_state = None

        if not keep_mcp_open and self.mcp:
            await self.mcp.close()
            self.mcp = None

    async def soft_reset(self):
        """Reset state but keep MCP connection open."""
        await self.reset(keep_mcp_open=True)

        if self.mcp:
            try:
                await self.mcp.call_tool("ManageJournal", {"action": "clear", "content": ""})
                self._trace("Journal cleared for next question", COLOR_CYAN)
            except Exception as e:
                self._trace(f"Failed to clear journal: {e}", COLOR_YELLOW)

    async def close(self):
        """Close the MCP server connection."""
        if self.mcp:
            self._trace("Closing MCP server connection...", COLOR_CYAN)
            await self.mcp.close()
            self.mcp = None
            self._trace("MCP server connection closed", COLOR_GREEN)

    def get_tool_call_summary(self) -> Dict[str, Any]:
        """Get summary of tool call durations."""
        if not self.tool_call_durations:
            return {"total_calls": 0, "total_duration_seconds": 0, "tool_breakdown": {}, "calls": []}

        total_duration = sum(call["duration_seconds"] for call in self.tool_call_durations)

        tool_breakdown = {}
        for call in self.tool_call_durations:
            tool_name = call["tool_name"]
            if tool_name not in tool_breakdown:
                tool_breakdown[tool_name] = {
                    "count": 0,
                    "total_duration": 0,
                    "avg_duration": 0,
                    "success_count": 0,
                    "failure_count": 0
                }
            tool_breakdown[tool_name]["count"] += 1
            tool_breakdown[tool_name]["total_duration"] += call["duration_seconds"]
            if call["success"]:
                tool_breakdown[tool_name]["success_count"] += 1
            else:
                tool_breakdown[tool_name]["failure_count"] += 1

        for tool_name in tool_breakdown:
            count = tool_breakdown[tool_name]["count"]
            tool_breakdown[tool_name]["avg_duration"] = round(
                tool_breakdown[tool_name]["total_duration"] / count, 3
            )
            tool_breakdown[tool_name]["total_duration"] = round(
                tool_breakdown[tool_name]["total_duration"], 3
            )

        return {
            "total_calls": len(self.tool_call_durations),
            "total_duration_seconds": round(total_duration, 3),
            "tool_breakdown": tool_breakdown,
            "tool_counts": self.tool_call_counts.copy(),
            "calls": self.tool_call_durations
        }

    def get_tool_call_counts(self) -> Dict[str, int]:
        """Get simple tool call counts."""
        return self.tool_call_counts.copy()


# ==============================================================================
# Main (for testing)
# ==============================================================================

if __name__ == "__main__":
    async def run_test():
        agent = SciQAAgent()
        try:
            answer = await agent.ask("What papers address the problem of text classification?")
            print(f"\n[SciQA Agent Answer]\n{answer}")

            counts = agent.get_tool_call_counts()
            print(f"\n[Tool Call Counts]")
            for tool_name, count in sorted(counts.items(), key=lambda x: x[1], reverse=True):
                print(f"  {tool_name}: {count}x")
        except Exception as e:
            print(f"Error in test run: {e}")
        finally:
            await agent.close()

    asyncio.run(run_test())
