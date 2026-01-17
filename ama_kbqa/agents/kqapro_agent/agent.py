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
import inspect
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

from ama_kbqa.agents.kqapro_agent.prompts import (
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
    # Reconfigure stdout to use UTF-8 with error handling
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    else:
        # Fallback for older Python versions
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, errors='replace')

load_dotenv(override=True)

# Import configuration utilities
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

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
# Configuration is now loaded from config.toml via ama_kbqa.config
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
                # Increased delay to allow subprocess cleanup and stdio buffer flushing
                # This prevents zombie processes and termination hangs
                await asyncio.sleep(0.5)  # Increased from 0.05s to 0.5s
            except (CancelledError, RuntimeError) as e:
                trace(
                    self.agent_name, f"{COLOR_YELLOW}WARNING: MCP-Close error on shutdown ignored ({type(e).__name__}).{COLOR_END}", COLOR_YELLOW)
            except Exception as e:
                trace(self.agent_name, f"{COLOR_RED}Error closing MCP client: {e}{COLOR_END}", COLOR_RED)
            finally:
                self._connected = False
                # Additional cleanup delay to ensure resources are fully released
                await asyncio.sleep(0.2)


# --- KQAProAgent ---

class KQAProAgent:
    """
    KQAPro Agent with deterministic pre/post processing hooks.

    The agent uses qtype-specific reasoning strategies that are injected
    dynamically based on the question classification.
    """

    def __init__(self, name: str = "kqapro_agent", session_id: str = "default"):
        self.name = name
        self.session_id = session_id
        self.mcp: Optional[MCPClient] = None

        # Get chat client and model from config.toml
        try:
            self.client = get_chat_client()
            self.model = get_chat_model_name()
        except (FileNotFoundError, ValueError, KeyError) as e:
            raise RuntimeError(
                f"Failed to initialize LLM client from config.toml: {e}"
            )

        # Get synthesis cPient and model from config.toml
        try:
            self.synthesis_client = get_synthesis_client()
            self.synthesis_model = get_synthesis_model_name()
        except (FileNotFoundError, ValueError, KeyError) as e:
            raise RuntimeError(
                f"Failed to initialize synthesis LLM client from config.toml: {e}"
            )

        self.request_timeout = REQUEST_TIMEOUT_SECONDS

        # NEW: Token Tracking
        self.token_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }

        # NEW: Tool Call Count Tracking (simple counter per tool)
        self.tool_call_counts: Dict[str, int] = {}

        # NEW: Tool Call Duration Tracking
        self.tool_call_durations: List[Dict[str, Any]] = []

        # Multi-Layered Loop Detection: Track multiple loop patterns
        self.tool_call_history = []  # List of (tool_name, args_json_str) tuples
        self.tool_sequence = []  # Track sequence of tool names (for oscillation detection)
        self.empty_result_count = 0  # Track consecutive empty/failed results
        self.last_journal_state = None  # Snapshot of journal to detect lack of progress

        self.system_prompt = SYSTEM_PROMPT

        self._messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt}
        ]

    def _trace(self, msg: str, color: str = COLOR_GREEN):
        trace(self.name, msg, color)

    def _load_fewshot_examples(self, max_per_type: int = 10, specific_qtype: Optional[str] = None) -> str:
        """
        Load few-shot examples from the fewshot-examples directory.

        Args:
            max_per_type: Maximum number of examples to load per question type
            specific_qtype: If provided, only load examples for this specific question type

        Returns:
            Formatted string containing few-shot examples, or empty string if none available
        """
        # Determine the fewshot examples directory
        repo_root = Path(__file__).resolve().parents[3]
        fewshot_dir = repo_root / "db" / "datasets" / "kqapro" / "fewshot-examples"

        if not fewshot_dir.exists():
            self._trace(f"Few-shot examples directory not found: {fewshot_dir}", COLOR_YELLOW)
            return ""

        # If specific_qtype is provided, only load that type's examples
        if specific_qtype:
            qtypes = [specific_qtype]
        else:
            qtypes = [
                "Count", "Verify", "SelectBetween", "SelectAmong",
                "QueryAttr", "QueryAttrQualifier", "QueryRelation",
                "QueryRelationQualifier", "QueryName"
            ]

        all_examples = []

        for qtype in qtypes:
            example_file = fewshot_dir / f"{qtype}.json"

            if not example_file.exists():
                continue

            try:
                with open(example_file, "r", encoding="utf-8") as f:
                    examples = json.load(f)

                if not examples:
                    continue

                # Limit to max_per_type examples
                examples = examples[:max_per_type]

                for example in examples:
                    all_examples.append({
                        "qtype": qtype,
                        "question": example.get("question", ""),
                        "reasoning": example.get("reasoning", ""),
                        "lesson_learned": example.get("lesson_learned", "")
                    })

            except (json.JSONDecodeError, Exception) as e:
                self._trace(f"Error loading {example_file}: {e}", COLOR_YELLOW)
                continue

        if not all_examples:
            return ""

        # Format examples for the prompt
        formatted_examples = "\n\n### Few-Shot Examples (Curated from Past Classifications)\n\n"
        formatted_examples += "Here are examples of correctly classified questions with reasoning:\n\n"

        for i, example in enumerate(all_examples, 1):
            formatted_examples += f"**Example {i}:**\n"
            formatted_examples += f"Question: {example['question']}\n"
            formatted_examples += f"Correct Type: {example['qtype']}\n"

            if example['reasoning']:
                formatted_examples += f"Reasoning: {example['reasoning']}\n"

            if example['lesson_learned']:
                formatted_examples += f"Lesson Learned: {example['lesson_learned']}\n"

            formatted_examples += "\n"

        return formatted_examples

    def _classify_question(self, question: str) -> Dict[str, str]:
        """
        Classify a question into KQAPro taxonomy and load relevant few-shot examples.

        Args:
            question: The question to classify

        Returns:
            Dict with 'question_type' and 'fewshot_examples' keys
        """
        # Build classification prompt from template
        prompt = CLASSIFICATION_PROMPT_TEMPLATE.format(question=question)

        try:
            # Call LLM with JSON mode
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": prompt}],
                temperature=get_chat_temperature(),
                response_format={"type": "json_object"},
                timeout=30.0
            )

            # Track token usage from classification
            if response.usage:
                self.token_usage["prompt_tokens"] += response.usage.prompt_tokens
                self.token_usage["completion_tokens"] += response.usage.completion_tokens
                self.token_usage["total_tokens"] += response.usage.total_tokens

            # Parse response
            json_content = response.choices[0].message.content
            result = json.loads(json_content)
            question_type = result.get("question_type", "Query")

            # Load few-shot examples for this question type
            fewshot_examples = self._load_fewshot_examples(max_per_type=10, specific_qtype=question_type)

            return {
                "question_type": question_type,
                "fewshot_examples": fewshot_examples
            }

        except Exception as e:
            self._trace(f"Question classification failed: {e}", COLOR_YELLOW)
            return {
                "question_type": "Query",
                "fewshot_examples": ""
            }

    def _extract_entities(self, question: str) -> Dict[str, List[str]]:
        """
        Extract entities/concepts and relations from a natural language query.

        Args:
            question: The question to analyze

        Returns:
            Dict with 'entities' and 'relations' keys, each containing a list of strings
        """
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

            # Track token usage from entity extraction
            if response.usage:
                self.token_usage["prompt_tokens"] += response.usage.prompt_tokens
                self.token_usage["completion_tokens"] += response.usage.completion_tokens
                self.token_usage["total_tokens"] += response.usage.total_tokens

            # Parse response
            json_content = response.choices[0].message.content
            result = json.loads(json_content)

            return {
                "entities": result.get("entities", []),
                "relations": result.get("relations", [])
            }

        except Exception as e:
            self._trace(f"Entity extraction failed: {e}", COLOR_YELLOW)
            return {
                "entities": [],
                "relations": []
            }

    def _detect_loops(self, func_name: str, func_args: dict) -> tuple[bool, str]:
        """
        Multi-layered loop detection to catch various infinite loop patterns.

        Args:
            func_name: Name of the tool being called
            func_args: Arguments for the tool call

        Returns:
            Tuple of (loop_detected: bool, reason: str)
        """
        # Create a hashable representation of this tool call
        args_str = json.dumps(func_args, sort_keys=True)
        current_call = (func_name, args_str)

        # Add to history
        self.tool_call_history.append(current_call)
        self.tool_sequence.append(func_name)

        # Keep only recent history (last 12 calls to avoid memory bloat)
        if len(self.tool_call_history) > 12:
            self.tool_call_history.pop(0)
        if len(self.tool_sequence) > 12:
            self.tool_sequence.pop(0)

        # DETECTION 1: Identical Repeated Calls (same tool + same params)
        if len(self.tool_call_history) >= 3:
            recent_calls = self.tool_call_history[-3:]
            if all(call == current_call for call in recent_calls):
                return True, f"Identical call repeated 3 times: {func_name}({list(func_args.keys())})"

        # DETECTION 2: Oscillating Pattern (A-B-A-B or A-B-C-A-B-C)
        if len(self.tool_sequence) >= 6:
            # Check for 2-tool oscillation (A-B-A-B-A-B)
            last_6 = self.tool_sequence[-6:]
            if last_6[0] == last_6[2] == last_6[4] and last_6[1] == last_6[3] == last_6[5] and last_6[0] != last_6[1]:
                return True, f"Oscillating between {last_6[0]} and {last_6[1]} (A-B-A-B-A-B pattern)"

            # Check for 3-tool oscillation (A-B-C-A-B-C)
            if last_6[0] == last_6[3] and last_6[1] == last_6[4] and last_6[2] == last_6[5]:
                return True, f"Oscillating between {last_6[0]}, {last_6[1]}, {last_6[2]} (A-B-C-A-B-C pattern)"

        # DETECTION 3: Same tool called 5+ times in last 6 calls (even with different params)
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
        """
        Provide tool-specific guidance when a loop is detected.

        Args:
            func_name: Name of the tool that's looping

        Returns:
            Specific recovery guidance for this tool
        """
        # Return specific guidance or generic fallback
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

            # 2. Add user query to message history
            self._messages.append({"role": "user", "content": query})

            # ═══════════════════════════════════════════════════════════════════════
            # PRE-AGENT HOOK: Deterministic Classification & Entity Extraction
            # This ALWAYS runs before the main loop to provide foundational analysis
            # ═══════════════════════════════════════════════════════════════════════
            self._trace("🎯 Starting pre-agent classification hook", COLOR_CYAN)

            # Step 1: Run classification and entity extraction (using internal methods)
            self._trace("🔍 Running question classification and entity extraction...", COLOR_YELLOW)

            # Call internal methods (no longer using MCP tools)
            qtype_data = self._classify_question(query)
            self._trace(f"✓ Question classification complete: {qtype_data['question_type']}", COLOR_GREEN)

            entities_data = self._extract_entities(query)
            self._trace(f"✓ Entity extraction complete: {len(entities_data['entities'])} entities, {len(entities_data['relations'])} relations", COLOR_GREEN)

            # Step 2: Extract and format the analysis data
            qtype = qtype_data.get("question_type", "Query")
            fewshot_examples = qtype_data.get("fewshot_examples", "")
            entities = entities_data.get("entities", [])
            relations = entities_data.get("relations", [])

            # Format entities and relations for readability
            formatted_entities = "\n".join([f"  - {e}" for e in entities]) if entities else "  (none identified)"
            formatted_relations = "\n".join([f"  - {r}" for r in relations]) if relations else "  (none identified)"

            # Step 4: Get the qtype-specific strategy
            qtype_strategy = QTYPE_STRATEGIES.get(qtype, QTYPE_STRATEGIES["Query"])

            # Step 5: Build the analysis context message
            analysis_context = ANALYSIS_CONTEXT_TEMPLATE.format(
                qtype=qtype,
                formatted_entities=formatted_entities,
                formatted_relations=formatted_relations,
                qtype_strategy=qtype_strategy
            )

            # Add fewshot examples if provided by QtypePrediction
            if fewshot_examples and fewshot_examples.strip():
                analysis_context += FEWSHOT_EXAMPLES_TEMPLATE.format(
                    qtype=qtype,
                    fewshot_examples=fewshot_examples
                )

            analysis_context += ANALYSIS_CONTEXT_SUFFIX

            # Step 6: Inject the analysis context as a user message
            self._messages.append({
                "role": "user",
                "content": analysis_context
            })

            self._trace(
                f"✅ Pre-agent hook complete - Type: {qtype}, Entities: {len(entities)}, Relations: {len(relations)}", COLOR_GREEN)

            # 3. Iterative tool-call loop
            iteration_count = 0
            max_iterations = 50  # Safety limit to prevent infinite loops
            while True:
                iteration_count += 1
                self._trace(f"🔄 Starting iteration {iteration_count}/{max_iterations}", COLOR_CYAN)

                # Safety check: prevent infinite loops
                if iteration_count > max_iterations:
                    self._trace(
                        f"⚠️  WARNING: Reached maximum iteration limit ({max_iterations}). Stopping.", COLOR_RED)
                    self._trace(f"⚠️  The agent may not have found a complete answer. Review logs for issues.", COLOR_RED)
                    return "Error: Agent reached maximum iteration limit without completing the task. This may indicate a problem with the reasoning loop or insufficient information in the knowledge base."

                # Warn when approaching iteration limit
                if iteration_count >= max_iterations - 5:
                    self._trace(f"⚠️  Approaching iteration limit! ({iteration_count}/{max_iterations})", COLOR_YELLOW)

                # Periodic status update
                if iteration_count % 10 == 0:
                    self._trace(
                        f"📊 Status: {self.token_usage['total_tokens']} tokens used, {len(self._messages)} messages in history", COLOR_CYAN)

                # Periodic journal refresh (every 5 iterations) + PROGRESS CHECK
                if iteration_count % 5 == 0 and iteration_count > 0:
                    self._trace("🔄 Injecting journal summary for working memory refresh", COLOR_CYAN)
                    try:
                        journal_refresh = await self.mcp.call_tool("GetJournalSummary", {})

                        # DETECTION 4: Check if journal has changed since last check (progress detection)
                        if self.last_journal_state is not None and journal_refresh == self.last_journal_state:
                            self._trace("⚠️  WARNING: Journal unchanged for 5 iterations - no progress!", COLOR_YELLOW)
                            # Inject a stronger intervention prompt
                            self._messages.append({
                                "role": "user",
                                "content": NO_PROGRESS_TEMPLATE.format(
                                    iteration_count=iteration_count,
                                    journal_refresh=journal_refresh
                                )
                            })
                        else:
                            # Normal refresh
                            self._messages.append({
                                "role": "user",
                                "content": JOURNAL_REFRESH_TEMPLATE.format(
                                    iteration_count=iteration_count,
                                    journal_refresh=journal_refresh
                                )
                            })

                        # Update last state for next comparison
                        self.last_journal_state = journal_refresh

                    except Exception as e:
                        self._trace(f"⚠️ Failed to inject journal refresh: {e}", COLOR_YELLOW)

                self._trace(
                    f"📤 Calling LLM with {len(self._messages)} messages and {len(openai_tools)} tools...", COLOR_YELLOW)
                response = self._llm_call(tools=openai_tools)
                message = response.choices[0].message

                finish_reason = response.choices[0].finish_reason
                self._trace(f"📥 LLM response received (finish_reason: {finish_reason})", COLOR_CYAN)

                # Warn on unexpected finish reasons
                if finish_reason not in ["stop", "tool_calls"]:
                    self._trace(
                        f"⚠️  Unexpected finish_reason: {finish_reason} (expected 'stop' or 'tool_calls')", COLOR_YELLOW)
                    if finish_reason == "length":
                        self._trace(
                            f"⚠️  LLM hit token limit! Response may be truncated. Consider increasing max_tokens.", COLOR_RED)

                # Update Token Usage
                if response.usage:
                    self.token_usage["prompt_tokens"] += response.usage.prompt_tokens
                    self.token_usage["completion_tokens"] += response.usage.completion_tokens
                    self.token_usage["total_tokens"] += response.usage.total_tokens

                # Logging intermediate thoughts (Thoughts/Text)
                if message.content:
                    self._trace(f"🧠 Thought/Text: {message.content}", COLOR_BLUE)

                # Final answer (no more tools) - break out of loop for synthesis
                if not message.tool_calls:
                    self._trace(f"🏁 No more tool calls - breaking to synthesis step", COLOR_GREEN)
                    # Store any intermediate reasoning the agent provided
                    if message.content:
                        self._messages.append({"role": "assistant", "content": message.content})
                    # Break out of loop to run deterministic synthesis
                    break

                # --- CORRECTION START ---
                # The assistant message MUST be added to the history ONCE
                # BEFORE processing the tools.
                # Convert ChatCompletionMessage to dict for message history
                msg_dict = {
                    "role": message.role,
                    "content": message.content,
                }
                if message.tool_calls:
                    # Convert tool_calls to proper format
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
                # --- CORRECTION END ---

                # Track if GetJournalSummary was called in this turn
                called_get_journal_summary = False

                # Execute tool call(s)
                self._trace(f"🔧 Processing {len(message.tool_calls)} tool call(s)", COLOR_YELLOW)
                for tool_call in message.tool_calls:
                    # Validate tool call has a valid name
                    if not tool_call.function or not tool_call.function.name:
                        self._trace(f"⚠️  Skipping invalid tool call with missing name", COLOR_RED)
                        # Add error response to maintain conversation flow
                        self._messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": "invalid_tool",
                            "content": "Error: Tool call has no valid name. Please try a different approach."
                        })
                        continue

                    func_name = tool_call.function.name

                    # Track if this is GetJournalSummary
                    if func_name == "GetJournalSummary":
                        called_get_journal_summary = True

                    try:
                        # Handle case where arguments might be None
                        args_str = tool_call.function.arguments
                        if args_str is None or args_str == "":
                            func_args = {}
                        else:
                            func_args = json.loads(args_str)
                    except json.JSONDecodeError as e:
                        self._trace(f"⚠️  Invalid JSON in tool arguments: {e}", COLOR_YELLOW)
                        func_args = {}

                    # Logging
                    args_pretty = json.dumps(func_args, indent=2, ensure_ascii=False)
                    self._trace(f"🛠️  Tool Call: {func_name}\n   Params: {args_pretty}", COLOR_YELLOW)

                    # MULTI-LAYERED LOOP DETECTION
                    loop_detected, loop_reason = self._detect_loops(func_name, func_args)

                    if loop_detected:
                        self._trace(
                            f"🔄 INFINITE LOOP DETECTED: {loop_reason}",
                            COLOR_RED
                        )

                        # Fetch current journal state to help agent recover
                        try:
                            journal_state = await self.mcp.call_tool("GetJournalSummary", {})
                            self._trace("📋 Fetched journal summary for loop recovery", COLOR_CYAN)
                        except Exception as e:
                            self._trace(f"⚠️ Failed to fetch journal for loop recovery: {e}", COLOR_YELLOW)
                            journal_state = "(Journal unavailable)"

                        # Provide tool-specific intervention message
                        tool_specific_guidance = self._get_tool_specific_loop_guidance(func_name)

                        tool_result = LOOP_INTERVENTION_TEMPLATE.format(
                            loop_reason=loop_reason,
                            func_name=func_name,
                            tool_specific_guidance=tool_specific_guidance,
                            journal_state=journal_state
                        )

                        # Clear recent history to allow fresh attempts
                        self.tool_call_history = []
                        self.tool_sequence = []
                    else:
                        # Normal execution - no loop detected
                        # Execution with error handling and duration tracking
                        tool_start_time = time.time()
                        try:
                            tool_result = await self.mcp.call_tool(func_name, func_args)
                            tool_end_time = time.time()
                            tool_duration = tool_end_time - tool_start_time

                            # Track tool call count (simple counter)
                            self.tool_call_counts[func_name] = self.tool_call_counts.get(func_name, 0) + 1

                            # Track tool call duration
                            self.tool_call_durations.append({
                                "tool_name": func_name,
                                "duration_seconds": round(tool_duration, 3),
                                "success": True,
                                "timestamp": datetime.now().isoformat()
                            })

                            # Logging result with duration
                            log_result = tool_result
                            if len(log_result) > 500:
                                log_result = log_result[:500] + f"... [truncated, total len: {len(tool_result)}]"
                            self._trace(f"🔙 Result ({func_name}) [{tool_duration:.3f}s]: {log_result}", COLOR_CYAN)

                        except Exception as tool_error:
                            tool_end_time = time.time()
                            tool_duration = tool_end_time - tool_start_time

                            # Track tool call count (even for failures)
                            self.tool_call_counts[func_name] = self.tool_call_counts.get(func_name, 0) + 1

                            # Track failed tool call duration
                            self.tool_call_durations.append({
                                "tool_name": func_name,
                                "duration_seconds": round(tool_duration, 3),
                                "success": False,
                                "error": str(tool_error),
                                "timestamp": datetime.now().isoformat()
                            })

                            # Handle tool execution errors gracefully
                            error_msg = str(tool_error)
                            self._trace(f"⚠️  Tool {func_name} failed [{tool_duration:.3f}s]: {error_msg}", COLOR_RED)

                            # Create error response that helps the LLM recover
                            tool_result = f"Error executing {func_name}: {error_msg}. Please try a different approach or verify the parameters."

                    # --- BUG WAS HERE (Removed: self._messages.append(message)) ---

                    # Only the tool result is appended in the loop
                    self._messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": func_name,
                        "content": tool_result
                    })

                # CRITICAL FIX: After ALL tools are processed, if GetJournalSummary was called,
                # inject a user prompt to force the agent to provide a final answer
                if called_get_journal_summary:
                    self._trace("📝 GetJournalSummary was called - injecting answer prompt", COLOR_CYAN)
                    self._messages.append({
                        "role": "user",
                        "content": JOURNAL_SUMMARY_ANSWER_PROMPT
                    })

            # ═══════════════════════════════════════════════════════════════════════
            # DETERMINISTIC POST-AGENT SYNTHESIS STEP
            # This ALWAYS runs after tool gathering is complete to ensure a final answer
            # ═══════════════════════════════════════════════════════════════════════
            self._trace("🎯 Starting deterministic synthesis step", COLOR_CYAN)

            # Step 1: Fetch complete journal summary with all discovered information
            self._trace("📋 Fetching journal summary with all discovered data...", COLOR_YELLOW)
            journal_summary = await self.mcp.call_tool("GetJournalSummary", {})
            self._trace(f"✓ Journal fetched ({len(journal_summary)} chars)", COLOR_GREEN)

            # Step 2: Inject synthesis prompt with full context
            synthesis_prompt = SYNTHESIS_PROMPT_TEMPLATE.format(
                journal_summary=journal_summary,
                query=query
            )

            self._messages.append({
                "role": "user",
                "content": synthesis_prompt
            })

            # Step 3: Make final synthesis call using synthesis-specific configuration
            self._trace("🤖 Making final synthesis LLM call (using synthesis config)...", COLOR_YELLOW)
            final_answer = self._llm_call_synthesis()

            # Step 4: Validate and return
            if final_answer and final_answer.strip():
                self._trace(f"✅ Synthesis complete - final answer generated ({len(final_answer)} chars)", COLOR_GREEN)
                self._messages.append({"role": "assistant", "content": final_answer})
                return final_answer.strip()
            else:
                # Fallback if synthesis fails
                self._trace("⚠️  WARNING: Synthesis call returned empty response", COLOR_RED)
                fallback_answer = "Unable to generate a final answer based on the gathered information. The investigation completed but synthesis failed."
                self._messages.append({"role": "assistant", "content": fallback_answer})
                return fallback_answer

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

                # Log tool call duration summary
                tool_summary = self.get_tool_call_summary()
                if tool_summary["total_calls"] > 0:
                    self._trace(
                        f"⏱️  TOOL CALL SUMMARY: {tool_summary['total_calls']} calls, "
                        f"total {tool_summary['total_duration_seconds']}s",
                        COLOR_CYAN
                    )
                    for tool_name, stats in tool_summary["tool_breakdown"].items():
                        self._trace(
                            f"   📊 {tool_name}: {stats['count']}x, "
                            f"total {stats['total_duration']}s, "
                            f"avg {stats['avg_duration']}s",
                            COLOR_CYAN
                        )

                # NOTE: MCP connection is NOT closed here anymore.
                # Use agent.close() explicitly when done with batch processing,
                # or agent.reset() (without keep_mcp_open=True) for single question mode.
                # This allows keeping MCP open across multiple questions in batch processing.
                self._trace("✅ Question complete (MCP connection preserved)", COLOR_GREEN)

    def _llm_call(self, tools: Optional[List[Dict[str, Any]]] = None):
        """Executes the actual API call (with tools)."""
        call_params = {
            "model": self.model,
            "messages": self._messages,
            "timeout": self.request_timeout,
            "temperature": get_chat_temperature(),
            "max_tokens": get_chat_max_tokens(),
            "tools": tools
        }

        # Add OpenRouter provider preferences if configured
        provider_prefs = get_provider_preferences()
        if provider_prefs:
            call_params["extra_body"] = {"provider": provider_prefs}

        self._trace(
            f"⏳ Calling {self.model} (timeout: {self.request_timeout}s, max_tokens: {call_params['max_tokens']})", COLOR_CYAN)

        try:
            response = self.client.chat.completions.create(**call_params)
            self._trace(f"✅ LLM call completed successfully", COLOR_GREEN)
            return response
        except Exception as e:
            self._trace(f"❌ LLM call failed: {e}", COLOR_RED)
            raise

    def _llm_call_text_only(self):
        """Executes the actual API call (without tools)."""
        call_params = {
            "model": self.model,
            "messages": self._messages,
            "timeout": self.request_timeout,
            "temperature": get_chat_temperature(),
            "max_tokens": get_chat_max_tokens(),
        }

        # Add OpenRouter provider preferences if configured
        provider_prefs = get_provider_preferences()
        if provider_prefs:
            call_params["extra_body"] = {"provider": provider_prefs}

        self._trace(f"⏳ Calling {self.model} (text-only, timeout: {self.request_timeout}s)", COLOR_CYAN)

        try:
            response = self.client.chat.completions.create(**call_params)
            self._trace(f"✅ LLM text-only call completed", COLOR_GREEN)

            if response.usage:
                self.token_usage["prompt_tokens"] += response.usage.prompt_tokens
                self.token_usage["completion_tokens"] += response.usage.completion_tokens
                self.token_usage["total_tokens"] += response.usage.total_tokens

            return response.choices[0].message.content
        except Exception as e:
            self._trace(f"❌ LLM text-only call failed: {e}", COLOR_RED)
            raise

    def _llm_call_synthesis(self):
        """Executes the synthesis API call using synthesis-specific configuration."""
        call_params = {
            "model": self.synthesis_model,
            "messages": self._messages,
            "timeout": self.request_timeout,
            "temperature": get_synthesis_temperature(),
            "max_tokens": get_synthesis_max_tokens(),
        }

        # Add OpenRouter provider preferences if configured
        provider_prefs = get_synthesis_provider_preferences()
        if provider_prefs:
            call_params["extra_body"] = {"provider": provider_prefs}

        self._trace(
            f"⏳ Calling {self.synthesis_model} for synthesis "
            f"(timeout: {self.request_timeout}s, max_tokens: {call_params['max_tokens']})",
            COLOR_CYAN
        )

        try:
            response = self.synthesis_client.chat.completions.create(**call_params)
            self._trace(f"✅ Synthesis LLM call completed", COLOR_GREEN)

            if response.usage:
                self.token_usage["prompt_tokens"] += response.usage.prompt_tokens
                self.token_usage["completion_tokens"] += response.usage.completion_tokens
                self.token_usage["total_tokens"] += response.usage.total_tokens

            return response.choices[0].message.content
        except Exception as e:
            self._trace(f"❌ Synthesis LLM call failed: {e}", COLOR_RED)
            raise

    async def reset(self, keep_mcp_open: bool = False):
        """
        Reset the agent state for a new question.

        Args:
            keep_mcp_open: If True, keeps the MCP server connection open (useful for batch processing)
        """
        self._messages = [
            {"role": "system", "content": self.system_prompt}
        ]
        self.token_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }
        # Clear tool call tracking
        self.tool_call_counts = {}
        self.tool_call_durations = []
        # Clear all loop detection tracking
        self.tool_call_history = []
        self.tool_sequence = []
        self.empty_result_count = 0
        self.last_journal_state = None

        if not keep_mcp_open and self.mcp:
            await self.mcp.close()
            self.mcp = None

    async def soft_reset(self):
        """
        Reset agent state but keep the MCP server connection open.
        Use this between questions in batch processing for efficiency.
        """
        await self.reset(keep_mcp_open=True)

        # Reset the journal/scratchpad on the MCP server side
        if self.mcp:
            try:
                await self.mcp.call_tool("ManageJournal", {"action": "clear", "content": ""})
                self._trace("📋 Journal cleared for next question", COLOR_CYAN)
            except Exception as e:
                self._trace(f"⚠️ Failed to clear journal: {e}", COLOR_YELLOW)

    async def close(self):
        """
        Explicitly close the MCP server connection.
        Call this when completely done with the agent (e.g., at end of batch processing).
        """
        if self.mcp:
            self._trace("🛑 Closing MCP server connection...", COLOR_CYAN)
            await self.mcp.close()
            self.mcp = None
            self._trace("✅ MCP server connection closed", COLOR_GREEN)

    def get_tool_call_summary(self) -> Dict[str, Any]:
        """
        Get a summary of tool call durations for the current question.

        Returns:
            Dictionary with tool call statistics including total time,
            per-tool breakdown, and individual call details.
        """
        if not self.tool_call_durations:
            return {"total_calls": 0, "total_duration_seconds": 0, "tool_breakdown": {}, "calls": []}

        total_duration = sum(call["duration_seconds"] for call in self.tool_call_durations)

        # Breakdown by tool
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

        # Calculate averages
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
        """
        Get a simple dictionary of tool call counts for the current question.

        Returns:
            Dictionary mapping tool names to their call counts.
        """
        return self.tool_call_counts.copy()


if __name__ == "__main__":
    async def run_test():
        agent = KQAProAgent()
        try:
            answer = await agent.ask("Who is the director of Inception?")
            print(f"\n[KQAPro Agent Answer]\n{answer}")

            # Print tool call counts (simple view)
            counts = agent.get_tool_call_counts()
            print(f"\n[Tool Call Counts]")
            for tool_name, count in sorted(counts.items(), key=lambda x: x[1], reverse=True):
                print(f"  {tool_name}: {count}x")

            # Print tool call summary (detailed view)
            summary = agent.get_tool_call_summary()
            print(f"\n[Tool Call Summary]")
            print(f"Total calls: {summary['total_calls']}")
            print(f"Total duration: {summary['total_duration_seconds']}s")
            for tool_name, stats in summary['tool_breakdown'].items():
                print(f"  {tool_name}: {stats['count']}x, avg {stats['avg_duration']}s")
        except Exception as e:
            print(f"Error in test run: {e}")
        finally:
            # Always close the MCP connection when done
            await agent.close()

    asyncio.run(run_test())
