"""
Abstract base class for KBQA agents.

Provides the core agent loop, tool calling, loop detection, and
synthesis functionality that all KBQA agents share.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
import os
import sys
import json
import time
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv
from mcp.types import Tool as McpTool

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
from ama_kbqa.framework.config import KnowledgeGraphConfig
from ama_kbqa.framework.mcp_client import MCPClient, trace

# Configure stdout to handle Unicode on Windows
if sys.platform == 'win32':
    import codecs
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    else:
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, errors='replace')

load_dotenv(override=True)

# Terminal colors
COLOR_BLUE = '\033[94m'
COLOR_GREEN = '\033[92m'
COLOR_RED = '\033[91m'
COLOR_YELLOW = '\033[93m'
COLOR_CYAN = '\033[96m'
COLOR_END = '\033[0m'

# Default timeout
REQUEST_TIMEOUT_SECONDS = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "60"))


class BaseKBQAAgent(ABC):
    """
    Abstract base class for KBQA agents.

    Provides:
    - MCP client management
    - Pre-agent hooks (classification, entity extraction)
    - 4-layer loop detection
    - Tool-calling loop with iteration limits
    - Post-agent synthesis
    - Token and tool tracking
    """

    def __init__(self, name: str = "kbqa_agent", session_id: str = "default"):
        """
        Initialize the base KBQA agent.

        Args:
            name: Agent name for tracing
            session_id: Session identifier
        """
        self.name = name
        self.session_id = session_id
        self.mcp: Optional[MCPClient] = None

        # Initialize LLM clients
        try:
            self.client = get_chat_client()
            self.model = get_chat_model_name()
        except (FileNotFoundError, ValueError, KeyError) as e:
            raise RuntimeError(f"Failed to initialize LLM client from config.toml: {e}")

        try:
            self.synthesis_client = get_synthesis_client()
            self.synthesis_model = get_synthesis_model_name()
        except (FileNotFoundError, ValueError, KeyError) as e:
            raise RuntimeError(f"Failed to initialize synthesis LLM client from config.toml: {e}")

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

        # Loop detection state
        self.tool_call_history: List[Tuple[str, str]] = []
        self.tool_sequence: List[str] = []
        self.empty_result_count = 0
        self.last_journal_state: Optional[str] = None

        # Message history
        self._messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self._get_system_prompt()}
        ]

    # =========================================================================
    # ABSTRACT METHODS - Must be implemented by subclasses
    # =========================================================================

    @abstractmethod
    def get_config(self) -> KnowledgeGraphConfig:
        """
        Get the knowledge graph configuration.

        Returns:
            Configuration for the specific knowledge graph
        """
        pass

    @abstractmethod
    def get_mcp_server_path(self) -> str:
        """
        Get the path to the MCP server for this agent.

        Returns:
            Absolute path to the MCP server Python file
        """
        pass

    # =========================================================================
    # TEMPLATE METHODS - Override to customize behavior
    # =========================================================================

    def _get_system_prompt(self) -> str:
        """
        Get the system prompt for this agent.

        Override in subclass to provide KG-specific prompt.

        Returns:
            System prompt string
        """
        return "You are a KBQA agent."

    def _get_qtype_strategies(self) -> Dict[str, str]:
        """
        Get question-type specific reasoning strategies.

        Override in subclass to provide KG-specific strategies.

        Returns:
            Dictionary mapping question types to strategy prompts
        """
        return {}

    def _get_classification_prompt(self, question: str) -> str:
        """
        Get the classification prompt for a question.

        Override in subclass for KG-specific classification.

        Args:
            question: The question to classify

        Returns:
            Classification prompt string
        """
        return f"""Classify this question into a category.
Question: {question}
Respond with JSON: {{"question_type": "category"}}"""

    def _get_entity_extraction_prompt(self) -> str:
        """
        Get the entity extraction prompt.

        Override in subclass for KG-specific extraction.

        Returns:
            Entity extraction prompt string
        """
        return """Extract entities and relations from the query.
Respond with JSON: {"entities": [], "relations": []}"""

    def _get_analysis_context_template(self) -> str:
        """
        Get the analysis context template.

        Override in subclass for KG-specific context.

        Returns:
            Analysis context template string
        """
        return """PRE-ANALYSIS
Question Type: {qtype}
Entities: {formatted_entities}
Relations: {formatted_relations}
{qtype_strategy}

Proceed with your investigation."""

    def _get_synthesis_prompt_template(self) -> str:
        """
        Get the synthesis prompt template.

        Override in subclass for KG-specific synthesis.

        Returns:
            Synthesis prompt template string
        """
        return """JOURNAL SUMMARY
{journal_summary}

Based on this information, answer: "{query}"

YOUR FINAL ANSWER:"""

    def _get_journal_refresh_template(self) -> str:
        """
        Get the journal refresh template.

        Returns:
            Journal refresh template string
        """
        return """WORKING MEMORY REFRESH (Iteration {iteration_count})

{journal_refresh}

Continue your investigation. Avoid revisiting what you've already explored."""

    def _get_no_progress_template(self) -> str:
        """
        Get the no-progress intervention template.

        Returns:
            No progress template string
        """
        return """WARNING: NO PROGRESS DETECTED (Iteration {iteration_count})

Your journal has NOT changed in 5 iterations.

{journal_refresh}

You MUST change your approach NOW."""

    def _get_tool_loop_guidance(self) -> Dict[str, str]:
        """
        Get tool-specific loop recovery guidance.

        Override in subclass for KG-specific guidance.

        Returns:
            Dictionary mapping tool names to guidance strings
        """
        return {}

    def _get_generic_loop_guidance(self) -> str:
        """
        Get generic loop recovery guidance.

        Returns:
            Generic loop guidance string
        """
        return """Generic Recovery:
- Try a different tool
- Review your journal
- Answer with available data
- The data might not exist"""

    def _get_loop_intervention_template(self) -> str:
        """
        Get the loop intervention template.

        Returns:
            Loop intervention template string
        """
        return """INFINITE LOOP DETECTED

Pattern: {loop_reason}

STOP using '{func_name}' and try:
{tool_specific_guidance}

WHAT YOU'VE DISCOVERED:
{journal_state}

Change strategy or acknowledge the data doesn't exist."""

    def _get_journal_summary_answer_prompt(self) -> str:
        """
        Get the prompt to inject after GetJournalSummary.

        Returns:
            Answer prompt string
        """
        return "Now provide your final answer. Do NOT call more tools."

    # =========================================================================
    # PRE-AGENT HOOKS
    # =========================================================================

    def _classify_question(self, question: str) -> Dict[str, str]:
        """
        Classify a question into the KG taxonomy.

        Args:
            question: The question to classify

        Returns:
            Dict with 'question_type' key
        """
        prompt = self._get_classification_prompt(question)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": prompt}],
                temperature=get_chat_temperature(),
                response_format={"type": "json_object"},
                timeout=30.0
            )

            if response.usage:
                self._track_token_usage(response.usage)

            json_content = response.choices[0].message.content
            result = json.loads(json_content)
            return {
                "question_type": result.get("question_type", "Query"),
                "fewshot_examples": ""
            }

        except Exception as e:
            self._trace(f"Question classification failed: {e}", COLOR_YELLOW)
            return {"question_type": "Query", "fewshot_examples": ""}

    def _extract_entities(self, question: str) -> Dict[str, List[str]]:
        """
        Extract entities and relations from a question.

        Args:
            question: The question to analyze

        Returns:
            Dict with 'entities' and 'relations' keys
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._get_entity_extraction_prompt()},
                    {"role": "user", "content": question}
                ],
                temperature=get_chat_temperature(),
                response_format={"type": "json_object"},
                timeout=30.0
            )

            if response.usage:
                self._track_token_usage(response.usage)

            json_content = response.choices[0].message.content
            result = json.loads(json_content)
            return {
                "entities": result.get("entities", []),
                "relations": result.get("relations", [])
            }

        except Exception as e:
            self._trace(f"Entity extraction failed: {e}", COLOR_YELLOW)
            return {"entities": [], "relations": []}

    def _build_analysis_context(
        self,
        qtype: str,
        entities: List[str],
        relations: List[str],
        fewshot_examples: str = ""
    ) -> str:
        """
        Build the analysis context message.

        Args:
            qtype: Question type
            entities: Extracted entities
            relations: Extracted relations
            fewshot_examples: Optional few-shot examples

        Returns:
            Analysis context string
        """
        formatted_entities = "\n".join([f"  - {e}" for e in entities]) if entities else "  (none)"
        formatted_relations = "\n".join([f"  - {r}" for r in relations]) if relations else "  (none)"

        strategies = self._get_qtype_strategies()
        qtype_strategy = strategies.get(qtype, strategies.get("Query", ""))

        context = self._get_analysis_context_template().format(
            qtype=qtype,
            formatted_entities=formatted_entities,
            formatted_relations=formatted_relations,
            qtype_strategy=qtype_strategy
        )

        if fewshot_examples:
            context += f"\n\nFew-shot Examples for {qtype}:\n{fewshot_examples}"

        return context

    # =========================================================================
    # LOOP DETECTION
    # =========================================================================

    def _detect_loops(self, func_name: str, func_args: dict) -> Tuple[bool, str]:
        """
        Multi-layered loop detection.

        Detects:
        1. Identical repeated calls (same tool + same params)
        2. Oscillating patterns (A-B-A-B or A-B-C-A-B-C)
        3. Same tool called too many times
        4. No progress (journal unchanged)

        Args:
            func_name: Name of the tool being called
            func_args: Arguments for the tool call

        Returns:
            Tuple of (loop_detected, reason)
        """
        args_str = json.dumps(func_args, sort_keys=True)
        current_call = (func_name, args_str)

        self.tool_call_history.append(current_call)
        self.tool_sequence.append(func_name)

        # Keep only recent history
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
            # A-B-A-B-A-B pattern
            if (last_6[0] == last_6[2] == last_6[4] and
                last_6[1] == last_6[3] == last_6[5] and
                last_6[0] != last_6[1]):
                return True, f"Oscillating between {last_6[0]} and {last_6[1]}"
            # A-B-C-A-B-C pattern
            if (last_6[0] == last_6[3] and
                last_6[1] == last_6[4] and
                last_6[2] == last_6[5]):
                return True, f"Oscillating between {last_6[0]}, {last_6[1]}, {last_6[2]}"

        # Detection 3: Same tool called too often
        if len(self.tool_sequence) >= 6:
            last_6 = self.tool_sequence[-6:]
            tool_counts: Dict[str, int] = {}
            for t in last_6:
                tool_counts[t] = tool_counts.get(t, 0) + 1
            for tool, count in tool_counts.items():
                if count >= 5:
                    return True, f"Tool '{tool}' called {count} times in last 6 iterations"

        # Detection 4: FindResource cap (configurable per agent)
        find_resource_cap = getattr(self, '_find_resource_cap', 8)
        if func_name == "FindResource":
            fr_count = self.tool_call_counts.get("FindResource", 0) + 1
            if fr_count > find_resource_cap:
                return True, (
                    f"FindResource called {fr_count} times (cap: {find_resource_cap}). "
                    f"Switch to SPARQL or other structured queries."
                )

        return False, ""

    def _get_tool_specific_loop_guidance(self, func_name: str) -> str:
        """
        Get tool-specific guidance for loop recovery.

        Args:
            func_name: Name of the looping tool

        Returns:
            Recovery guidance string
        """
        guidance = self._get_tool_loop_guidance()
        return guidance.get(func_name, self._get_generic_loop_guidance())

    # =========================================================================
    # CORE AGENT LOOP
    # =========================================================================

    async def ask(self, query: str) -> str:
        """
        Answer a question using the knowledge graph.

        Args:
            query: The natural language question

        Returns:
            The agent's answer
        """
        self._trace(f"Incoming query: '{query}'", COLOR_GREEN)

        try:
            # Initialize MCP connection
            await self._init_mcp()

            if not self.mcp:
                self._trace(f"{COLOR_YELLOW}Tool server not available.{COLOR_END}", COLOR_YELLOW)
                self._messages.append({"role": "user", "content": query})
                return self._llm_call_text_only()

            # Get tools
            mcp_tools = await self.mcp.list_tools()
            openai_tools = self.mcp.convert_tools_to_openai_format(mcp_tools)
            self._trace(f"Found {len(openai_tools)} tools.")

            # Populate known tool names for validation (Fix 6)
            self._known_tool_names = {t.name for t in mcp_tools}

            # Read per-agent config for FindResource cap (Fix 5)
            config = self.get_config()
            self._find_resource_cap = config.domain_settings.get("find_resource_cap", 8)
            self._context_limit = config.domain_settings.get("context_limit", 100000)

            # Add query to messages
            self._messages.append({"role": "user", "content": query})

            # Run pre-agent hooks
            self._trace("Starting pre-agent classification hook", COLOR_CYAN)

            qtype_data = self._classify_question(query)
            self._trace(f"Question classification: {qtype_data['question_type']}", COLOR_GREEN)

            entities_data = self._extract_entities(query)
            self._trace(
                f"Entity extraction: {len(entities_data['entities'])} entities, "
                f"{len(entities_data['relations'])} relations",
                COLOR_GREEN
            )

            # Build and inject analysis context
            qtype = qtype_data.get("question_type", "Query")
            fewshot_examples = qtype_data.get("fewshot_examples", "")
            entities = entities_data.get("entities", [])
            relations = entities_data.get("relations", [])

            analysis_context = self._build_analysis_context(
                qtype, entities, relations, fewshot_examples
            )
            self._messages.append({"role": "user", "content": analysis_context})

            self._trace(f"Pre-agent hook complete - Type: {qtype}", COLOR_GREEN)

            # Run tool loop (config already loaded above)
            max_iterations = config.domain_settings.get("max_iterations", 50)
            refresh_interval = config.domain_settings.get("journal_refresh_interval", 5)

            answer = await self._run_tool_loop(
                query, openai_tools, max_iterations, refresh_interval
            )

            return answer

        except Exception as e:
            self._trace(f"{COLOR_RED}Error in agent loop: {e}{COLOR_END}", COLOR_RED)
            raise

        finally:
            await self._finalize_question()

    async def _run_tool_loop(
        self,
        query: str,
        tools: List[Dict],
        max_iterations: int,
        refresh_interval: int
    ) -> str:
        """
        Run the main tool-calling loop.

        Args:
            query: Original query
            tools: OpenAI-format tools
            max_iterations: Maximum loop iterations
            refresh_interval: Iterations between journal refreshes

        Returns:
            Final answer string
        """
        iteration_count = 0

        while True:
            iteration_count += 1
            self._trace(f"Starting iteration {iteration_count}/{max_iterations}", COLOR_CYAN)

            # Safety check
            if iteration_count > max_iterations:
                self._trace(f"WARNING: Reached max iterations ({max_iterations})", COLOR_RED)
                return "Error: Agent reached maximum iteration limit."

            # Manage context window before LLM call
            self._manage_context_window()

            # Periodic journal refresh
            if iteration_count % refresh_interval == 0 and iteration_count > 0:
                await self._inject_journal_refresh(iteration_count)

            # Call LLM
            self._trace(f"Calling LLM with {len(self._messages)} messages...", COLOR_YELLOW)
            response = self._llm_call(tools=tools)
            message = response.choices[0].message
            finish_reason = response.choices[0].finish_reason

            self._trace(f"LLM response (finish_reason: {finish_reason})", COLOR_CYAN)

            if response.usage:
                self._track_token_usage(response.usage)

            if message.content:
                self._trace(f"Thought: {message.content}", COLOR_BLUE)

            # No tool calls - break for synthesis
            if not message.tool_calls:
                self._trace("No more tool calls - breaking to synthesis", COLOR_GREEN)
                if message.content:
                    self._messages.append({"role": "assistant", "content": message.content})
                break

            # Add assistant message to history
            msg_dict: Dict[str, Any] = {"role": message.role, "content": message.content}
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
            called_get_journal_summary = await self._execute_tool_calls(message.tool_calls)

            # Inject answer prompt if GetJournalSummary was called
            if called_get_journal_summary:
                self._trace("GetJournalSummary called - injecting answer prompt", COLOR_CYAN)
                self._messages.append({
                    "role": "user",
                    "content": self._get_journal_summary_answer_prompt()
                })

        # Run synthesis
        return await self._run_synthesis(query)

    async def _execute_tool_calls(self, tool_calls: List) -> bool:
        """
        Execute a batch of tool calls.

        Args:
            tool_calls: List of tool calls from LLM response

        Returns:
            True if GetJournalSummary was called
        """
        called_get_journal_summary = False
        self._trace(f"Processing {len(tool_calls)} tool call(s)", COLOR_YELLOW)

        for tool_call in tool_calls:
            if not tool_call.function or not tool_call.function.name:
                self._messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": "invalid_tool",
                    "content": "Error: Invalid tool call."
                })
                continue

            func_name = tool_call.function.name

            # Validate tool name exists
            known_tools = getattr(self, '_known_tool_names', set())
            if known_tools and func_name not in known_tools:
                tool_result = (
                    f"Error: Tool '{func_name}' does not exist. "
                    f"Available tools: {', '.join(sorted(known_tools))}"
                )
                self._trace(f"Unknown tool called: {func_name}", COLOR_RED)
                self._messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": func_name,
                    "content": tool_result
                })
                continue

            if func_name == "GetJournalSummary":
                called_get_journal_summary = True

            # Parse arguments
            try:
                args_str = tool_call.function.arguments
                func_args = json.loads(args_str) if args_str else {}
            except json.JSONDecodeError:
                func_args = {}

            args_pretty = json.dumps(func_args, indent=2, ensure_ascii=False)
            self._trace(f"Tool Call: {func_name}\n   Params: {args_pretty}", COLOR_YELLOW)

            # Check for loops
            loop_detected, loop_reason = self._detect_loops(func_name, func_args)

            if loop_detected:
                tool_result = await self._handle_loop_detected(
                    func_name, loop_reason
                )
            else:
                tool_result = await self._execute_single_tool(func_name, func_args)

            self._messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": func_name,
                "content": tool_result
            })

        return called_get_journal_summary

    async def _execute_single_tool(self, func_name: str, func_args: Dict) -> str:
        """
        Execute a single tool call with tracking.

        Args:
            func_name: Tool name
            func_args: Tool arguments

        Returns:
            Tool result string
        """
        tool_start_time = time.time()

        try:
            tool_result = await self.mcp.call_tool(func_name, func_args)
            tool_duration = time.time() - tool_start_time

            # Track success
            self.tool_call_counts[func_name] = self.tool_call_counts.get(func_name, 0) + 1
            self.tool_call_durations.append({
                "tool_name": func_name,
                "duration_seconds": round(tool_duration, 3),
                "success": True,
                "timestamp": datetime.now().isoformat()
            })

            # Log result
            log_result = tool_result
            if len(log_result) > 500:
                log_result = log_result[:500] + "... [truncated]"
            self._trace(f"Result ({func_name}) [{tool_duration:.3f}s]: {log_result}", COLOR_CYAN)

            return tool_result

        except Exception as tool_error:
            tool_duration = time.time() - tool_start_time

            # Track failure
            self.tool_call_counts[func_name] = self.tool_call_counts.get(func_name, 0) + 1
            self.tool_call_durations.append({
                "tool_name": func_name,
                "duration_seconds": round(tool_duration, 3),
                "success": False,
                "error": str(tool_error),
                "timestamp": datetime.now().isoformat()
            })

            self._trace(f"Tool {func_name} failed: {tool_error}", COLOR_RED)
            return f"Error executing {func_name}: {tool_error}. Try a different approach."

    async def _handle_loop_detected(self, func_name: str, loop_reason: str) -> str:
        """
        Handle a detected loop by injecting intervention.

        Args:
            func_name: Name of the looping tool
            loop_reason: Reason for loop detection

        Returns:
            Intervention message string
        """
        self._trace(f"LOOP DETECTED: {loop_reason}", COLOR_RED)

        # Get journal state
        try:
            journal_state = await self.mcp.call_tool("GetJournalSummary", {})
        except Exception:
            journal_state = "(Journal unavailable)"

        # Build intervention message
        tool_specific_guidance = self._get_tool_specific_loop_guidance(func_name)

        intervention = self._get_loop_intervention_template().format(
            loop_reason=loop_reason,
            func_name=func_name,
            tool_specific_guidance=tool_specific_guidance,
            journal_state=journal_state
        )

        # Clear loop history
        self.tool_call_history = []
        self.tool_sequence = []

        return intervention

    async def _inject_journal_refresh(self, iteration_count: int) -> None:
        """
        Inject a journal refresh message.

        Args:
            iteration_count: Current iteration number
        """
        self._trace("Injecting journal refresh", COLOR_CYAN)

        try:
            journal_refresh = await self.mcp.call_tool("GetJournalSummary", {})

            # Check for progress
            if (self.last_journal_state is not None and
                journal_refresh == self.last_journal_state):
                self._trace("WARNING: No progress in last 5 iterations!", COLOR_YELLOW)
                template = self._get_no_progress_template()
            else:
                template = self._get_journal_refresh_template()

            self._messages.append({
                "role": "user",
                "content": template.format(
                    iteration_count=iteration_count,
                    journal_refresh=journal_refresh
                )
            })

            self.last_journal_state = journal_refresh

        except Exception as e:
            self._trace(f"Failed to inject journal refresh: {e}", COLOR_YELLOW)

    def _manage_context_window(self) -> None:
        """
        Manage context window by summarizing old tool results when approaching limits.

        Uses a rough heuristic of len(text) / 3.5 to estimate token count.
        Preserves the system message and the most recent 6 message pairs.
        """
        context_limit = getattr(self, '_context_limit', 100000)

        # Estimate total tokens
        total_chars = sum(
            len(msg.get("content", "") or "") for msg in self._messages
        )
        estimated_tokens = total_chars / 3.5

        if estimated_tokens < context_limit * 0.8:
            return  # Under threshold, no action needed

        self._trace(
            f"Context management: ~{int(estimated_tokens)} tokens "
            f"({int(estimated_tokens / context_limit * 100)}% of {context_limit} limit)",
            COLOR_YELLOW
        )

        # Determine how aggressively to trim
        truncate_len = 200 if estimated_tokens < context_limit * 0.9 else 100

        # Never touch system message (index 0) or last 12 messages (~6 pairs)
        protected_tail = 12
        if len(self._messages) <= protected_tail + 1:
            return  # Not enough messages to trim

        trimmed_count = 0
        for i in range(1, len(self._messages) - protected_tail):
            msg = self._messages[i]
            content = msg.get("content", "") or ""

            # Only trim tool results (they tend to be the largest)
            if msg.get("role") == "tool" and len(content) > truncate_len + 50:
                self._messages[i] = {
                    **msg,
                    "content": content[:truncate_len] + "... [summarized]"
                }
                trimmed_count += 1

        if trimmed_count > 0:
            self._trace(f"Trimmed {trimmed_count} old tool results to {truncate_len} chars", COLOR_YELLOW)

    async def _run_synthesis(self, query: str) -> str:
        """
        Run the deterministic synthesis step.

        Args:
            query: Original query

        Returns:
            Final answer string
        """
        self._trace("Starting synthesis step", COLOR_CYAN)

        # Get journal summary
        journal_summary = await self.mcp.call_tool("GetJournalSummary", {})
        self._trace(f"Journal fetched ({len(journal_summary)} chars)", COLOR_GREEN)

        # Build synthesis prompt
        synthesis_prompt = self._get_synthesis_prompt_template().format(
            journal_summary=journal_summary,
            query=query
        )

        self._messages.append({"role": "user", "content": synthesis_prompt})

        # Make synthesis call
        self._trace("Making final synthesis LLM call...", COLOR_YELLOW)
        final_answer = self._llm_call_synthesis()

        if final_answer and final_answer.strip():
            # Verification pass: if synthesis indicates failure but journal has data, re-prompt
            failure_phrases = ["cannot answer", "no data", "not found", "insufficient",
                               "unable to determine", "could not find", "no information"]
            answer_lower = final_answer.lower()
            if any(phrase in answer_lower for phrase in failure_phrases):
                # Check if journal actually has useful data
                has_data = ("found_values" in journal_summary.lower() or
                           "verified_facts" in journal_summary.lower() or
                           "orkgr:" in journal_summary.lower() or
                           "R" in journal_summary)
                journal_seems_empty = (
                    "none" in journal_summary.lower()
                    and len(journal_summary) < 200
                )

                if has_data and not journal_seems_empty:
                    self._trace("Synthesis indicated failure but journal has data - re-prompting", COLOR_YELLOW)
                    self._messages.append({"role": "assistant", "content": final_answer})
                    self._messages.append({
                        "role": "user",
                        "content": (
                            "Your answer indicates you could not find data, but your journal "
                            "contains discovered values and resource IDs. Please re-read the "
                            "journal summary above carefully and provide an answer based on "
                            "the data you DID find. Use the specific values from the journal."
                        )
                    })
                    final_answer = self._llm_call_synthesis()
                    if final_answer and final_answer.strip():
                        self._trace(f"Re-synthesis complete ({len(final_answer)} chars)", COLOR_GREEN)

            self._trace(f"Synthesis complete ({len(final_answer)} chars)", COLOR_GREEN)
            self._messages.append({"role": "assistant", "content": final_answer})
            return final_answer.strip()
        else:
            self._trace("WARNING: Synthesis returned empty", COLOR_RED)
            fallback = "Unable to generate answer. Investigation completed but synthesis failed."
            self._messages.append({"role": "assistant", "content": fallback})
            return fallback

    async def _finalize_question(self) -> None:
        """Finalize question processing with cleanup and logging."""
        if not self.mcp:
            return

        try:
            final_state = await self.mcp.call_tool(
                "ManageJournal", {"action": "read", "content": "Final"}
            )
            self._trace(f"FINAL SCRATCHPAD STATE:\n{final_state}", COLOR_CYAN)
        except Exception:
            pass

        # Log tool summary
        tool_summary = self.get_tool_call_summary()
        if tool_summary["total_calls"] > 0:
            self._trace(
                f"TOOL SUMMARY: {tool_summary['total_calls']} calls, "
                f"total {tool_summary['total_duration_seconds']}s",
                COLOR_CYAN
            )

        self._trace("Question complete (MCP preserved)", COLOR_GREEN)

    # =========================================================================
    # LLM CALLS
    # =========================================================================

    def _llm_call(self, tools: Optional[List[Dict[str, Any]]] = None):
        """Execute LLM call with tools."""
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

    def _llm_call_text_only(self) -> str:
        """Execute LLM call without tools."""
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
                self._track_token_usage(response.usage)

            return response.choices[0].message.content
        except Exception as e:
            self._trace(f"LLM text-only call failed: {e}", COLOR_RED)
            raise

    def _llm_call_synthesis(self) -> str:
        """Execute synthesis LLM call."""
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
                self._track_token_usage(response.usage)

            return response.choices[0].message.content
        except Exception as e:
            self._trace(f"Synthesis LLM call failed: {e}", COLOR_RED)
            raise

    # =========================================================================
    # MCP MANAGEMENT
    # =========================================================================

    async def _init_mcp(self) -> None:
        """Initialize the MCP connection."""
        if self.mcp:
            return

        try:
            server_path = self.get_mcp_server_path()
            self._trace(f"Starting MCP server: {server_path}")
            self.mcp = MCPClient(server_path, self.name)
            await self.mcp.start()
            self._trace("MCP connected")
        except Exception as e:
            self._trace(f"{COLOR_RED}MCP error: {e}{COLOR_END}", COLOR_RED)
            self.mcp = None
            raise

    async def reset(self, keep_mcp_open: bool = False) -> None:
        """
        Reset the agent state for a new question.

        Args:
            keep_mcp_open: If True, keep MCP connection open
        """
        self._messages = [{"role": "system", "content": self._get_system_prompt()}]
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

    async def soft_reset(self) -> None:
        """Reset state but keep MCP connection open."""
        await self.reset(keep_mcp_open=True)

        if self.mcp:
            try:
                await self.mcp.call_tool("ManageJournal", {"action": "clear", "content": ""})
                self._trace("Journal cleared for next question", COLOR_CYAN)
            except Exception as e:
                self._trace(f"Failed to clear journal: {e}", COLOR_YELLOW)

    async def close(self) -> None:
        """Close the MCP server connection."""
        if self.mcp:
            self._trace("Closing MCP server connection...", COLOR_CYAN)
            await self.mcp.close()
            self.mcp = None
            self._trace("MCP server connection closed", COLOR_GREEN)

    # =========================================================================
    # TRACKING & UTILITIES
    # =========================================================================

    def _trace(self, msg: str, color: str = COLOR_GREEN) -> None:
        """Log a trace message."""
        trace(self.name, msg, color)

    def _track_token_usage(self, usage) -> None:
        """Track token usage from an API response."""
        self.token_usage["prompt_tokens"] += usage.prompt_tokens
        self.token_usage["completion_tokens"] += usage.completion_tokens
        self.token_usage["total_tokens"] += usage.total_tokens

    def get_tool_call_summary(self) -> Dict[str, Any]:
        """Get summary of tool call durations."""
        if not self.tool_call_durations:
            return {"total_calls": 0, "total_duration_seconds": 0, "tool_breakdown": {}, "calls": []}

        total_duration = sum(call["duration_seconds"] for call in self.tool_call_durations)

        tool_breakdown: Dict[str, Dict[str, Any]] = {}
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
