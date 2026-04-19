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
    get_auto_inject_journal,
    get_synthesis_client,
    get_synthesis_enabled,
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
    elif hasattr(sys.stdout, 'buffer'):
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

    def __init__(self, name: str = "kbqa_agent", session_id: str = "default", use_fewshot: bool = True):
        """
        Initialize the base KBQA agent.

        Args:
            name: Agent name for tracing
            session_id: Session identifier
            use_fewshot: Whether to inject few-shot examples during classification
        """
        self.use_fewshot = use_fewshot
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

    def _get_synthesis_system_prompt(self) -> str:
        """Return the system message used during the final synthesis call.

        Switches between benchmark (terse) and conversational (verbose) based
        on the `synthesis.synthesis_mode` setting in config.toml.
        """
        try:
            from ama_kbqa.config import get_synthesis_mode
            mode = get_synthesis_mode()
        except Exception:
            mode = "benchmark"

        if mode == "conversational":
            return (
                "You are a helpful assistant answering a user's question using "
                "the provided journal data. Write a clear, friendly, human-readable "
                "response. Lead with the direct answer, then add brief supporting "
                "context from the data. Do not invent facts beyond the journal."
            )
        return (
            "You are a precise question-answering system. Answer based strictly "
            "on the provided journal data. Give only the answer value."
        )

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

    def _get_allowed_tools_for_qtype(self, qtype: str) -> Optional[set]:
        """
        Return set of allowed tool names for this question type.
        Return None to allow all tools (default).
        Override in subclass for qtype-specific tool filtering.
        """
        return None

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

    def _classify_and_extract(self, question: str) -> Dict[str, Any]:
        """
        Combined classification + entity extraction in a single LLM call.
        Saves ~2-4k tokens by avoiding a second round-trip.

        Args:
            question: The question to classify and analyze

        Returns:
            Dict with 'question_type', 'entities', and 'relations' keys
        """
        prompt = self._get_classification_prompt(question)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": prompt}],
                temperature=get_chat_temperature(),
                response_format={"type": "json_object"},
                max_tokens=300,
                timeout=30.0
            )

            if response.usage:
                self._track_token_usage(response.usage)

            json_content = response.choices[0].message.content
            result = json.loads(json_content)
            return {
                "question_type": result.get("question_type", "Query"),
                "entities": result.get("entities", []),
                "relations": result.get("relations", [])
            }

        except Exception as e:
            self._trace(f"Classification+extraction failed: {e}", COLOR_YELLOW)
            return {"question_type": "Query", "entities": [], "relations": []}

    def _classify_question(self, question: str) -> Dict[str, Any]:
        """
        Classify question and extract entities in one LLM call.
        Subclasses can override to enrich the result (e.g. add fewshot examples).
        """
        result = self._classify_and_extract(question)
        return {
            "question_type": result["question_type"],
            "entities": result.get("entities", []),
            "relations": result.get("relations", []),
            "fewshot_examples": "",
        }

    def _extract_entities(self, question: str) -> Dict[str, List[str]]:
        """Legacy: delegates to combined call."""
        result = self._classify_and_extract(question)
        return {"entities": result["entities"], "relations": result["relations"]}

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

        # Detection 5: RunORKGSPARQL cap (configurable per agent)
        sparql_cap = getattr(self, '_sparql_cap', 10)
        if func_name == "RunORKGSPARQL":
            sq_count = self.tool_call_counts.get("RunORKGSPARQL", 0) + 1
            if sq_count > sparql_cap:
                return True, (
                    f"RunORKGSPARQL called {sq_count} times (cap: {sparql_cap}). "
                    f"Use GetComparisonContributions or GetResourceSummary instead."
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

            # Read per-agent config for tool caps
            config = self.get_config()
            self._find_resource_cap = config.domain_settings.get("find_resource_cap", 8)
            self._sparql_cap = config.domain_settings.get("sparql_cap", 10)
            self._context_limit = config.domain_settings.get("context_limit", 100000)

            # Add query to messages
            self._messages.append({"role": "user", "content": query})

            # Run pre-agent hooks (combined classification + extraction = 1 LLM call)
            self._trace("Starting pre-agent classification hook", COLOR_CYAN)

            # Call _classify_question (overrideable by subclasses for fewshot loading etc.)
            qtype_data = self._classify_question(query)
            qtype = qtype_data.get("question_type", "Query")
            fewshot_examples = qtype_data.get("fewshot_examples", "")
            entities = qtype_data.get("entities", [])
            relations = qtype_data.get("relations", [])

            self._trace(
                f"Classification: {qtype} | "
                f"Entities: {len(entities)}, Relations: {len(relations)}",
                COLOR_GREEN
            )

            analysis_context = self._build_analysis_context(
                qtype, entities, relations, fewshot_examples
            )
            self._messages.append({"role": "user", "content": analysis_context})

            self._trace(f"Pre-agent hook complete - Type: {qtype}", COLOR_GREEN)

            # === FAST PATH: Skip agent loop for simple 1-hop questions ===
            fast_path_types = {"QueryAttr", "QueryRelation", "QueryName"}
            if (qtype in fast_path_types
                    and len(entities) == 1
                    and len(relations) <= 1
                    and config.domain_settings.get("enable_fast_path", True)):
                self._trace(f"FAST PATH: Simple {qtype} with 1 entity", COLOR_GREEN)
                fast_answer = await self._try_fast_path(query, qtype, entities, relations)
                if fast_answer is not None:
                    self._trace(f"Fast path succeeded ({len(fast_answer)} chars)", COLOR_GREEN)
                    return fast_answer
                self._trace("Fast path failed - falling back to full loop", COLOR_YELLOW)

            # Filter tools by question type (saves ~2-3k tokens per iteration)
            allowed_tools = self._get_allowed_tools_for_qtype(qtype)
            if allowed_tools is not None:
                filtered_tools = [t for t in openai_tools if t["function"]["name"] in allowed_tools]
                self._trace(
                    f"Tool filtering: {len(openai_tools)} → {len(filtered_tools)} tools for {qtype}",
                    COLOR_GREEN
                )
                openai_tools = filtered_tools

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

    async def _try_fast_path(
        self,
        query: str,
        qtype: str,
        entities: List[str],
        relations: List[str],
    ) -> Optional[str]:
        """
        Attempt to answer simple 1-hop questions without the full agent loop.

        Executes: FindNode → GetNodeSummary → Synthesis
        Returns None if the fast path cannot answer (fallback to full loop).
        """
        entity_name = entities[0]

        try:
            # Step 1: Find the entity
            find_result = await self.mcp.call_tool("FindNode", {"semantic_node_name": entity_name})

            # Parse the result to get node_id
            import re as _re
            id_match = _re.search(r'"original_id":\s*"([^"]+)"', find_result)
            if not id_match:
                return None
            node_id = id_match.group(1)

            # Step 2: Get full node summary
            summary_result = await self.mcp.call_tool("GetNodeSummary", {"node_id": node_id})

            # Track tool calls
            self.tool_call_counts["FindNode"] = self.tool_call_counts.get("FindNode", 0) + 1
            self.tool_call_counts["GetNodeSummary"] = self.tool_call_counts.get("GetNodeSummary", 0) + 1

            # Step 3: If relation-specific, also get relation details
            relation_result = ""
            if relations and qtype == "QueryRelation":
                try:
                    relation_result = await self.mcp.call_tool(
                        "GetRelationDetails",
                        {"base_node_id": node_id, "relation_name": relations[0]}
                    )
                    self.tool_call_counts["GetRelationDetails"] = self.tool_call_counts.get("GetRelationDetails", 0) + 1
                except Exception:
                    pass

            # Step 4: Synthesize answer from gathered data
            data_context = f"Entity: {entity_name} (ID: {node_id})\n"
            data_context += f"Node Summary:\n{summary_result}\n"
            if relation_result:
                data_context += f"Relation Details:\n{relation_result}\n"

            synthesis_prompt = self._get_synthesis_prompt_template().format(
                journal_summary=data_context,
                query=query
            )

            synthesis_messages = [
                {"role": "system", "content": self._get_synthesis_system_prompt()},
                {"role": "user", "content": synthesis_prompt}
            ]

            answer = self._llm_call_synthesis(messages_override=synthesis_messages)

            if answer and answer.strip():
                # Check for failure indicators
                failure_phrases = ["cannot answer", "no data", "not found", "insufficient",
                                   "unable to determine", "could not find"]
                if any(phrase in answer.lower() for phrase in failure_phrases):
                    return None  # Fallback to full loop
                return answer.strip()

            return None

        except Exception as e:
            self._trace(f"Fast path error: {e}", COLOR_YELLOW)
            return None

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
        final_agent_content: Optional[str] = None

        while True:
            iteration_count += 1
            self._trace(f"Starting iteration {iteration_count}/{max_iterations}", COLOR_CYAN)

            # Safety check
            if iteration_count > max_iterations:
                self._trace(f"WARNING: Reached max iterations ({max_iterations})", COLOR_RED)
                return "Error: Agent reached maximum iteration limit."

            # Manage context window before LLM call
            self._manage_context_window()

            # Periodic journal refresh (can be disabled via agent.auto_inject_journal)
            if (
                get_auto_inject_journal()
                and iteration_count % refresh_interval == 0
                and iteration_count > 0
            ):
                await self._inject_journal_refresh(iteration_count)

            # Call LLM - use tool_choice="required" for early iterations
            # to force the model to call a tool instead of "thinking"
            tc = "required" if iteration_count <= 3 else "auto"
            self._trace(f"Calling LLM with {len(self._messages)} messages (tool_choice={tc})...", COLOR_YELLOW)
            response = self._llm_call(tools=tools, tool_choice=tc)
            message = response.choices[0].message
            finish_reason = response.choices[0].finish_reason

            self._trace(f"LLM response (finish_reason: {finish_reason})", COLOR_CYAN)

            if response.usage:
                self._track_token_usage(response.usage)

            if message.content:
                self._trace(f"Thought: {message.content}", COLOR_BLUE)
                final_agent_content = message.content

            # No tool calls - break for synthesis
            if not message.tool_calls:
                self._trace("No more tool calls - breaking to synthesis", COLOR_GREEN)
                if message.content:
                    self._messages.append({"role": "assistant", "content": message.content})
                final_agent_content = message.content
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
            # (can be disabled via agent.auto_inject_journal)
            if called_get_journal_summary and get_auto_inject_journal():
                self._trace("GetJournalSummary called - injecting answer prompt", COLOR_CYAN)
                self._messages.append({
                    "role": "user",
                    "content": self._get_journal_summary_answer_prompt()
                })

            # Early exit: nudge agent to wrap up after iteration 15
            if iteration_count >= 15 and iteration_count % 5 == 0:
                self._trace(f"Iteration {iteration_count} - injecting wrap-up nudge", COLOR_YELLOW)
                self._messages.append({
                    "role": "user",
                    "content": (
                        f"You are on iteration {iteration_count}. If you have found relevant data, "
                        "call GetJournalSummary and provide your answer now. "
                        "Only continue if you have a concrete next step that will yield new information."
                    )
                })

        # Synthesis can be bypassed via config (synthesis.synthesis_enabled = false)
        # to use the agent's own final message as the answer. This saves a
        # second LLM call at the cost of losing the deterministic answer shaping.
        if not get_synthesis_enabled():
            if final_agent_content and final_agent_content.strip():
                self._trace(
                    "Synthesis bypassed - using agent's final message as answer",
                    COLOR_CYAN,
                )
                return final_agent_content.strip()
            self._trace(
                "Synthesis bypassed but agent produced no final content - falling back to synthesis",
                COLOR_YELLOW,
            )

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

    # Marker prefix used to identify journal refresh messages for replace-not-append
    _JOURNAL_REFRESH_MARKER = "<!-- JOURNAL_REFRESH -->"

    async def _inject_journal_refresh(self, iteration_count: int) -> None:
        """
        Inject a journal refresh message, replacing any previous refresh.

        Uses replace-not-append strategy: removes all prior journal refresh
        messages from self._messages before adding the new one, so only one
        (current) refresh exists at any time.

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

            # Remove all previous journal refresh messages (replace-not-append)
            self._messages = [
                msg for msg in self._messages
                if not (msg.get("role") == "user"
                        and isinstance(msg.get("content"), str)
                        and msg["content"].startswith(self._JOURNAL_REFRESH_MARKER))
            ]

            self._messages.append({
                "role": "user",
                "content": self._JOURNAL_REFRESH_MARKER + template.format(
                    iteration_count=iteration_count,
                    journal_refresh=journal_refresh
                )
            })

            self.last_journal_state = journal_refresh

        except Exception as e:
            self._trace(f"Failed to inject journal refresh: {e}", COLOR_YELLOW)

    def _manage_context_window(self) -> None:
        """
        Manage context window by truncating old tool results.

        Two-tier approach:
        - At 50% capacity: truncate old tool results to 150 chars
        - At 75% capacity: truncate aggressively to 80 chars and drop old user injection messages
        """
        context_limit = getattr(self, '_context_limit', 100000)

        # Estimate total tokens (rough heuristic)
        total_chars = sum(
            len(msg.get("content", "") or "") for msg in self._messages
        )
        estimated_tokens = total_chars / 3.5

        if estimated_tokens < context_limit * 0.5:
            return  # Under threshold

        self._trace(
            f"Context management: ~{int(estimated_tokens)} tokens "
            f"({int(estimated_tokens / context_limit * 100)}% of {context_limit} limit)",
            COLOR_YELLOW
        )

        # Tier 1: Moderate trimming at 50%
        truncate_len = 150
        # Tier 2: Aggressive at 75%
        if estimated_tokens >= context_limit * 0.75:
            truncate_len = 80

        # Never touch system message (index 0) or last 8 messages (~4 pairs)
        protected_tail = 8
        if len(self._messages) <= protected_tail + 1:
            return

        trimmed_count = 0
        for i in range(1, len(self._messages) - protected_tail):
            msg = self._messages[i]
            content = msg.get("content", "") or ""

            # Trim tool results (largest messages)
            if msg.get("role") == "tool" and len(content) > truncate_len + 50:
                self._messages[i] = {
                    **msg,
                    "content": content[:truncate_len] + "...[trimmed]"
                }
                trimmed_count += 1

            # At tier 2, also trim verbose user injection messages (journal refreshes, etc.)
            if estimated_tokens >= context_limit * 0.75:
                if msg.get("role") == "user" and len(content) > 500 and i > 2:
                    self._messages[i] = {
                        **msg,
                        "content": content[:200] + "...[trimmed]"
                    }
                    trimmed_count += 1

        if trimmed_count > 0:
            self._trace(f"Trimmed {trimmed_count} messages to {truncate_len} chars", COLOR_YELLOW)

    async def _run_synthesis(self, query: str) -> str:
        """
        Run the deterministic synthesis step.

        Uses a MINIMAL message set (system + journal + query) instead of the
        full conversation history. This saves 30-80k tokens per question.

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

        # Use MINIMAL messages for synthesis instead of full history
        # This is the single biggest token saving in the pipeline
        synthesis_messages = [
            {"role": "system", "content": self._get_synthesis_system_prompt()},
            {"role": "user", "content": synthesis_prompt}
        ]

        # Make synthesis call with minimal context
        self._trace("Making final synthesis LLM call (minimal context)...", COLOR_YELLOW)
        final_answer = self._llm_call_synthesis(messages_override=synthesis_messages)

        if final_answer and final_answer.strip():
            # Verification pass: if synthesis indicates failure but journal has data, re-prompt
            failure_phrases = ["cannot answer", "no data", "not found", "insufficient",
                               "unable to determine", "could not find", "no information"]
            answer_lower = final_answer.lower()
            if any(phrase in answer_lower for phrase in failure_phrases):
                # Check if journal actually has useful data. Key off markers
                # the renderer emits, not arbitrary substrings.
                summary_lower = journal_summary.lower()
                has_discovered_values = (
                    "discovered values" in summary_lower
                    and "no values discovered yet" not in summary_lower
                )
                has_verified_facts = "verified facts" in summary_lower
                has_partial = "partial answer:" in summary_lower
                has_orkg = "orkgr:" in summary_lower
                has_data = (
                    has_discovered_values
                    or has_verified_facts
                    or has_partial
                    or has_orkg
                )
                journal_seems_empty = (
                    "no values discovered yet" in summary_lower
                    and not has_verified_facts
                    and not has_partial
                )

                if has_data and not journal_seems_empty:
                    self._trace("Synthesis indicated failure but journal has data - re-prompting", COLOR_YELLOW)
                    synthesis_messages.append({"role": "assistant", "content": final_answer})
                    synthesis_messages.append({
                        "role": "user",
                        "content": (
                            "Your answer indicates you could not find data, but the journal "
                            "contains discovered values. Re-read the journal and answer using "
                            "the specific values found."
                        )
                    })
                    final_answer = self._llm_call_synthesis(messages_override=synthesis_messages)
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

    def _llm_call(
        self,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str] = None,
    ):
        """Execute LLM call with tools and optional tool_choice."""
        call_params = {
            "model": self.model,
            "messages": self._messages,
            "timeout": self.request_timeout,
            "temperature": get_chat_temperature(),
            "max_tokens": get_chat_max_tokens(),
            "tools": tools
        }

        if tool_choice and tools:
            call_params["tool_choice"] = tool_choice

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

    def _llm_call_synthesis(self, messages_override: Optional[List[Dict[str, Any]]] = None) -> str:
        """Execute synthesis LLM call with optional minimal message set."""
        call_params = {
            "model": self.synthesis_model,
            "messages": messages_override if messages_override is not None else self._messages,
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
