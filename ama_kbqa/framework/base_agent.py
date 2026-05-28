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
import re
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
    get_zero_tool_call_retry,
    get_zero_tool_call_retry_max,
    get_synthesis_client,
    get_synthesis_enabled,
    get_synthesis_model_name,
    get_synthesis_temperature,
    get_synthesis_max_tokens,
    get_synthesis_provider_preferences,
)
from ama_kbqa.framework.config import KnowledgeGraphConfig
from ama_kbqa.framework.mcp_client import MCPClient, trace
from ama_kbqa.framework.trace import (
    JOURNAL_MUTATING_TOOLS,
    TraceRecorder,
    _current_span_id,
)

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

    def __init__(
        self,
        name: str = "kbqa_agent",
        session_id: str = "default",
        use_fewshot: bool = True,
        parent_recorder: Optional[TraceRecorder] = None,
        parent_span_id: Optional[str] = None,
    ):
        """
        Initialize the base KBQA agent.

        Args:
            name: Agent name for tracing
            session_id: Session identifier
            use_fewshot: Whether to inject few-shot examples during classification
            parent_recorder: If supplied (e.g. by an Orchestrator delegating to
                this sub-agent), append all events to that recorder instead of
                creating an independent one. Produces a single nested trace.
            parent_span_id: Span id under which this agent's root span should
                nest. Only honoured when `parent_recorder` is also supplied.
        """
        self.use_fewshot = use_fewshot
        self.name = name
        self.session_id = session_id
        self.mcp: Optional[MCPClient] = None

        # Trace recorder — owns this run's events. Sub-agents share their
        # parent's recorder when `parent_recorder` is supplied.
        self.recorder: TraceRecorder = parent_recorder or TraceRecorder()
        self._parent_span_id_override: Optional[str] = parent_span_id

        # Live graph snapshots — populated as the agent investigates.
        # Each entry: {"ts": float, "iteration": int, "trigger": str, "state": dict}
        self.journal_snapshots: List[Dict[str, Any]] = []
        self._last_journal_summary_hash: Optional[str] = None

        # Initialize LLM clients
        try:
            self.client = get_chat_client()
            self.model = get_chat_model_name()
        except (FileNotFoundError, ValueError, KeyError) as e:
            raise RuntimeError(f"Failed to initialize LLM client from config.toml: {e}")

        # The synthesis client is initialised lazily (see the synthesis_client /
        # synthesis_model properties below). Synthesis can be disabled entirely
        # via config (synthesis.synthesis_enabled = false), and the synthesis
        # provider can differ from the chat provider. Initialising it eagerly
        # here meant an agent could not even be constructed when synthesis was
        # off but the synthesis provider's API key happened to be unset — e.g.
        # chat_provider = "kit" works while synthesis_provider = "openrouter"
        # has no OPENROUTER_API_KEY. Deferring construction until synthesis
        # actually runs keeps the agent loadable in that (valid) configuration.
        self._synthesis_client = None
        self._synthesis_model = None

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

        # Some models on some endpoints (e.g. minimax-m2.7 on KIT) don't emit
        # OpenAI-style structured tool_calls. Detect and append a plain-text
        # tool-call format instruction so the client-side parser can pick them up.
        from ama_kbqa.framework.text_tool_calls import (
            needs_text_tool_calls,
            TEXT_TOOL_CALL_INSTRUCTION,
        )
        self._text_tool_call_mode = needs_text_tool_calls(self.model)
        if self._text_tool_call_mode:
            self._messages.append({"role": "system", "content": TEXT_TOOL_CALL_INSTRUCTION})

    def _ensure_synthesis_client(self) -> None:
        """Lazily build the synthesis LLM client on first use.

        Kept out of __init__ so an agent stays constructible when synthesis is
        disabled or its provider's API key is absent; the cost (and the
        config/key requirement) is only paid if synthesis actually runs.
        """
        if self._synthesis_client is not None:
            return
        try:
            self._synthesis_client = get_synthesis_client()
            self._synthesis_model = get_synthesis_model_name()
        except (FileNotFoundError, ValueError, KeyError) as e:
            raise RuntimeError(f"Failed to initialize synthesis LLM client from config.toml: {e}")

    @property
    def synthesis_client(self):
        self._ensure_synthesis_client()
        return self._synthesis_client

    @property
    def synthesis_model(self):
        self._ensure_synthesis_client()
        return self._synthesis_model

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

    @staticmethod
    def _strip_think_blocks(text: str) -> str:
        """Remove model-internal reasoning blocks from user-facing answers."""
        if not text:
            return ""
        cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)
        cleaned = re.sub(r"</?think>", "", cleaned, flags=re.IGNORECASE)
        return cleaned.strip()

    @staticmethod
    def _normalize_verify_answer(answer: str) -> Optional[str]:
        """Normalize verification answers to the benchmark yes/no contract."""
        cleaned = BaseKBQAAgent._strip_think_blocks(answer)
        compact = cleaned.strip().lower().strip(" .!?:;\"'")
        if compact in {"yes", "true"}:
            return "yes"
        if compact in {"no", "false"}:
            return "no"

        explicit = re.search(
            r"\b(?:answer|final answer|result|verdict)\s*[:\-]\s*(yes|no|true|false)\b",
            compact,
        )
        if explicit:
            return "yes" if explicit.group(1) in {"yes", "true"} else "no"

        negative_phrases = (
            "does not",
            "do not",
            "did not",
            "is not",
            "are not",
            "was not",
            "were not",
            "not true",
            "not correct",
            "not verified",
            "not satisfied",
            "doesn't",
            "isn't",
            "wasn't",
            "false",
        )
        if any(phrase in compact for phrase in negative_phrases):
            return "no"

        affirmative_phrases = (
            "yes,",
            "yes.",
            "is true",
            "is correct",
            "is verified",
            "verified",
            "confirmed",
            "satisfies",
            "meets the condition",
            "greater than",
            "less than",
            "at least",
            "at most",
            "over ",
            "under ",
            "before ",
            "after ",
        )
        if any(phrase in compact for phrase in affirmative_phrases):
            return "yes"

        return None

    def _finalize_answer_text(self, answer: str, qtype: str = "") -> str:
        """Apply final answer-shape cleanup without changing non-Verify semantics."""
        cleaned = self._strip_think_blocks(answer)
        if qtype == "Verify":
            normalized = self._normalize_verify_answer(cleaned)
            if normalized:
                return normalized
        return cleaned

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

    @staticmethod
    def _extract_json_object(content: str) -> Optional[Dict[str, Any]]:
        """
        Pull a JSON object out of an LLM response that may include surrounding
        prose, `<think>...</think>` blocks (minimax-m2.7 emits these even with
        `response_format=json_object`), or markdown code fences.

        Returns the parsed dict, or None if no valid JSON object is found.
        """
        if not content:
            return None
        # Strip <think>...</think> blocks (minimax/reasoning-style prefixes).
        # We use re.DOTALL so embedded newlines don't break the match.
        cleaned = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
        # Strip markdown code fences.
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned.strip(), flags=re.MULTILINE)
        cleaned = re.sub(r"\s*```$", "", cleaned, flags=re.MULTILINE)
        cleaned = cleaned.strip()
        # Try direct parse first (cheap path for well-behaved models).
        try:
            obj = json.loads(cleaned)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
        # Fallback: find the first balanced {...} block via brace counting.
        start = cleaned.find("{")
        if start == -1:
            return None
        depth = 0
        for i, ch in enumerate(cleaned[start:], start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = cleaned[start:i + 1]
                    try:
                        obj = json.loads(candidate)
                        if isinstance(obj, dict):
                            return obj
                    except json.JSONDecodeError:
                        return None
        return None

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

        with self.recorder.span_sync(
            "classify",
            self.model,
            attributes={"model": self.model},
            payload={"prompt": prompt, "question": question},
        ) as _cls_span:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "system", "content": prompt}],
                    temperature=get_chat_temperature(),
                    response_format={"type": "json_object"},
                    # Bumped from 300 to 1500: minimax-m2.7 emits a
                    # `<think>...</think>` reasoning prefix before the JSON
                    # object even with response_format=json_object. With
                    # max_tokens=300 the response routinely truncates inside
                    # the thinking block, leaving no JSON to parse and
                    # silently defaulting question_type to "Query" for every
                    # question. 1500 tokens leaves headroom for the think
                    # block plus the actual JSON.
                    max_tokens=1500,
                    timeout=30.0,
                )

                if response.usage:
                    self._track_token_usage(response.usage)
                    _cls_span.update_attributes({
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                    })

                json_content = response.choices[0].message.content
                _cls_span.set_payload("response_content", json_content)
                # Use the brace-balanced extractor to handle <think>...</think>
                # prefixes and markdown fences. json.loads alone fails on those.
                result = self._extract_json_object(json_content)
                if result is None:
                    self._trace(
                        f"Classification: no parseable JSON in response "
                        f"({len(json_content) if json_content else 0} chars), "
                        f"defaulting to Query",
                        COLOR_YELLOW,
                    )
                    _cls_span.set_attribute("parse_failed", True)
                    return {"question_type": "Query", "entities": [], "relations": []}
                out = {
                    "question_type": result.get("question_type", "Query"),
                    "entities": result.get("entities", []),
                    "relations": result.get("relations", []),
                }
                _cls_span.update_attributes({
                    "question_type": out["question_type"],
                    "n_entities": len(out["entities"]),
                    "n_relations": len(out["relations"]),
                })
                return out

            except Exception as e:
                self._trace(f"Classification+extraction failed: {e}", COLOR_YELLOW)
                _cls_span.set_attribute("error", str(e))
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
        fewshot_examples: str = "",
        query: str = "",
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

        exact_constraints = self._extract_exact_attribute_constraints(query)
        if exact_constraints:
            formatted_constraints = "\n".join(
                f"  - {c['attribute_name']} = {c['value']}"
                for c in exact_constraints
            )
            context += (
                "\n\nEXACT ATTRIBUTE CONSTRAINTS DETECTED:\n"
                f"{formatted_constraints}\n"
                "Use exact reverse lookup (`FindByAttribute`) or explicit verification "
                "for these constraints before semantic entity search or final synthesis. "
                "If several entities share a name, prefer the one satisfying all exact constraints."
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

        # Root span: wraps the entire run. If a parent_span_id was supplied
        # (orchestrator delegation), set it as the current parent so this
        # span nests correctly under the parent's "delegate" span.
        _parent_token = None
        if self._parent_span_id_override is not None:
            _parent_token = _current_span_id.set(self._parent_span_id_override)

        try:
            async with self.recorder.span(
                "agent_run",
                self.name,
                attributes={
                    "agent": self.name,
                    "model": self.model,
                    "query": query[:500],
                },
                payload={"query": query},
            ) as _root_span:
                return await self._ask_impl(query, _root_span)
        finally:
            if _parent_token is not None:
                _current_span_id.reset(_parent_token)

    async def _ask_impl(self, query: str, _root_span) -> str:
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

            # For text-mode models: attach a plain-text tool catalog so the
            # model can produce <tool_call>...</tool_call> blocks without us
            # passing OpenAI-style `tools=...` on the API call (which makes
            # some endpoints expect native function-call output and silently
            # revert to prose when the model can't comply).
            if self._text_tool_call_mode:
                from ama_kbqa.framework.text_tool_calls import build_text_mode_tool_catalog
                catalog = build_text_mode_tool_catalog(openai_tools)
                self._messages.append({"role": "system", "content": catalog})
                self._trace("Injected text-mode tool catalog", COLOR_CYAN)

            # Read per-agent config for tool caps
            config = self.get_config()
            self._find_resource_cap = config.domain_settings.get("find_resource_cap", 8)
            self._sparql_cap = config.domain_settings.get("sparql_cap", 10)
            self._context_limit = config.domain_settings.get("context_limit", 100000)
            self._max_tool_calls = config.domain_settings.get("max_tool_calls", 0)

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
                qtype, entities, relations, fewshot_examples, query=query
            )
            self._messages.append({"role": "user", "content": analysis_context})

            self._trace(f"Pre-agent hook complete - Type: {qtype}", COLOR_GREEN)

            # === FAST PATH: Skip agent loop for simple 1-hop questions ===
            fast_path_types = {"QueryAttr", "QueryRelation", "QueryName"}
            if (qtype in fast_path_types
                    and len(entities) == 1
                    and len(relations) <= 1
                    and not self._should_skip_fast_path(query, qtype, entities, relations)
                    and config.domain_settings.get("enable_fast_path", True)):
                self._trace(f"FAST PATH: Simple {qtype} with 1 entity", COLOR_GREEN)
                async with self.recorder.span(
                    "fast_path",
                    qtype,
                    attributes={
                        "qtype": qtype,
                        "entity": entities[0] if entities else None,
                        "relation": relations[0] if relations else None,
                    },
                ) as _fp_span:
                    fast_answer = await self._try_fast_path(query, qtype, entities, relations)
                    _fp_span.set_attribute("succeeded", fast_answer is not None)
                if fast_answer is not None:
                    self._trace(f"Fast path succeeded ({len(fast_answer)} chars)", COLOR_GREEN)
                    return self._finalize_answer_text(fast_answer, qtype)
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
                query, openai_tools, max_iterations, refresh_interval, qtype=qtype
            )

            return answer

        except Exception as e:
            self._trace(f"{COLOR_RED}Error in agent loop: {e}{COLOR_END}", COLOR_RED)
            raise

        finally:
            await self._finalize_question()

    def _should_skip_fast_path(
        self,
        query: str,
        qtype: str,
        entities: List[str],
        relations: List[str],
    ) -> bool:
        """Block fast path for deceptively one-hop-looking qualifier questions."""
        q = query.lower()
        award_work_question = (
            ("nominated for" in q or "award" in q or "received" in q)
            and any(phrase in q for phrase in (
                "what film",
                "which film",
                "what movie",
                "which movie",
                "what work",
                "which work",
                "for which",
            ))
        )
        if award_work_question:
            self._trace("Skipping fast path: award/work qualifier pattern", COLOR_YELLOW)
            return True
        return False

    def _extract_exact_attribute_constraints(self, query: str) -> List[Dict[str, str]]:
        """
        Return KG-specific exact attribute/value constraints mentioned in the query.

        Subclasses override this with their schema vocabulary. The base agent only
        consumes the generic contract: each item has `attribute_name` and `value`.
        """
        return []

    def _extract_fast_path_attribute_lookup(self, query: str) -> Optional[Dict[str, str]]:
        """Return the first exact attribute lookup suitable for fast-path search."""
        constraints = self._extract_exact_attribute_constraints(query)
        return constraints[0] if constraints else None

    async def _try_fast_path(
        self,
        query: str,
        qtype: str,
        entities: List[str],
        relations: List[str],
    ) -> Optional[str]:
        """
        Attempt to answer simple 1-hop questions without the full agent loop.

        Executes a small, trace-visible lookup and returns only directly
        extracted values. It deliberately does not ask the LLM to synthesize from
        partial evidence; if the exact requested value is not present, the full
        loop gets the recorded tool evidence and continues normally.
        Returns None if the fast path cannot answer (fallback to full loop).
        """
        entity_name = entities[0]

        try:
            # Step 1: Find the entity. Exact code/identifier questions should
            # use reverse attribute lookup before semantic search.
            exact_lookup = self._extract_fast_path_attribute_lookup(query)
            if exact_lookup:
                self._trace(
                    "FAST PATH: exact attribute lookup "
                    f"{exact_lookup['attribute_name']}={exact_lookup['value']}",
                    COLOR_GREEN,
                )
                find_result = await self._execute_fast_path_tool("FindByAttribute", exact_lookup)
            else:
                find_result = await self._execute_fast_path_tool(
                    "FindNode",
                    {"semantic_node_name": entity_name},
                )

            # Parse the result to get node_id
            node_id, _resolved_entity_name = self._extract_first_match_identity(find_result)
            if not node_id:
                return None

            # Step 2: Try the exact requested relation/attribute first. Direct
            # values are safe to return; empty results fall back to the full loop.
            relation_name = relations[0] if relations else ""
            if qtype == "QueryAttr" and relation_name:
                attr_result = await self._execute_fast_path_tool(
                    "GetAttributeDetails",
                    {"base_node_id": node_id, "attribute_name": relation_name},
                )
                values = self._extract_attribute_values(attr_result)
                if values:
                    return self._format_fast_path_values(values)

            if qtype == "QueryRelation" and relation_name:
                relation_result = await self._execute_fast_path_tool(
                    "GetRelationDetails",
                    {"base_node_id": node_id, "relation_name": relation_name},
                )
                related_ids = self._extract_relation_ids(relation_result)
                if related_ids:
                    labels_result = await self._execute_fast_path_tool(
                        "BatchGetNodeLabels",
                        {"node_ids": related_ids[:10]},
                    )
                    labels = self._extract_batch_labels(labels_result)
                    values = [labels[rid] for rid in related_ids if rid in labels]
                    if values:
                        return self._format_fast_path_values(values)

            # Step 3: Preserve broad evidence for the fallback loop, but do not
            # let a synthesis-only fast path answer from this summary.
            await self._execute_fast_path_tool("GetNodeSummary", {"node_id": node_id})

            return None

        except Exception as e:
            self._trace(f"Fast path error: {e}", COLOR_YELLOW)
            return None

    async def _execute_fast_path_tool(self, func_name: str, func_args: Dict[str, Any]) -> str:
        """Execute a fast-path tool through normal tracing and mirror it into messages."""
        result = await self._execute_single_tool(func_name, func_args)
        self._append_fast_path_tool_messages(func_name, func_args, result)
        return result

    def _append_fast_path_tool_messages(
        self,
        func_name: str,
        func_args: Dict[str, Any],
        result: str,
    ) -> None:
        """Add synthetic assistant/tool messages so fast-path evidence is visible."""
        args_json = json.dumps(func_args, ensure_ascii=False)
        if self._text_tool_call_mode:
            call = json.dumps({"name": func_name, "arguments": func_args}, ensure_ascii=False)
            self._messages.append({
                "role": "assistant",
                "content": f"<tool_call>{call}</tool_call>",
            })
            self._messages.append({
                "role": "user",
                "content": f"<tool_result name=\"{func_name}\">{result}</tool_result>",
            })
            return

        tool_call_id = f"fast_path_{len(self.tool_call_durations)}_{func_name}"
        self._messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": tool_call_id,
                "type": "function",
                "function": {
                    "name": func_name,
                    "arguments": args_json,
                },
            }],
        })
        self._messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": func_name,
            "content": result,
        })

    @staticmethod
    def _parse_tool_json(raw: str) -> Dict[str, Any]:
        """Best-effort parser for JSON returned by MCP tools."""
        if not raw:
            return {}
        try:
            data = json.loads(raw)
            return data if isinstance(data, dict) else {}
        except Exception:
            extracted = BaseKBQAAgent._extract_json_object(str(raw))
            return extracted or {}

    @classmethod
    def _extract_first_match_identity(cls, raw: str) -> Tuple[str, str]:
        data = cls._parse_tool_json(raw)
        matches = data.get("matches") or []
        if not matches or not isinstance(matches[0], dict):
            return "", ""
        first = matches[0]
        return str(first.get("original_id") or ""), str(first.get("name") or "")

    @classmethod
    def _extract_attribute_values(cls, raw: str) -> List[str]:
        data = cls._parse_tool_json(raw)
        values = []
        for item in data.get("values") or []:
            if isinstance(item, dict):
                value = item.get("value")
                if value is None:
                    continue
                unit = item.get("unit")
                values.append(f"{value} {unit}" if unit else str(value))
            elif item is not None:
                values.append(str(item))
        return values

    @classmethod
    def _extract_relation_ids(cls, raw: str) -> List[str]:
        data = cls._parse_tool_json(raw)
        ids = []
        for item in data.get("triples") or []:
            if isinstance(item, dict) and item.get("related_id"):
                rid = str(item["related_id"])
                if rid not in ids:
                    ids.append(rid)
        return ids

    @classmethod
    def _extract_batch_labels(cls, raw: str) -> Dict[str, str]:
        data = cls._parse_tool_json(raw)
        resolved = data.get("resolved") or {}
        return {str(k): str(v) for k, v in resolved.items()} if isinstance(resolved, dict) else {}

    @staticmethod
    def _format_fast_path_values(values: List[str]) -> str:
        clean = [str(v).strip() for v in values if str(v).strip()]
        return " ".join(clean)

    async def _run_tool_loop(
        self,
        query: str,
        tools: List[Dict],
        max_iterations: int,
        refresh_interval: int,
        qtype: str = "",
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
        # Fast-path evidence is collected before the loop. Count it here so a
        # model may synthesize from already-recorded tool evidence without being
        # mislabeled as a true zero-tool answer.
        total_tool_calls_made = sum(self.tool_call_counts.values())
        zero_tool_call_retries = 0
        zero_tool_call_retry_max = get_zero_tool_call_retry_max() if get_zero_tool_call_retry() else 0

        while True:
            iteration_count += 1
            self._trace(f"Starting iteration {iteration_count}/{max_iterations}", COLOR_CYAN)

            # Safety check
            if iteration_count > max_iterations:
                self._trace(f"WARNING: Reached max iterations ({max_iterations})", COLOR_RED)
                self.recorder.event(
                    "intervention",
                    "max_iterations_reached",
                    attributes={"iteration_count": iteration_count, "max": max_iterations},
                )
                return "Error: Agent reached maximum iteration limit."

            # Per-iteration point-in-time event. We don't wrap the iteration
            # in an interval span because the loop has many `continue`/`break`
            # paths that would make exit-handling fragile. The LLM-call and
            # tool-call spans inside this iteration give us the structure;
            # this event just marks the iteration boundary in the timeline.
            self.recorder.event(
                "tool_loop_iter",
                f"iter:{iteration_count}",
                attributes={
                    "iteration": iteration_count,
                    "n_messages": len(self._messages),
                },
            )

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
            # Text-mode: don't pass tools= to the API call. The catalog is in
            # the system prompt and the format is in the format-instruction.
            if self._text_tool_call_mode:
                response = self._llm_call(tools=None, tool_choice=None)
            else:
                response = self._llm_call(tools=tools, tool_choice=tc)
            message = response.choices[0].message
            finish_reason = response.choices[0].finish_reason

            self._trace(f"LLM response (finish_reason: {finish_reason})", COLOR_CYAN)

            if response.usage:
                self._track_token_usage(response.usage)

            if message.content:
                self._trace(f"Thought: {message.content}", COLOR_BLUE)
                final_agent_content = message.content

            # Text-mode fallback: model didn't emit structured tool_calls but
            # may have written `<tool_call>{...}</tool_call>` blocks in content.
            # Parse them into synthetic tool_calls so the rest of the loop runs
            # unchanged. Applied only for known text-mode models so well-behaved
            # endpoints aren't double-parsed.
            text_mode_synthetic = False
            if self._text_tool_call_mode and not message.tool_calls and message.content:
                from ama_kbqa.framework.text_tool_calls import parse_text_tool_calls
                synthetic = parse_text_tool_calls(message.content)
                if synthetic:
                    self._trace(
                        f"Parsed {len(synthetic)} text tool_call(s) from content",
                        COLOR_CYAN,
                    )
                    message.tool_calls = synthetic
                    text_mode_synthetic = True
                    # Keep content as-is (do NOT strip) — for text-mode models
                    # the conversation history needs to preserve the model's
                    # original `<tool_call>...</tool_call>` blocks so the next
                    # turn sees its own format and continues to comply.

            # No tool calls - break for synthesis
            if not message.tool_calls:
                # Truncated text-mode tool call: minimax-m2.7-kit occasionally
                # stops generation mid-`<tool_call>{...}` (no closing tag, JSON
                # not parseable). The synthetic-parse step above returned
                # zero, so we'd fall through to "final answer" — except the
                # "answer" is a partial JSON fragment. Re-prompt once so the
                # model retries the call cleanly. Counts against the same
                # retry budget as zero-tool-call.
                if (
                    self._text_tool_call_mode
                    and zero_tool_call_retries < zero_tool_call_retry_max
                ):
                    from ama_kbqa.framework.text_tool_calls import has_truncated_tool_call
                    if has_truncated_tool_call(message.content):
                        zero_tool_call_retries += 1
                        self.recorder.event(
                            "intervention",
                            "truncated_tool_call_retry",
                            attributes={
                                "retry": zero_tool_call_retries,
                                "max": zero_tool_call_retry_max,
                            },
                        )
                        self._trace(
                            f"Truncated tool-call detected — re-prompting "
                            f"(retry {zero_tool_call_retries}/{zero_tool_call_retry_max})",
                            COLOR_RED,
                        )
                        # Persist the broken assistant turn so the model sees
                        # what it produced and can correct, then prompt it.
                        self._messages.append({
                            "role": "assistant",
                            "content": message.content or "",
                        })
                        self._messages.append({
                            "role": "user",
                            "content": (
                                "Your previous response contained an unclosed or unparseable "
                                "<tool_call> block. Re-emit the tool call you intended as ONE "
                                "complete block on a single line, with valid JSON: "
                                "<tool_call>{\"name\": \"ToolName\", \"arguments\": {...}}</tool_call>. "
                                "Do not include partial JSON or unclosed brackets. After the "
                                "tool result returns, continue the investigation."
                            ),
                        })
                        continue
                # RULE 0 enforcement: if the agent is about to emit a final
                # answer without ever having queried the KG, re-prompt it
                # once. Pure-prompt RULE 0 / RULE 0a do not bind reliably for
                # gemma; observed ~7% zero-tool-call hallucinations in the
                # 2026-05-03 fixbundle audit.
                if (
                    total_tool_calls_made == 0
                    and zero_tool_call_retries < zero_tool_call_retry_max
                ):
                    zero_tool_call_retries += 1
                    self.recorder.event(
                        "intervention",
                        "zero_tool_call_retry",
                        attributes={
                            "retry": zero_tool_call_retries,
                            "max": zero_tool_call_retry_max,
                        },
                    )
                    self._trace(
                        f"Zero-tool-call answer detected — re-prompting "
                        f"(retry {zero_tool_call_retries}/{zero_tool_call_retry_max})",
                        COLOR_RED,
                    )
                    self._messages.append({
                        "role": "user",
                        "content": (
                            "STOP. You produced a final answer without calling any tools, "
                            "which violates RULE 0 (MANDATORY TOOL USE). The knowledge graph "
                            "almost certainly has the answer; you have not yet looked. "
                            "Discard your previous response. "
                            "Now: pick one entity from the question and call FindNode "
                            "(semantic name) or FindByAttribute (exact code/URL/ID/ISNI). "
                            "Then proceed with normal lookup. Do NOT answer in prose until "
                            "you have queried the KG."
                        ),
                    })
                    continue
                if total_tool_calls_made == 0:
                    self.recorder.event(
                        "intervention",
                        "zero_tool_call_hard_stop",
                        attributes={
                            "retry": zero_tool_call_retries,
                            "max": zero_tool_call_retry_max,
                        },
                    )
                    self._trace(
                        "Zero-tool-call answer persisted after retry budget - hard stopping",
                        COLOR_RED,
                    )
                    return "Error: Agent attempted final answer with zero tool calls."
                self._trace("No more tool calls - breaking to synthesis", COLOR_GREEN)
                if message.content:
                    self._messages.append({"role": "assistant", "content": message.content})
                final_agent_content = message.content
                break

            total_tool_calls_made += len(message.tool_calls)

            # Add assistant message to history. For text-mode we keep the
            # original content (with `<tool_call>` blocks) and DON'T attach the
            # OpenAI-style structured `tool_calls` field — minimax-style models
            # weren't trained on that shape and including it confuses replies.
            msg_dict: Dict[str, Any] = {"role": message.role, "content": message.content}
            if message.tool_calls and not text_mode_synthetic:
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

            # Execute tool calls. Tell the executor whether to wrap results as
            # `role=tool` (native) or `role=user` plain prose (text-mode).
            called_get_journal_summary = await self._execute_tool_calls(
                message.tool_calls, as_user_messages=text_mode_synthetic,
            )

            # Inject answer prompt if GetJournalSummary was called
            # (can be disabled via agent.auto_inject_journal)
            if called_get_journal_summary and get_auto_inject_journal():
                self._trace("GetJournalSummary called - injecting answer prompt", COLOR_CYAN)
                self._messages.append({
                    "role": "user",
                    "content": self._get_journal_summary_answer_prompt()
                })

            max_tool_calls = getattr(self, "_max_tool_calls", 0) or 0
            if max_tool_calls and total_tool_calls_made >= max_tool_calls:
                self.recorder.event(
                    "intervention",
                    "max_tool_calls_reached",
                    attributes={
                        "total_tool_calls": total_tool_calls_made,
                        "max_tool_calls": max_tool_calls,
                    },
                )
                self._trace(
                    f"Reached max tool-call cap ({total_tool_calls_made}/{max_tool_calls}) - forcing synthesis",
                    COLOR_RED,
                )
                return await self._run_synthesis(query, qtype=qtype)

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
                return self._finalize_answer_text(final_agent_content, qtype)
            self._trace(
                "Synthesis bypassed but agent produced no final content - falling back to synthesis",
                COLOR_YELLOW,
            )

        # Run synthesis
        return await self._run_synthesis(query, qtype=qtype)

    async def _execute_tool_calls(self, tool_calls: List, as_user_messages: bool = False) -> bool:
        """
        Execute a batch of tool calls.

        Args:
            tool_calls: List of tool calls from LLM response
            as_user_messages: If True, append results as `role=user` prose
                instead of `role=tool` records. Used for text-mode models that
                weren't trained on the OpenAI tool-message schema.

        Returns:
            True if GetJournalSummary was called
        """
        called_get_journal_summary = False
        self._trace(f"Processing {len(tool_calls)} tool call(s)", COLOR_YELLOW)

        def _append_tool_result(tc, name: str, content: str) -> None:
            if as_user_messages:
                self._messages.append({
                    "role": "user",
                    "content": f"<tool_result name=\"{name}\">{content}</tool_result>",
                })
            else:
                self._messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": name,
                    "content": content,
                })

        for tool_call in tool_calls:
            if not tool_call.function or not tool_call.function.name:
                _append_tool_result(tool_call, "invalid_tool", "Error: Invalid tool call.")
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
                _append_tool_result(tool_call, func_name, tool_result)
                continue

            # Parse arguments
            try:
                args_str = tool_call.function.arguments
                func_args = json.loads(args_str) if args_str else {}
            except json.JSONDecodeError:
                func_args = {}

            args_pretty = json.dumps(func_args, indent=2, ensure_ascii=False)
            self._trace(f"Tool Call: {func_name}\n   Params: {args_pretty}", COLOR_YELLOW)

            if func_name == "GetJournalSummary":
                if func_args:
                    self._trace(
                        "GetJournalSummary called with unexpected arguments; "
                        "not injecting answer prompt until a no-argument summary call succeeds",
                        COLOR_YELLOW,
                    )
                else:
                    called_get_journal_summary = True

            # Check for loops
            loop_detected, loop_reason = self._detect_loops(func_name, func_args)

            if loop_detected:
                tool_result = await self._handle_loop_detected(
                    func_name, loop_reason
                )
            else:
                tool_result = await self._execute_single_tool(func_name, func_args)

            _append_tool_result(tool_call, func_name, tool_result)

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

        async with self.recorder.span(
            "tool_call",
            func_name,
            attributes={
                "tool_name": func_name,
                "args_keys": sorted(list(func_args.keys()))[:8],
            },
            payload={"arguments": func_args},
        ) as _span:
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

                _span.update_attributes({
                    "duration_seconds": round(tool_duration, 3),
                    "success": True,
                    "result_chars": len(tool_result) if tool_result else 0,
                })
                _span.set_payload("result", tool_result)

                # Log result
                log_result = tool_result
                if len(log_result) > 500:
                    log_result = log_result[:500] + "... [truncated]"
                self._trace(f"Result ({func_name}) [{tool_duration:.3f}s]: {log_result}", COLOR_CYAN)

                # Snapshot the journal if this tool mutates it. Cheap: one
                # extra MCP call only for known-mutating tools, not every
                # iteration.
                if func_name in JOURNAL_MUTATING_TOOLS:
                    await self._snapshot_journal(trigger=f"after:{func_name}")

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

                _span.update_attributes({
                    "duration_seconds": round(tool_duration, 3),
                    "success": False,
                    "error": str(tool_error),
                })

                self._trace(f"Tool {func_name} failed: {tool_error}", COLOR_RED)
                return f"Error executing {func_name}: {tool_error}. Try a different approach."

    async def _snapshot_journal(self, trigger: str) -> None:
        """Fetch the current server-side journal as a structured dict.

        Cheap when called only at journal-mutating boundaries. Silently no-op
        on failure (e.g. server doesn't have the GetJournalStateJSON tool yet
        — fall back to nothing rather than break the agent loop).
        """
        if not self.mcp:
            return
        try:
            raw = await self.mcp.call_tool("GetJournalStateJSON", {})
        except Exception:
            return
        if not raw:
            return
        try:
            state = json.loads(raw)
        except Exception:
            return
        # Skip duplicates: cheaper than a hash; just compare object identity
        # via the last snapshot's state dict.
        if self.journal_snapshots and self.journal_snapshots[-1].get("state") == state:
            return
        self.journal_snapshots.append({
            "ts": time.time(),
            "trigger": trigger,
            "state": state,
        })

    async def _handle_loop_detected(self, func_name: str, loop_reason: str) -> str:
        """
        Handle a detected loop by injecting intervention.

        Args:
            func_name: Name of the looping tool
            loop_reason: Reason for loop detection

        Returns:
            Intervention message string
        """
        self.recorder.event(
            "loop_detected",
            func_name,
            attributes={"tool_name": func_name, "reason": loop_reason},
        )
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
            no_progress = (
                self.last_journal_state is not None
                and journal_refresh == self.last_journal_state
            )
            if no_progress:
                self._trace("WARNING: No progress in last 5 iterations!", COLOR_YELLOW)
                template = self._get_no_progress_template()
            else:
                template = self._get_journal_refresh_template()

            self.recorder.event(
                "journal_refresh",
                f"iter:{iteration_count}",
                attributes={
                    "iteration": iteration_count,
                    "no_progress": no_progress,
                    "summary_chars": len(journal_refresh) if journal_refresh else 0,
                },
            )

            # Also snapshot structured journal state on refresh — only if it
            # has actually changed since the last snapshot.
            if not no_progress:
                await self._snapshot_journal(trigger=f"refresh:iter{iteration_count}")

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
            is_tool_result = msg.get("role") == "tool"
            is_text_mode_tool_result = (
                msg.get("role") == "user"
                and isinstance(content, str)
                and content.startswith("<tool_result")
            )
            if (is_tool_result or is_text_mode_tool_result) and len(content) > truncate_len + 50:
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
            self.recorder.event(
                "context_trim",
                f"trimmed_{trimmed_count}",
                attributes={
                    "trimmed_count": trimmed_count,
                    "truncate_len": truncate_len,
                    "estimated_tokens": int(estimated_tokens),
                    "context_limit": context_limit,
                },
            )

    async def _run_synthesis(self, query: str, qtype: str = "") -> str:
        """
        Run the deterministic synthesis step.

        Uses a MINIMAL message set (system + journal + query) instead of the
        full conversation history. This saves 30-80k tokens per question.

        Args:
            query: Original query

        Returns:
            Final answer string
        """
        async with self.recorder.span(
            "synthesis",
            self.synthesis_model,
            attributes={"model": self.synthesis_model},
            payload={"query": query},
        ) as _syn_span:
            return await self._run_synthesis_impl(query, _syn_span, qtype=qtype)

    async def _run_synthesis_impl(self, query: str, _syn_span, qtype: str = "") -> str:
        self._trace("Starting synthesis step", COLOR_CYAN)

        # Get journal summary
        journal_summary = await self.mcp.call_tool("GetJournalSummary", {})
        self._trace(f"Journal fetched ({len(journal_summary)} chars)", COLOR_GREEN)
        _syn_span.set_attribute("journal_chars", len(journal_summary) if journal_summary else 0)
        _syn_span.set_payload("journal_summary", journal_summary)
        # Take a final structured snapshot for the graph view.
        await self._snapshot_journal(trigger="synthesis")

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

            final_answer = self._finalize_answer_text(final_answer, qtype)
            self._trace(f"Synthesis complete ({len(final_answer)} chars)", COLOR_GREEN)
            self._messages.append({"role": "assistant", "content": final_answer})
            return final_answer
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
        call_params: Dict[str, Any] = {
            "model": self.model,
            "messages": self._messages,
            "timeout": self.request_timeout,
            "temperature": get_chat_temperature(),
            "max_tokens": get_chat_max_tokens(),
        }
        # Only include tools= when we actually have some — for text-mode models
        # we pass tools=None on purpose, and some servers reject a literal null.
        if tools:
            call_params["tools"] = tools

        if tool_choice and tools:
            call_params["tool_choice"] = tool_choice

        provider_prefs = get_provider_preferences()
        if provider_prefs:
            call_params["extra_body"] = {"provider": provider_prefs}

        self._trace(f"Calling {self.model} (timeout: {self.request_timeout}s)", COLOR_CYAN)

        with self.recorder.span_sync(
            "llm_call",
            self.model,
            attributes={
                "model": self.model,
                "n_messages": len(self._messages),
                "n_tools": len(tools) if tools else 0,
                "tool_choice": tool_choice,
            },
            payload={"messages": self._messages_for_payload()},
        ) as _llm_span:
            try:
                response = self.client.chat.completions.create(**call_params)
                self._trace("LLM call completed", COLOR_GREEN)
                if response.usage:
                    _llm_span.update_attributes({
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens,
                    })
                if response.choices:
                    _llm_span.set_attribute(
                        "finish_reason", response.choices[0].finish_reason
                    )
                    msg = response.choices[0].message
                    _llm_span.set_payload(
                        "assistant_content",
                        (msg.content or "")[:4000],
                    )
                    if getattr(msg, "tool_calls", None):
                        _llm_span.set_payload(
                            "tool_calls",
                            [
                                {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments,
                                }
                                for tc in msg.tool_calls
                            ],
                        )
                return response
            except Exception as e:
                self._trace(f"LLM call failed: {e}", COLOR_RED)
                raise

    def _messages_for_payload(self) -> List[Dict[str, Any]]:
        """Snapshot self._messages for trace payload, truncating long content."""
        snap: List[Dict[str, Any]] = []
        for m in self._messages:
            content = m.get("content")
            if isinstance(content, str) and len(content) > 4000:
                content = content[:4000] + "...[truncated]"
            entry = {"role": m.get("role"), "content": content}
            if "tool_calls" in m:
                entry["tool_calls"] = m["tool_calls"]
            if "tool_call_id" in m:
                entry["tool_call_id"] = m["tool_call_id"]
            if "name" in m:
                entry["name"] = m["name"]
            snap.append(entry)
        return snap

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

        with self.recorder.span_sync(
            "llm_call",
            f"{self.model} (text-only)",
            attributes={"model": self.model, "n_messages": len(self._messages), "mode": "text_only"},
            payload={"messages": self._messages_for_payload()},
        ) as _span:
            try:
                response = self.client.chat.completions.create(**call_params)

                if response.usage:
                    self._track_token_usage(response.usage)
                    _span.update_attributes({
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                    })

                content = response.choices[0].message.content
                _span.set_payload("assistant_content", (content or "")[:4000])
                return content
            except Exception as e:
                self._trace(f"LLM text-only call failed: {e}", COLOR_RED)
                raise

    def _llm_call_synthesis(self, messages_override: Optional[List[Dict[str, Any]]] = None) -> str:
        """Execute synthesis LLM call with optional minimal message set."""
        msgs = messages_override if messages_override is not None else self._messages
        call_params = {
            "model": self.synthesis_model,
            "messages": msgs,
            "timeout": self.request_timeout,
            "temperature": get_synthesis_temperature(),
            "max_tokens": get_synthesis_max_tokens(),
        }

        provider_prefs = get_synthesis_provider_preferences()
        if provider_prefs:
            call_params["extra_body"] = {"provider": provider_prefs}

        self._trace(f"Calling {self.synthesis_model} for synthesis", COLOR_CYAN)

        with self.recorder.span_sync(
            "llm_call",
            f"{self.synthesis_model} (synthesis)",
            attributes={
                "model": self.synthesis_model,
                "n_messages": len(msgs),
                "mode": "synthesis",
            },
            payload={"messages": [
                {"role": m.get("role"), "content": (m.get("content") or "")[:4000]}
                for m in msgs
            ]},
        ) as _span:
            try:
                response = self.synthesis_client.chat.completions.create(**call_params)
                self._trace("Synthesis LLM call completed", COLOR_GREEN)

                if response.usage:
                    self._track_token_usage(response.usage)
                    _span.update_attributes({
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                    })

                content = response.choices[0].message.content
                _span.set_payload("assistant_content", (content or "")[:4000])
                return content
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
        # Reset trace + snapshots so a reused agent starts a clean trace.
        # When this agent shares a parent recorder, only clear our snapshots
        # and let the parent keep its events.
        if self._parent_span_id_override is None:
            self.recorder = TraceRecorder()
        self.journal_snapshots = []
        self._last_journal_summary_hash = None

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
