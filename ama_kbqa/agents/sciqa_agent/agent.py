"""
SciQA Agent using the generic KBQA framework.

This agent uses the Open Research Knowledge Graph (ORKG) loaded from
the SciQA dataset to answer scientific research questions.
"""

from __future__ import annotations
import asyncio
import os
import re
from pathlib import Path
from typing import Dict, List, Optional

from ama_kbqa.config import get_chat_temperature

from ama_kbqa.framework.base_agent import BaseKBQAAgent
from ama_kbqa.framework.config import KnowledgeGraphConfig
from ama_kbqa.framework.adapters.sciqa_adapter import SciQAAdapter

from ama_kbqa.agents.sciqa_agent.prompts import (
    QTYPE_STRATEGIES,
    SYSTEM_PROMPT,
    CLASSIFICATION_PROMPT_TEMPLATE,
    ENTITY_EXTRACTION_PROMPT,
    ANALYSIS_CONTEXT_TEMPLATE,
    QUESTION_CONTEXT_TEMPLATE,
    FEWSHOT_EXAMPLES_TEMPLATE,
    ANALYSIS_CONTEXT_SUFFIX,
    JOURNAL_REFRESH_TEMPLATE,
    NO_PROGRESS_TEMPLATE,
    SYNTHESIS_PROMPT_TEMPLATE,
    SYNTHESIS_PROMPT_TEMPLATE_CONVERSATIONAL,
    JOURNAL_SUMMARY_ANSWER_PROMPT,
    TOOL_LOOP_GUIDANCE,
    GENERIC_LOOP_GUIDANCE,
    LOOP_INTERVENTION_TEMPLATE,
    FEWSHOT_EXAMPLES,
)


# Tools always included regardless of question type
CORE_TOOLS = {
    "FindResource", "FindPredicate", "GetPredicateReference",
    "GetResourceDetails", "GetResourceSummary",
    "GetResourceLabel", "BatchGetResourceLabels", "GetRelationTargets",
    "FollowRelationPath", "FindByPredicateValue", "RunORKGSPARQL",
    "ManageJournal", "GetJournalSummary", "GetJournalStateJSON",
}

# Extra tools per question type (on top of CORE_TOOLS). "General" is
# deliberately absent: unknown/general questions keep the full tool set.
QTYPE_TOOL_MAP: Dict[str, set] = {
    "Factoid":     {"GetPaperContributions", "GetPaperAuthors", "GetContributionMethods",
                    "GetResearchFieldPapers", "FindAuthorPapers", "FindCoAuthors",
                    "GetComparisonContributions"},
    "Count":       {"GetResearchFieldPapers", "FindAuthorPapers", "FindCoAuthors",
                    "GetPaperAuthors", "FindFrequentValues", "GetComparisonContributions",
                    "InspectComparisonSchema", "QueryComparisonRows"},
    "List":        {"GetPaperContributions", "GetPaperAuthors", "GetResearchFieldPapers",
                    "FindAuthorPapers", "FindCoAuthors", "GetComparisonContributions",
                    "InspectComparisonSchema", "QueryComparisonRows"},
    "Boolean":     {"VerifyNumericCondition", "CompareResources", "GetPaperAuthors",
                    "GetPaperContributions", "FindAuthorPapers",
                    "InspectComparisonSchema", "QueryComparisonRows"},
    "Comparison":  {"InspectComparisonSchema", "QueryComparisonRows",
                    "AggregateComparisonValues", "DiagnoseComparisonAggregation",
                    "GetComparisonContributions", "CompareResources",
                    "FindFrequentValues", "VerifyNumericCondition"},
    "Superlative": {"InspectComparisonSchema", "QueryComparisonRows",
                    "AggregateComparisonValues", "DiagnoseComparisonAggregation",
                    "GetComparisonContributions", "CompareResources",
                    "FindFrequentValues", "GetResearchFieldPapers"},
    "Aggregation": {"InspectComparisonSchema", "QueryComparisonRows",
                    "AggregateComparisonValues", "DiagnoseComparisonAggregation",
                    "GetComparisonContributions", "FindFrequentValues",
                    "VerifyNumericCondition"},
}


def allowed_tools_for_qtype(qtype: str) -> Optional[set]:
    """
    Resolve a (possibly multi-label) SciQA qtype to its allowed tool set.

    The classifier sometimes emits multi-label strings such as
    "Factoid\\nSuperlative" or "Factoid, Count"; tools from EVERY matching
    label are unioned so the specific operation labels always land.
    Returns None (all tools) when no label matches, including "General".
    """
    if not qtype:
        return None
    labels = [c.strip() for c in re.split(r"[\n,/+|;]+", qtype) if c.strip()]
    extra: set = set()
    matched = False
    for label in labels:
        norm = label[:1].upper() + label[1:]
        tools = QTYPE_TOOL_MAP.get(norm)
        if tools is not None:
            matched = True
            extra |= tools
    if not matched:
        return None
    return CORE_TOOLS | extra


class SciQAAgent(BaseKBQAAgent):
    """
    SciQA Agent for ORKG Knowledge Graph Question Answering.

    Uses the SciQA MCP server to query the Open Research Knowledge Graph
    and answer scientific research questions.
    """

    def __init__(self, name: str = "sciqa_agent", session_id: str = "default", use_fewshot: bool = True):
        """
        Initialize the SciQA agent.

        Args:
            name: Agent name for tracing
            session_id: Session identifier
            use_fewshot: Whether to inject few-shot examples during classification
        """
        self._adapter = SciQAAdapter()
        super().__init__(name=name, session_id=session_id, use_fewshot=use_fewshot)

    # =========================================================================
    # ABSTRACT METHOD IMPLEMENTATIONS
    # =========================================================================

    def get_config(self) -> KnowledgeGraphConfig:
        """Get the SciQA/ORKG configuration."""
        return self._adapter.config

    def get_mcp_server_path(self) -> str:
        """Get the path to the SciQA MCP server."""
        current_file = Path(__file__).resolve()
        ama_kbqa_root = current_file.parents[2]
        return str(ama_kbqa_root / "server" / "sciqa_server.py")

    # =========================================================================
    # TEMPLATE METHOD OVERRIDES
    # =========================================================================

    def _get_system_prompt(self) -> str:
        """Get the SciQA system prompt."""
        return SYSTEM_PROMPT

    def _get_qtype_strategies(self) -> Dict[str, str]:
        """Get SciQA question-type strategies."""
        return QTYPE_STRATEGIES

    def _get_classification_prompt(self, question: str) -> str:
        """Get the classification prompt for a question."""
        return CLASSIFICATION_PROMPT_TEMPLATE.format(question=question)

    def _get_entity_extraction_prompt(self) -> str:
        """Get the entity extraction prompt."""
        return ENTITY_EXTRACTION_PROMPT

    def _get_analysis_context_template(self) -> str:
        """Get the analysis context template."""
        return ANALYSIS_CONTEXT_TEMPLATE

    def _get_synthesis_prompt_template(self) -> str:
        """Get the synthesis prompt template — style depends on synthesis_mode."""
        from ama_kbqa.config import get_synthesis_mode
        if get_synthesis_mode() == "conversational":
            return SYNTHESIS_PROMPT_TEMPLATE_CONVERSATIONAL
        return SYNTHESIS_PROMPT_TEMPLATE

    def _get_journal_refresh_template(self) -> str:
        """Get the journal refresh template."""
        return JOURNAL_REFRESH_TEMPLATE

    def _get_no_progress_template(self) -> str:
        """Get the no-progress intervention template."""
        return NO_PROGRESS_TEMPLATE

    def _get_tool_loop_guidance(self) -> Dict[str, str]:
        """Get tool-specific loop recovery guidance."""
        return TOOL_LOOP_GUIDANCE

    def _get_generic_loop_guidance(self) -> str:
        """Get generic loop recovery guidance."""
        return GENERIC_LOOP_GUIDANCE

    def _get_loop_intervention_template(self) -> str:
        """Get the loop intervention template."""
        return LOOP_INTERVENTION_TEMPLATE

    def _get_journal_summary_answer_prompt(self) -> str:
        """Get the prompt to inject after GetJournalSummary."""
        return JOURNAL_SUMMARY_ANSWER_PROMPT

    def _get_allowed_tools_for_qtype(self, qtype: str) -> Optional[set]:
        """Return set of tool names allowed for this question type, or None for all."""
        return allowed_tools_for_qtype(qtype)

    def _get_denied_tool_names(self) -> set:
        """Gate raw SPARQL off when AMA_SCIQA_DISABLE_RAW_SPARQL is set (truthy).

        Off by default. Used to A/B-test retiring RunORKGSPARQL and as the
        last-resort gating lever: the tool runs ~65% unproductive, so a run can
        force the agent onto the dedicated comparison/aggregation tools.
        """
        flag = os.environ.get("AMA_SCIQA_DISABLE_RAW_SPARQL", "").strip().lower()
        if flag in {"1", "true", "yes", "on"}:
            return {"RunORKGSPARQL"}
        return set()

    def _classify_question(self, question: str) -> Dict[str, str]:
        """
        Classify a question and inject few-shot examples based on type.

        Overrides base class to populate fewshot_examples from FEWSHOT_EXAMPLES dict.
        """
        prompt = self._get_classification_prompt(question)

        try:
            call_params: Dict[str, object] = {
                "model": self.model,
                "messages": [{"role": "system", "content": prompt}],
                "temperature": get_chat_temperature(),
                "response_format": {"type": "json_object"},
                # Bumped from default to leave headroom for minimax-m2.7's
                # `<think>...</think>` reasoning prefix that precedes JSON.
                # Without this, the response truncates inside the think block
                # and qtype silently defaults to "General" — which means
                # FEWSHOT_EXAMPLES.get("General", "") returns "" and the
                # SciQA agent flies blind on aggregation/comparison/count
                # questions. Verified empirically (KQAPro side: 100/100
                # questions classified as "Query" before this fix).
                "max_tokens": 1500,
                "timeout": 30.0,
            }
            # Same stepped-backoff retry as the base agent's tool loop/synthesis
            # calls (self._retry, set up in BaseKBQAAgent.__init__).
            response = self._create_with_retry(self.client, call_params, label="SciQA classification")

            if response.usage:
                self._track_token_usage(response.usage)

            json_content = response.choices[0].message.content
            # Use the brace-balanced extractor on the base class to handle
            # `<think>...</think>` prefixes and markdown fences.
            result = self._extract_json_object(json_content)
            if result is None:
                self._trace(
                    "SciQA classification: no parseable JSON in response, "
                    "defaulting to General",
                    "\033[93m",
                )
                return {"question_type": "General", "fewshot_examples": ""}
            qtype = result.get("question_type", "General")

            # Inject few-shot examples for the detected question type (unless disabled).
            # Some classifier responses come back as multi-label strings like
            # "Factoid\nSuperlative" or "Factoid, Count" — split on common separators
            # and concatenate fewshots from EVERY matching label. The specific
            # operation labels (Count, Superlative, Aggregation, ...) carry the
            # high-leverage guidance, so they should always land even when the
            # classifier emits them after a generic "Factoid"/"Non-factoid" prefix.
            fewshot = ""
            chosen_label = qtype
            if self.use_fewshot:
                direct = FEWSHOT_EXAMPLES.get(qtype, "")
                if direct:
                    fewshot = direct
                elif qtype:
                    candidates = [
                        c.strip() for c in re.split(r"[\n,/+|;]+", qtype) if c.strip()
                    ]
                    seen = set()
                    parts: List[str] = []
                    for cand in candidates:
                        cand_norm = cand[:1].upper() + cand[1:] if cand else cand
                        if cand_norm in FEWSHOT_EXAMPLES and cand_norm not in seen:
                            seen.add(cand_norm)
                            parts.append(FEWSHOT_EXAMPLES[cand_norm])
                            if chosen_label == qtype:
                                chosen_label = cand_norm
                    fewshot = "\n".join(parts)

            return {
                "question_type": chosen_label,
                "fewshot_examples": fewshot
            }

        except Exception as e:
            self._trace(f"Question classification failed: {e}", "\033[93m")
            return {"question_type": "General", "fewshot_examples": ""}

    # =========================================================================
    # SCIQA-SPECIFIC METHODS
    # =========================================================================

    def _build_static_qtype_context(
        self,
        qtype: str,
        fewshot_examples: str = "",
    ) -> str:
        """
        Build the STATIC, qtype-only portion of the analysis context:
        reasoning strategy + few-shot examples. This depends only on
        `qtype`, never on the specific question's text/entities/relations,
        so it is byte-identical across every question of the same qtype.

        `_ask_impl` appends this as its own message BEFORE the
        per-question context (`_build_question_context`) so this block
        stays a shared, cacheable prefix across consecutive same-qtype
        questions instead of sitting behind per-question text. See
        `.agent/Tasks/active/prompt-cache-utilization.md`.

        Args:
            qtype: Question type
            fewshot_examples: Optional few-shot examples for this qtype

        Returns:
            Static analysis context string
        """
        qtype_strategy = QTYPE_STRATEGIES.get(qtype, QTYPE_STRATEGIES.get("General", ""))

        context = qtype_strategy

        if fewshot_examples and fewshot_examples.strip():
            context += FEWSHOT_EXAMPLES_TEMPLATE.format(
                qtype=qtype,
                fewshot_examples=fewshot_examples
            )

        return context

    def _build_question_context(
        self,
        qtype: str,
        entities: List[str],
        relations: List[str],
        query: str = "",
    ) -> str:
        """
        Build the PER-QUESTION portion of the analysis context: the
        type/entities/relations header. Must be appended to the message
        stack AFTER the static qtype context (`_build_static_qtype_context`)
        so that block stays a stable, cacheable prefix.

        Args:
            qtype: Question type
            entities: Extracted entities
            relations: Extracted relations
            query: The raw question text (unused here; SciQA has no
                per-query exact-constraint extraction, unlike KQAPro)

        Returns:
            Per-question analysis context string
        """
        formatted_entities = "\n".join([f"  - {e}" for e in entities]) if entities else "  (none identified)"
        formatted_relations = "\n".join([f"  - {r}" for r in relations]) if relations else "  (none identified)"

        context = QUESTION_CONTEXT_TEMPLATE.format(
            qtype=qtype,
            formatted_entities=formatted_entities,
            formatted_relations=formatted_relations,
        )

        # ANALYSIS_CONTEXT_SUFFIX ("Use this pre-analysis to guide...") reads
        # naturally right after this question's specific entities/relations,
        # so it stays at the tail of the per-question block rather than in
        # the static block.
        context += ANALYSIS_CONTEXT_SUFFIX

        return context

    def _build_analysis_context(
        self,
        qtype: str,
        entities: List[str],
        relations: List[str],
        fewshot_examples: str = "",
        query: str = "",
    ) -> str:
        """
        Backward-compatible combined builder (static context followed by
        per-question context). `_ask_impl` calls the two builders above
        directly, as separate messages, so the static block can be placed
        ahead of per-question content for prefix-cache reuse; this wrapper
        is kept for any other caller (e.g. tests) that still wants one
        combined string.

        Args:
            qtype: Question type
            entities: Extracted entities
            relations: Extracted relations
            fewshot_examples: Optional few-shot examples

        Returns:
            Analysis context string
        """
        static = self._build_static_qtype_context(qtype, fewshot_examples)
        question = self._build_question_context(qtype, entities, relations, query)
        if static and question:
            return f"{static}\n\n{question}"
        return static or question


# =============================================================================
# Main entry point for testing
# =============================================================================

if __name__ == "__main__":
    async def run_test():
        agent = SciQAAgent()
        try:
            answer = await agent.ask("What papers address the problem of text classification?")
            print(f"\n[SciQA Agent Answer]\n{answer}")

            counts = agent.get_tool_call_counts()
            print("\n[Tool Call Counts]")
            for tool_name, count in sorted(counts.items(), key=lambda x: x[1], reverse=True):
                print(f"  {tool_name}: {count}x")

            summary = agent.get_tool_call_summary()
            print("\n[Tool Call Summary]")
            print(f"Total calls: {summary['total_calls']}")
            print(f"Total duration: {summary['total_duration_seconds']}s")
            for tool_name, stats in summary['tool_breakdown'].items():
                print(f"  {tool_name}: {stats['count']}x, avg {stats['avg_duration']}s")
        except Exception as e:
            print(f"Error in test run: {e}")
        finally:
            await agent.close()

    asyncio.run(run_test())
