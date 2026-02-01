"""
KQAPro Agent using the generic KBQA framework.

This agent uses the KQAPro knowledge graph to answer natural language
questions about general domain facts.
"""

from __future__ import annotations
import asyncio
import json
from pathlib import Path
from typing import Dict, List, Optional

from ama_kbqa.framework.base_agent import BaseKBQAAgent
from ama_kbqa.framework.config import KnowledgeGraphConfig
from ama_kbqa.framework.adapters.kqapro_adapter import KQAProAdapter

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


class KQAProAgent(BaseKBQAAgent):
    """
    KQAPro Agent for Knowledge Graph Question Answering.

    Uses the KQAPro MCP server to query the knowledge graph and
    answer natural language questions.
    """

    def __init__(self, name: str = "kqapro_agent", session_id: str = "default"):
        """
        Initialize the KQAPro agent.

        Args:
            name: Agent name for tracing
            session_id: Session identifier
        """
        self._adapter = KQAProAdapter()
        super().__init__(name=name, session_id=session_id)

    # =========================================================================
    # ABSTRACT METHOD IMPLEMENTATIONS
    # =========================================================================

    def get_config(self) -> KnowledgeGraphConfig:
        """Get the KQAPro configuration."""
        return self._adapter.config

    def get_mcp_server_path(self) -> str:
        """Get the path to the KQAPro MCP server."""
        current_file = Path(__file__).resolve()
        ama_kbqa_root = current_file.parents[2]
        return str(ama_kbqa_root / "server" / "kqapro_server.py")

    # =========================================================================
    # TEMPLATE METHOD OVERRIDES
    # =========================================================================

    def _get_system_prompt(self) -> str:
        """Get the KQAPro system prompt."""
        return SYSTEM_PROMPT

    def _get_qtype_strategies(self) -> Dict[str, str]:
        """Get KQAPro question-type strategies."""
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
        """Get the synthesis prompt template."""
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

    # =========================================================================
    # KQAPRO-SPECIFIC METHODS
    # =========================================================================

    def _load_fewshot_examples(
        self,
        max_per_type: int = 10,
        specific_qtype: Optional[str] = None
    ) -> str:
        """
        Load few-shot examples from the fewshot-examples directory.

        Args:
            max_per_type: Maximum examples per question type
            specific_qtype: Load only this question type if specified

        Returns:
            Formatted few-shot examples string
        """
        repo_root = Path(__file__).resolve().parents[3]
        fewshot_dir = repo_root / "db" / "datasets" / "kqapro" / "fewshot-examples"

        if not fewshot_dir.exists():
            return ""

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

                examples = examples[:max_per_type]

                for example in examples:
                    all_examples.append({
                        "qtype": qtype,
                        "question": example.get("question", ""),
                        "reasoning": example.get("reasoning", ""),
                        "lesson_learned": example.get("lesson_learned", "")
                    })

            except (json.JSONDecodeError, Exception):
                continue

        if not all_examples:
            return ""

        formatted = "\n\n### Few-Shot Examples\n\n"
        for i, example in enumerate(all_examples, 1):
            formatted += f"**Example {i}:**\n"
            formatted += f"Question: {example['question']}\n"
            formatted += f"Type: {example['qtype']}\n"
            if example['reasoning']:
                formatted += f"Reasoning: {example['reasoning']}\n"
            if example['lesson_learned']:
                formatted += f"Lesson: {example['lesson_learned']}\n"
            formatted += "\n"

        return formatted

    def _classify_question(self, question: str) -> Dict[str, str]:
        """
        Classify a question and load relevant few-shot examples.

        Override to add KQAPro-specific few-shot example loading.

        Args:
            question: The question to classify

        Returns:
            Dict with 'question_type' and 'fewshot_examples' keys
        """
        # Call parent classification
        result = super()._classify_question(question)

        # Load few-shot examples for this question type
        qtype = result.get("question_type", "Query")
        fewshot_examples = self._load_fewshot_examples(
            max_per_type=10,
            specific_qtype=qtype
        )
        result["fewshot_examples"] = fewshot_examples

        return result

    def _build_analysis_context(
        self,
        qtype: str,
        entities: List[str],
        relations: List[str],
        fewshot_examples: str = ""
    ) -> str:
        """
        Build the analysis context message with KQAPro templates.

        Args:
            qtype: Question type
            entities: Extracted entities
            relations: Extracted relations
            fewshot_examples: Optional few-shot examples

        Returns:
            Analysis context string
        """
        formatted_entities = "\n".join([f"  - {e}" for e in entities]) if entities else "  (none identified)"
        formatted_relations = "\n".join([f"  - {r}" for r in relations]) if relations else "  (none identified)"

        qtype_strategy = QTYPE_STRATEGIES.get(qtype, QTYPE_STRATEGIES.get("Query", ""))

        context = ANALYSIS_CONTEXT_TEMPLATE.format(
            qtype=qtype,
            formatted_entities=formatted_entities,
            formatted_relations=formatted_relations,
            qtype_strategy=qtype_strategy
        )

        if fewshot_examples and fewshot_examples.strip():
            context += FEWSHOT_EXAMPLES_TEMPLATE.format(
                qtype=qtype,
                fewshot_examples=fewshot_examples
            )

        context += ANALYSIS_CONTEXT_SUFFIX

        return context


# =============================================================================
# Main entry point for testing
# =============================================================================

if __name__ == "__main__":
    async def run_test():
        agent = KQAProAgent()
        try:
            answer = await agent.ask("Who is the director of Inception?")
            print(f"\n[KQAPro Agent Answer]\n{answer}")

            counts = agent.get_tool_call_counts()
            print(f"\n[Tool Call Counts]")
            for tool_name, count in sorted(counts.items(), key=lambda x: x[1], reverse=True):
                print(f"  {tool_name}: {count}x")

            summary = agent.get_tool_call_summary()
            print(f"\n[Tool Call Summary]")
            print(f"Total calls: {summary['total_calls']}")
            print(f"Total duration: {summary['total_duration_seconds']}s")
            for tool_name, stats in summary['tool_breakdown'].items():
                print(f"  {tool_name}: {stats['count']}x, avg {stats['avg_duration']}s")
        except Exception as e:
            print(f"Error in test run: {e}")
        finally:
            await agent.close()

    asyncio.run(run_test())
