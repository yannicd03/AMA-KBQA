"""
KQAPro Agent using the generic KBQA framework.

This agent uses the KQAPro knowledge graph to answer natural language
questions about general domain facts.
"""

from __future__ import annotations
import asyncio
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

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
    SYNTHESIS_PROMPT_TEMPLATE_CONVERSATIONAL,
    JOURNAL_SUMMARY_ANSWER_PROMPT,
    TOOL_LOOP_GUIDANCE,
    GENERIC_LOOP_GUIDANCE,
    LOOP_INTERVENTION_TEMPLATE,
    GENERAL_GUIDANCE_TEMPLATE,
    TOOL_TIPS_TEMPLATE,
)


    # Tools always included regardless of question type
CORE_TOOLS = {
    "FindNode", "GetNodeSummary", "GetAttributeDetails", "GetRelationDetails",
    "ManageJournal", "GetJournalSummary", "RunSPARQL", "GetNodeLabel",
    "BatchGetNodeLabels", "FilterEntities", "FindByAttribute",
}

# Extra tools per question type (on top of CORE_TOOLS)
QTYPE_TOOL_MAP: Dict[str, set] = {
    "Count":                 {"CompareEntities", "FindByAttribute", "FindEntitiesByRelationPath",
                              "CountEntities", "CountUnion"},
    "Verify":                {"VerifyNumericCondition", "VerifyString", "CompareEntities", "FindByAttribute"},
    "SelectBetween":         {"CompareEntities", "GetSchemaForAttribute"},
    "SelectAmong":           {"CompareEntities", "GetSchemaForAttribute", "FindByAttribute"},
    "QueryAttr":             {"FindByAttribute", "GetSchemaForAttribute",
                              "GetQualifierValue", "GetQualifiersByPredicate"},
    "QueryAttrQualifier":    {"GetEdgeQualifiers", "GetQualifiersByPredicate",
                              "GetAttributeWithQualifiers", "TemporalAttributeQuery",
                              "GetSchemaForAttribute", "QualifierFilter",
                              "GetQualifierValue"},
    "QueryRelation":         {"ExploreNeighborhood", "FindEntitiesByRelationPath",
                              "GetRelationBetween", "GetQualifierValue",
                              "GetQualifiersByPredicate"},
    "QueryRelationQualifier":{"GetEdgeQualifiers", "GetQualifiersByPredicate",
                              "ExploreNeighborhood", "QualifierFilter",
                              "GetQualifierValue"},
    "QueryName":             {"FindByAttribute", "FindEntitiesByRelationPath",
                              "ExploreNeighborhood", "GetQualifierValue",
                              "GetQualifiersByPredicate"},
}


class KQAProAgent(BaseKBQAAgent):
    """
    KQAPro Agent for Knowledge Graph Question Answering.

    Uses the KQAPro MCP server to query the knowledge graph and
    answer natural language questions.
    """

    def __init__(self, name: str = "kqapro_agent", session_id: str = "default", use_fewshot: bool = True):
        """
        Initialize the KQAPro agent.

        Args:
            name: Agent name for tracing
            session_id: Session identifier
            use_fewshot: Whether to inject few-shot examples during classification
        """
        self._adapter = KQAProAdapter()
        super().__init__(name=name, session_id=session_id, use_fewshot=use_fewshot)

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

    def _extract_exact_attribute_constraints(self, query: str) -> List[Dict[str, str]]:
        """Extract reusable exact attribute/value constraints for KQAPro."""
        constraints: List[Dict[str, str]] = []

        def add(attribute_name: str, value: str) -> None:
            attribute_name = re.sub(r"\s+", " ", attribute_name.strip())
            value = value.strip().strip('"').strip("'").rstrip("?.!,;:")
            if not attribute_name or not value:
                return
            aliases = {
                "iscw": "ISWC",
                "iswc": "ISWC",
                "isni": "ISNI",
                "umls cui": "UMLS CUI",
                "icd-10-cm": "ICD-10-CM",
                "official name": "official name",
                "date of birth": "date of birth",
            }
            attribute_name = aliases.get(attribute_name.lower(), attribute_name)
            item = {"attribute_name": attribute_name, "value": value}
            if item not in constraints:
                constraints.append(item)

        # Exact quoted values: "has official name \"Land Force Command\"".
        for match in re.finditer(
            r"\b(?:has|with|whose|having|that has)\s+(?:the\s+)?"
            r"(?P<attr>official name|date of birth|IAB code|ICD-10-CM|UMLS CUI|ISWC|ISCW|ISNI|[A-Za-z][A-Za-z0-9 -]{0,40}? code)"
            r"\s+(?:is\s+|=|:)?\"(?P<value>[^\"]+)\"",
            query,
            flags=re.I,
        ):
            add(match.group("attr"), match.group("value"))

        # Exact unquoted values, ending at punctuation or a relative clause.
        for match in re.finditer(
            r"\b(?:has|with|whose|having|that has)\s+(?:the\s+)?"
            r"(?P<attr>official name|date of birth|IAB code|ICD-10-CM|UMLS CUI|ISWC|ISCW|ISNI|[A-Za-z][A-Za-z0-9 -]{0,40}? code)"
            r"\s+(?:is\s+|=|:)?(?P<value>[A-Za-z0-9][A-Za-z0-9 ._:/+-]*?)"
            r"(?=,|\?|;|$|\s+(?:and|or|that|which|whose|who)\b)",
            query,
            flags=re.I,
        ):
            add(match.group("attr"), match.group("value"))

        # Identifier wording that does not use "has/with": "known under ISWC T-...".
        for match in re.finditer(
            r"\b(?:known under|identified by|recorded under)\s+"
            r"(?P<attr>ISWC|ISCW|ISNI|UMLS CUI|ICD-10-CM|[A-Za-z][A-Za-z0-9 -]{0,40}? code)"
            r"\s+(?P<value>[A-Za-z0-9][A-Za-z0-9 ._:/+-]*?)"
            r"(?=,|\?|;|$|\s+(?:and|or|that|which|whose|who)\b)",
            query,
            flags=re.I,
        ):
            add(match.group("attr"), match.group("value"))

        # Entity disambiguation phrasing: "John Powell born 1936-03-10".
        birth_match = re.search(r"\bborn\s+(?P<value>\d{4}-\d{2}-\d{2})\b", query, flags=re.I)
        if birth_match:
            add("date of birth", birth_match.group("value"))

        return constraints

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
        extra = QTYPE_TOOL_MAP.get(qtype)
        if extra is None:
            # Unknown qtype → allow all tools
            return None
        return CORE_TOOLS | extra

    # =========================================================================
    # KQAPRO-SPECIFIC METHODS
    # =========================================================================

    def _load_fewshot_examples(
        self,
        max_per_type: int = 3,
        specific_qtype: Optional[str] = None
    ) -> str:
        """
        Load tool-trace few-shot examples from the fewshot-examples directory.

        Args:
            max_per_type: Maximum examples per question type
            specific_qtype: Load only this question type if specified

        Returns:
            Formatted tool-trace examples string
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
                        "answer": example.get("answer", ""),
                        "trace": example.get("trace", []),
                        "lesson": example.get("lesson", ""),
                        "pitfall": example.get("pitfall", ""),
                    })

            except (json.JSONDecodeError, Exception):
                continue

        if not all_examples:
            return ""

        formatted = ""
        for i, example in enumerate(all_examples, 1):
            answer = example["answer"]
            formatted += f'--- Example {i}: "{example["question"]}" -> {answer}\n'

            trace = example.get("trace", [])
            if trace:
                formatted += "Trace: "
                for j, step in enumerate(trace):
                    prefix = "       " if j > 0 else ""
                    tool = step.get("tool", "?")
                    args = step.get("args", "")
                    result = step.get("result", "")
                    formatted += f'{prefix}{tool}("{args}") -> {result}\n'

            if example["lesson"]:
                formatted += f'Lesson: {example["lesson"]}\n'
            if example.get("pitfall"):
                formatted += f'Pitfall: {example["pitfall"]}\n'

            formatted += "\n"

        return formatted

    def _load_general_guidance(self) -> str:
        """Load cross-type general guidance from _general.json."""
        repo_root = Path(__file__).resolve().parents[3]
        general_file = repo_root / "db" / "datasets" / "kqapro" / "fewshot-examples" / "_general.json"

        if not general_file.exists():
            return ""

        try:
            with open(general_file, "r", encoding="utf-8") as f:
                examples = json.load(f)

            if not examples:
                return ""

            lines = []
            for ex in examples[:5]:
                title = ex.get("title", "")
                guidance = ex.get("guidance", "")
                applies = ", ".join(ex.get("applies_to", ["all"]))
                lines.append(f"- [{applies}] {title}: {guidance}")

            return "\n".join(lines)
        except (json.JSONDecodeError, Exception):
            return ""

    def _load_tool_tips(self, qtype: Optional[str] = None) -> str:
        """Load tool tips from _tool_tips.json, optionally filtered by qtype relevance."""
        repo_root = Path(__file__).resolve().parents[3]
        tips_file = repo_root / "db" / "datasets" / "kqapro" / "fewshot-examples" / "_tool_tips.json"

        if not tips_file.exists():
            return ""

        try:
            with open(tips_file, "r", encoding="utf-8") as f:
                tips = json.load(f)

            if not tips:
                return ""

            lines = []
            for tip in tips[:10]:
                tool = tip.get("tool_name", "")
                pattern = tip.get("problem_pattern", "")
                guidance = tip.get("guidance", "")
                lines.append(f"- {tool} | When: {pattern} | Do: {guidance}")

            return "\n".join(lines)
        except (json.JSONDecodeError, Exception):
            return ""

    def _classify_question(self, question: str) -> Dict[str, Any]:
        """
        Classify a question and load relevant few-shot examples.
        Enriches base result with KQAPro-specific fewshot examples.
        """
        result = super()._classify_question(question)

        if self.use_fewshot:
            qtype = result.get("question_type", "Query")
            result["fewshot_examples"] = self._load_fewshot_examples(
                max_per_type=3,
                specific_qtype=qtype
            )
        else:
            result["fewshot_examples"] = ""

        return result

    def _build_analysis_context(
        self,
        qtype: str,
        entities: List[str],
        relations: List[str],
        fewshot_examples: str = "",
        query: str = "",
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

        if fewshot_examples and fewshot_examples.strip():
            context += FEWSHOT_EXAMPLES_TEMPLATE.format(
                qtype=qtype,
                fewshot_examples=fewshot_examples
            )

        # Append general guidance if available
        if self.use_fewshot:
            general_guidance = self._load_general_guidance()
            if general_guidance:
                context += GENERAL_GUIDANCE_TEMPLATE.format(
                    general_guidance=general_guidance
                )

            tool_tips = self._load_tool_tips(qtype=qtype)
            if tool_tips:
                context += TOOL_TIPS_TEMPLATE.format(
                    tool_tips=tool_tips
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
