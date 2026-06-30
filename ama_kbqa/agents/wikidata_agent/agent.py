"""Wikidata agent: the existing BaseKBQAAgent tool-loop applied to Wikidata.

This reuses the framework's agentic machinery (tool loop, journal, synthesis)
and points it at ``wikidata_server.py`` so the agent can DISCOVER properties and
paths by probing the live challenge endpoint. The synthesis step is overridden to
emit a SPARQL query rather than a natural-language answer.

Two knobs differ from the KQAPro agent:
  * model/provider can be overridden per-instance (config.toml's chat_model is
    stale, and we benchmark several models);
  * fast-path and KQAPro fewshots are disabled (no KQAPro schema here).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from ama_kbqa.agents.wikidata_agent.prompts import (
    ANALYSIS_CONTEXT,
    EXTENDED_CONVENTIONS,
    SYNTHESIS_PROMPT_TEMPLATE,
    SYNTHESIS_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
)
from ama_kbqa.framework.base_agent import BaseKBQAAgent
from ama_kbqa.framework.config import (
    GraphConfig,
    KnowledgeGraphConfig,
    NamespaceConfig,
    VectorConfig,
)
from ama_kbqa.wikikgqa.endpoint import WIKIDATA_PREFIXES, resolve_endpoint

_WD = "http://www.wikidata.org/"


class WikidataAgent(BaseKBQAAgent):
    """Agentic SPARQL generator for Wikidata, built on the shared framework."""

    def __init__(
        self,
        name: str = "wikidata_agent",
        session_id: str = "default",
        model: Optional[str] = None,
        provider: Optional[str] = None,
        conventions: str = "full",
    ):
        # conventions: "full" includes the extended R7-R10 modeling rules; "minimal"
        # uses only R1-R6 (for held-out A/B testing of whether R7-R10 generalize).
        self._conventions = conventions
        super().__init__(name=name, session_id=session_id, use_fewshot=False)

        # config.toml's chat_model is stale; allow a per-run override and keep
        # the text-tool-call mode consistent with the actual model in use.
        if provider:
            from ama_kbqa.config import _create_client

            self.client = _create_client(provider, "chat")
        if model:
            self.model = model
            try:
                from ama_kbqa.framework.text_tool_calls import needs_text_tool_calls

                self._text_tool_call_mode = needs_text_tool_calls(self.model)
            except Exception:
                pass

    # --- required abstract methods ---
    def get_config(self) -> KnowledgeGraphConfig:
        namespaces = NamespaceConfig(
            entity_prefix=f"{_WD}entity/",
            property_prefix=f"{_WD}prop/direct/",
            attribute_prefix=f"{_WD}prop/direct/",
            qualifier_prefix=f"{_WD}prop/qualifier/",
            class_prefix=f"{_WD}entity/",
            sparql_prefixes=WIKIDATA_PREFIXES,
        )
        vectors = VectorConfig(
            entity_collection="wikidata-entities",   # placeholders; no Qdrant here
            relation_collection="wikidata-properties",
        )
        graph = GraphConfig(
            endpoint=resolve_endpoint(),
            supports_reification=True,
            has_temporal_data=True,
            timeout_ms=90000,
        )
        return KnowledgeGraphConfig(
            name="Wikidata",
            code="wikidata",
            namespaces=namespaces,
            vectors=vectors,
            graph=graph,
            domain_settings={
                "enable_fast_path": False,   # always explore; no KQAPro fast path
                # Tool-call budget the agent is told about (see prompt). At the cap the
                # framework FORCES synthesis (emits the best validated query) rather than
                # letting the model refine until the wall-clock cancels it with nothing.
                "max_tool_calls": 20,
                "max_iterations": 24,
                "sparql_cap": 20,
                "find_resource_cap": 12,
                "context_limit": 100000,
            },
        )

    def get_mcp_server_path(self) -> str:
        ama_kbqa_root = Path(__file__).resolve().parents[2]
        return str(ama_kbqa_root / "server" / "wikidata_server.py")

    # --- prompt overrides ---
    def _get_system_prompt(self) -> str:
        if getattr(self, "_conventions", "full") == "minimal":
            return SYSTEM_PROMPT
        return SYSTEM_PROMPT + "\n" + EXTENDED_CONVENTIONS

    def _get_synthesis_prompt_template(self) -> str:
        return SYNTHESIS_PROMPT_TEMPLATE

    def _get_synthesis_system_prompt(self) -> str:
        return SYNTHESIS_SYSTEM_PROMPT

    def _get_allowed_tools_for_qtype(self, qtype: str) -> Optional[set]:
        return None  # all exploration tools available for every question

    # --- bypass the KQAPro-style classifier: no LLM call, derive ids from text ---
    def _classify_question(self, question: str) -> Dict[str, Any]:
        return {
            "question_type": "Query",
            "entities": list(dict.fromkeys(re.findall(r"Q\d+", question))),
            "relations": list(dict.fromkeys(re.findall(r"P\d+", question))),
            "fewshot_examples": "",
        }

    def _build_analysis_context(
        self,
        qtype: str,
        entities: List[str],
        relations: List[str],
        fewshot_examples: str = "",
        query: str = "",
    ) -> str:
        ents = ", ".join(entities) if entities else "(none given)"
        rels = ", ".join(relations) if relations else "(none given)"
        return f"Given entity ids: {ents}\nGiven property ids: {rels}\n\n{ANALYSIS_CONTEXT}"
