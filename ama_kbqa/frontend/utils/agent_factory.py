"""Agent metadata and factory for the Streamlit frontend."""

from __future__ import annotations
from typing import Any


AGENT_INFO = {
    "Orchestrator": {
        "tagline": "Auto-routes to the best specialist. Use when you're not sure which agent fits.",
        "description": "Routes questions to the best specialist agent (KQAPro or SciQA) automatically.",
        "databases": "KQAPro KG + ORKG (via sub-agents)",
        "tools": "Delegates to sub-agents",
    },
    "KQAPro": {
        "tagline": "Facts from a general knowledge graph (films, places, people). Use for factual lookups.",
        "description": "Answers factual questions over the KQAPro knowledge graph (movies, geography, science facts).",
        "databases": "Qdrant (entities/relations) + Virtuoso (SPARQL)",
        "tools": "29 MCP tools",
    },
    "SciQA": {
        "tagline": "Scientific research via the ORKG. Use for research papers and contributions.",
        "description": "Answers scientific research questions using the Open Research Knowledge Graph (ORKG).",
        "databases": "Qdrant (ORKG entities/relations) + Virtuoso (ORKG SPARQL)",
        "tools": "27 MCP tools",
    },
}

AGENT_SUGGESTIONS = {
    "Orchestrator": {
        ":blue[:material/movie:] Director of Inception": "Who is the director of Inception?",
        ":green[:material/science:] COVID-19 research": "What research contributions address COVID-19 detection?",
        ":orange[:material/location_on:] Einstein's birthplace": "In which city was Albert Einstein born?",
    },
    "KQAPro": {
        ":blue[:material/movie:] Director of Inception": "Who is the director of Inception?",
        ":green[:material/music_note:] Heavy metal bands like Queen": "How many heavy metal groups are in the genre of Queen?",
        ":orange[:material/location_on:] Einstein's birthplace": "In which city was Albert Einstein born?",
    },
    "SciQA": {
        ":green[:material/science:] COVID-19 research": "What research contributions address COVID-19 detection?",
        ":blue[:material/biotech:] Machine learning benchmarks": "What benchmarks are used for evaluating machine learning models?",
        ":orange[:material/article:] NLP contributions": "What are the main contributions in natural language processing?",
    },
}


def create_agent(name: str) -> Any:
    """Create an agent instance by name.

    Args:
        name: One of 'Orchestrator', 'KQAPro', 'SciQA'

    Returns:
        An agent instance with an async ask() method.
    """
    if name == "Orchestrator":
        from ama_kbqa.agents.orchestrator_agent.agent import Orchestrator
        return Orchestrator()
    elif name == "KQAPro":
        from ama_kbqa.agents.kqapro_agent.agent import KQAProAgent
        return KQAProAgent()
    elif name == "SciQA":
        from ama_kbqa.agents.sciqa_agent.agent import SciQAAgent
        return SciQAAgent()
    else:
        raise ValueError(f"Unknown agent: {name}")
