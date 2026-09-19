"""Agent metadata and factory for the Streamlit frontend."""

from __future__ import annotations
from typing import Any


# Picker entries that resolve to the Orchestrator, mapped to the `federation`
# flag passed to its constructor. Router is the paper's single-dispatch
# behaviour (exactly one specialist per question); Federated is the
# experimental multi-specialist path (the router may pick up to
# `max_specialists` specialists, run them concurrently, and fuse their
# answers with one LLM call).
ORCHESTRATOR_MODES: dict[str, bool] = {
    "Orchestrator (Router)": False,
    "Orchestrator (Federated)": True,
}


def is_orchestrator(name: str) -> bool:
    """True when ``name`` is one of the Orchestrator picker entries."""
    return name in ORCHESTRATOR_MODES


AGENT_INFO = {
    "Orchestrator (Router)": {
        "tagline": "Picks the single best specialist for your question, as evaluated in the paper.",
        "description": "Routes questions to the best specialist agent (KQAPro or SciQA) automatically.",
        "databases": "KQAPro KG + ORKG (via sub-agents)",
        "tools": "Delegates to sub-agents",
    },
    "Orchestrator (Federated)": {
        "tagline": (
            "Experimental: may ask both specialists in parallel and combine "
            "their answers. Slower and uses about twice the tokens; the "
            "paper's numbers are single-dispatch."
        ),
        "description": (
            "Routes questions to one or more specialist agents (KQAPro and/or "
            "SciQA); when more than one is picked they run concurrently and "
            "their answers are fused into one response."
        ),
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
    # Same suggestions as before: single-domain questions the router handles
    # by picking exactly one specialist.
    "Orchestrator (Router)": {
        ":blue[:material/movie:] Director of Inception": "Who is the director of Inception?",
        ":green[:material/science:] COVID-19 research": "What research contributions address COVID-19 detection?",
        ":orange[:material/location_on:] Einstein's birthplace": "In which city was Albert Einstein born?",
    },
    # Federated suggestions are anchored on topics that genuinely exist in
    # BOTH graphs, so each half of the fused answer is grounded rather than
    # improvised. Breast cancer and epilepsy are diseases KQAPro knows as
    # Wikidata entities (with `notable_people_with_this_condition` links) and
    # that ORKG covers as research problems carrying real contributions and
    # papers. Both were verified end to end against the demo's own data: the
    # router fans out to kqapro_agent + sciqa_agent and the fusion step has
    # two grounded answers to combine. Expect roughly 2.5 to 3 minutes per
    # federated answer, since the slower specialist gates the whole run.
    #
    # Do not swap in a topic without checking both graphs first. Questions
    # phrased around ORKG *models* or *benchmarks* tempt the specialist into
    # answering from the LLM's own knowledge, and topics that postdate the
    # KQAPro Wikidata snapshot (COVID-19, for one) have no general-knowledge
    # half at all.
    #
    # The third entry is a clearly single-domain question, to show the router
    # still picks just one specialist when federation isn't warranted.
    "Orchestrator (Federated)": {
        ":violet[:material/hub:] Breast cancer, people and papers": (
            "Which notable people have had breast cancer, and what research "
            "contributions address breast cancer?"
        ),
        ":violet[:material/hub:] Epilepsy and antiepileptic drugs": (
            "Which notable people have had epilepsy, and what research "
            "contributions address the effectiveness of antiepileptic drugs?"
        ),
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
        name: One of 'Orchestrator (Router)', 'Orchestrator (Federated)',
            'KQAPro', 'SciQA'.

    Returns:
        An agent instance with an async ask() method.
    """
    if is_orchestrator(name):
        from ama_kbqa.agents.orchestrator_agent.agent import Orchestrator
        return Orchestrator(federation=ORCHESTRATOR_MODES[name])
    elif name == "KQAPro":
        from ama_kbqa.agents.kqapro_agent.agent import KQAProAgent
        return KQAProAgent()
    elif name == "SciQA":
        from ama_kbqa.agents.sciqa_agent.agent import SciQAAgent
        return SciQAAgent()
    else:
        raise ValueError(f"Unknown agent: {name}")
