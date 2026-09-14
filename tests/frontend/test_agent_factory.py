"""Tests for the agent picker's metadata and factory.

The picker offers two Orchestrator entries — "Orchestrator (Router)" (the
paper's single-dispatch behaviour) and "Orchestrator (Federated)"
(experimental multi-specialist dispatch + fusion) — plus the two direct
specialists (KQAPro, SciQA).
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from ama_kbqa.frontend.utils.agent_factory import (
    AGENT_INFO,
    AGENT_SUGGESTIONS,
    ORCHESTRATOR_MODES,
    create_agent,
    is_orchestrator,
)


class TestOrchestratorModes:
    def test_mapping_has_exactly_router_and_federated(self):
        assert ORCHESTRATOR_MODES == {
            "Orchestrator (Router)": False,
            "Orchestrator (Federated)": True,
        }

    @pytest.mark.parametrize("name,expected", [
        ("Orchestrator (Router)", True),
        ("Orchestrator (Federated)", True),
        ("KQAPro", False),
        ("SciQA", False),
        ("Orchestrator", False),  # stale pre-split name
        ("", False),
        (None, False),
    ])
    def test_is_orchestrator(self, name, expected):
        assert is_orchestrator(name) is expected


class TestAgentInfoAndSuggestions:
    def test_agent_info_has_four_entries(self):
        assert set(AGENT_INFO.keys()) == {
            "Orchestrator (Router)",
            "Orchestrator (Federated)",
            "KQAPro",
            "SciQA",
        }

    def test_agent_info_and_suggestions_have_identical_keys(self):
        assert set(AGENT_INFO.keys()) == set(AGENT_SUGGESTIONS.keys())

    def test_every_agent_info_entry_has_a_tagline(self):
        for name, meta in AGENT_INFO.items():
            assert meta.get("tagline"), f"{name} missing a tagline"

    def test_federated_tagline_flags_experimental_and_cost(self):
        tagline = AGENT_INFO["Orchestrator (Federated)"]["tagline"]
        assert "Experimental" in tagline
        assert "twice the tokens" in tagline

    def test_router_tagline_matches_paper_single_dispatch(self):
        tagline = AGENT_INFO["Orchestrator (Router)"]["tagline"]
        assert "single best specialist" in tagline
        assert "paper" in tagline

    def test_router_suggestions_are_single_domain_questions(self):
        # Unchanged from the pre-split Orchestrator suggestions.
        assert AGENT_SUGGESTIONS["Orchestrator (Router)"] == {
            ":blue[:material/movie:] Director of Inception": "Who is the director of Inception?",
            ":green[:material/science:] COVID-19 research": "What research contributions address COVID-19 detection?",
            ":orange[:material/location_on:] Einstein's birthplace": "In which city was Albert Einstein born?",
        }

    def test_federated_suggestions_include_a_dual_domain_and_a_single_domain_question(self):
        suggestions = AGENT_SUGGESTIONS["Orchestrator (Federated)"]
        assert len(suggestions) >= 2
        questions = list(suggestions.values())
        # At least one question plausibly needs both the KQAPro general
        # knowledge graph AND the SciQA/ORKG research graph...
        dual_domain = [
            q for q in questions
            if ("research" in q.lower() or "benchmark" in q.lower() or "cite" in q.lower())
            and any(term in q for term in ("Einstein", "Inception", "director", "born"))
        ]
        assert dual_domain, questions
        # ...plus at least one clearly single-domain question, to show the
        # router can still pick just one specialist under federation.
        assert "In which city was Albert Einstein born?" in questions

    def test_kqapro_and_sciqa_suggestions_unchanged(self):
        assert AGENT_SUGGESTIONS["KQAPro"] == {
            ":blue[:material/movie:] Director of Inception": "Who is the director of Inception?",
            ":green[:material/music_note:] Heavy metal bands like Queen": "How many heavy metal groups are in the genre of Queen?",
            ":orange[:material/location_on:] Einstein's birthplace": "In which city was Albert Einstein born?",
        }
        assert AGENT_SUGGESTIONS["SciQA"] == {
            ":green[:material/science:] COVID-19 research": "What research contributions address COVID-19 detection?",
            ":blue[:material/biotech:] Machine learning benchmarks": "What benchmarks are used for evaluating machine learning models?",
            ":orange[:material/article:] NLP contributions": "What are the main contributions in natural language processing?",
        }


class _FakeOrchestrator:
    """Records constructor kwargs instead of building a real Orchestrator.

    The real `Orchestrator` on this branch does not yet accept a
    `federation` kwarg (that lands via a parallel branch), so tests must not
    instantiate it directly for the federated path.
    """

    instances: list["_FakeOrchestrator"] = []

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        _FakeOrchestrator.instances.append(self)


class TestCreateAgent:
    def setup_method(self):
        _FakeOrchestrator.instances = []

    @pytest.mark.parametrize("name,expected_federation", [
        ("Orchestrator (Router)", False),
        ("Orchestrator (Federated)", True),
    ])
    def test_create_agent_passes_federation_flag(self, name, expected_federation):
        with patch(
            "ama_kbqa.agents.orchestrator_agent.agent.Orchestrator",
            _FakeOrchestrator,
        ):
            agent = create_agent(name)
        assert isinstance(agent, _FakeOrchestrator)
        assert agent.kwargs == {"federation": expected_federation}

    def test_kqapro_and_sciqa_construction_unchanged(self):
        from ama_kbqa.agents.kqapro_agent.agent import KQAProAgent
        from ama_kbqa.agents.sciqa_agent.agent import SciQAAgent

        assert isinstance(create_agent("KQAPro"), KQAProAgent)
        assert isinstance(create_agent("SciQA"), SciQAAgent)

    def test_unknown_agent_raises(self):
        with pytest.raises(ValueError):
            create_agent("Orchestrator")  # stale pre-split name
        with pytest.raises(ValueError):
            create_agent("NotAnAgent")
