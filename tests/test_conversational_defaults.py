"""Tests for conversational-mode defaults and the "How I found this" step summary.

Two behaviours:
1. synthesis_mode defaults to "conversational" (frontend standard) when the
   config omits it or sets an unknown value.
2. In conversational mode each agent's post-journal answer prompt asks for a
   brief "How I found this" step summary; benchmark mode stays terse.
"""

from __future__ import annotations

import ama_kbqa.config as config
from ama_kbqa.agents.kqapro_agent.agent import KQAProAgent
from ama_kbqa.agents.sciqa_agent.agent import SciQAAgent


def test_synthesis_mode_defaults_to_conversational(monkeypatch):
    monkeypatch.setattr(config, "load_config", lambda: {})
    assert config.get_synthesis_mode() == "conversational"


def test_unknown_synthesis_mode_falls_back_to_conversational(monkeypatch):
    monkeypatch.setattr(
        config, "load_config", lambda: {"synthesis": {"synthesis_mode": "bogus"}}
    )
    assert config.get_synthesis_mode() == "conversational"


def _answer_prompt(agent_cls, mode, monkeypatch):
    # The method reads get_synthesis_mode() but no instance state, so bypass
    # the production __init__ entirely.
    agent = agent_cls.__new__(agent_cls)
    monkeypatch.setattr(config, "get_synthesis_mode", lambda: mode)
    return agent._get_journal_summary_answer_prompt()


def test_kqapro_conversational_prompt_requests_step_summary(monkeypatch):
    conv = _answer_prompt(KQAProAgent, "conversational", monkeypatch)
    assert "How I found this" in conv
    bench = _answer_prompt(KQAProAgent, "benchmark", monkeypatch)
    assert "How I found this" not in bench


def test_sciqa_conversational_prompt_requests_step_summary(monkeypatch):
    conv = _answer_prompt(SciQAAgent, "conversational", monkeypatch)
    assert "How I found this" in conv
    bench = _answer_prompt(SciQAAgent, "benchmark", monkeypatch)
    assert "How I found this" not in bench


def _effective_system_prompt(agent_cls, mode, monkeypatch):
    agent = agent_cls.__new__(agent_cls)
    monkeypatch.setattr(config, "get_synthesis_mode", lambda: mode)
    return agent._get_effective_system_prompt()


def test_system_prompt_carries_how_i_found_this_in_conversational_mode(monkeypatch):
    # Regression: some models (e.g. gemma4) answer without calling
    # GetJournalSummary, so the explanation must live in the system prompt — not
    # only in the post-GetJournalSummary injection — to reach every model.
    for agent_cls in (KQAProAgent, SciQAAgent):
        conv = _effective_system_prompt(agent_cls, "conversational", monkeypatch)
        assert "How I found this" in conv
        # The subclass system prompt is still present (directive is appended).
        base = agent_cls.__new__(agent_cls)._get_system_prompt()
        assert base in conv

        bench = _effective_system_prompt(agent_cls, "benchmark", monkeypatch)
        assert "How I found this" not in bench
        assert bench == base
