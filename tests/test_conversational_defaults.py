"""Tests for the "How I found this" conversational step summary.

In conversational mode each agent's post-journal answer prompt asks for a
brief "How I found this" step summary; benchmark mode stays terse.
"""

from __future__ import annotations

import ama_kbqa.config as config
from ama_kbqa.agents.kqapro_agent.agent import KQAProAgent
from ama_kbqa.agents.sciqa_agent.agent import SciQAAgent


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
