"""Tests for the "Reproduce with SPARQL" answer contract.

Every user-facing (conversational) final answer must end with a fenced
```sparql block that retrieves the answer from the graph, so a booth visitor
can reproduce it. Benchmark mode must stay byte-identical: the paper's numbers
were measured with the terse prompts, so the section text must not leak into
them.
"""

from __future__ import annotations

import ama_kbqa.config as config
from ama_kbqa.agents.kqapro_agent.agent import KQAProAgent
from ama_kbqa.agents.sciqa_agent.agent import SciQAAgent

SECTION_TITLE = "Reproduce with SPARQL:"


def _effective_system_prompt(agent_cls, mode, monkeypatch):
    # Both methods read get_synthesis_mode() but no instance state, so bypass
    # the production __init__ entirely (mirrors test_conversational_defaults).
    agent = agent_cls.__new__(agent_cls)
    monkeypatch.setattr(config, "get_synthesis_mode", lambda: mode)
    return agent._get_effective_system_prompt()


def _answer_prompt(agent_cls, mode, monkeypatch):
    agent = agent_cls.__new__(agent_cls)
    monkeypatch.setattr(config, "get_synthesis_mode", lambda: mode)
    return agent._get_journal_summary_answer_prompt()


# ---------------------------------------------------------------------------
# Conversational mode carries the full contract
# ---------------------------------------------------------------------------

def test_conversational_system_prompt_states_the_section_contract(monkeypatch):
    for agent_cls in (KQAProAgent, SciQAAgent):
        prompt = _effective_system_prompt(agent_cls, "conversational", monkeypatch)
        assert SECTION_TITLE in prompt
        # Exactly one fenced sparql block, and the fence language is named.
        assert "EXACTLY ONE fenced code" in prompt
        assert "sparql" in prompt
        # Verification policy: one attempt, explicit fallback, never omitted.
        assert "ONE verification attempt" in prompt
        assert "(not executed)" in prompt
        assert "Verified against the knowledge graph." in prompt
        assert "NEVER omit this section" in prompt
        assert "no query could be formed" in prompt
        # Shape rules for the three answer kinds the demo can produce.
        assert "ASK query" in prompt
        assert "COUNT" in prompt


def test_conversational_system_prompt_carries_the_real_uri_scheme(monkeypatch):
    """The query must be runnable as-is, so the graph's real scheme is taught."""
    kqapro = _effective_system_prompt(KQAProAgent, "conversational", monkeypatch)
    assert "PREFIX ex:   <http://kqapro.org/entity/>" in kqapro
    assert "PREFIX prop: <http://kqapro.org/property/>" in kqapro
    assert "PREFIX attr: <http://kqapro.org/attribute/>" in kqapro
    assert "FROM <http://kqapro.org/kb>" in kqapro
    assert "RunSPARQL" in kqapro

    sciqa = _effective_system_prompt(SciQAAgent, "conversational", monkeypatch)
    assert "PREFIX orkgr: <http://orkg.org/orkg/resource/>" in sciqa
    assert "PREFIX orkgp: <http://orkg.org/orkg/predicate/>" in sciqa
    assert "GRAPH <http://sciqa.org/kg>" in sciqa
    assert "RunORKGSPARQL" in sciqa


def test_conversational_journal_summary_prompt_repeats_the_section(monkeypatch):
    """The journal-summary step forbids tool calls, so it must still demand the
    section and name the "(not executed)" fallback."""
    for agent_cls in (KQAProAgent, SciQAAgent):
        prompt = _answer_prompt(agent_cls, "conversational", monkeypatch)
        assert SECTION_TITLE in prompt
        assert "```sparql" in prompt
        assert "(not executed)" in prompt
        assert "no query could be formed" in prompt


# ---------------------------------------------------------------------------
# Benchmark mode is untouched
# ---------------------------------------------------------------------------

def test_benchmark_system_prompt_has_no_section_text(monkeypatch):
    for agent_cls in (KQAProAgent, SciQAAgent):
        bench = _effective_system_prompt(agent_cls, "benchmark", monkeypatch)
        assert SECTION_TITLE not in bench
        assert "```sparql" not in bench
        assert "(not executed)" not in bench
        assert "REPRODUCE-WITH-SPARQL DETAILS" not in bench
        # Byte-identical to the subclass prompt: nothing at all is appended.
        assert bench == agent_cls.__new__(agent_cls)._get_system_prompt()


def test_benchmark_journal_summary_prompt_has_no_section_text(monkeypatch):
    for agent_cls in (KQAProAgent, SciQAAgent):
        bench = _answer_prompt(agent_cls, "benchmark", monkeypatch)
        assert SECTION_TITLE not in bench
        assert "sparql" not in bench.lower()


# ---------------------------------------------------------------------------
# The tool the prompt asks for has to survive per-qtype tool filtering
# ---------------------------------------------------------------------------

def test_raw_sparql_tool_survives_qtype_filtering_for_every_question_type():
    """The contract demands a verification run for every conversational answer,
    so the raw SPARQL tool must be in the filtered tool set of EVERY qtype, not
    just the ones whose strategy happens to mention it."""
    from ama_kbqa.agents.kqapro_agent.agent import QTYPE_TOOL_MAP as KQAPRO_QTYPES
    from ama_kbqa.agents.sciqa_agent.agent import QTYPE_TOOL_MAP as SCIQA_QTYPES

    kqapro = KQAProAgent.__new__(KQAProAgent)
    for qtype in KQAPRO_QTYPES:
        assert "RunSPARQL" in kqapro._get_allowed_tools_for_qtype(qtype), qtype
    # Unknown qtype keeps the full tool set.
    assert kqapro._get_allowed_tools_for_qtype("NotAQtype") is None

    sciqa = SciQAAgent.__new__(SciQAAgent)
    for qtype in SCIQA_QTYPES:
        assert "RunORKGSPARQL" in sciqa._get_allowed_tools_for_qtype(qtype), qtype
    assert sciqa._get_allowed_tools_for_qtype("General") is None


def test_sciqa_hint_degrades_when_raw_sparql_is_gated_off(monkeypatch):
    """With RunORKGSPARQL denied the URI scheme must still reach the model —
    only the verification instruction changes."""
    monkeypatch.setenv("AMA_SCIQA_DISABLE_RAW_SPARQL", "1")
    hint = SciQAAgent.__new__(SciQAAgent)._get_sparql_reproduction_hint()
    assert "GRAPH <http://sciqa.org/kg>" in hint
    assert "Raw SPARQL execution is disabled" in hint
    assert "(not executed)" in hint
