"""Tests for the "Reproduce with SPARQL" answer contract.

Every user-facing (conversational) final answer must end with a fenced
```sparql block that retrieves the answer from the graph, so a booth visitor
can reproduce it. Benchmark mode must stay byte-identical: the paper's numbers
were measured with the terse prompts, so the section text must not leak into
them.
"""

from __future__ import annotations

import pytest

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


# ---------------------------------------------------------------------------
# Synthesis is a SEPARATE final-answer path and owes the same section
# ---------------------------------------------------------------------------

def _synthesis_template(agent_cls, mode, monkeypatch):
    """The synthesis user prompt as base_agent actually assembles it: the
    format()-ed template plus the brace-heavy hint appended afterwards."""
    agent = agent_cls.__new__(agent_cls)
    monkeypatch.setattr(config, "get_synthesis_mode", lambda: mode)
    return agent._get_synthesis_prompt_template() + agent._get_synthesis_sparql_hint()


def _synthesis_system_prompt(agent_cls, mode, monkeypatch):
    agent = agent_cls.__new__(agent_cls)
    monkeypatch.setattr(config, "get_synthesis_mode", lambda: mode)
    return agent._get_synthesis_system_prompt()


def test_conversational_synthesis_template_states_the_section_contract(monkeypatch):
    """Regression: the max-tool-call / max-iteration exits answer through
    synthesis, not the agent loop, and used to drop the section entirely."""
    for agent_cls in (KQAProAgent, SciQAAgent):
        template = _synthesis_template(agent_cls, "conversational", monkeypatch)
        assert SECTION_TITLE in template
        assert "```sparql" in template
        # No tools in this step, so the default marker is "(not executed)".
        assert "(not executed)" in template
        assert "NO tools in this step" in template
        assert "no query could be formed" in template
        # The placeholders the synthesis call formats must survive.
        assert "{journal_summary}" in template
        assert "{query}" in template


def test_conversational_synthesis_template_carries_the_real_uri_scheme(monkeypatch):
    kqapro = _synthesis_template(KQAProAgent, "conversational", monkeypatch)
    assert "PREFIX ex:   <http://kqapro.org/entity/>" in kqapro
    assert "FROM <http://kqapro.org/kb>" in kqapro

    sciqa = _synthesis_template(SciQAAgent, "conversational", monkeypatch)
    assert "PREFIX orkgr: <http://orkg.org/orkg/resource/>" in sciqa
    assert "GRAPH <http://sciqa.org/kg>" in sciqa


def test_conversational_synthesis_system_prompt_states_the_section_contract(monkeypatch):
    """The system message covers the re-synthesis re-prompt too, which only
    says "answer using the specific values found"."""
    for agent_cls in (KQAProAgent, SciQAAgent):
        prompt = _synthesis_system_prompt(agent_cls, "conversational", monkeypatch)
        assert SECTION_TITLE in prompt
        assert "```sparql" in prompt
        assert "(not executed)" in prompt
        assert "including any rewrite" in prompt


def test_benchmark_synthesis_template_has_no_section_text(monkeypatch):
    for agent_cls in (KQAProAgent, SciQAAgent):
        template = _synthesis_template(agent_cls, "benchmark", monkeypatch)
        assert SECTION_TITLE not in template
        assert "sparql" not in template.lower()
        assert "(not executed)" not in template


def test_benchmark_synthesis_system_prompt_has_no_section_text(monkeypatch):
    for agent_cls in (KQAProAgent, SciQAAgent):
        prompt = _synthesis_system_prompt(agent_cls, "benchmark", monkeypatch)
        assert SECTION_TITLE not in prompt
        assert "sparql" not in prompt.lower()


def test_synthesis_hint_tells_the_model_it_cannot_run_the_query():
    """The loop variant asks for a verification run; the synthesis variant must
    not, because that call is made without any tools."""
    from ama_kbqa.agents.kqapro_agent.prompts import (
        SPARQL_REPRODUCTION_HINT as KQAPRO_LOOP,
        SPARQL_REPRODUCTION_HINT_SYNTHESIS as KQAPRO_SYNTH,
    )
    from ama_kbqa.agents.sciqa_agent.prompts import (
        SPARQL_REPRODUCTION_HINT as SCIQA_LOOP,
        SPARQL_REPRODUCTION_HINT_SYNTHESIS as SCIQA_SYNTH,
    )

    assert "single verification run" in KQAPRO_LOOP
    assert "single verification run" not in KQAPRO_SYNTH
    assert "NO tools in this step" in KQAPRO_SYNTH

    assert "single verification run" in SCIQA_LOOP
    assert "single verification run" not in SCIQA_SYNTH
    assert "NO tools in this step" in SCIQA_SYNTH

    # Both variants still teach the same URI scheme.
    for loop, synth in ((KQAPRO_LOOP, KQAPRO_SYNTH), (SCIQA_LOOP, SCIQA_SYNTH)):
        assert "URI scheme:" in loop and "URI scheme:" in synth


def test_synthesis_template_still_formats_with_its_two_fields(monkeypatch):
    """Regression: the SPARQL hint is full of literal `{` / `}` from the query
    examples. Appending it to the template BEFORE str.format made synthesis
    raise KeyError: '\\n    GRAPH <http' and return no answer at all. The hint
    must therefore never be part of the formatted string."""
    for agent_cls in (KQAProAgent, SciQAAgent):
        agent = agent_cls.__new__(agent_cls)
        monkeypatch.setattr(config, "get_synthesis_mode", lambda: "conversational")

        # The template alone formats cleanly...
        formatted = agent._get_synthesis_prompt_template().format(
            journal_summary="Discovered values: Q25191 / R187008",
            query="Who directed Inception?",
        )
        assert "Who directed Inception?" in formatted
        assert "R187008" in formatted

        # ...and the hint, which cannot be formatted, is appended after.
        hint = agent._get_synthesis_sparql_hint()
        assert "{" in hint  # literal SPARQL braces, i.e. genuinely unformattable
        with pytest.raises((KeyError, IndexError, ValueError)):
            hint.format(journal_summary="x", query="y")

        assert SECTION_TITLE in (formatted + hint) or "URI scheme:" in hint


def test_benchmark_synthesis_hint_is_empty(monkeypatch):
    for agent_cls in (KQAProAgent, SciQAAgent):
        agent = agent_cls.__new__(agent_cls)
        monkeypatch.setattr(config, "get_synthesis_mode", lambda: "benchmark")
        assert agent._get_synthesis_sparql_hint() == ""
