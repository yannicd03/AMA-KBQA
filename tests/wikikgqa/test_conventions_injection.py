"""Tests for conditional (per-question) extended-convention injection.

The 2026-07-03 held-out A/B found that injecting the full extended rule bundle
(R7-R10) on EVERY question hurts overall score (0.7311 minimal vs 0.6833 full) -
most questions don't need those rules, so always-on injection is noise. As of
2026-07-15, `conventions="minimal"` (the default) instead injects three of those
rules CONDITIONALLY: per-question, only when the question text trips a narrow
trigger regex (see `get_conditional_conventions` in
`ama_kbqa/agents/wikidata_agent/prompts.py`, wired into
`WikidataAgent._build_analysis_context` in `agent.py`). `conventions="full"` is
unaffected: it keeps injecting the entire EXTENDED_CONVENTIONS bundle
unconditionally via the system prompt.

These tests avoid a full agent init (no MCP subprocess, no LLM client) by building
a minimal stub carrying only the attributes `_build_analysis_context` /
`_get_system_prompt` touch, then calling the unbound WikidataAgent methods on it -
the same pattern used in test_entity_search.py's
test_agent_sets_env_flag_and_injects_guidance.
"""

from __future__ import annotations

import importlib

from ama_kbqa.agents.wikidata_agent.prompts import (
    CURRENTLY_TRIGGER,
    EXTENDED_CONVENTIONS,
    LOCATION_TRIGGER,
    QUANTITY_TRIGGER,
    get_conditional_conventions,
)

_agent_mod = importlib.import_module("ama_kbqa.agents.wikidata_agent.agent")
WikidataAgent = _agent_mod.WikidataAgent


class _Stub:
    """Carries only the attributes `_build_analysis_context`/`_get_system_prompt` read."""

    _conventions = "minimal"
    _entity_search = False
    _build_analysis_context = WikidataAgent._build_analysis_context
    _get_system_prompt = WikidataAgent._get_system_prompt


def _stub(conventions: str = "minimal") -> _Stub:
    s = _Stub()
    s._conventions = conventions
    return s


# --- trigger regex matching (positive + negative) ---------------------------------


def test_quantity_trigger_matches_positive_examples():
    for q in [
        "How much does a blue whale weigh?",
        "What is the net worth of Elon Musk?",
        "How tall is the Eiffel Tower?",
        "How heavy is a baseball?",
    ]:
        assert QUANTITY_TRIGGER.search(q), q


def test_quantity_trigger_does_not_match_unrelated_questions():
    for q in [
        "Who is the president of France?",
        "When was Marie Curie born?",
        "Which countries border Germany?",
    ]:
        assert not QUANTITY_TRIGGER.search(q), q


def test_currently_trigger_matches_positive_examples():
    for q in [
        "Which countries currently have multiple capitals?",
        "Is the bridge still in use?",
        "What present-day nations were part of the empire?",
        "Which volcanoes are nowadays active?",
    ]:
        assert CURRENTLY_TRIGGER.search(q), q


def test_currently_trigger_does_not_match_unrelated_questions():
    for q in [
        "Which countries have a female head of state?",
        "What is the capital of Japan?",
    ]:
        assert not CURRENTLY_TRIGGER.search(q), q


def test_location_trigger_matches_positive_examples():
    for q in [
        "Which countries is the tepui found in?",
        "How many species of orchid are there in Brazil?",
        "Which countries have a monarchy?",
    ]:
        assert LOCATION_TRIGGER.search(q), q


def test_location_trigger_does_not_match_unrelated_questions():
    for q in [
        "Who directed Inception?",
        "What is the population of Canada?",
    ]:
        assert not LOCATION_TRIGGER.search(q), q


# --- get_conditional_conventions ----------------------------------------------------


def test_get_conditional_conventions_empty_for_non_matching_question():
    assert get_conditional_conventions("Who directed Inception?") == ""


def test_get_conditional_conventions_includes_quantity_rule_text():
    out = get_conditional_conventions("How much does a blue whale weigh?")
    assert "psn:Pxxx/wikibase:quantityAmount" in out
    assert "MINUS" not in out  # currently-rule must not leak in
    assert "P131*/P17" not in out  # location-rule must not leak in


def test_get_conditional_conventions_includes_currently_rule_and_its_warning():
    out = get_conditional_conventions("Which countries currently have multiple capitals?")
    assert "MINUS { ?x wdt:P582 ?e }" in out
    # the warning half must travel with the rule, not just the positive instruction
    assert "never add it" in out.lower()


def test_get_conditional_conventions_includes_location_rule_text():
    out = get_conditional_conventions("Which countries is the tepui found in?")
    assert "wdt:P131*/wdt:P17" in out


def test_get_conditional_conventions_can_combine_multiple_triggers():
    # "currently" + "which countries have" both fire on one question.
    out = get_conditional_conventions("Currently, which countries have a monarchy?")
    assert "MINUS { ?x wdt:P582 ?e }" in out
    assert "wdt:P131*/wdt:P17" in out


# --- injection into the outgoing per-question analysis context ----------------------


def test_analysis_context_injects_quantity_rule_for_matching_question():
    ctx = _stub("minimal")._build_analysis_context(
        "Query", [], [], query="How tall is the Eiffel Tower?"
    )
    assert "psn:Pxxx/wikibase:quantityAmount" in ctx


def test_analysis_context_omits_conditional_rules_for_non_matching_question():
    ctx = _stub("minimal")._build_analysis_context(
        "Query", [], [], query="Who directed Inception?"
    )
    assert "ADDITIONAL MODELING CONVENTIONS FOR THIS QUESTION" not in ctx
    assert "psn:Pxxx/wikibase:quantityAmount" not in ctx
    assert "MINUS { ?x wdt:P582 ?e }" not in ctx
    assert "wdt:P131*/wdt:P17" not in ctx


def test_analysis_context_never_injects_currently_rule_without_its_trigger():
    """R9 must never fire on a question that doesn't say currently/still/nowadays/
    present-day, even one that is semantically about an ongoing state - the A/B
    lesson (q81) was exactly this over-application."""
    ctx = _stub("minimal")._build_analysis_context(
        "Query", [], [], query="Which countries have a female head of state?"
    )
    assert "MINUS { ?x wdt:P582 ?e }" not in ctx
    assert "MINUS" not in ctx


def test_analysis_context_different_questions_get_different_injections_same_agent():
    """The same stub (agent) instance must inject different rules per call,
    keyed off the question text passed to that call."""
    stub = _stub("minimal")
    ctx_quantity = stub._build_analysis_context(
        "Query", [], [], query="How heavy is a baseball?"
    )
    ctx_plain = stub._build_analysis_context(
        "Query", [], [], query="Who directed Inception?"
    )
    assert "psn:Pxxx/wikibase:quantityAmount" in ctx_quantity
    assert "psn:Pxxx/wikibase:quantityAmount" not in ctx_plain


# --- conventions="full" is unaffected -------------------------------------------


def test_full_conventions_system_prompt_has_extended_bundle_unconditionally():
    prompt = _stub("full")._get_system_prompt()
    assert EXTENDED_CONVENTIONS.strip() in prompt


def test_full_conventions_analysis_context_does_not_duplicate_conditional_injection():
    """conventions='full' already gets the whole bundle in the system prompt;
    _build_analysis_context must not also append the conditional per-question
    snippet (that would just duplicate the quantity rule text)."""
    ctx = _stub("full")._build_analysis_context(
        "Query", [], [], query="How heavy is a baseball?"
    )
    assert "ADDITIONAL MODELING CONVENTIONS FOR THIS QUESTION" not in ctx


def test_minimal_conventions_system_prompt_excludes_extended_bundle():
    prompt = _stub("minimal")._get_system_prompt()
    assert EXTENDED_CONVENTIONS.strip() not in prompt
