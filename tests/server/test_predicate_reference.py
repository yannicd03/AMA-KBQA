"""Tests for the GetPredicateReference tool and the system-prompt split."""

import asyncio

from ama_kbqa.agents.sciqa_agent.prompts import SYSTEM_PROMPT
from ama_kbqa.server import sciqa_server as sci


def _call(domain=""):
    fn = getattr(sci.GetPredicateReference, "fn", sci.GetPredicateReference)
    return asyncio.run(fn(domain=domain))


def test_empty_domain_returns_full_reference():
    out = _call()
    for marker in ("P31", "P43133", "P41740", "P41923", "P23161", "compareContribution"):
        assert marker in out


def test_domain_filter_returns_only_matching_section():
    out = _call("energy")
    assert "P43133" in out and "P43135" in out
    assert "P41740" not in out  # chemistry stays out
    assert "P41923" not in out  # benchmarks stays out


def test_domain_filter_matches_substrings_of_topic():
    out = _call("energy sources in the comparison")
    assert "P43133" in out  # 'energy' matched
    assert "P5038" in out   # 'comparison' matched


def test_unknown_domain_lists_available_sections():
    out = _call("astrophysics")
    assert "No predicate reference section matches" in out
    assert "energy" in out and "FindPredicate" in out


def test_system_prompt_no_longer_embeds_domain_predicates():
    """The domain-specific IDs moved behind the tool; the prompt keeps only
    the core navigation predicates and points at GetPredicateReference."""
    for domain_only_id in ("P43133", "P41740", "P35148", "P41923", "P37458"):
        assert domain_only_id not in SYSTEM_PROMPT
    assert "P31" in SYSTEM_PROMPT  # core navigation stays
    assert "GetPredicateReference" in SYSTEM_PROMPT
