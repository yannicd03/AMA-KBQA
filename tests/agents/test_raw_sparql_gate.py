"""Tests for the AMA_SCIQA_DISABLE_RAW_SPARQL gate.

The SciQA agent drops RunORKGSPARQL from the advertised tool set when the env
flag is truthy; off by default it changes nothing. The base-agent denylist hook
returns an empty set by default.
"""

import os

import pytest


@pytest.fixture(autouse=True)
def _clear_flag():
    saved = os.environ.pop("AMA_SCIQA_DISABLE_RAW_SPARQL", None)
    yield
    os.environ.pop("AMA_SCIQA_DISABLE_RAW_SPARQL", None)
    if saved is not None:
        os.environ["AMA_SCIQA_DISABLE_RAW_SPARQL"] = saved


def _denied_names():
    # Resolve the bound method without constructing the agent (which needs an
    # MCP/LLM client): call the function on a lightweight stand-in.
    from ama_kbqa.agents.sciqa_agent.agent import SciQAAgent
    return SciQAAgent._get_denied_tool_names(object())


def test_default_off_denies_nothing():
    assert _denied_names() == set()


@pytest.mark.parametrize("val", ["1", "true", "TRUE", "yes", "on", "On"])
def test_truthy_flag_denies_raw_sparql(val):
    os.environ["AMA_SCIQA_DISABLE_RAW_SPARQL"] = val
    assert _denied_names() == {"RunORKGSPARQL"}


@pytest.mark.parametrize("val", ["0", "false", "no", "off", "", "  "])
def test_falsy_flag_denies_nothing(val):
    os.environ["AMA_SCIQA_DISABLE_RAW_SPARQL"] = val
    assert _denied_names() == set()


def test_base_agent_default_denylist_empty():
    from ama_kbqa.framework.base_agent import BaseKBQAAgent
    assert BaseKBQAAgent._get_denied_tool_names(object()) == set()


def test_denylist_filters_openai_tool_dicts():
    """The denylist contract: tools whose function.name is denied are dropped."""
    tools = [
        {"function": {"name": "RunORKGSPARQL"}},
        {"function": {"name": "AggregateComparisonValues"}},
        {"function": {"name": "QueryComparisonRows"}},
    ]
    denied = {"RunORKGSPARQL"}
    kept = [t for t in tools if t["function"]["name"] not in denied]
    assert [t["function"]["name"] for t in kept] == [
        "AggregateComparisonValues",
        "QueryComparisonRows",
    ]
