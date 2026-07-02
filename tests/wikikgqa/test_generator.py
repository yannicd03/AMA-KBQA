"""Unit tests for the WikiKGQA generator's pure logic (no network/LLM).

The SPARQL extraction has to survive models that wrap queries in markdown
fences or prepend chatty prose, since the query is executed verbatim.
"""

from __future__ import annotations

from ama_kbqa.wikikgqa.dataset import Mention
from ama_kbqa.wikikgqa.generator import (
    GeneratedQuery,
    _best_query_from_snapshots,
    strip_sparql,
)
from ama_kbqa.wikikgqa.prompts import format_mentions_block


class _FakeAgent:
    """Minimal stand-in exposing the journal_snapshots the recovery reads."""

    def __init__(self, snapshots):
        self.journal_snapshots = snapshots


def _snap(found_values):
    return {"state": {"found_values": found_values}}


def test_recover_picks_latest_nonempty_query():
    agent = _FakeAgent([
        _snap({
            "sparql_result_1": {"query": "SELECT ?a WHERE {}", "result_count": 3},
            "sparql_result_2": {"query": "SELECT ?b WHERE {}", "result_count": 5},
        }),
    ])
    # Highest-numbered validated (non-empty) query wins: the agent's latest attempt.
    assert _best_query_from_snapshots(agent) == "SELECT ?b WHERE {}"


def test_recover_skips_empty_result_queries():
    agent = _FakeAgent([
        _snap({
            "sparql_result_1": {"query": "SELECT ?a WHERE {}", "result_count": 4},
            "sparql_result_2": {"query": "SELECT ?bad WHERE {}", "result_count": 0},
        }),
    ])
    # The later query returned 0 rows (wrong path), so fall back to the earlier good one.
    assert _best_query_from_snapshots(agent) == "SELECT ?a WHERE {}"


def test_recover_returns_none_without_validated_query():
    assert _best_query_from_snapshots(_FakeAgent([])) is None
    assert _best_query_from_snapshots(_FakeAgent([_snap({})])) is None
    empty = _FakeAgent([_snap({"sparql_result_1": {"query": "SELECT ?x {}", "result_count": 0}})])
    assert _best_query_from_snapshots(empty) is None


def test_strip_fenced_sparql():
    assert strip_sparql("```sparql\nSELECT ?x WHERE {}\n```") == "SELECT ?x WHERE {}"


def test_strip_plain_fence():
    assert strip_sparql("```\nASK { wd:Q1 wdt:P1 wd:Q2 }\n```") == "ASK { wd:Q1 wdt:P1 wd:Q2 }"


def test_strip_leading_prose():
    out = strip_sparql("Sure, here is the query:\nSELECT ?c WHERE { ?c wdt:P31 wd:Q5 }")
    assert out.startswith("SELECT ?c")


def test_strip_keeps_prefixes():
    q = "PREFIX wd: <http://www.wikidata.org/entity/>\nSELECT ?x WHERE {}"
    assert strip_sparql(q) == q


def test_generated_query_answer_falls_back_to_empty():
    g = GeneratedQuery(qid=1, sparql="SELECT ?x WHERE {}", result=None)
    assert g.answer == {"head": {"vars": []}, "results": {"bindings": []}}
    assert g.ok is False


def test_format_mentions_block_marks_inverse():
    mentions = [
        Mention(string="Stranger Things", language="en", entity="Q19798734"),
        Mention(string="played", language="en", entity="Q222749", property="P161", inverse=True),
    ]
    block = format_mentions_block(mentions)
    assert "entity=Q19798734" in block
    assert "property=P161 (inverse)" in block


def test_format_mentions_block_empty():
    assert "none provided" in format_mentions_block([])
