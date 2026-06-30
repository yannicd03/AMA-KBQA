"""Unit tests for the WikiKGQA generator's pure logic (no network/LLM).

The SPARQL extraction has to survive models that wrap queries in markdown
fences or prepend chatty prose, since the query is executed verbatim.
"""

from __future__ import annotations

from ama_kbqa.wikikgqa.dataset import Mention
from ama_kbqa.wikikgqa.generator import GeneratedQuery, strip_sparql
from ama_kbqa.wikikgqa.prompts import format_mentions_block


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
