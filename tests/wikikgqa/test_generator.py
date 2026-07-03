"""Unit tests for the WikiKGQA generator's pure logic (no network/LLM).

The SPARQL extraction has to survive models that wrap queries in markdown
fences or prepend chatty prose, since the query is executed verbatim.
"""

from __future__ import annotations

import ama_kbqa.wikikgqa.generator as generator_mod
from ama_kbqa.wikikgqa.dataset import Mention, WikiKGQAQuestion
from ama_kbqa.wikikgqa.endpoint import SparqlResult
from ama_kbqa.wikikgqa.generator import (
    AgentSparqlGenerator,
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


def test_strip_no_sparql_returns_empty():
    # Agent-loop error prose must not be treated as a query (the caller falls
    # back to journal recovery when strip_sparql returns "").
    assert strip_sparql("Error: Agent reached maximum iteration limit.") == ""


def test_strip_empty_and_blank_return_empty():
    assert strip_sparql("") == ""
    assert strip_sparql("   \n  ") == ""
    assert strip_sparql(None) == ""


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


# --- AgentSparqlGenerator: journal-recovery swap + ask_votes self-consistency ---
# WikidataAgent (the real tool-loop) is monkeypatched out entirely; these tests
# only exercise the pure decision logic in _generate_once/generate, no
# network/LLM/MCP involved.


def _question(qid: int = 1, text: str = "Some question?") -> WikiKGQAQuestion:
    return WikiKGQAQuestion(id=qid, questions={"en": text}, mentions=[])


def _select_result(rows: bool = True) -> SparqlResult:
    bindings = [{"x": {"type": "uri", "value": "http://www.wikidata.org/entity/Q1"}}] if rows else []
    return SparqlResult(ok=True, json={"head": {"vars": ["x"]}, "results": {"bindings": bindings}})


def _error_result() -> SparqlResult:
    return SparqlResult(ok=False, json=None, error="boom")


def _bool_result(value: bool) -> SparqlResult:
    return SparqlResult(ok=True, json={"head": {"vars": []}, "boolean": value})


def _fake_agent_cls(raw: str):
    """Stand-in for WikidataAgent: returns a fixed synthesis string, no MCP/LLM."""

    class _FakeAgent:
        def __init__(self, **kwargs):
            self.journal_snapshots: list = []

        async def ask(self, augmented: str) -> str:
            return raw

        async def close(self) -> None:
            pass

    return _FakeAgent


def test_generate_once_keeps_committed_query_when_it_has_rows(monkeypatch):
    monkeypatch.setattr(
        "ama_kbqa.agents.wikidata_agent.agent.WikidataAgent",
        _fake_agent_cls("```sparql\nSELECT ?x WHERE { ?x wdt:P31 wd:Q5 }\n```"),
    )
    monkeypatch.setattr(generator_mod, "execute", lambda q, endpoint=None, timeout=120: _select_result(rows=True))
    # A "better" recovered query exists but must not be consulted since rows were found.
    monkeypatch.setattr(generator_mod, "_best_query_from_snapshots", lambda agent: "SELECT ?other WHERE {}")

    out = AgentSparqlGenerator()._generate_once(_question())
    assert out.sparql == "SELECT ?x WHERE { ?x wdt:P31 wd:Q5 }"
    assert out.attempts == 1
    assert out.ok is True


def test_generate_once_swaps_to_recovered_query_when_committed_is_empty(monkeypatch):
    recovered = "SELECT ?y WHERE { ?y wdt:P31 wd:Q5 }"
    monkeypatch.setattr(
        "ama_kbqa.agents.wikidata_agent.agent.WikidataAgent",
        _fake_agent_cls("```sparql\nSELECT ?x WHERE { ?x wdt:BAD wd:Q5 }\n```"),
    )
    monkeypatch.setattr(
        generator_mod, "execute",
        lambda q, endpoint=None, timeout=120: _select_result(rows=(q.strip() == recovered)),
    )
    monkeypatch.setattr(generator_mod, "_best_query_from_snapshots", lambda agent: recovered)

    out = AgentSparqlGenerator()._generate_once(_question())
    assert out.sparql == recovered
    assert out.attempts == 2
    assert out.ok is True


def test_generate_once_no_swap_when_recovered_equals_committed_query(monkeypatch):
    same_query = "SELECT ?x WHERE { ?x wdt:P31 wd:Q5 }"
    monkeypatch.setattr(
        "ama_kbqa.agents.wikidata_agent.agent.WikidataAgent",
        _fake_agent_cls(f"```sparql\n{same_query}\n```"),
    )
    monkeypatch.setattr(generator_mod, "execute", lambda q, endpoint=None, timeout=120: _error_result())
    monkeypatch.setattr(generator_mod, "_best_query_from_snapshots", lambda agent: same_query)

    out = AgentSparqlGenerator()._generate_once(_question())
    assert out.sparql == same_query
    assert out.attempts == 1
    assert out.ok is False


def test_generate_once_falls_back_to_journal_when_synthesis_empty(monkeypatch):
    recovered = "SELECT ?y WHERE { ?y wdt:P31 wd:Q5 }"
    monkeypatch.setattr(
        "ama_kbqa.agents.wikidata_agent.agent.WikidataAgent",
        _fake_agent_cls("Error: Agent reached maximum iteration limit."),
    )
    monkeypatch.setattr(generator_mod, "execute", lambda q, endpoint=None, timeout=120: _select_result(rows=True))
    monkeypatch.setattr(generator_mod, "_best_query_from_snapshots", lambda agent: recovered)

    out = AgentSparqlGenerator()._generate_once(_question())
    assert out.sparql == recovered
    assert out.attempts == 2


def test_generate_once_returns_no_query_without_synthesis_or_recovery(monkeypatch):
    monkeypatch.setattr(
        "ama_kbqa.agents.wikidata_agent.agent.WikidataAgent",
        _fake_agent_cls("Error: Agent reached maximum iteration limit."),
    )
    monkeypatch.setattr(generator_mod, "_best_query_from_snapshots", lambda agent: None)
    calls: list = []
    monkeypatch.setattr(generator_mod, "execute", lambda *a, **k: calls.append(1))

    out = AgentSparqlGenerator()._generate_once(_question())
    assert out.sparql is None
    assert out.result is None
    assert out.attempts == 1
    assert not calls  # execute() must never run without a query to run


def test_ask_votes_default_off_returns_single_run():
    gen = AgentSparqlGenerator(ask_votes=1)
    first = GeneratedQuery(qid=1, sparql="ASK {}", result=_bool_result(True))
    calls: list = []

    def fake_once(question):
        calls.append(question)
        return first

    gen._generate_once = fake_once
    out = gen.generate(_question())
    assert out is first
    assert len(calls) == 1


def test_ask_votes_skips_revoting_for_non_boolean_result():
    gen = AgentSparqlGenerator(ask_votes=3)
    first = GeneratedQuery(qid=1, sparql="SELECT ?x WHERE {}", result=_select_result(rows=False))
    calls: list = []

    def fake_once(question):
        calls.append(question)
        return first

    gen._generate_once = fake_once
    out = gen.generate(_question())
    assert out is first
    assert len(calls) == 1  # SELECT results never trigger self-consistency re-runs


def test_ask_votes_majority_returns_earliest_matching_run():
    gen = AgentSparqlGenerator(ask_votes=3)
    runs = [
        GeneratedQuery(qid=1, sparql="ASK {}", result=_bool_result(False)),
        GeneratedQuery(qid=1, sparql="ASK {}", result=_bool_result(True)),
        GeneratedQuery(qid=1, sparql="ASK {}", result=_bool_result(True)),
    ]
    it = iter(runs)
    gen._generate_once = lambda question: next(it)
    out = gen.generate(_question())
    # majority is True (2-1); the earliest True-valued run is runs[1], not runs[0].
    assert out is runs[1]


def test_ask_votes_tie_keeps_first_runs_answer():
    gen = AgentSparqlGenerator(ask_votes=2)
    runs = [
        GeneratedQuery(qid=1, sparql="ASK {}", result=_bool_result(True)),
        GeneratedQuery(qid=1, sparql="ASK {}", result=_bool_result(False)),
    ]
    it = iter(runs)
    gen._generate_once = lambda question: next(it)
    out = gen.generate(_question())
    assert out is runs[0]


def test_ask_votes_ignores_reruns_that_return_non_boolean():
    gen = AgentSparqlGenerator(ask_votes=3)
    runs = [
        GeneratedQuery(qid=1, sparql="ASK {}", result=_bool_result(True)),
        GeneratedQuery(qid=1, sparql="SELECT ?x WHERE {}", result=_select_result(rows=False)),
        GeneratedQuery(qid=1, sparql="ASK {}", result=_bool_result(False)),
    ]
    it = iter(runs)
    gen._generate_once = lambda question: next(it)
    out = gen.generate(_question())
    # votes counted = [True, False] (the non-boolean re-run is dropped) -> tie -> first run wins.
    assert out is runs[0]
