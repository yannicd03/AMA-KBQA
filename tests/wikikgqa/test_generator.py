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


# --- AgentSparqlGenerator: journal-recovery swap + answer-set self-consistency ---
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


def _select_result_values(*qids: str) -> SparqlResult:
    bindings = [
        {"x": {"type": "uri", "value": f"http://www.wikidata.org/entity/{q}"}} for q in qids
    ]
    return SparqlResult(ok=True, json={"head": {"vars": ["x"]}, "results": {"bindings": bindings}})


def test_votes_default_off_returns_single_run():
    gen = AgentSparqlGenerator(votes=1)
    first = GeneratedQuery(qid=1, sparql="ASK {}", result=_bool_result(True))
    calls: list = []

    def fake_once(question):
        calls.append(question)
        return first

    gen._generate_once = fake_once
    out = gen.generate(_question())
    assert out is first
    assert len(calls) == 1


def test_votes_early_consensus_stops_after_two_agreeing_select_runs():
    gen = AgentSparqlGenerator(votes=3)
    # Different queries, SAME answer set -> they vote together and consensus
    # is reached after two runs (the third generation never happens).
    runs = [
        GeneratedQuery(qid=1, sparql="SELECT ?x WHERE { a }", result=_select_result_values("Q1", "Q2")),
        GeneratedQuery(qid=1, sparql="SELECT ?y WHERE { b }", result=_select_result_values("Q2", "Q1")),
    ]
    it = iter(runs)
    calls: list = []

    def fake_once(question):
        calls.append(question)
        return next(it)

    gen._generate_once = fake_once
    out = gen.generate(_question())
    assert out is runs[0]  # earliest run matching the modal answer set
    assert len(calls) == 2  # early exit: no third run


def test_votes_majority_returns_earliest_matching_run():
    gen = AgentSparqlGenerator(votes=3)
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


def test_votes_tie_keeps_first_runs_answer():
    gen = AgentSparqlGenerator(votes=2)
    runs = [
        GeneratedQuery(qid=1, sparql="ASK {}", result=_bool_result(True)),
        GeneratedQuery(qid=1, sparql="ASK {}", result=_bool_result(False)),
    ]
    it = iter(runs)
    gen._generate_once = lambda question: next(it)
    out = gen.generate(_question())
    assert out is runs[0]


def test_votes_tie_prefers_sane_answer_over_unsane():
    # Two runs, 1-1 tie by vote count, but the first-seen run's answer set is
    # unsane (a statement-node URI) and the second's is a clean entity id. A
    # tie must resolve to the sane answer, not the earliest-seen one, since an
    # unsane answer set can never match gold.
    gen = AgentSparqlGenerator(votes=2)
    runs = [
        GeneratedQuery(qid=1, sparql="SELECT ?x WHERE { a }", result=_statement_uri_result()),
        GeneratedQuery(qid=1, sparql="SELECT ?y WHERE { b }", result=_select_result_values("Q1")),
    ]
    it = iter(runs)
    gen._generate_once = lambda question: next(it)
    out = gen.generate(_question())
    assert out is runs[1]


def test_votes_empty_runs_never_win():
    gen = AgentSparqlGenerator(votes=3)
    runs = [
        GeneratedQuery(qid=1, sparql="ASK {}", result=_bool_result(True)),
        GeneratedQuery(qid=1, sparql="SELECT ?x WHERE {}", result=_select_result(rows=False)),
        GeneratedQuery(qid=1, sparql="ASK {}", result=_bool_result(False)),
    ]
    it = iter(runs)
    gen._generate_once = lambda question: next(it)
    out = gen.generate(_question())
    # keys counted = [True, False] (the empty run keys to None and is dropped)
    # -> tie -> first run wins.
    assert out is runs[0]


def test_votes_two_acts_as_retry_when_first_run_is_empty():
    gen = AgentSparqlGenerator(votes=2)
    runs = [
        GeneratedQuery(qid=1, sparql="SELECT ?x WHERE {}", result=_select_result(rows=False)),
        GeneratedQuery(qid=1, sparql="SELECT ?y WHERE { b }", result=_select_result_values("Q5")),
    ]
    it = iter(runs)
    gen._generate_once = lambda question: next(it)
    out = gen.generate(_question())
    # The empty first run cannot win (None key); the valid second run is modal.
    assert out is runs[1]


def test_votes_all_runs_empty_returns_first():
    gen = AgentSparqlGenerator(votes=2)
    runs = [
        GeneratedQuery(qid=1, sparql="SELECT ?x WHERE {}", result=_select_result(rows=False)),
        GeneratedQuery(qid=1, sparql=None, result=None),
    ]
    it = iter(runs)
    gen._generate_once = lambda question: next(it)
    out = gen.generate(_question())
    assert out is runs[0]


# --- answer-sanity guard ---
# Every gold answer is a clean Q/P id, literal, or boolean -- never a statement
# node, blank node, or Special:EntityData URL. _result_is_sane names that
# failure mode (q113 incident: a committed 50-row result was all
# "statement/Qxxx-UUID" junk) so it can be treated the same way a 0-row result
# already is: known-wrong, worth a journal-alternate recovery attempt.


def _bnode_result() -> SparqlResult:
    return SparqlResult(
        ok=True,
        json={"head": {"vars": ["x"]}, "results": {"bindings": [{"x": {"type": "bnode", "value": "b0"}}]}},
    )


def _statement_uri_result() -> SparqlResult:
    return SparqlResult(
        ok=True,
        json={
            "head": {"vars": ["x"]},
            "results": {
                "bindings": [
                    {
                        "x": {
                            "type": "uri",
                            "value": "http://www.wikidata.org/entity/statement/Q1061678-4137053F",
                        }
                    }
                ]
            },
        },
    )


def _special_entitydata_result() -> SparqlResult:
    return SparqlResult(
        ok=True,
        json={
            "head": {"vars": ["x"]},
            "results": {
                "bindings": [
                    {"x": {"type": "uri", "value": "https://www.wikidata.org/wiki/Special:EntityData/Q42"}}
                ]
            },
        },
    )


def test_result_is_sane_false_for_bnode():
    assert generator_mod._result_is_sane(_bnode_result().json) is False


def test_result_is_sane_false_for_statement_node_uri():
    assert generator_mod._result_is_sane(_statement_uri_result().json) is False


def test_result_is_sane_false_for_special_entitydata_url():
    assert generator_mod._result_is_sane(_special_entitydata_result().json) is False


def test_result_is_sane_true_for_clean_entity_rows():
    assert generator_mod._result_is_sane(_select_result_values("Q1", "Q2").json) is True


def test_result_is_sane_true_for_literal_rows():
    r = {
        "head": {"vars": ["n"]},
        "results": {"bindings": [{"n": {"type": "literal", "value": "3"}}]},
    }
    assert generator_mod._result_is_sane(r) is True


def test_result_is_sane_true_for_boolean_results():
    assert generator_mod._result_is_sane(_bool_result(True).json) is True
    assert generator_mod._result_is_sane(_bool_result(False).json) is True


def test_result_is_sane_false_for_slash_junk_after_normalization():
    # Catch-all: gold-scan of data/wikikgqa/wikikgqa.json (497 questions) found
    # no gold answer value containing "/", so a literal that still has one
    # after to_codabench_answers normalization is unmapped URI junk, not a
    # legitimate literal/URL answer.
    r = {
        "head": {"vars": ["x"]},
        "results": {"bindings": [{"x": {"type": "literal", "value": "foo/bar"}}]},
    }
    assert generator_mod._result_is_sane(r) is False


def test_result_is_sane_false_for_none_or_empty():
    assert generator_mod._result_is_sane(None) is False


def test_generate_once_swaps_to_sane_alternate_when_committed_is_unsane(monkeypatch):
    committed = "SELECT ?x WHERE { ?x wdt:P31 wd:Q5 }"
    recovered = "SELECT ?y WHERE { ?y wdt:P31 wd:Q5 }"
    monkeypatch.setattr(
        "ama_kbqa.agents.wikidata_agent.agent.WikidataAgent",
        _fake_agent_cls(f"```sparql\n{committed}\n```"),
    )

    def _exec(q, endpoint=None, timeout=120):
        if q.strip() == recovered:
            return _select_result_values("Q1")  # sane, has rows
        return _statement_uri_result()  # committed: has rows but unsane

    monkeypatch.setattr(generator_mod, "execute", _exec)
    monkeypatch.setattr(generator_mod, "_best_query_from_snapshots", lambda agent: recovered)

    out = AgentSparqlGenerator()._generate_once(_question())
    assert out.sparql == recovered
    assert out.result.json == _select_result_values("Q1").json
    assert out.attempts == 2


def test_generate_once_keeps_unsane_committed_when_no_alternate(monkeypatch):
    committed = "SELECT ?x WHERE { ?x wdt:P31 wd:Q5 }"
    monkeypatch.setattr(
        "ama_kbqa.agents.wikidata_agent.agent.WikidataAgent",
        _fake_agent_cls(f"```sparql\n{committed}\n```"),
    )
    monkeypatch.setattr(generator_mod, "execute", lambda q, endpoint=None, timeout=120: _statement_uri_result())
    monkeypatch.setattr(generator_mod, "_best_query_from_snapshots", lambda agent: None)  # no journal alternate

    out = AgentSparqlGenerator()._generate_once(_question())
    # No sane alternate exists: a bad answer is not worse than a bad answer, so
    # the original (unsane) committed result is kept rather than blanked out.
    assert out.sparql == committed
    assert out.result.json == _statement_uri_result().json
    assert out.attempts == 1


# --- class-closure expansion repair (ANSWER_CONVENTIONS.md rule 10) ---
# Committed queries that constrain class membership with bare `wdt:P31 wd:QX`
# score precision=1.0/recall~=0 against gold's transitive closure. These tests
# cover the rewrite/escalation helpers directly, then the end-to-end wiring in
# _generate_once (same monkeypatched-execute style as the tests above).


def test_next_closure_escalation_bare_to_single_closure():
    bare = "SELECT DISTINCT ?x WHERE { ?x wdt:P31 wd:Q22645 }"
    out = generator_mod._next_closure_escalation(bare)
    assert out == "SELECT DISTINCT ?x WHERE { ?x wdt:P31/wdt:P279* wd:Q22645 }"


def test_next_closure_escalation_single_to_starred_closure():
    level1 = "SELECT DISTINCT ?x WHERE { ?x wdt:P31/wdt:P279* wd:Q22645 }"
    out = generator_mod._next_closure_escalation(level1)
    assert out == "SELECT DISTINCT ?x WHERE { ?x wdt:P31*/wdt:P279* wd:Q22645 }"


def test_next_closure_escalation_at_ceiling_returns_none():
    level2 = "SELECT DISTINCT ?x WHERE { ?x wdt:P31*/wdt:P279* wd:Q22645 }"
    assert generator_mod._next_closure_escalation(level2) is None


def test_next_closure_escalation_no_membership_pattern_returns_none():
    q = "SELECT DISTINCT ?x WHERE { ?x wdt:P106 wd:Q5 }"
    assert generator_mod._next_closure_escalation(q) is None


def test_next_closure_escalation_does_not_touch_string_literals():
    # A literal that happens to spell out "wdt:P31" must survive untouched;
    # only the real triple pattern outside the string gets escalated.
    q = 'SELECT ?x WHERE { ?x rdfs:label "wdt:P31 fake" . ?x wdt:P31 wd:Q5 }'
    out = generator_mod._next_closure_escalation(q)
    assert out == 'SELECT ?x WHERE { ?x rdfs:label "wdt:P31 fake" . ?x wdt:P31/wdt:P279* wd:Q5 }'


def test_closure_expansion_eligible_true_for_plain_select_of_uris():
    q = "SELECT DISTINCT ?x WHERE { ?x wdt:P31 wd:Q5 }"
    result_json = _select_result_values("Q1").json
    assert generator_mod._closure_expansion_eligible(q, result_json) is True


def test_closure_expansion_eligible_false_for_ask():
    q = "ASK { ?x wdt:P31 wd:Q5 }"
    result_json = _bool_result(True).json
    assert generator_mod._closure_expansion_eligible(q, result_json) is False


def test_closure_expansion_eligible_false_for_count_aggregate():
    q = "SELECT (COUNT(?x) AS ?c) WHERE { ?x wdt:P31 wd:Q5 }"
    result_json = {"head": {"vars": ["c"]}, "results": {"bindings": [{"c": {"type": "literal", "value": "3"}}]}}
    assert generator_mod._closure_expansion_eligible(q, result_json) is False


def test_closure_expansion_eligible_false_for_group_by():
    q = "SELECT ?x WHERE { ?x wdt:P31 wd:Q5 } GROUP BY ?x"
    result_json = _select_result_values("Q1").json
    assert generator_mod._closure_expansion_eligible(q, result_json) is False


def test_closure_expansion_eligible_false_for_having():
    q = "SELECT ?x WHERE { ?x wdt:P31 wd:Q5 } GROUP BY ?x HAVING (COUNT(?x) > 1)"
    result_json = _select_result_values("Q1").json
    assert generator_mod._closure_expansion_eligible(q, result_json) is False


def test_closure_expansion_eligible_false_for_literal_valued_select():
    q = "SELECT ?label WHERE { ?x wdt:P31 wd:Q5 ; rdfs:label ?label }"
    result_json = {"head": {"vars": ["label"]}, "results": {"bindings": [{"label": {"type": "literal", "value": "foo"}}]}}
    assert generator_mod._closure_expansion_eligible(q, result_json) is False


def test_closure_expansion_eligible_false_for_zero_rows():
    q = "SELECT ?x WHERE { ?x wdt:P31 wd:Q5 }"
    result_json = _select_result(rows=False).json
    assert generator_mod._closure_expansion_eligible(q, result_json) is False


def test_is_strict_superset_true_when_candidate_adds_rows():
    committed = _select_result_values("Q1").json
    candidate = _select_result_values("Q1", "Q2").json
    assert generator_mod._is_strict_superset(candidate, committed) is True


def test_is_strict_superset_false_when_sets_are_equal():
    same = _select_result_values("Q1").json
    assert generator_mod._is_strict_superset(same, same) is False


def test_is_strict_superset_false_when_candidate_does_not_contain_committed():
    committed = _select_result_values("Q1").json
    candidate = _select_result_values("Q2").json
    assert generator_mod._is_strict_superset(candidate, committed) is False


def _closure_exec_map(mapping: dict, calls: list):
    """execute() stand-in: dispatch by exact (stripped) query text, log every call."""

    def _exec(q, endpoint=None, timeout=120):
        key = q.strip()
        calls.append(key)
        return mapping.get(key, SparqlResult(ok=False, json=None, error=f"unmapped query: {key!r}"))

    return _exec


def test_generate_once_adopts_one_closure_level_when_strict_superset(monkeypatch):
    bare = "SELECT DISTINCT ?x WHERE { ?x wdt:P31 wd:Q22645 }"
    level1 = generator_mod._next_closure_escalation(bare)
    level2 = generator_mod._next_closure_escalation(level1)
    monkeypatch.setattr(
        "ama_kbqa.agents.wikidata_agent.agent.WikidataAgent",
        _fake_agent_cls(f"```sparql\n{bare}\n```"),
    )
    calls: list = []
    monkeypatch.setattr(
        generator_mod, "execute",
        _closure_exec_map(
            {
                bare: _select_result_values("Q1"),
                level1: _select_result_values("Q1", "Q2"),  # strict superset -> adopt
                level2: _select_result_values("Q1", "Q2"),  # same set as level1 -> reject, stop
            },
            calls,
        ),
    )

    out = AgentSparqlGenerator()._generate_once(_question())
    assert out.sparql == level1
    assert out.result.json == _select_result_values("Q1", "Q2").json
    assert calls == [bare, level1, level2]  # tried both levels, stopped after level2 rejected


def test_generate_once_escalates_two_levels_when_both_strict_supersets(monkeypatch):
    bare = "SELECT DISTINCT ?x WHERE { ?x wdt:P31 wd:Q729 }"
    level1 = generator_mod._next_closure_escalation(bare)
    level2 = generator_mod._next_closure_escalation(level1)
    monkeypatch.setattr(
        "ama_kbqa.agents.wikidata_agent.agent.WikidataAgent",
        _fake_agent_cls(f"```sparql\n{bare}\n```"),
    )
    calls: list = []
    monkeypatch.setattr(
        generator_mod, "execute",
        _closure_exec_map(
            {
                bare: _select_result_values("Q1"),
                level1: _select_result_values("Q1", "Q2"),
                level2: _select_result_values("Q1", "Q2", "Q3"),
            },
            calls,
        ),
    )

    out = AgentSparqlGenerator()._generate_once(_question())
    assert out.sparql == level2
    assert out.result.json == _select_result_values("Q1", "Q2", "Q3").json
    assert calls == [bare, level1, level2]


def test_generate_once_rejects_non_superset_escalation_keeps_committed(monkeypatch):
    bare = "SELECT DISTINCT ?x WHERE { ?x wdt:P31 wd:Q22645 }"
    level1 = generator_mod._next_closure_escalation(bare)
    monkeypatch.setattr(
        "ama_kbqa.agents.wikidata_agent.agent.WikidataAgent",
        _fake_agent_cls(f"```sparql\n{bare}\n```"),
    )
    calls: list = []
    monkeypatch.setattr(
        generator_mod, "execute",
        _closure_exec_map(
            {
                bare: _select_result_values("Q1"),
                level1: _select_result_values("Q9"),  # disjoint set -> not a superset
            },
            calls,
        ),
    )

    out = AgentSparqlGenerator()._generate_once(_question())
    assert out.sparql == bare
    assert out.result.json == _select_result_values("Q1").json
    assert calls == [bare, level1]  # tried once, rejected, never tried level2


def test_generate_once_rejects_unsane_escalation_candidate_keeps_committed(monkeypatch):
    # level1's answer set is a strict superset of bare's (so the plain
    # superset check alone would adopt it) but one of the added rows is a
    # statement-node URI -- unsane, so it must NOT be adopted even though it
    # nominally "grows" the answer set.
    bare = "SELECT DISTINCT ?x WHERE { ?x wdt:P31 wd:Q22645 }"
    level1 = generator_mod._next_closure_escalation(bare)
    monkeypatch.setattr(
        "ama_kbqa.agents.wikidata_agent.agent.WikidataAgent",
        _fake_agent_cls(f"```sparql\n{bare}\n```"),
    )
    unsane_superset = SparqlResult(
        ok=True,
        json={
            "head": {"vars": ["x"]},
            "results": {
                "bindings": [
                    {"x": {"type": "uri", "value": "http://www.wikidata.org/entity/Q1"}},
                    {
                        "x": {
                            "type": "uri",
                            "value": "http://www.wikidata.org/entity/statement/Q1-UUID",
                        }
                    },
                ]
            },
        },
    )
    calls: list = []
    monkeypatch.setattr(
        generator_mod, "execute",
        _closure_exec_map(
            {
                bare: _select_result_values("Q1"),
                level1: unsane_superset,  # superset but unsane -> not adopted
            },
            calls,
        ),
    )

    out = AgentSparqlGenerator()._generate_once(_question())
    assert out.sparql == bare
    assert out.result.json == _select_result_values("Q1").json


def test_generate_once_equal_set_escalation_passes_through_to_next_level(monkeypatch):
    # Regression for the q110 shape (verified live on the challenge endpoint):
    # P31 -> P31/P279* keeps the same row set, and only P31*/P279* widens to
    # gold's. An equal-set escalation is a no-op answer-wise and must NOT stop
    # the loop, or the ceiling level is never tried.
    bare = "SELECT DISTINCT ?x WHERE { ?x wdt:P31 wd:Q22645 }"
    level1 = generator_mod._next_closure_escalation(bare)
    level2 = generator_mod._next_closure_escalation(level1)
    monkeypatch.setattr(
        "ama_kbqa.agents.wikidata_agent.agent.WikidataAgent",
        _fake_agent_cls(f"```sparql\n{bare}\n```"),
    )
    calls: list = []
    monkeypatch.setattr(
        generator_mod, "execute",
        _closure_exec_map(
            {
                bare: _select_result_values("Q1"),
                level1: _select_result_values("Q1"),  # equal set: pass through
                level2: _select_result_values("Q1", "Q2", "Q3"),
            },
            calls,
        ),
    )

    out = AgentSparqlGenerator()._generate_once(_question())
    assert out.sparql == level2
    assert out.result.json == _select_result_values("Q1", "Q2", "Q3").json
    assert calls == [bare, level1, level2]


def test_generate_once_error_on_escalation_keeps_committed_result(monkeypatch):
    bare = "SELECT DISTINCT ?x WHERE { ?x wdt:P31 wd:Q22645 }"
    level1 = generator_mod._next_closure_escalation(bare)
    monkeypatch.setattr(
        "ama_kbqa.agents.wikidata_agent.agent.WikidataAgent",
        _fake_agent_cls(f"```sparql\n{bare}\n```"),
    )
    calls: list = []
    monkeypatch.setattr(
        generator_mod, "execute",
        _closure_exec_map(
            {
                bare: _select_result_values("Q1"),
                level1: SparqlResult(ok=False, json=None, error="timeout"),
            },
            calls,
        ),
    )

    out = AgentSparqlGenerator()._generate_once(_question())
    assert out.sparql == bare
    assert out.result.json == _select_result_values("Q1").json
    assert calls == [bare, level1]


def test_generate_once_skips_closure_expansion_for_aggregate_query(monkeypatch):
    committed = "SELECT (COUNT(?x) AS ?c) WHERE { ?x wdt:P31 wd:Q22645 }"
    monkeypatch.setattr(
        "ama_kbqa.agents.wikidata_agent.agent.WikidataAgent",
        _fake_agent_cls(f"```sparql\n{committed}\n```"),
    )
    count_result = SparqlResult(
        ok=True, json={"head": {"vars": ["c"]}, "results": {"bindings": [{"c": {"type": "literal", "value": "3"}}]}}
    )
    calls: list = []
    monkeypatch.setattr(generator_mod, "execute", _closure_exec_map({committed: count_result}, calls))

    out = AgentSparqlGenerator()._generate_once(_question())
    assert out.sparql == committed
    assert calls == [committed]  # no escalation query was ever attempted


def test_generate_once_closure_expansion_disabled_by_constructor_flag(monkeypatch):
    bare = "SELECT DISTINCT ?x WHERE { ?x wdt:P31 wd:Q22645 }"
    monkeypatch.setattr(
        "ama_kbqa.agents.wikidata_agent.agent.WikidataAgent",
        _fake_agent_cls(f"```sparql\n{bare}\n```"),
    )
    calls: list = []
    monkeypatch.setattr(generator_mod, "execute", _closure_exec_map({bare: _select_result_values("Q1")}, calls))

    out = AgentSparqlGenerator(closure_expansion=False)._generate_once(_question())
    assert out.sparql == bare
    assert calls == [bare]  # constructor flag off -> no extra execution at all
