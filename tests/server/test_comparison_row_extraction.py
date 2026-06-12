"""Hermetic tests for the 2026-06-13 SciQA deep-dive fixes:

- QueryComparisonRows: dict-form filters, path predicates with inverse hops,
  nested fallback matching, and the default paper column.
- AggregateComparisonValues: paired grouping when group and intermediate share
  the same path (no cross-join), and the degenerate-grouping warning.
- RunORKGSPARQL: unanchored compareContribution lint.
- Prompt guards for the new rules.

No live Virtuoso: a fake SPARQLWrapper returns canned bindings and records the
generated query for inspection.
"""

import asyncio
import json
from types import SimpleNamespace

from ama_kbqa.server import sciqa_server as sci


class FakeSPARQL:
    def __init__(self, bindings):
        self._bindings = bindings
        self.last_query = None

    def setQuery(self, query):
        self.last_query = query

    def query(self):
        return self

    def convert(self):
        return {
            "head": {"vars": ["x"]},
            "results": {"bindings": self._bindings},
        }


def _context(sparql):
    return SimpleNamespace(
        request_context=SimpleNamespace(
            lifespan_context=SimpleNamespace(sparql=sparql)
        )
    )


def _fn(tool):
    return getattr(tool, "fn", tool)


def _run_rows(bindings, **kwargs):
    sparql = FakeSPARQL(bindings)
    coro = _fn(sci.QueryComparisonRows)(
        _context(sparql), comparison_id="R112387", **kwargs
    )
    return json.loads(asyncio.run(coro)), sparql


# =============================================================================
# QueryComparisonRows
# =============================================================================

ROW_BINDING = {
    "contrib": {"value": "http://orkg.org/orkg/resource/R76800"},
    "paper": {"value": "http://orkg.org/orkg/resource/R76799"},
    "paperLabel": {"value": "User feedback paper"},
    "retObj0": {"value": "http://orkg.org/orkg/resource/R99001"},
    "retLabel0": {"value": "0.72"},
}


def test_dict_filters_are_accepted():
    payload, sparql = _run_rows(
        [ROW_BINDING],
        filters={"P15006": "Naive bayes", "P36075": "Bag of words"},
        return_predicates=["P37365"],
    )
    assert payload["row_count"] == 1
    assert {f["predicate"] for f in payload["filters"]} == {"P15006", "P36075"}
    assert "orkgp:P15006" in sparql.last_query
    assert "orkgp:P36075" in sparql.last_query


def test_nested_fallback_unions_deeper_hops():
    _, sparql = _run_rows(
        [ROW_BINDING],
        filters=[{"predicate": "P15006", "value": "Naive bayes"}],
    )
    # Direct match plus one- and two-hop nested alternatives.
    assert sparql.last_query.count("UNION") >= 2
    assert "?f0m1 orkgp:P15006" in sparql.last_query
    assert "?f0m3 orkgp:P15006" in sparql.last_query


def test_nested_fallback_can_be_disabled():
    _, sparql = _run_rows(
        [ROW_BINDING],
        filters=[{"predicate": "P15006", "value": "Naive bayes"}],
        nested_fallback=False,
    )
    assert "UNION" not in sparql.last_query


def test_path_filter_with_inverse_hop():
    _, sparql = _run_rows(
        [ROW_BINDING],
        filters=[{"predicate": "^P31,P29", "value": "2020"}],
    )
    # Inverse first hop: the intermediate points AT the contribution.
    assert "?f0p0 orkgp:P31 ?contrib ." in sparql.last_query
    assert "?f0p0 orkgp:P29 ?fobj0 ." in sparql.last_query
    # Paths never get the anonymous nested fallback.
    assert "UNION" not in sparql.last_query


def test_rows_include_paper_by_default():
    payload, sparql = _run_rows(
        [ROW_BINDING],
        filters=[{"predicate": "P15006", "value": "Naive bayes"}],
        return_predicates=["P37365"],
    )
    row = payload["rows"][0]
    assert row["paper"] == "R76799"
    assert row["paper_label"] == "User feedback paper"
    assert "?paper orkgp:P31 ?contrib ." in sparql.last_query


def test_include_paper_false_omits_paper_clause():
    payload, sparql = _run_rows(
        [ROW_BINDING],
        filters=[{"predicate": "P15006", "value": "Naive bayes"}],
        include_paper=False,
    )
    assert "?paper" not in sparql.last_query
    assert "paper" not in payload["rows"][0]


def test_string_return_predicates_treated_as_one_path():
    _, sparql = _run_rows(
        [ROW_BINDING],
        filters=[{"predicate": "P15006", "value": "Naive bayes"}],
        return_predicates="P37586,P35205",
    )
    # The two steps chain through one intermediate variable.
    assert "?contrib orkgp:P37586 ?r0p0 ." in sparql.last_query
    assert "?r0p0 orkgp:P35205 ?retObj0 ." in sparql.last_query


# =============================================================================
# AggregateComparisonValues paired grouping
# =============================================================================

def _agg_binding(contrib, intermediate, value):
    return {
        "contrib": {"value": f"http://orkg.org/orkg/resource/{contrib}"},
        "valuePred": {"value": "http://orkg.org/orkg/predicate/P43133"},
        "intermediate": {"value": f"http://orkg.org/orkg/resource/I{intermediate}"},
        "intermediateLabel": {"value": intermediate},
        "group": {"value": f"http://orkg.org/orkg/resource/I{intermediate}"},
        "groupLabel": {"value": intermediate},
        "valueObj": {"value": str(value)},
    }


PAIRED_BINDINGS = [
    _agg_binding("C1", "photovoltaics", 100),
    _agg_binding("C2", "photovoltaics", 200),
    _agg_binding("C1", "all sources", 400),
    _agg_binding("C2", "all sources", 600),
]


def _run_agg(bindings, **kwargs):
    sparql = FakeSPARQL(bindings)
    coro = _fn(sci.AggregateComparisonValues)(
        _context(sparql),
        comparison_id="R153799",
        value_predicate="P43133",
        **kwargs,
    )
    return json.loads(asyncio.run(coro)), sparql


def test_same_axis_grouping_binds_group_to_intermediate():
    payload, sparql = _run_agg(
        PAIRED_BINDINGS,
        agg="avg",
        group_by_predicate="P43135",
        intermediate_predicate="P43135",
    )
    assert "BIND(?intermediate AS ?group)" in sparql.last_query
    # The group path is not walked a second time with a fresh variable.
    assert "?contrib orkgp:P43135 ?group" not in sparql.last_query
    by_group = {r["group"]: r for r in payload["result"]}
    assert by_group["photovoltaics"]["value"] == 150.0
    assert by_group["all sources"]["value"] == 500.0


def test_same_axis_with_group_by_intermediate_has_single_axis():
    payload, _ = _run_agg(
        PAIRED_BINDINGS,
        agg="avg",
        group_by_predicate="P43135",
        intermediate_predicate="P43135",
        group_by_intermediate=True,
    )
    groups = [r["group"] for r in payload["result"]]
    # Plain labels, not {"intermediate": ..., "P43135": ...} pairs.
    assert all(isinstance(g, str) for g in groups)
    assert set(groups) == {"photovoltaics", "all sources"}


def test_degenerate_cross_join_warns():
    # Three groups, identical aggregate and identical n: the cross-join shape.
    bindings = []
    for grp in ("a", "b", "c"):
        for val in (10, 20):
            b = {
                "contrib": {"value": "http://orkg.org/orkg/resource/C1"},
                "valuePred": {"value": "http://orkg.org/orkg/predicate/P43133"},
                "group": {"value": f"http://orkg.org/orkg/resource/G{grp}"},
                "groupLabel": {"value": grp},
                "valueObj": {"value": str(val)},
            }
            bindings.append(b)
    payload, _ = _run_agg(bindings, agg="avg", group_by_predicate="P43135")
    assert "WARNING" in payload["status"]
    assert "cross-join" in payload["status"]


def test_distinct_axes_keep_separate_walk():
    payload, sparql = _run_agg(
        PAIRED_BINDINGS,
        agg="avg",
        group_by_predicate="P43139",
        intermediate_predicate="P43135",
    )
    assert "BIND(?intermediate AS ?group)" not in sparql.last_query
    assert "orkgp:P43139 ?group" in sparql.last_query


# =============================================================================
# RunORKGSPARQL anchor lint
# =============================================================================

def _run_sparql(query):
    sparql = FakeSPARQL([])
    coro = _fn(sci.RunORKGSPARQL)(_context(sparql), query=query)
    return json.loads(asyncio.run(coro))


def test_unanchored_compare_contribution_warns():
    payload = _run_sparql(
        "SELECT ?drug (COUNT(?drug) AS ?c) WHERE { "
        "?comparison orkgp:compareContribution ?contribution . "
        "?contribution orkgp:P37578 ?drug . } GROUP BY ?drug"
    )
    assert payload["note"] and "spans" in payload["note"]


def test_anchored_compare_contribution_passes():
    payload = _run_sparql(
        "SELECT ?drug WHERE { "
        "orkgr:R155621 orkgp:compareContribution ?contribution . "
        "?contribution orkgp:P37578 ?drug . }"
    )
    assert not (payload.get("note") and "spans" in str(payload["note"]))


def test_values_anchored_variable_passes():
    payload = _run_sparql(
        "SELECT ?drug WHERE { VALUES ?cmp { orkgr:R155621 } "
        "?cmp orkgp:compareContribution ?contribution . "
        "?contribution orkgp:P37578 ?drug . }"
    )
    assert not (payload.get("note") and "spans" in str(payload["note"]))


# =============================================================================
# Prompt guards
# =============================================================================

def test_prompt_has_anchor_and_p31_rules():
    from ama_kbqa.agents.sciqa_agent.prompts import SYSTEM_PROMPT
    assert "SCOPE AND ANCHORING RULES" in SYSTEM_PROMPT
    assert "?paper orkgp:P31 ?contrib" in SYSTEM_PROMPT


def test_prompt_has_final_answer_contract():
    from ama_kbqa.agents.sciqa_agent.prompts import SYSTEM_PROMPT
    assert "FINAL ANSWER CONTRACT" in SYSTEM_PROMPT
    assert "FULL precision" in SYSTEM_PROMPT
    assert "all sources" in SYSTEM_PROMPT


def test_prompt_has_ranking_recipes():
    from ama_kbqa.agents.sciqa_agent.prompts import SYSTEM_PROMPT
    assert "RANKING RECIPES" in SYSTEM_PROMPT
    assert "ORDER BY DESC" in SYSTEM_PROMPT
