"""Hermetic tests for CountEntities hardening and AggregateComparisonValues
range bucketing.

No live Virtuoso/Qdrant: a fake SPARQLWrapper returns canned bindings and
records the generated query for inspection.
"""

import asyncio
import json
from types import SimpleNamespace

from ama_kbqa.server import kqapro_server as kqa
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
        return {"results": {"bindings": self._bindings}}


def _context(sparql):
    return SimpleNamespace(
        request_context=SimpleNamespace(
            lifespan_context=SimpleNamespace(sparql=sparql)
        )
    )


def _fn(tool):
    """The plain callable behind the @mcp.tool decorator."""
    return getattr(tool, "fn", tool)


# =============================================================================
# CountEntities hardening
# =============================================================================

COUNT_BINDINGS = [{"c": {"value": "7"}}]


def test_count_success_is_trusted():
    sparql = FakeSPARQL(COUNT_BINDINGS)
    resp = _fn(kqa.CountEntities)(_context(sparql), concept="TV series")
    assert resp.count == 7
    assert resp.trusted is True
    assert "trusted count" in resp.status


def test_incomplete_primary_filter_dropped_when_duplicated_in_not_conditions():
    """attribute_name without value + same attribute in not_conditions must not
    add an existence-only triple that excludes entities lacking the attribute."""
    sparql = FakeSPARQL(COUNT_BINDINGS)
    resp = _fn(kqa.CountEntities)(
        _context(sparql),
        concept="TV series",
        attribute_name="start_time",
        not_conditions=[
            {"attribute_name": "start_time", "attribute_value": "2005", "operator": "="}
        ],
    )
    assert resp.count == 7
    assert "Ignored" in resp.status and "start_time" in resp.status
    # The attribute may appear only inside FILTER NOT EXISTS, not as a
    # top-level existence requirement.
    query = sparql.last_query
    attr_uri = "<http://kqapro.org/attribute/start_time>"
    outside_not_exists = query.split("FILTER NOT EXISTS")[0]
    assert attr_uri not in outside_not_exists
    assert attr_uri in query


def test_bare_existence_check_without_not_conditions_is_kept():
    """An attribute_name with no value and no not_conditions is a legitimate
    'how many X have attribute Y' existence count and must keep filtering."""
    sparql = FakeSPARQL(COUNT_BINDINGS)
    resp = _fn(kqa.CountEntities)(
        _context(sparql), concept="human", attribute_name="Twitter username"
    )
    assert resp.count == 7
    assert "Ignored" not in resp.status
    assert "<http://kqapro.org/attribute/Twitter_username>" in sparql.last_query


def test_incomplete_or_and_not_conditions_are_dropped():
    sparql = FakeSPARQL(COUNT_BINDINGS)
    resp = _fn(kqa.CountEntities)(
        _context(sparql),
        concept="film",
        or_conditions=[{"attribute_value": "2000", "operator": ">"}],
        not_conditions=[{"attribute_value": "1990"}],
    )
    assert resp.count == 7
    assert resp.status.count("without attribute_name") == 2


def test_all_filters_incomplete_refuses_untrusted():
    sparql = FakeSPARQL(COUNT_BINDINGS)
    resp = _fn(kqa.CountEntities)(
        _context(sparql),
        or_conditions=[{"attribute_value": "2000"}],
    )
    assert resp.count == 0
    assert resp.trusted is False
    assert "incomplete" in resp.status


def test_count_union_is_trusted():
    sparql = FakeSPARQL(COUNT_BINDINGS)
    resp = _fn(kqa.CountUnion)(
        branches=[{"concept": "film"}, {"concept": "TV series"}],
        context=_context(sparql),
    )
    assert resp.count == 7
    assert resp.trusted is True


# =============================================================================
# AggregateComparisonValues range bucketing
# =============================================================================

def _agg_binding(contrib, year, value):
    return {
        "contrib": {"value": f"http://orkg.org/orkg/resource/{contrib}"},
        "valuePred": {"value": "http://orkg.org/orkg/predicate/P1"},
        "group": {"value": f"http://orkg.org/orkg/resource/Y{year}"},
        "groupLabel": {"value": str(year)},
        "valueObj": {"value": str(value)},
    }


AGG_BINDINGS = [
    _agg_binding("C1", 2006, 10),
    _agg_binding("C2", 2007, 20),
    _agg_binding("C3", 2011, 30),
    _agg_binding("C4", 2016, 40),
]


def _run_aggregate(**kwargs):
    sparql = FakeSPARQL(kwargs.pop("bindings", AGG_BINDINGS))
    coro = _fn(sci.AggregateComparisonValues)(
        _context(sparql),
        comparison_id="R1",
        value_predicate="P1",
        group_by_predicate="P_year",
        **kwargs,
    )
    return json.loads(asyncio.run(coro)), sparql


def test_bucketed_grouping_bins_years_into_intervals():
    payload, _ = _run_aggregate(agg="avg", group_bucket_size=5)
    assert payload["group_bucket"] == {"size": 5, "start": 2006}
    result = payload["result"]
    assert [r["group"] for r in result] == ["2006-2010", "2011-2015", "2016-2020"]
    assert [r["value"] for r in result] == [15.0, 30.0, 40.0]
    assert [r["n"] for r in result] == [2, 1, 1]
    assert "5-wide buckets" in payload["status"]


def test_bucket_start_override():
    payload, _ = _run_aggregate(agg="avg", group_bucket_size=5, group_bucket_start="2005")
    result = payload["result"]
    assert [r["group"] for r in result] == ["2005-2009", "2010-2014", "2015-2019"]


def test_bucket_size_zero_keeps_plain_grouping():
    payload, _ = _run_aggregate(agg="avg")
    assert payload["group_bucket"] is None
    groups = {r["group"] for r in payload["result"]}
    assert groups == {"2006", "2007", "2011", "2016"}


def test_system_prompt_advertises_bucketing():
    """Adoption guard: in the 2026-06-12 validation run the model never used
    group_bucket_size because only the tool docstring mentioned it. The
    decision tree in the system prompt must name the parameter explicitly."""
    from ama_kbqa.agents.sciqa_agent.prompts import SYSTEM_PROMPT
    assert SYSTEM_PROMPT.count("group_bucket_size") >= 2


def test_inverse_group_path_walks_backward_to_paper_year():
    """group_by_path='^P31,P29' must emit paper --P31--> contrib and
    paper --P29--> year (the gold pattern for interval questions)."""
    sparql = FakeSPARQL(AGG_BINDINGS)
    coro = _fn(sci.AggregateComparisonValues)(
        _context(sparql),
        comparison_id="R1",
        value_predicate="P1",
        group_by_path="^P31,P29",
        group_bucket_size=5,
    )
    payload = json.loads(asyncio.run(coro))
    query = sparql.last_query
    assert "?groupPath0 orkgp:P31 ?contrib ." in query
    assert "?groupPath0 orkgp:P29 ?group ." in query
    # Bucketing still applies on the path-derived group values.
    assert payload["group_bucket"] == {"size": 5, "start": 2006}


def test_forward_group_path_unchanged():
    sparql = FakeSPARQL(AGG_BINDINGS)
    coro = _fn(sci.AggregateComparisonValues)(
        _context(sparql),
        comparison_id="R1",
        value_predicate="P1",
        group_by_path="P37581,P43139",
    )
    asyncio.run(coro)
    query = sparql.last_query
    assert "?contrib orkgp:P37581 ?groupPath0 ." in query
    assert "?groupPath0 orkgp:P43139 ?group ." in query


def test_prompt_advertises_inverse_year_path():
    from ama_kbqa.agents.sciqa_agent.prompts import SYSTEM_PROMPT
    assert '^P31,P29' in SYSTEM_PROMPT


def test_non_numeric_group_values_keep_label():
    bindings = AGG_BINDINGS + [
        {
            "contrib": {"value": "http://orkg.org/orkg/resource/C5"},
            "valuePred": {"value": "http://orkg.org/orkg/predicate/P1"},
            "group": {"value": "http://orkg.org/orkg/resource/Yx"},
            "groupLabel": {"value": "unknown period"},
            "valueObj": {"value": "50"},
        }
    ]
    payload, _ = _run_aggregate(agg="avg", group_bucket_size=5, bindings=bindings)
    groups = [r["group"] for r in payload["result"]]
    assert groups == ["2006-2010", "2011-2015", "2016-2020", "unknown period"]
