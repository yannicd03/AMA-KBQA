"""Hermetic tests for FindFrequentValues scope-ambiguity and truncation
signals.

Bug context: the tool's ``scope`` parameter defaults to "comparisons", which
silently restricts aggregation to contributions that belong to some Featured
Comparison — excluding every paper never included in one. This caused a
systematic undercount for graph-wide questions ("Top five used research
fields in papers"), even though the tool was called correctly by name in all
three benchmark runs that hit it. Separately, ``limit_subjects`` (default
5000) truncates the scan silently with no indication in the response.

These tests use a routing FakeSPARQL that returns different canned bindings
depending on the query text, since a single FindFrequentValues call can now
issue multiple SPARQL round-trips: scope-population probes, the main
aggregation query, and (when truncated) a total-count probe.
"""

import asyncio
import json
from types import SimpleNamespace

from ama_kbqa.server import sciqa_server as sci


class RoutingFakeSPARQL:
    """Dispatches canned bindings based on the query text of each call."""

    def __init__(self, router):
        self.router = router
        self.queries = []
        self.last_query = None

    def setQuery(self, query):
        self.last_query = query
        self.queries.append(query)

    def query(self):
        return self

    def convert(self):
        return {"results": {"bindings": self.router(self.last_query)}}


def _context(sparql):
    return SimpleNamespace(
        request_context=SimpleNamespace(
            lifespan_context=SimpleNamespace(sparql=sparql)
        )
    )


def _fn(tool):
    """The plain callable behind the @mcp.tool decorator."""
    return getattr(tool, "fn", tool)


def _run(sparql, **kwargs):
    coro = _fn(sci.FindFrequentValues)(_context(sparql), **kwargs)
    return json.loads(asyncio.run(coro))


MAIN_BINDINGS = [
    {
        "valueSubject": {"value": "http://orkg.org/orkg/resource/C1"},
        "valueObj": {"value": "http://orkg.org/orkg/resource/RF1"},
        "valueLabel": {"value": "Machine Learning"},
    },
    {
        "valueSubject": {"value": "http://orkg.org/orkg/resource/C2"},
        "valueObj": {"value": "http://orkg.org/orkg/resource/RF1"},
        "valueLabel": {"value": "Machine Learning"},
    },
    {
        "valueSubject": {"value": "http://orkg.org/orkg/resource/C3"},
        "valueObj": {"value": "http://orkg.org/orkg/resource/RF2"},
        "valueLabel": {"value": "NLP"},
    },
]


def _count_router(comparisons_count, papers_count, main_bindings=MAIN_BINDINGS):
    def router(query):
        if "COUNT(DISTINCT ?valueSubject)" in query:
            if "compareContribution" in query:
                return [{"total": {"value": str(comparisons_count)}}]
            return [{"total": {"value": str(papers_count)}}]
        return main_bindings

    return router


# =============================================================================
# Scope-difference signal
# =============================================================================

def test_scope_autoswitches_to_broader_papers_scope_with_warning():
    """papers scope covers more subjects than the default comparisons scope
    -> the tool must use 'papers' and say so, not silently undercount."""
    sparql = RoutingFakeSPARQL(_count_router(comparisons_count=2, papers_count=10))
    payload = _run(sparql, value_predicate="P30", agg="mode_top")

    assert payload["scope"] == "all_paper_contributions"
    assert payload["scope_population"] == {"comparisons": 2, "papers": 10}
    assert payload["scope_warning"] is not None
    assert "papers" in payload["scope_warning"]
    # The main query actually ran against the papers scope clause.
    main_queries = [q for q in sparql.queries if "SELECT DISTINCT ?valueSubject" in q]
    assert len(main_queries) == 1
    assert "orkgp:P31 ?contrib" in main_queries[0]
    assert "compareContribution" not in main_queries[0]


def test_scope_stays_comparisons_when_it_is_the_broader_one():
    sparql = RoutingFakeSPARQL(_count_router(comparisons_count=10, papers_count=2))
    payload = _run(sparql, value_predicate="P30", agg="mode_top")

    assert payload["scope"] == "all_comparisons"
    assert payload["scope_population"] == {"comparisons": 10, "papers": 2}
    assert payload["scope_warning"] is not None
    assert "comparisons" in payload["scope_warning"]
    main_queries = [q for q in sparql.queries if "SELECT DISTINCT ?valueSubject" in q]
    assert "compareContribution" in main_queries[0]


def test_scope_no_warning_when_populations_are_equal():
    sparql = RoutingFakeSPARQL(_count_router(comparisons_count=5, papers_count=5))
    payload = _run(sparql, value_predicate="P30", agg="mode_top")

    assert payload["scope"] == "all_comparisons"
    assert payload["scope_population"] == {"comparisons": 5, "papers": 5}
    assert payload["scope_warning"] is None


def test_scope_probe_skipped_when_explicit_scope_papers_requested():
    """An explicit scope='papers' call trusts the caller and must not issue
    the extra probe round-trips or override the choice."""
    sparql = RoutingFakeSPARQL(_count_router(comparisons_count=999, papers_count=1))
    payload = _run(sparql, value_predicate="P30", agg="mode_top", scope="papers")

    assert payload["scope"] == "all_paper_contributions"
    assert payload["scope_population"] is None
    assert payload["scope_warning"] is None
    # Only the main query ran — no COUNT probes.
    assert not any("COUNT(DISTINCT ?valueSubject)" in q for q in sparql.queries)


def test_scope_probe_skipped_when_research_field_id_set():
    sparql = RoutingFakeSPARQL(_count_router(comparisons_count=999, papers_count=1))
    payload = _run(
        sparql, value_predicate="P30", agg="mode_top", research_field_id="R132"
    )
    assert payload["scope_population"] is None
    assert payload["scope_warning"] is None


# =============================================================================
# Truncation flag
# =============================================================================

def test_truncation_flag_set_when_limit_subjects_is_hit():
    def router(query):
        if "COUNT(DISTINCT ?valueSubject)" in query:
            return [{"total": {"value": "5"}}]
        return MAIN_BINDINGS[:2]  # exactly hits limit_subjects=2

    sparql = RoutingFakeSPARQL(router)
    payload = _run(
        sparql,
        value_predicate="P30",
        agg="mode_top",
        research_field_id="R132",  # skip scope probing to isolate truncation
        limit_subjects=2,
    )

    assert payload["truncated"] is True
    assert payload["scanned_subjects"] == 2
    assert payload["total_subjects"] == 5
    assert payload["truncation_note"] is not None
    assert "limit_subjects" in payload["truncation_note"]
    assert "TRUNCATED" in payload["status"]


def test_no_truncation_flag_when_under_limit():
    def router(query):
        if "COUNT(DISTINCT ?valueSubject)" in query:
            raise AssertionError("truncation probe should not run when under the limit")
        return MAIN_BINDINGS  # 3 rows, well under the default 5000 cap

    sparql = RoutingFakeSPARQL(router)
    payload = _run(
        sparql,
        value_predicate="P30",
        agg="mode_top",
        research_field_id="R132",
    )

    assert payload["truncated"] is False
    assert payload["total_subjects"] is None
    assert payload["truncation_note"] is None
    assert "TRUNCATED" not in payload["status"]
