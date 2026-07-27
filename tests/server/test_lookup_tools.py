"""Tests for the lexical-lookup tooling added on top of
ama_kbqa/retrieval/lookup.py:

- KQAPro's `_find_node_impl` Phase 1 refactor onto the shared helper (must
  keep exact prior behavior — see tests/server/test_kqapro_server_bugfixes.py
  for the pre-existing Fix B ambiguity tests, which exercise the same code
  path and must keep passing unmodified).
- The new `LookupEntityByName` tool (KQAPro).
- SciQA FindResource's new automatic exact-match phase (the KQAPro/SciQA
  asymmetry the retrieval A/B flagged).
- The new `LookupResourceByLabel` tool (SciQA).

Hermetic: fake Qdrant/SPARQL clients, no live services.
"""

import asyncio
import json
from types import SimpleNamespace

from ama_kbqa.framework.state import JournalState
from ama_kbqa.server import kqapro_server as kqa
from ama_kbqa.server import sciqa_server as sci


def _fn(tool):
    """The plain callable behind the @mcp.tool()/@mcp.tool decorator."""
    return getattr(tool, "fn", tool)


# =============================================================================
# Shared fakes
# =============================================================================

class FakePoint:
    def __init__(self, payload, point_id=0, score=1.0):
        self.id = point_id
        self.payload = payload
        self.score = score


def _matches(payload: dict, should_conditions) -> bool:
    from qdrant_client import models
    for cond in should_conditions:
        if isinstance(cond.match, models.MatchValue) and payload.get(cond.key) == cond.match.value:
            return True
    return False


class FakeQdrant:
    """Evaluates should-filters against an in-memory point list; records
    every scroll call so probe counts can be asserted."""

    def __init__(self, points):
        self.points = points
        self.scroll_calls = []

    def scroll(self, collection_name, scroll_filter, limit, with_payload=True):
        should = scroll_filter.should or []
        self.scroll_calls.append(limit)
        matched = [p for p in self.points if _matches(p.payload or {}, should)]
        return matched[:limit], None


class EmptyFakeSPARQL:
    """Always returns zero bindings — used where a test only cares about the
    Qdrant-side exact-match phase and wants the SPARQL-based predicate/
    lexical-fallback lookups to be harmless no-ops."""

    def setQuery(self, query):
        self.last_query = query

    def setReturnFormat(self, fmt):
        pass

    def query(self):
        return self

    def convert(self):
        return {"results": {"bindings": []}}


def _reset_kqa_journal():
    kqa.session_journal = JournalState()


def _reset_sci_journal():
    sci.session_journal = JournalState()


# =============================================================================
# KQAPro: _find_node_impl Phase 1 refactor (parity is covered by the
# pre-existing test_kqapro_server_bugfixes.py Fix B tests; these add
# coverage for the shared-helper details specifically)
# =============================================================================

def _kqa_context(qdrant):
    return SimpleNamespace(
        request_context=SimpleNamespace(
            lifespan_context=SimpleNamespace(qdrant=qdrant, embedding_client=None)
        )
    )


def test_find_node_impl_delegates_to_shared_lookup(monkeypatch):
    """The refactor must still issue a single scroll call at
    EXACT_MATCH_SCROLL_LIMIT, matching the pre-refactor behavior exactly."""
    _reset_kqa_journal()
    monkeypatch.setattr(kqa.retrieval, "embed_query", lambda client, text: [0.0])
    monkeypatch.setattr(kqa.retrieval, "search", lambda *a, **k: [])

    points = [FakePoint({"original_id": "Q1", "name": "Duran Duran", "attributes": [], "relations": []})]
    qdrant = FakeQdrant(points)

    resp = kqa._find_node_impl("Duran Duran", _kqa_context(qdrant))

    assert qdrant.scroll_calls == [kqa.EXACT_MATCH_SCROLL_LIMIT]
    assert resp.disambiguation_notice is None
    assert [m.original_id for m in resp.matches] == ["Q1"]


# =============================================================================
# KQAPro: LookupEntityByName
# =============================================================================

def test_lookup_entity_by_name_prefix_finds_over_specified_mention():
    """Motivating case: agent holds 'Texas metropolitan area', KB entity is
    named exactly 'Texas'."""
    _reset_kqa_journal()
    qdrant = FakeQdrant([FakePoint({"original_id": "Q1", "name": "Texas", "attributes": [], "relations": []})])

    resp = _fn(kqa.LookupEntityByName)("Texas metropolitan area", _kqa_context(qdrant), mode="prefix")

    assert resp.result_count == 1
    assert resp.matches[0].original_id == "Q1"
    assert resp.matches[0].name == "Texas"
    assert resp.disambiguation_notice is None


def test_lookup_entity_by_name_flags_ambiguous_labels():
    _reset_kqa_journal()
    points = [
        FakePoint({"original_id": f"Q{i}", "name": "Roger Moore", "attributes": [], "relations": []})
        for i in range(13)
    ]
    qdrant = FakeQdrant(points)

    resp = _fn(kqa.LookupEntityByName)("Roger Moore", _kqa_context(qdrant), mode="exact")

    assert resp.disambiguation_notice is not None
    assert "13" in resp.disambiguation_notice
    assert len(resp.matches) == 13


def test_lookup_entity_by_name_no_match_reports_non_exhaustive():
    """A long, entirely-unmatched query should say so without claiming the
    entity is confirmed absent."""
    _reset_kqa_journal()
    qdrant = FakeQdrant([])
    long_query = " ".join(f"word{i}" for i in range(10))

    resp = _fn(kqa.LookupEntityByName)(long_query, _kqa_context(qdrant), mode="contains")

    assert resp.result_count == 0
    assert resp.disambiguation_notice is not None
    assert "does not confirm" in resp.disambiguation_notice


# =============================================================================
# SciQA: FindResource's automatic exact-match phase
# =============================================================================

def _sci_context(qdrant, sparql):
    return SimpleNamespace(
        request_context=SimpleNamespace(
            lifespan_context=SimpleNamespace(qdrant=qdrant, sparql=sparql, embedding_client=None)
        )
    )


def _run_find_resource(qdrant, sparql, semantic_query, **kwargs):
    coro = _fn(sci.FindResource)(_sci_context(qdrant, sparql), semantic_query, **kwargs)
    return json.loads(asyncio.run(coro))


def test_find_resource_exact_phase_surfaces_ambiguity(monkeypatch):
    """SciQA previously had no equivalent to KQAPro's Phase 1 disambiguation
    — this is the new automatic exact-match phase closing that gap."""
    _reset_sci_journal()
    monkeypatch.setattr(sci.retrieval, "embed_query", lambda client, text: [0.0])
    monkeypatch.setattr(sci.retrieval, "search", lambda *a, **k: [])

    points = [
        FakePoint({"uri": f"http://orkg.org/orkg/resource/R{i}", "name": "Text Summarization", "node_type": "paper"})
        for i in range(3)
    ]
    qdrant = FakeQdrant(points)
    sparql = EmptyFakeSPARQL()

    result = _run_find_resource(qdrant, sparql, "Text Summarization")

    assert result["disambiguation_notice"] is not None
    assert "3" in result["disambiguation_notice"]
    assert len(result["matches"]) == 3


def test_find_resource_exact_phase_single_match_merged_no_notice(monkeypatch):
    _reset_sci_journal()
    monkeypatch.setattr(sci.retrieval, "embed_query", lambda client, text: [0.0])
    monkeypatch.setattr(sci.retrieval, "search", lambda *a, **k: [])

    points = [FakePoint({"uri": "http://orkg.org/orkg/resource/R1", "name": "Text Summarization", "node_type": "paper"})]
    qdrant = FakeQdrant(points)
    sparql = EmptyFakeSPARQL()

    result = _run_find_resource(qdrant, sparql, "Text Summarization")

    assert result["disambiguation_notice"] is None
    assert any(m["original_id"] == "R1" for m in result["matches"])


def test_find_resource_exact_phase_respects_node_type_filter(monkeypatch):
    """An exact label match of the wrong ORKG class must not be surfaced
    when a node_type_filter is given."""
    _reset_sci_journal()
    monkeypatch.setattr(sci.retrieval, "embed_query", lambda client, text: [0.0])
    monkeypatch.setattr(sci.retrieval, "search", lambda *a, **k: [])

    points = [FakePoint({"uri": "http://orkg.org/orkg/resource/R1", "name": "Renewables", "node_type": "author"})]
    qdrant = FakeQdrant(points)
    sparql = EmptyFakeSPARQL()

    result = _run_find_resource(qdrant, sparql, "Renewables", node_type_filter="Comparison")

    assert not any(m["original_id"] == "R1" for m in result["matches"])


# =============================================================================
# SciQA: LookupResourceByLabel
# =============================================================================

def _run_lookup_resource(qdrant, sparql, label, **kwargs):
    coro = _fn(sci.LookupResourceByLabel)(_sci_context(qdrant, sparql), label, **kwargs)
    return json.loads(asyncio.run(coro))


def test_lookup_resource_by_label_prefix_finds_over_specified_mention():
    _reset_sci_journal()
    qdrant = FakeQdrant([FakePoint({"uri": "http://orkg.org/orkg/resource/R1", "name": "Text Summarization", "node_type": "paper"})])
    sparql = EmptyFakeSPARQL()

    result = _run_lookup_resource(qdrant, sparql, "text summarization survey 2020", mode="prefix")

    assert result["result_count"] == 1
    assert result["matches"][0]["original_id"] == "R1"
    assert result["disambiguation_notice"] is None


def test_lookup_resource_by_label_ambiguous():
    _reset_sci_journal()
    points = [
        FakePoint({"uri": f"http://orkg.org/orkg/resource/R{i}", "name": "Text Summarization", "node_type": "paper"})
        for i in range(4)
    ]
    qdrant = FakeQdrant(points)
    sparql = EmptyFakeSPARQL()

    result = _run_lookup_resource(qdrant, sparql, "Text Summarization", mode="exact")

    assert result["disambiguation_notice"] is not None
    assert "4" in result["disambiguation_notice"]
    assert len(result["matches"]) == 4


def test_lookup_resource_by_label_no_match_reports_non_exhaustive():
    _reset_sci_journal()
    qdrant = FakeQdrant([])
    sparql = EmptyFakeSPARQL()
    long_label = " ".join(f"word{i}" for i in range(10))

    result = _run_lookup_resource(qdrant, sparql, long_label, mode="contains")

    assert result["result_count"] == 0
    assert result["disambiguation_notice"] is not None
    assert "does not confirm" in result["disambiguation_notice"]
