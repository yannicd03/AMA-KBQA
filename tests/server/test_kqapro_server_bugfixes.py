"""Hermetic tests for four independently-verified kqapro_server bug fixes
(2026-07-25):

- Fix A: GetEdgeQualifiers / GetQualifierValue's attribute-mode branch only
  matched blank-node-wrapped attribute values (`?e attr:k ?bnode ; ?bnode
  rdf:value ?v`). Many KQAPro attributes (dates, ID numbers, plain numbers)
  are stored as direct literals instead (`?e attr:k "literal"`), which
  can't be RDF subjects, so the qualifier join silently failed. Fixed by
  UNION'ing a second pattern matching the reified statement whose
  `rdf:object` is the literal directly.
- Fix B: FindNode's exact-match Qdrant scroll was capped at limit=5, making
  entities with common duplicate labels (e.g. 13 KG entities named "Roger
  Moore") structurally unreachable — whichever 5 Qdrant happened to return
  first silently won, and vector similarity can't discriminate identical
  labels. Fixed by raising the scroll limit and surfacing a
  disambiguation_notice when more than one exact match exists.
- Fix C: CountEntities.or_conditions only accepted attribute-shaped
  branches; relation-shaped branches (main_subject, location, etc. — which
  live under prop:, not attr:) were silently dropped while the tool still
  reported trusted=True. Fixed by accepting relation-shaped or_conditions
  (reusing CountUnion's relation-block builder) and by never silently
  dropping any branch: malformed/nonexistent branches now surface an
  explicit error and force trusted=False.
- Fix D: GetJournalSummary's formatter assumed specific key shapes
  (value/related_id for found_values, subject/relation/related_id for
  verified_facts) that many tools — RunSPARQL chief among them — don't
  produce, rendering literal "?" placeholders. Fixed by handling the actual
  shapes each tool writes and falling back to the raw entry instead of "?".
- Fix E: FindByAttribute spliced the raw attribute value directly into a
  URI literal (`?attrValue = <{NS_ENTITY}{value}>`) while only guarding it
  with a runtime `!CONTAINS(value, " ")` FILTER. That guard is evaluated at
  SPARQL runtime, but the `<...>` fragment is static query TEXT, so any
  value containing a space (e.g. ISNI codes like "0000 0003 6864 0824")
  produced an unparseable IRI and the whole query failed with
  QueryBadFormed before the FILTER ever ran. Fixed by deciding in Python,
  at query-construction time, whether a value could possibly be a valid
  entity ID (no whitespace or reserved IRI characters) and only emitting
  the URI-comparison branch then. Also hardened the literal-comparison
  branch's escaping, which only escaped double quotes and left backslash,
  newline, and CR unescaped.

No live Virtuoso/Qdrant: fakes return canned bindings/points and record the
generated query for inspection, following the pattern in
tests/server/test_count_and_aggregate_hardening.py.
"""

import asyncio
import json
from types import SimpleNamespace

from ama_kbqa.framework.state import JournalState
from ama_kbqa.server import kqapro_server as kqa


def _fn(tool):
    """The plain callable behind the @mcp.tool()/@mcp.tool decorator."""
    return getattr(tool, "fn", tool)


def _context(sparql, **extra):
    return SimpleNamespace(
        request_context=SimpleNamespace(
            lifespan_context=SimpleNamespace(sparql=sparql, **extra)
        )
    )


def _reset_journal():
    """CountEntities/GetEdgeQualifiers/etc. write to a module-level
    singleton journal; reset it so tests don't see each other's state."""
    kqa.session_journal = JournalState()


# =============================================================================
# Fix A: GetEdgeQualifiers / GetQualifierValue direct-literal attribute shape
# =============================================================================

class FakeSPARQL:
    def __init__(self, bindings):
        self._bindings = bindings
        self.last_query = None

    def setQuery(self, query):
        self.last_query = query

    def setReturnFormat(self, fmt):
        pass

    def query(self):
        return self

    def convert(self):
        return {"results": {"bindings": self._bindings}}


def test_get_edge_qualifiers_bnode_wrapped_shape():
    """Q166776 / attr:ranking "64" is bnode-wrapped and carries
    qual:review_score_by "FIFA" — the pre-existing shape must keep working."""
    _reset_journal()
    bindings = [{
        "value": {"type": "literal", "value": "64"},
        "qkey": {"type": "literal", "value": "review_score_by"},
        "qval": {"type": "literal", "value": "FIFA"},
    }]
    sparql = FakeSPARQL(bindings)
    coro = _fn(kqa.GetEdgeQualifiers)(
        _context(sparql),
        base_node_id="Q166776",
        attribute_name="ranking",
        attribute_value="64",
    )
    result = json.loads(asyncio.run(coro))

    query = sparql.last_query
    # The fix must UNION both storage shapes in a single query.
    assert "?bnode rdf:value ?value" in query
    assert "isLiteral(?value)" in query

    # The value filter MUST compare on STR(). Verified against live Virtuoso:
    # attr:ranking values are stored as xsd:decimal, so a bare
    # `FILTER(?value = "64")` matches nothing and silently returns zero
    # qualifiers for data that is present. This is what made idx 68
    # ("who was the reviewer... ranking of 64" -> FIFA) unanswerable.
    assert 'FILTER(STR(?value) = "64")' in query
    assert 'FILTER(?value = "64")' not in query

    assert result["qualifiers"]["review_score_by"][0]["value"] == "FIFA"


def test_get_edge_qualifiers_direct_literal_shape():
    """Q63366 / attr:exploitation_visa_number "116454" is a DIRECT literal
    (no bnode) and carries qual:start_time "2006-11-20" on the reification
    statement whose rdf:object is the literal itself. Before the fix this
    returned {} because literals can't be RDF subjects for the bnode join."""
    _reset_journal()
    bindings = [{
        "value": {"type": "literal", "value": "116454"},
        "qkey": {"type": "literal", "value": "start_time"},
        "qval": {"type": "literal", "value": "2006-11-20"},
    }]
    sparql = FakeSPARQL(bindings)
    coro = _fn(kqa.GetEdgeQualifiers)(
        _context(sparql),
        base_node_id="Q63366",
        attribute_name="exploitation visa number",
        attribute_value="116454",
    )
    result = json.loads(asyncio.run(coro))

    query = sparql.last_query
    assert "attr:exploitation_visa_number" in query
    assert "isLiteral(?value)" in query
    # STR()-normalised for the same reason as the bnode case above: direct
    # literals are typed too (this one is xsd:date-adjacent ID data).
    assert 'FILTER(STR(?value) = "116454")' in query

    assert result["qualifiers"]["start_time"][0]["value"] == "2006-11-20"
    assert result["value"] == "116454"


def test_get_qualifier_value_attribute_bnode_wrapped_shape():
    _reset_journal()
    bindings = [{"qval": {"type": "literal", "value": "FIFA"}}]
    sparql = FakeSPARQL(bindings)
    coro = _fn(kqa.GetQualifierValue)(
        _context(sparql),
        subject_id="Q166776",
        predicate="ranking",
        target="64",
        qualifier_name="review score by",
        predicate_type="attribute",
    )
    result = json.loads(asyncio.run(coro))

    query = sparql.last_query
    assert "?bnode rdf:value ?targetValue" in query
    assert "isLiteral(?targetValue)" in query

    assert result["values"][0]["value"] == "FIFA"
    assert result["direction"] == "forward"


def test_get_qualifier_value_attribute_direct_literal_shape():
    _reset_journal()
    bindings = [{"qval": {"type": "literal", "value": "2006-11-20"}}]
    sparql = FakeSPARQL(bindings)
    coro = _fn(kqa.GetQualifierValue)(
        _context(sparql),
        subject_id="Q63366",
        predicate="exploitation visa number",
        target="116454",
        qualifier_name="start time",
        predicate_type="attribute",
    )
    result = json.loads(asyncio.run(coro))

    query = sparql.last_query
    assert "isLiteral(?targetValue)" in query
    assert 'FILTER(STR(?targetValue) = "116454")' in query

    assert result["values"][0]["value"] == "2006-11-20"
    assert result["direction"] == "forward"


# =============================================================================
# Fix B: FindNode exact-match ambiguity (duplicate labels)
# =============================================================================

class FakePoint:
    def __init__(self, payload, score=1.0):
        self.payload = payload
        self.score = score


class FakeQdrant:
    def __init__(self, points):
        self._points = points
        self.scroll_limits = []

    def scroll(self, collection_name, scroll_filter, limit, with_payload):
        self.scroll_limits.append(limit)
        return self._points[:limit], None


def _search_context(qdrant):
    return SimpleNamespace(
        request_context=SimpleNamespace(
            lifespan_context=SimpleNamespace(qdrant=qdrant, embedding_client=None)
        )
    )


def test_find_node_surfaces_ambiguity_and_raises_scroll_limit(monkeypatch):
    """13 KG entities are named 'Roger Moore'; the one born 1943 previously
    fell outside the hardcoded limit=5 scroll and was structurally
    unreachable. The scroll limit must rise well above 5, and the response
    must flag the ambiguity instead of silently trusting the first hits."""
    _reset_journal()
    monkeypatch.setattr(kqa.retrieval, "embed_query", lambda client, text: [0.0])
    monkeypatch.setattr(kqa.retrieval, "search", lambda *a, **k: [])

    points = [
        FakePoint({"original_id": f"Q{i}", "name": "Roger Moore", "attributes": [], "relations": []})
        for i in range(13)
    ]
    qdrant = FakeQdrant(points)

    resp = kqa._find_node_impl("Roger Moore", _search_context(qdrant))

    assert qdrant.scroll_limits == [50]
    assert resp.disambiguation_notice is not None
    assert "13" in resp.disambiguation_notice
    returned_ids = {m.original_id for m in resp.matches}
    # Previously only ~5 of 13 would have survived; the target entity must
    # now be reachable regardless of Qdrant's internal scroll ordering.
    assert "Q10" in returned_ids
    assert "Q12" in returned_ids
    assert len(resp.matches) == 13


def test_find_node_single_exact_match_unaffected(monkeypatch):
    """The common case (one exact match) must keep its existing shape: no
    disambiguation_notice, TOP_N-bounded merge with semantic results."""
    _reset_journal()
    monkeypatch.setattr(kqa.retrieval, "embed_query", lambda client, text: [0.0])
    monkeypatch.setattr(kqa.retrieval, "search", lambda *a, **k: [])

    points = [FakePoint({"original_id": "Q1", "name": "Duran Duran", "attributes": [], "relations": []})]
    qdrant = FakeQdrant(points)

    resp = kqa._find_node_impl("Duran Duran", _search_context(qdrant))

    assert resp.disambiguation_notice is None
    assert [m.original_id for m in resp.matches] == ["Q1"]


# =============================================================================
# Fix C: CountEntities.or_conditions relation-shaped branches
# =============================================================================

COUNT_BINDINGS = [{"c": {"value": "7"}}]


class FakeSPARQLAsk:
    """FakeSPARQL for CountEntities: answers ASK existence probes from a
    caller-supplied set of predicate URIs that "exist" in the KG, and
    returns canned bindings for the final SELECT COUNT query."""

    def __init__(self, count_bindings, existing_uris=()):
        self._count_bindings = count_bindings
        self._existing_uris = set(existing_uris)
        self.last_query = None
        self.queries = []

    def setQuery(self, query):
        self.last_query = query
        self.queries.append(query)

    def setReturnFormat(self, fmt):
        pass

    def query(self):
        return self

    def convert(self):
        if "ASK {" in (self.last_query or ""):
            for uri in self._existing_uris:
                if uri in self.last_query:
                    return {"boolean": True}
            return {"boolean": False}
        return {"results": {"bindings": self._count_bindings}}


def test_count_entities_or_conditions_relation_shaped_branch_counts():
    """main_subject/location exist only under prop:, never attr: (1,135 vs 0
    occurrences in the KG). A relation-shaped or_conditions branch must be
    accepted and folded into the UNION, not silently dropped."""
    _reset_journal()
    sparql = FakeSPARQLAsk(COUNT_BINDINGS)
    resp = _fn(kqa.CountEntities)(
        _context(sparql),
        concept="film",
        or_conditions=[{"relation_name": "main_subject", "relation_target_id": "Q100"}],
    )
    assert resp.count == 7
    assert resp.trusted is True
    assert "UNTRUSTED" not in resp.status
    assert "trusted count" in resp.status

    query = sparql.last_query
    assert "<http://kqapro.org/property/main_subject>" in query
    assert "<http://kqapro.org/entity/Q100>" in query


def test_count_union_relation_block_unchanged_after_refactor():
    """CountUnion's relation-block building was refactored into a shared
    helper (_relation_branch_sparql); the generated query must be byte-for-
    byte equivalent to before."""
    _reset_journal()
    sparql = FakeSPARQLAsk(COUNT_BINDINGS)
    resp = _fn(kqa.CountUnion)(
        branches=[{"relation_name": "main_subject", "relation_target_id": "Q100"}],
        context=_context(sparql),
    )
    assert resp.count == 7
    assert resp.trusted is True
    query = sparql.last_query
    assert "<http://kqapro.org/property/main_subject>" in query
    assert "?relTarget_u0" in query


def test_count_entities_or_conditions_malformed_branch_is_not_silently_dropped():
    """A branch that is neither attribute-shaped nor relation-shaped must
    surface an explicit error and force trusted=False, never a confident
    (and wrong) count."""
    _reset_journal()
    sparql = FakeSPARQLAsk(COUNT_BINDINGS)
    resp = _fn(kqa.CountEntities)(
        _context(sparql),
        concept="film",
        or_conditions=[{"foo": "bar"}],
    )
    assert resp.trusted is False
    assert "or_condition" in resp.status
    assert "attribute_name" in resp.status and "relation_name" in resp.status
    # The count still executes over the remaining valid filters (concept
    # alone here) — it just must not be trusted as the final answer.
    assert resp.count == 7


def test_count_entities_or_conditions_nonexistent_attribute_flags_prop_hint():
    """An or_conditions branch that names a real predicate under the wrong
    namespace (attribute_name='main_subject', which only exists under
    prop:) must be rejected with a hint instead of silently contributing
    zero matches to the union."""
    _reset_journal()
    sparql = FakeSPARQLAsk(
        COUNT_BINDINGS,
        existing_uris=["http://kqapro.org/property/main_subject"],
    )
    resp = _fn(kqa.CountEntities)(
        _context(sparql),
        concept="film",
        or_conditions=[{"attribute_name": "main_subject", "attribute_value": "Q100"}],
    )
    assert resp.trusted is False
    assert "main_subject" in resp.status
    assert "does not exist" in resp.status
    assert "prop:" in resp.status


def test_count_entities_or_conditions_unknown_predicate_no_hint():
    """When neither attr: nor prop: has the predicate, the branch is still
    rejected (trusted=False) but without a misleading prop: hint."""
    _reset_journal()
    sparql = FakeSPARQLAsk(COUNT_BINDINGS, existing_uris=[])
    resp = _fn(kqa.CountEntities)(
        _context(sparql),
        concept="film",
        or_conditions=[{"attribute_name": "totally_made_up_field", "attribute_value": "x"}],
    )
    assert resp.trusted is False
    assert "totally_made_up_field" in resp.status
    assert "prop:" not in resp.status


# =============================================================================
# Fix D: GetJournalSummary rendering of RunSPARQL-sourced journal entries
# =============================================================================

def _journal_context():
    return SimpleNamespace(request_context=SimpleNamespace(lifespan_context=SimpleNamespace()))


def test_journal_summary_renders_runsparql_found_values_without_placeholder():
    """RunSPARQL's found_values entries are {query, results, result_count,
    ...} where `results` is a list of raw {sparql_var: value} row dicts —
    not the {value, related_id} qualifier shape. No line may render a bare
    "?" placeholder."""
    _reset_journal()
    kqa.session_journal.found_values["sparql_result_1"] = {
        "query": "SELECT ?c WHERE { ... }",
        "results": [{"c": "42"}],
        "result_count": 1,
        "returned_count": 1,
        "truncated": False,
    }
    summary = _fn(kqa.GetJournalSummary)(_journal_context())

    assert "✓ results: ?" not in summary
    assert "c=42" in summary


def test_journal_summary_renders_runsparql_verified_fact_without_placeholder():
    """RunSPARQL's verified_facts entry is {query_type, variables,
    result_count, ..., source: 'RunSPARQL'} — none of subject/relation/
    related_id. The old formatter rendered '• ? (?) ->[?]-> ?'."""
    _reset_journal()
    kqa.session_journal.verified_facts.append({
        "query_type": "SPARQL",
        "variables": ["c"],
        "result_count": 1,
        "returned_count": 1,
        "truncated": False,
        "results": [{"c": "42"}],
        "source": "RunSPARQL",
    })
    summary = _fn(kqa.GetJournalSummary)(_journal_context())

    assert "• ? (?) ->[?]-> ?" not in summary
    assert "RunSPARQL" in summary
    assert "1 result" in summary


def test_journal_summary_renders_fact_shaped_entries():
    """The {"fact": ..., "source": ...} shape used by CountEntities,
    VerifyFact, SelectExtreme, etc. must render the fact text, not "?"."""
    _reset_journal()
    kqa.session_journal.verified_facts.append({
        "fact": "TRUSTED COUNT[type=film] = 7 (exact COUNT(DISTINCT) result)",
        "source": "CountEntities",
    })
    summary = _fn(kqa.GetJournalSummary)(_journal_context())

    assert "• ? (?) ->[?]-> ?" not in summary
    assert "TRUSTED COUNT[type=film] = 7" in summary


def test_journal_summary_still_renders_relation_triples():
    """Regression guard: the pre-existing subject/relation/related_id shape
    (GetRelationDetails/GetRelationBetween) must still render as a triple."""
    _reset_journal()
    kqa.session_journal.visited_nodes["Q1"] = "Duran Duran"
    kqa.session_journal.visited_nodes["Q2"] = "United Kingdom"
    kqa.session_journal.verified_facts.append({
        "subject": "Q1",
        "relation": "country",
        "related_id": "Q2",
        "direction": "forward",
        "source": "GetRelationDetails",
    })
    summary = _fn(kqa.GetJournalSummary)(_journal_context())

    assert "Duran Duran (Q1) ->[country]-> United Kingdom (Q2)" in summary


# =============================================================================
# Fix E: FindByAttribute parse-time-vs-runtime URI construction
# =============================================================================

def test_find_by_attribute_whitespace_value_omits_uri_branch():
    """An ISNI-shaped value ("0000 0003 6864 0824") contains spaces and can
    never be a valid entity ID. The URI-comparison branch must not be
    generated at all — not merely guarded by a runtime FILTER, since the
    `<...>` fragment is static query text and a space inside it makes the
    whole query fail to parse."""
    _reset_journal()
    isni = "0000 0003 6864 0824"
    sparql = FakeSPARQL([{
        "entity": {"type": "uri", "value": "http://kqapro.org/entity/Q42"},
        "entityName": {"type": "literal", "value": "Some Person"},
    }])
    resp = _fn(kqa.FindByAttribute)(
        value=isni,
        attribute_name="ISNI",
        context=_context(sparql),
    )
    query = sparql.last_query

    # No URI branch at all for a whitespace-bearing value (PREFIX ex: is
    # always present, but the ?attrValue = <...> comparison must not be).
    assert "?attrValue = <" not in query
    # The literal-comparison branch is preserved, value intact (no escaping
    # needed here since there's no quote/backslash).
    assert f'STR(?attrValue) = "{isni}"' in query
    # No leftover debug comments from the earlier broken fix attempt.
    assert "HIER" not in query
    assert "DAS L" not in query
    assert resp.result_count == 1


def test_find_by_attribute_entity_id_shape_includes_uri_branch():
    """A plain, whitespace-free value like a NUTS code still gets compared
    against a URI built from the value, exactly as before the fix."""
    _reset_journal()
    sparql = FakeSPARQL([{
        "entity": {"type": "uri", "value": "http://kqapro.org/entity/Q99"},
        "entityName": {"type": "literal", "value": "Some City"},
    }])
    _fn(kqa.FindByAttribute)(
        value="UKE11",
        attribute_name="NUTS code",
        context=_context(sparql),
    )
    query = sparql.last_query

    assert "?attrValue = <http://kqapro.org/entity/UKE11>" in query
    assert 'STR(?attrValue) = "UKE11"' in query


def test_find_by_attribute_quote_bearing_value_is_escaped():
    """A value containing a double quote must not break the generated
    string literal; it must be backslash-escaped, and (since a quote is not
    valid inside an IRI) the URI branch must be omitted."""
    _reset_journal()
    value = 'the "best" one'
    sparql = FakeSPARQL([])
    _fn(kqa.FindByAttribute)(
        value=value,
        attribute_name="nickname",
        context=_context(sparql),
    )
    query = sparql.last_query

    assert 'STR(?attrValue) = "the \\"best\\" one"' in query
    assert '"the "best" one"' not in query  # the naive, unescaped form
    assert "?attrValue = <" not in query


def test_find_by_attribute_backslash_and_newline_are_escaped():
    """Backslashes and embedded newlines must also be escaped in the string
    literal branch — the pre-fix code only escaped double quotes, so a raw
    backslash or newline could still desync the SPARQL string literal."""
    _reset_journal()
    value = 'a\\b\nc'
    sparql = FakeSPARQL([])
    _fn(kqa.FindByAttribute)(
        value=value,
        attribute_name="raw_id",
        context=_context(sparql),
    )
    query = sparql.last_query

    assert 'STR(?attrValue) = "a\\\\b\\nc"' in query
