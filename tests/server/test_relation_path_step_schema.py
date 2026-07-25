"""Regression tests for the relation_path tool-schema robustness fix
(2026-07-25):

`FindEntitiesByRelationPath` (kqapro_server.py) and `FollowRelationPath`
(sciqa_server.py) both took `relation_path: list[Dict[str, str]]` and then
did unguarded `step["relation"]` / `step["predicate"]` dict subscripting
inside the loop that builds the SPARQL query. Two compounding problems were
observed in a live benchmark log:

1. Schema under-specification: pydantic validated "a dict" for each list
   item, but not that it *contains* the relation/predicate key. A caller
   that (reasonably) guessed a different key name — e.g. {"predicate": ...}
   instead of {"relation": ...} — passed schema validation and then blew up
   with a raw `KeyError` once the loop tried `step["relation"]`.
2. Unhelpful error: pydantic's `dict_type` message for a bare-string list
   item never told the caller what keys were expected, so a model reading
   the error had no way to self-correct other than by guessing.

The fix replaces the loose `list[Dict[str, str]]` with a typed
`RelationPathStep` model (one per server module) that:
  - requires a relation-name field, accepting the aliases 'relation',
    'predicate', 'relation_name', 'property' (kqapro) — 'predicate' is the
    primary name for sciqa, with the same aliases as synonyms;
  - defaults `direction` to "forward";
  - runs a `model_validator(mode="before")` that turns "not a dict" and
    "dict missing the relation-name key" into a ValueError naming the
    accepted keys, instead of letting a raw KeyError reach query
    construction.

These tests exercise the fix at two levels, mirroring how the previous
kqapro bugfix test file (test_kqapro_server_bugfixes.py) handles servers
that go through FastMCP's Context plumbing:

  - Schema level: validate raw (dict-like) input through
    `TypeAdapter(list[RelationPathStep])`, exactly what FastMCP's
    `Tool.run()` does internally via `get_cached_typeadapter(fn)` before
    invoking the underlying function. This is what actually reproduces the
    original bug (confirmed by hand against `tool.run(...)` before writing
    these tests) and is the layer where the fix must show up.
  - Function-body level: call the underlying tool function directly with
    already-validated `RelationPathStep` instances (the shape the function
    receives in production once FastMCP's validation succeeds), using the
    same FakeSPARQL/_context hermetic pattern as
    test_kqapro_server_bugfixes.py, to confirm the loop body's switch from
    dict subscripting to attribute access still produces the same query and
    response shape.
"""

import asyncio
import json
from types import SimpleNamespace

from pydantic import TypeAdapter, ValidationError

from ama_kbqa.framework.state import JournalState
from ama_kbqa.server import kqapro_server as kqa
from ama_kbqa.server import sciqa_server as sq


def _fn(tool):
    """The plain callable behind the @mcp.tool()/@mcp.tool decorator."""
    return getattr(tool, "fn", tool)


def _context(sparql, **extra):
    return SimpleNamespace(
        request_context=SimpleNamespace(
            lifespan_context=SimpleNamespace(sparql=sparql, **extra)
        )
    )


def _reset_kqa_journal():
    kqa.session_journal = JournalState()


def _reset_sq_journal():
    sq.session_journal = JournalState()


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


# =============================================================================
# Schema-level: kqapro_server.RelationPathStep
# =============================================================================

KQA_STEP_LIST = TypeAdapter(list[kqa.RelationPathStep])


def test_kqapro_relation_path_step_correct_form_still_works():
    steps = KQA_STEP_LIST.validate_python(
        [{"relation": "location of formation", "direction": "forward"}]
    )
    assert steps[0].relation == "location of formation"
    assert steps[0].direction == "forward"


def test_kqapro_relation_path_step_direction_defaults_to_forward():
    steps = KQA_STEP_LIST.validate_python([{"relation": "shares border with"}])
    assert steps[0].direction == "forward"


def test_kqapro_relation_path_step_predicate_alias_accepted():
    """The exact scenario from the live benchmark log: the agent guessed
    {"predicate": "country_of_origin"} after being told relation_path needs
    dicts. That guess must now succeed instead of raising KeyError."""
    steps = KQA_STEP_LIST.validate_python([{"predicate": "country_of_origin"}])
    assert steps[0].relation == "country_of_origin"
    assert steps[0].direction == "forward"


def test_kqapro_relation_path_step_other_aliases_accepted():
    steps = KQA_STEP_LIST.validate_python([{"relation_name": "owner of"}])
    assert steps[0].relation == "owner of"
    steps = KQA_STEP_LIST.validate_python([{"property": "owner of"}])
    assert steps[0].relation == "owner of"


def test_kqapro_relation_path_step_missing_key_gives_clear_error_not_keyerror():
    try:
        KQA_STEP_LIST.validate_python([{"foo": "bar"}])
        assert False, "expected a ValidationError"
    except ValidationError as exc:
        message = str(exc)
    assert "relation" in message and "predicate" in message
    assert "relation_name" in message and "property" in message
    assert "KeyError" not in message


def test_kqapro_relation_path_step_bare_string_rejected_names_expected_keys():
    """A bare string item (e.g. relation_path=["country_of_origin"]) must be
    rejected with a message naming the accepted keys, not just pydantic's
    generic 'Input should be a valid dictionary'."""
    try:
        KQA_STEP_LIST.validate_python(["country_of_origin"])
        assert False, "expected a ValidationError"
    except ValidationError as exc:
        message = str(exc)
    assert "object" in message or "dict" in message
    assert "relation" in message and "predicate" in message


# =============================================================================
# Schema-level: sciqa_server.RelationPathStep
# =============================================================================

SQ_STEP_LIST = TypeAdapter(list[sq.RelationPathStep])


def test_sciqa_relation_path_step_correct_form_still_works():
    steps = SQ_STEP_LIST.validate_python([{"predicate": "P31", "direction": "forward"}])
    assert steps[0].predicate == "P31"
    assert steps[0].direction == "forward"


def test_sciqa_relation_path_step_direction_defaults_to_forward():
    steps = SQ_STEP_LIST.validate_python([{"predicate": "P31"}])
    assert steps[0].direction == "forward"


def test_sciqa_relation_path_step_relation_alias_accepted():
    steps = SQ_STEP_LIST.validate_python([{"relation": "P31"}])
    assert steps[0].predicate == "P31"


def test_sciqa_relation_path_step_missing_key_gives_clear_error_not_keyerror():
    try:
        SQ_STEP_LIST.validate_python([{"foo": "bar"}])
        assert False, "expected a ValidationError"
    except ValidationError as exc:
        message = str(exc)
    assert "predicate" in message and "relation" in message
    assert "KeyError" not in message


def test_sciqa_relation_path_step_bare_string_rejected_names_expected_keys():
    try:
        SQ_STEP_LIST.validate_python(["P31"])
        assert False, "expected a ValidationError"
    except ValidationError as exc:
        message = str(exc)
    assert "object" in message or "dict" in message
    assert "predicate" in message


# =============================================================================
# Function-body level: FindEntitiesByRelationPath (kqapro_server.py)
# =============================================================================

def test_find_entities_by_relation_path_still_navigates_with_typed_steps():
    """Post-fix, the function body reads `step.relation` / `step.direction`
    (attribute access) instead of dict subscripting. Confirm the SPARQL
    query and response shape are unchanged for the common case."""
    _reset_kqa_journal()
    bindings = [
        {"hop0": {"value": "http://kqapro.org/entity/Q678410"},
         "hop1": {"value": "http://kqapro.org/entity/Q213474"}},
    ]
    sparql = FakeSPARQL(bindings)
    steps = KQA_STEP_LIST.validate_python(
        [{"relation": "location of formation", "direction": "forward"}]
    )

    result = _fn(kqa.FindEntitiesByRelationPath)(
        start_node_id="Q678410",
        relation_path=steps,
        context=_context(sparql),
    )

    query = sparql.last_query
    assert "<http://kqapro.org/property/location_of_formation>" in query
    assert "?hop0 <http://kqapro.org/property/location_of_formation> ?hop1 ." in query

    assert result["entities_found"] == ["Q213474"]
    # relation_path in the response is plain-dict serializable (not a
    # RelationPathStep instance) so downstream JSON/journal writers keep
    # working unchanged.
    assert result["relation_path"] == [{"relation": "location of formation", "direction": "forward"}]
    json.dumps(result)  # must be plain-JSON-serializable


def test_find_entities_by_relation_path_backward_direction_with_predicate_alias():
    """End-to-end: a step built from the {"predicate": ...} alias (the
    exact shape the agent retried with in the live log) must produce the
    same backward-direction query as the canonical {"relation": ...} form."""
    _reset_kqa_journal()
    sparql = FakeSPARQL([])
    steps = KQA_STEP_LIST.validate_python(
        [{"predicate": "country_of_origin", "direction": "backward"}]
    )

    result = _fn(kqa.FindEntitiesByRelationPath)(
        start_node_id="Q1",
        relation_path=steps,
        context=_context(sparql),
    )

    query = sparql.last_query
    assert "?hop1 <http://kqapro.org/property/country_of_origin> ?hop0 ." in query
    assert result["status"] == "No entities found following this path"
    assert result["relation_path"] == [{"relation": "country_of_origin", "direction": "backward"}]


# =============================================================================
# Function-body level: FollowRelationPath (sciqa_server.py)
# =============================================================================

def test_follow_relation_path_still_navigates_with_typed_steps():
    _reset_sq_journal()
    bindings = [
        {"hop0": {"value": "http://orkg.org/orkg/resource/R1", "type": "uri"},
         "hop1": {"value": "http://orkg.org/orkg/resource/R2", "type": "uri"}},
    ]
    sparql = FakeSPARQL(bindings)
    steps = SQ_STEP_LIST.validate_python([{"predicate": "P31", "direction": "forward"}])

    app = SimpleNamespace(sparql=sparql)
    context = SimpleNamespace(request_context=SimpleNamespace(lifespan_context=app))

    coro = _fn(sq.FollowRelationPath)(
        app_context=context,
        start_resource_id="R1",
        relation_path=steps,
    )
    result = json.loads(asyncio.run(coro))

    query = sparql.last_query
    assert "?hop0 orkgp:P31 ?hop1 ." in query
    assert result["entities_found"] == [{"id": "R2", "label": "R2"}]
    assert result["relation_path"] == [{"predicate": "P31", "direction": "forward"}]


def test_follow_relation_path_relation_alias_produces_same_query():
    _reset_sq_journal()
    sparql = FakeSPARQL([])
    steps = SQ_STEP_LIST.validate_python([{"relation": "P31", "direction": "backward"}])

    app = SimpleNamespace(sparql=sparql)
    context = SimpleNamespace(request_context=SimpleNamespace(lifespan_context=app))

    coro = _fn(sq.FollowRelationPath)(
        app_context=context,
        start_resource_id="R1",
        relation_path=steps,
    )
    result = json.loads(asyncio.run(coro))

    query = sparql.last_query
    assert "?hop1 orkgp:P31 ?hop0 ." in query
    assert result["relation_path"] == [{"predicate": "P31", "direction": "backward"}]
