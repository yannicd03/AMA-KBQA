"""Unit tests for submission._bare_id's Wikidata URI-prefix reduction.

Regression guard for the q113 incident: a committed query whose result
contained `prop/statement/`- and `prop/qualifier/`-namespaced property URIs
(and entity STATEMENT-node URIs) got serialized into the submission as junk
like "statement/Pxxx" instead of the bare "Pxxx" -- which can never match
gold. See ama_kbqa/wikikgqa/dataset.py's `_PROPERTY_PREFIXES` for the fuller
prefix list this aligns with; `to_codabench_answers` formatting itself is
covered separately in test_submission_format.py.
"""

from __future__ import annotations

from ama_kbqa.wikikgqa.submission import _bare_id


def test_bare_id_strips_prop_direct_prefix():
    assert _bare_id("http://www.wikidata.org/prop/direct/P569") == "P569"


def test_bare_id_strips_prop_statement_prefix():
    # Previously fell through to the generic "prop/" prefix and reduced to
    # "statement/P569" instead of the bare id.
    assert _bare_id("http://www.wikidata.org/prop/statement/P569") == "P569"


def test_bare_id_strips_prop_qualifier_prefix():
    # Previously fell through to the generic "prop/" prefix and reduced to
    # "qualifier/P585" instead of the bare id.
    assert _bare_id("http://www.wikidata.org/prop/qualifier/P585") == "P585"


def test_bare_id_strips_generic_prop_prefix():
    assert _bare_id("http://www.wikidata.org/prop/P569") == "P569"


def test_bare_id_strips_entity_prefix():
    assert _bare_id("http://www.wikidata.org/entity/Q42") == "Q42"


def test_bare_id_entity_statement_node_passes_through_as_statement_remainder():
    # A statement node (http://www.wikidata.org/entity/statement/Qxxx-UUID) is
    # not an entity and has no correct bare form -- _bare_id must NOT invent
    # one (e.g. must not silently produce a bare "Qxxx"). It's left with the
    # "statement/..." remainder attached so downstream callers (the
    # generator's _result_is_sane guard) can recognize and reject it rather
    # than let it slip through looking like a plausible answer.
    value = (
        "http://www.wikidata.org/entity/statement/"
        "Q1061678-4137053F-4100-42FE-B13A-89EEFCF8B94E"
    )
    out = _bare_id(value)
    assert out == "statement/Q1061678-4137053F-4100-42FE-B13A-89EEFCF8B94E"
    assert not out.startswith("Q")


def test_bare_id_leaves_literals_untouched():
    assert _bare_id("42") == "42"
    assert _bare_id("1230-01-01T00:00:00Z") == "1230-01-01T00:00:00Z"
