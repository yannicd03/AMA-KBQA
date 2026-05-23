"""Tests for SPARQL result compaction helpers."""

from __future__ import annotations

from ama_kbqa.utils.sparql_results import compact_sparql_select_results


def test_compact_sparql_select_results_truncates_bindings_and_preserves_counts():
    raw_results = {
        "head": {"vars": ["item"]},
        "results": {
            "ordered": True,
            "bindings": [
                {"item": {"type": "uri", "value": "http://orkg.org/orkg/resource/R1"}},
                {"item": {"type": "uri", "value": "http://orkg.org/orkg/resource/R2"}},
                {"item": {"type": "uri", "value": "http://orkg.org/orkg/resource/R3"}},
            ],
        },
    }

    compact = compact_sparql_select_results(
        raw_results,
        max_bindings=2,
        uri_prefixes_to_strip=("http://orkg.org/orkg/resource/",),
    )

    assert compact["bindings"] == [{"item": "R1"}, {"item": "R2"}]
    assert compact["result_count"] == 3
    assert compact["returned_count"] == 2
    assert compact["truncated"] is True
    assert "truncated to 2 of 3 rows" in compact["note"]
    assert len(compact["raw_json"]["results"]["bindings"]) == 2
    assert compact["raw_json"]["results"]["ordered"] is True


def test_compact_sparql_select_results_does_not_add_note_for_short_results():
    raw_results = {
        "head": {"vars": ["count"]},
        "results": {
            "bindings": [
                {"count": {"type": "literal", "value": "7"}},
            ],
        },
    }

    compact = compact_sparql_select_results(raw_results, max_bindings=2)

    assert compact["bindings"] == [{"count": "7"}]
    assert compact["result_count"] == 1
    assert compact["returned_count"] == 1
    assert compact["truncated"] is False
    assert compact["note"] is None
