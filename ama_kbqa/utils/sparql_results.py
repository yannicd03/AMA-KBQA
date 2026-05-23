"""Utilities for making raw SPARQL responses safe for agent context."""

from __future__ import annotations

from typing import Any, Iterable


DEFAULT_MAX_SPARQL_BINDINGS = 50


def _shorten_value(value: str, uri_prefixes_to_strip: Iterable[str]) -> str:
    for prefix in uri_prefixes_to_strip:
        if value.startswith(prefix):
            return value.removeprefix(prefix).strip("/")
    return value


def compact_sparql_select_results(
    raw_results: dict[str, Any],
    vars_list: list[str] | None = None,
    *,
    max_bindings: int = DEFAULT_MAX_SPARQL_BINDINGS,
    uri_prefixes_to_strip: Iterable[str] = (),
) -> dict[str, Any]:
    """Return bounded, simplified SELECT results plus truncation metadata."""
    vars_list = vars_list or raw_results.get("head", {}).get("vars", [])
    raw_result_block = raw_results.get("results", {})
    raw_bindings = raw_result_block.get("bindings", [])
    returned_raw_bindings = raw_bindings[:max_bindings]

    simplified_bindings: list[dict[str, str]] = []
    for binding in returned_raw_bindings:
        row = {}
        for var in vars_list:
            if var in binding:
                value = binding[var].get("value", "")
                row[var] = _shorten_value(value, uri_prefixes_to_strip)
        simplified_bindings.append(row)

    compact_raw_results = {
        key: value
        for key, value in raw_results.items()
        if key != "results"
    }
    compact_raw_results["results"] = {
        key: value
        for key, value in raw_result_block.items()
        if key != "bindings"
    }
    compact_raw_results["results"]["bindings"] = returned_raw_bindings

    result_count = len(raw_bindings)
    returned_count = len(returned_raw_bindings)
    truncated = result_count > returned_count
    note = None
    if truncated:
        note = (
            f"SPARQL result truncated to {returned_count} of {result_count} rows. "
            "Refine the query with COUNT, GROUP BY, LIMIT/OFFSET, or a dedicated tool."
        )

    compact_raw_results["result_count"] = result_count
    compact_raw_results["returned_count"] = returned_count
    compact_raw_results["truncated"] = truncated
    if note:
        compact_raw_results["note"] = note

    return {
        "vars": vars_list,
        "bindings": simplified_bindings,
        "raw_json": compact_raw_results,
        "result_count": result_count,
        "returned_count": returned_count,
        "truncated": truncated,
        "note": note,
    }
