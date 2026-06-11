"""Tests for URL-artifact gold answer dereferencing."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from ama_kbqa.utils import artifact_golds
from ama_kbqa.utils.artifact_golds import (
    _format_result_table,
    _query_from_embed_url,
    extract_artifact_url,
    materialize_artifact_gold,
)

EMBED_URL = (
    "https://www.orkg.org/orkg/sparql/embed.html#"
    "select%20%3Flabel%20%3Fvalue%20where%20%7B%20%3Fs%20rdfs%3Alabel%20%3Flabel%20%7D"
)

RAW_RESULTS = {
    "head": {"vars": ["label", "value"]},
    "results": {
        "bindings": [
            {"label": {"value": "all sources"}, "value": {"value": "177.95"}},
            {"label": {"value": "photovoltaics"}, "value": {"value": "50.375"}},
        ]
    },
}


@pytest.fixture(autouse=True)
def clear_cache():
    artifact_golds._materialized_cache.clear()
    yield
    artifact_golds._materialized_cache.clear()


def test_extract_artifact_url_finds_tinyurl_in_prose():
    gold = "Short URL to the result https://tinyurl.com/y4v8w5vb"
    assert extract_artifact_url(gold) == "https://tinyurl.com/y4v8w5vb"


def test_extract_artifact_url_prefers_direct_embed_link():
    assert extract_artifact_url(f"see {EMBED_URL}") == EMBED_URL


def test_extract_artifact_url_ignores_plain_golds_and_orkg_resources():
    assert extract_artifact_url("367.5708") is None
    assert extract_artifact_url('"http://orkg.org/orkg/resource/R6228","QA eval"') is None
    assert extract_artifact_url(None) is None


def test_query_from_embed_url_decodes_fragment():
    query = _query_from_embed_url(EMBED_URL)
    assert query is not None
    assert query.startswith("select ?label ?value")


def test_query_from_embed_url_rejects_non_embed_and_non_select():
    assert _query_from_embed_url("https://example.com/page#fragment") is None
    assert _query_from_embed_url(
        "https://www.orkg.org/orkg/sparql/embed.html#not%20a%20query"
    ) is None


def test_format_result_table_renders_rows_and_truncates():
    table = _format_result_table(RAW_RESULTS, "https://tinyurl.com/x", max_rows=1)
    assert "label | value" in table
    assert "all sources | 177.95" in table
    assert "1 more rows omitted" in table


def test_format_result_table_returns_none_for_empty_results():
    empty = {"head": {"vars": ["x"]}, "results": {"bindings": []}}
    assert _format_result_table(empty, "https://tinyurl.com/x") is None


def test_materialize_artifact_gold_full_path():
    with patch.object(
        artifact_golds, "_resolve_redirect", return_value=EMBED_URL
    ) as resolve, patch.object(
        artifact_golds, "_execute_query", return_value=RAW_RESULTS
    ) as execute:
        gold = "Short URL to the result https://tinyurl.com/y4v8w5vb"
        table = materialize_artifact_gold(gold)

    assert table is not None
    assert "all sources | 177.95" in table
    assert "https://tinyurl.com/y4v8w5vb" in table
    resolve.assert_called_once()
    execute.assert_called_once()


def test_materialize_artifact_gold_skips_non_artifact_golds_without_network():
    with patch.object(artifact_golds, "_resolve_redirect") as resolve:
        assert materialize_artifact_gold("Loyola University Chicago") is None
    resolve.assert_not_called()


def test_materialize_artifact_gold_returns_none_on_resolution_failure():
    with patch.object(artifact_golds, "_resolve_redirect", return_value=None):
        assert materialize_artifact_gold("https://tinyurl.com/dead") is None


def test_materialize_artifact_gold_caches_per_url():
    with patch.object(
        artifact_golds, "_resolve_redirect", return_value=EMBED_URL
    ) as resolve, patch.object(
        artifact_golds, "_execute_query", return_value=RAW_RESULTS
    ):
        first = materialize_artifact_gold("https://tinyurl.com/y4v8w5vb")
        second = materialize_artifact_gold("again: https://tinyurl.com/y4v8w5vb")

    assert first == second
    resolve.assert_called_once()
