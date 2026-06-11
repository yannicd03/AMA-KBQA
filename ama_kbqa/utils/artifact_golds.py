"""Dereference URL-artifact gold answers into judgeable result tables.

Some SciQA benchmark gold answers are not answers at all but pointers:
"Short URL to the result https://tinyurl.com/..." redirecting to an ORKG
SPARQL embed page whose URL fragment contains the gold query. An LLM judge
comparing a predicted table against such an opaque link marks every answer
wrong. This module resolves the link, extracts the query, executes it
against the project's Virtuoso store, and renders the rows as a compact
table the judge can actually compare against.

Every step degrades gracefully: any failure (no URL, network error, no
query fragment, empty result) returns None and the caller keeps the raw
gold answer, so evaluation never gets worse than the status quo.
"""

from __future__ import annotations

import re
import urllib.parse
from typing import Any, Dict, List, Optional

import requests

TINYURL_RE = re.compile(r"https?://(?:www\.)?tinyurl\.com/[A-Za-z0-9]+", re.IGNORECASE)
ORKG_EMBED_RE = re.compile(
    r"https?://(?:www\.)?orkg\.org/orkg/sparql/embed\.html#\S+", re.IGNORECASE
)

DEFAULT_MAX_GOLD_ROWS = 40
_RESOLVE_TIMEOUT_SECONDS = 15
_QUERY_TIMEOUT_SECONDS = 60

# Per-process cache: benchmark runs judge the same artifact gold repeatedly.
_materialized_cache: Dict[str, Optional[str]] = {}


def extract_artifact_url(gold_answer: Optional[str]) -> Optional[str]:
    """Return the first artifact URL in a gold answer, or None."""
    text = str(gold_answer or "")
    embed = ORKG_EMBED_RE.search(text)
    if embed:
        return embed.group(0)
    tiny = TINYURL_RE.search(text)
    if tiny:
        return tiny.group(0)
    return None


def _resolve_redirect(url: str, timeout: float = _RESOLVE_TIMEOUT_SECONDS) -> Optional[str]:
    """Follow one redirect hop without losing the URL fragment.

    requests/urllib drop or mangle fragments when auto-following redirects,
    and the ORKG embed link carries the gold query in its fragment, so the
    Location header must be read directly.
    """
    try:
        response = requests.head(url, allow_redirects=False, timeout=timeout)
    except requests.RequestException:
        return None
    location = response.headers.get("Location")
    if location:
        return location
    if response.status_code == 200:
        return url
    return None


def _query_from_embed_url(url: str) -> Optional[str]:
    """Extract the SPARQL query from an ORKG sparql/embed.html fragment."""
    if not url or "sparql/embed.html#" not in url.lower():
        return None
    fragment = url.split("#", 1)[1]
    query = urllib.parse.unquote(fragment).strip()
    if not re.search(r"\bselect\b", query, re.IGNORECASE):
        return None
    return query


def _execute_query(
    query: str, endpoint: Optional[str], timeout: float = _QUERY_TIMEOUT_SECONDS
) -> Optional[Dict[str, Any]]:
    if endpoint is None:
        from ama_kbqa.config import get_virtuoso_endpoint

        endpoint = get_virtuoso_endpoint()
    try:
        response = requests.post(
            endpoint,
            data={"query": query, "format": "application/sparql-results+json"},
            timeout=timeout,
        )
        response.raise_for_status()
        return response.json()
    except (requests.RequestException, ValueError):
        return None


def _format_result_table(
    raw_results: Dict[str, Any], source_url: str, max_rows: int = DEFAULT_MAX_GOLD_ROWS
) -> Optional[str]:
    vars_list: List[str] = raw_results.get("head", {}).get("vars", [])
    bindings = raw_results.get("results", {}).get("bindings", [])
    if not vars_list or not bindings:
        return None
    lines = [
        "The gold answer is the result of a SPARQL query the benchmark links to "
        f"({source_url}). Judge the predicted answer against this result table:",
        " | ".join(vars_list),
    ]
    for binding in bindings[:max_rows]:
        lines.append(
            " | ".join(binding.get(var, {}).get("value", "") for var in vars_list)
        )
    if len(bindings) > max_rows:
        lines.append(f"... ({len(bindings) - max_rows} more rows omitted)")
    return "\n".join(lines)


def materialize_artifact_gold(
    gold_answer: Optional[str],
    *,
    endpoint: Optional[str] = None,
    max_rows: int = DEFAULT_MAX_GOLD_ROWS,
) -> Optional[str]:
    """Turn a URL-artifact gold answer into a judgeable result table.

    Returns the rendered table, or None when the gold is not an artifact or
    any resolution step fails (caller should fall back to the raw gold).
    """
    artifact_url = extract_artifact_url(gold_answer)
    if artifact_url is None:
        return None

    cache_key = artifact_url
    if cache_key in _materialized_cache:
        return _materialized_cache[cache_key]

    materialized: Optional[str] = None
    resolved = artifact_url
    if TINYURL_RE.match(artifact_url):
        resolved = _resolve_redirect(artifact_url) or ""
    query = _query_from_embed_url(resolved)
    if query:
        raw_results = _execute_query(query, endpoint)
        if raw_results:
            materialized = _format_result_table(raw_results, artifact_url, max_rows)

    _materialized_cache[cache_key] = materialized
    return materialized
