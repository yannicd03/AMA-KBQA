"""SPARQL execution against the WikiKGQA evaluation endpoint.

The challenge answers must be the W3C SPARQL-JSON results produced by running a
query against the official endpoint (a fixed Wikidata version). This module is
the single execution seam:

* ``default`` endpoint is public WDQS, used for local development. The official
  challenge endpoint URL (Codabench, registration-gated) is dropped in via the
  ``endpoint`` argument or the ``WIKIKGQA_ENDPOINT`` env var when available.
* ``execute()`` returns the raw SPARQL-JSON dict so it can be written straight
  into a submission's ``answers`` field with no reformatting.
* Final-answer execution must NOT truncate (some gold sets exceed 800k rows),
  so there is no row cap here. The agent's *exploratory* SPARQL truncates
  elsewhere; this path is for the committed answer query.

Wikidata's standard query-service prefixes (wd/wdt/p/ps/pq/...) are not part of
SPARQL itself; WDQS predefines them, but a neutral endpoint may not. We prepend
them when missing so the same query text works on either.
"""

from __future__ import annotations

import base64
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any

try:  # make .env (WIKIKGQA_ENDPOINT / WIKIKGQA_USER / WIKIKGQA_PASSWORD) visible
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # dotenv optional; env vars may already be exported
    pass

DEFAULT_ENDPOINT = "https://query.wikidata.org/sparql"
# The challenge endpoint (QLever, behind a WAF) 403s non-browser User-Agents, so
# we present a browser-like UA. WDQS also accepts it.
USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36 AMAKBQA-wikikgqa/0.1"
)

# Wikidata query-service prefixes, prepended when a query omits them.
WIKIDATA_PREFIXES = """PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX p: <http://www.wikidata.org/prop/>
PREFIX ps: <http://www.wikidata.org/prop/statement/>
PREFIX psv: <http://www.wikidata.org/prop/statement/value/>
PREFIX psn: <http://www.wikidata.org/prop/statement/value-normalized/>
PREFIX pq: <http://www.wikidata.org/prop/qualifier/>
PREFIX pqv: <http://www.wikidata.org/prop/qualifier/value/>
PREFIX wikibase: <http://wikiba.se/ontology#>
PREFIX bd: <http://www.bigdata.com/rdf#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
PREFIX schema: <http://schema.org/>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
"""


def resolve_endpoint(endpoint: str | None = None) -> str:
    """Endpoint precedence: explicit arg > WIKIKGQA_ENDPOINT env > public WDQS."""
    return endpoint or os.environ.get("WIKIKGQA_ENDPOINT") or DEFAULT_ENDPOINT


def resolve_basic_auth() -> str | None:
    """HTTP Basic auth header value for the challenge endpoint, if configured.

    Reads ``WIKIKGQA_USER`` + ``WIKIKGQA_PASSWORD`` (or a combined
    ``WIKIKGQA_AUTH=user:pass``) from the environment / .env. Returns None when
    no credentials are set (e.g. when developing against public WDQS).
    """
    combined = os.environ.get("WIKIKGQA_AUTH")
    if not combined:
        user = os.environ.get("WIKIKGQA_USER")
        password = os.environ.get("WIKIKGQA_PASSWORD")
        if user and password is not None:
            combined = f"{user}:{password}"
    if not combined:
        return None
    token = base64.b64encode(combined.encode()).decode()
    return f"Basic {token}"


def with_prefixes(query: str) -> str:
    """Prepend Wikidata prefixes unless the query already declares its own."""
    if "PREFIX" in query.upper():
        return query
    return f"{WIKIDATA_PREFIXES}\n{query}"


@dataclass
class SparqlResult:
    ok: bool
    json: dict[str, Any] | None      # SPARQL-JSON results, or None on error
    error: str | None = None
    elapsed_s: float = 0.0


def execute(
    query: str,
    endpoint: str | None = None,
    timeout: int = 120,
    retries: int = 2,
    backoff_s: float = 2.0,
) -> SparqlResult:
    """Run a SPARQL query and return its SPARQL-JSON results.

    Retries transient HTTP/network failures with linear backoff. Never truncates
    results — the caller owns size concerns.
    """
    url = resolve_endpoint(endpoint)
    full_query = with_prefixes(query)
    data = urllib.parse.urlencode({"query": full_query}).encode()
    headers = {
        "Accept": "application/sparql-results+json",
        "User-Agent": USER_AGENT,
        "Content-Type": "application/x-www-form-urlencoded",
    }
    auth = resolve_basic_auth()
    if auth:
        headers["Authorization"] = auth

    last_err: str | None = None
    for attempt in range(retries + 1):
        start = time.monotonic()
        try:
            req = urllib.request.Request(url, data=data, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                import json

                payload = json.loads(resp.read().decode())
            return SparqlResult(ok=True, json=payload, elapsed_s=time.monotonic() - start)
        except urllib.error.HTTPError as exc:
            body = ""
            try:
                body = exc.read().decode(errors="replace")[:300]
            except Exception:
                pass
            last_err = f"HTTP {exc.code}: {body}"
            # 4xx (bad query) won't fix on retry; bail early.
            if 400 <= exc.code < 500 and exc.code != 429:
                break
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_err = f"{type(exc).__name__}: {exc}"
        except Exception as exc:  # JSON decode, etc.
            last_err = f"{type(exc).__name__}: {exc}"
            break
        if attempt < retries:
            time.sleep(backoff_s * (attempt + 1))

    return SparqlResult(ok=False, json=None, error=last_err)
