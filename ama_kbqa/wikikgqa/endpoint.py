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
import re
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

# Gold answers were frozen against the challenge KG at a fixed reference instant.
# Reverse-engineered from gold data: gold ages for Stan Lee (103, born 1922-12-28)
# and Justin Bieber (32, born 1994-03-01) bound the window, and every gold
# "birthday today" entity was born on April 8 -- together pinning the reference
# date to 2026-04-08.
REFERENCE_TIME = "2026-04-08T00:00:00Z"


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


# Matches SPARQL/IRI/string-literal spans that must NOT have NOW() rewritten
# inside them: long triple-quoted strings, short single/double-quoted strings
# (with backslash escapes), and IRIREFs. Used to protect those spans from the
# NOW()-call regex below, which otherwise has no notion of SPARQL syntax.
_STRING_OR_IRI_RE = re.compile(
    r"""
    '''(?:\\.|(?!''').)*'''            # long single-quoted string
    | \"\"\"(?:\\.|(?!\"\"\").)*\"\"\"  # long double-quoted string
    | '(?:\\.|[^'\\\n])*'              # short single-quoted string
    | "(?:\\.|[^"\\\n])*"              # short double-quoted string
    | <[A-Za-z][\w+.-]*:[^<>\s]*>      # IRIREF (scheme required, so a space-free
                                       # comparison like ?a<NOW()&&?b>?c is not
                                       # mistaken for an IRI and left unrewritten)
    """,
    re.VERBOSE | re.DOTALL,
)

# The NOW() function-call form: case-insensitive, whitespace-tolerant between
# NOW and the parens. The negative lookbehind excludes prefixed names (ex:now),
# variables/parameters (?now, $now), and identifiers where "now" is a
# substring -- \w, ":", "?", or "$" immediately before "NOW" means it isn't a
# standalone function-call token.
_NOW_CALL_RE = re.compile(r"(?<![\w:?$])NOW\s*\(\s*\)", re.IGNORECASE)


def rewrite_now(query: str, reference_time: str = REFERENCE_TIME) -> str:
    """Replace every SPARQL ``NOW()`` call with the frozen ``REFERENCE_TIME``.

    Gold answers were computed once against a frozen KG snapshot using the
    reference instant baked into ``REFERENCE_TIME`` (see its docstring). Gold
    SPARQL uses ``NOW()``, which is fine at freeze time but drifts when a
    system re-runs the query later: ages tick over, "born today" filters land
    on the wrong day, and open-ended future-date filters admit rows gold never
    saw. Rewriting ``NOW()`` to the frozen instant keeps outgoing queries
    reproducing the gold evaluation instant regardless of when they run.

    Uses the full ``xsd:dateTime`` IRI (not the ``xsd:`` prefix) so the
    replacement is correct even if the query has no prefix declarations and
    is sent before ``with_prefixes()`` runs.

    Only the ``NOW()`` function-call form is rewritten (case-insensitive,
    whitespace-tolerant, e.g. ``NOW ( )``). Occurrences inside string
    literals or IRIs are left untouched, as are identifiers that merely
    look like "now" (a variable ``?known``/``?now`` or a prefixed name like
    ``ex:now``) since those aren't calls to the NOW() function.
    """
    literal = f'"{reference_time}"^^<http://www.w3.org/2001/XMLSchema#dateTime>'
    protected = [(m.start(), m.end()) for m in _STRING_OR_IRI_RE.finditer(query)]

    def _in_protected(pos: int) -> bool:
        return any(start <= pos < end for start, end in protected)

    def _replace(match: re.Match[str]) -> str:
        if _in_protected(match.start()):
            return match.group(0)
        return literal

    return _NOW_CALL_RE.sub(_replace, query)


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
    pin_now: bool | None = None,
) -> SparqlResult:
    """Run a SPARQL query and return its SPARQL-JSON results.

    Retries transient HTTP/network failures with linear backoff. Never truncates
    results — the caller owns size concerns.

    ``pin_now`` rewrites ``NOW()`` in the query to the frozen ``REFERENCE_TIME``
    gold was computed against, so results stay reproducible regardless of when
    the query actually runs; see ``rewrite_now()``. Default (``None``) resolves
    from the ``WIKIKGQA_PIN_NOW`` env var (unset/"1" = on; "0" = off, set by
    ``benchmark.py --no-pin-now``) — env rather than parameter threading because
    execute() is reached through several layers (generator commit path, the
    agent's RunSPARQL tool via wikidata_server, submission checks) that would
    all need the plumbing. This module is challenge-only code, hence the
    on-by-default; pass ``pin_now=False`` to send the query as written.
    """
    if pin_now is None:
        pin_now = os.environ.get("WIKIKGQA_PIN_NOW", "1") != "0"
    url = resolve_endpoint(endpoint)
    full_query = with_prefixes(rewrite_now(query) if pin_now else query)
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
