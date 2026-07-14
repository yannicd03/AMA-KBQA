"""Unit tests for endpoint.py's pure helpers: prefix injection, endpoint
resolution precedence, basic-auth header construction, and the NOW()-pinning
rewrite.

No real HTTP calls are made anywhere in this file. ``execute()``'s network
path is covered indirectly through the generator tests (which monkeypatch it
out entirely) plus the ``execute()`` tests below, which stub
``urllib.request.urlopen`` to inspect the outgoing query text without hitting
the network.
"""

from __future__ import annotations

import urllib.parse

from ama_kbqa.wikikgqa.endpoint import (
    DEFAULT_ENDPOINT,
    REFERENCE_TIME,
    WIKIDATA_PREFIXES,
    execute,
    resolve_basic_auth,
    resolve_endpoint,
    rewrite_now,
    with_prefixes,
)

_DATETIME_LITERAL = f'"{REFERENCE_TIME}"^^<http://www.w3.org/2001/XMLSchema#dateTime>'

# --- with_prefixes ---


def test_with_prefixes_adds_prefixes_when_missing():
    query = "SELECT ?x WHERE { ?x wdt:P31 wd:Q5 }"
    out = with_prefixes(query)
    assert out.startswith(WIKIDATA_PREFIXES)
    assert out.endswith(query)


def test_with_prefixes_leaves_query_untouched_when_prefix_already_declared():
    query = "PREFIX wd: <http://www.wikidata.org/entity/>\nSELECT ?x WHERE {}"
    assert with_prefixes(query) == query


def test_with_prefixes_is_case_insensitive_for_existing_prefix():
    query = "prefix wd: <http://www.wikidata.org/entity/>\nSELECT ?x WHERE {}"
    assert with_prefixes(query) == query


# --- resolve_endpoint ---


def test_resolve_endpoint_prefers_explicit_argument(monkeypatch):
    monkeypatch.setenv("WIKIKGQA_ENDPOINT", "https://env.example/sparql")
    assert resolve_endpoint("https://explicit.example/sparql") == "https://explicit.example/sparql"


def test_resolve_endpoint_falls_back_to_env_var(monkeypatch):
    monkeypatch.setenv("WIKIKGQA_ENDPOINT", "https://env.example/sparql")
    assert resolve_endpoint(None) == "https://env.example/sparql"


def test_resolve_endpoint_falls_back_to_default_when_nothing_set(monkeypatch):
    monkeypatch.delenv("WIKIKGQA_ENDPOINT", raising=False)
    assert resolve_endpoint(None) == DEFAULT_ENDPOINT


def test_resolve_endpoint_treats_empty_explicit_arg_as_unset(monkeypatch):
    monkeypatch.setenv("WIKIKGQA_ENDPOINT", "https://env.example/sparql")
    assert resolve_endpoint("") == "https://env.example/sparql"


# --- resolve_basic_auth ---


def test_resolve_basic_auth_none_without_credentials(monkeypatch):
    monkeypatch.delenv("WIKIKGQA_AUTH", raising=False)
    monkeypatch.delenv("WIKIKGQA_USER", raising=False)
    monkeypatch.delenv("WIKIKGQA_PASSWORD", raising=False)
    assert resolve_basic_auth() is None


def test_resolve_basic_auth_from_user_and_password(monkeypatch):
    monkeypatch.delenv("WIKIKGQA_AUTH", raising=False)
    monkeypatch.setenv("WIKIKGQA_USER", "alice")
    monkeypatch.setenv("WIKIKGQA_PASSWORD", "secret")
    header = resolve_basic_auth()
    assert header is not None
    assert header.startswith("Basic ")
    import base64

    decoded = base64.b64decode(header.removeprefix("Basic ")).decode()
    assert decoded == "alice:secret"


def test_resolve_basic_auth_from_combined_env_var_takes_precedence(monkeypatch):
    monkeypatch.setenv("WIKIKGQA_AUTH", "bob:hunter2")
    monkeypatch.setenv("WIKIKGQA_USER", "alice")
    monkeypatch.setenv("WIKIKGQA_PASSWORD", "secret")
    header = resolve_basic_auth()
    import base64

    decoded = base64.b64decode(header.removeprefix("Basic ")).decode()
    assert decoded == "bob:hunter2"


def test_resolve_basic_auth_none_when_password_missing(monkeypatch):
    monkeypatch.delenv("WIKIKGQA_AUTH", raising=False)
    monkeypatch.setenv("WIKIKGQA_USER", "alice")
    monkeypatch.delenv("WIKIKGQA_PASSWORD", raising=False)
    assert resolve_basic_auth() is None


def test_resolve_basic_auth_accepts_empty_password(monkeypatch):
    # password="" is a legitimate (if unusual) value, distinct from "unset".
    monkeypatch.delenv("WIKIKGQA_AUTH", raising=False)
    monkeypatch.setenv("WIKIKGQA_USER", "alice")
    monkeypatch.setenv("WIKIKGQA_PASSWORD", "")
    header = resolve_basic_auth()
    assert header is not None
    import base64

    decoded = base64.b64decode(header.removeprefix("Basic ")).decode()
    assert decoded == "alice:"


# --- rewrite_now ---


def test_rewrite_now_replaces_simple_call():
    query = "SELECT * WHERE { ?s :born ?d . FILTER(?d < NOW()) }"
    out = rewrite_now(query)
    assert "NOW()" not in out
    assert _DATETIME_LITERAL in out


def test_rewrite_now_is_case_insensitive_and_tolerates_whitespace():
    for call in ("now()", "Now()", "NOW()", "NOW ( )", "now  (  )", "nOw(\t)"):
        query = f"FILTER(?d < {call})"
        out = rewrite_now(query)
        assert out == f"FILTER(?d < {_DATETIME_LITERAL})", call


def test_rewrite_now_replaces_multiple_occurrences():
    query = "FILTER(?a < NOW() && ?b > now() && ?c = Now())"
    out = rewrite_now(query)
    assert out.count(_DATETIME_LITERAL) == 3
    assert "now" not in out.lower()


def test_rewrite_now_does_not_replace_inside_single_quoted_string():
    query = "SELECT ?s WHERE { ?s rdfs:label 'call NOW() here' }"
    assert rewrite_now(query) == query


def test_rewrite_now_does_not_replace_inside_double_quoted_string():
    query = 'SELECT ?s WHERE { ?s rdfs:label "call NOW() here" }'
    assert rewrite_now(query) == query


def test_rewrite_now_does_not_replace_inside_long_quoted_string():
    query = 'SELECT ?s WHERE { ?s rdfs:comment """multi\nline NOW() text""" }'
    assert rewrite_now(query) == query


def test_rewrite_now_does_not_replace_inside_iri():
    query = "SELECT * WHERE { ?s ?p <http://example.org/NOW()> }"
    assert rewrite_now(query) == query


def test_rewrite_now_still_replaces_call_outside_string_when_string_present():
    query = 'SELECT ?s WHERE { ?s rdfs:label "not a NOW() call" . FILTER(?d < NOW()) }'
    out = rewrite_now(query)
    assert 'label "not a NOW() call"' in out
    assert out.count(_DATETIME_LITERAL) == 1


def test_rewrite_now_no_false_positive_on_variable_identifier():
    query = "SELECT ?now WHERE { ?s ex:knownAt ?now }"
    assert rewrite_now(query) == query


def test_rewrite_now_no_false_positive_on_prefixed_name():
    query = "SELECT * WHERE { ?s ex:now ?o }"
    assert rewrite_now(query) == query


def test_rewrite_now_no_false_positive_on_prefixed_name_followed_by_parens():
    # ex:now() isn't valid SPARQL, but the lookbehind must still reject it --
    # it is not a bare NOW() function call.
    query = "SELECT * WHERE { ?s ex:now() ?o }"
    assert rewrite_now(query) == query


def test_rewrite_now_accepts_custom_reference_time():
    out = rewrite_now("FILTER(NOW() > ?d)", reference_time="2020-01-01T00:00:00Z")
    assert out == 'FILTER("2020-01-01T00:00:00Z"^^<http://www.w3.org/2001/XMLSchema#dateTime> > ?d)'


# --- execute() NOW()-pinning wiring ---


class _FakeResponse:
    def __init__(self, body: bytes):
        self._body = body

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def read(self):
        return self._body


def _stub_urlopen(monkeypatch, captured: dict):
    def fake_urlopen(req, timeout=None):
        captured["data"] = req.data
        return _FakeResponse(b'{"head": {"vars": []}, "results": {"bindings": []}}')

    monkeypatch.setattr(
        "ama_kbqa.wikikgqa.endpoint.urllib.request.urlopen", fake_urlopen
    )


def test_execute_rewrites_now_in_outgoing_query_by_default(monkeypatch):
    captured: dict = {}
    _stub_urlopen(monkeypatch, captured)

    result = execute("SELECT * WHERE { FILTER(?d < NOW()) }")

    assert result.ok
    sent = urllib.parse.parse_qs(captured["data"].decode())["query"][0]
    assert "NOW()" not in sent
    assert _DATETIME_LITERAL in sent


def test_execute_pin_now_false_sends_query_unmodified(monkeypatch):
    captured: dict = {}
    _stub_urlopen(monkeypatch, captured)

    result = execute("SELECT * WHERE { FILTER(?d < NOW()) }", pin_now=False)

    assert result.ok
    sent = urllib.parse.parse_qs(captured["data"].decode())["query"][0]
    assert "NOW()" in sent
    assert _DATETIME_LITERAL not in sent


def test_execute_pin_now_env_opt_out(monkeypatch):
    captured: dict = {}
    _stub_urlopen(monkeypatch, captured)
    monkeypatch.setenv("WIKIKGQA_PIN_NOW", "0")

    result = execute("SELECT * WHERE { FILTER(?d < NOW()) }")

    assert result.ok
    sent = urllib.parse.parse_qs(captured["data"].decode())["query"][0]
    assert "NOW()" in sent
    assert _DATETIME_LITERAL not in sent


def test_rewrite_now_not_fooled_by_space_free_comparison():
    # ?a<NOW()&&?b>?c must not be mistaken for an IRIREF span
    q = "SELECT ?a WHERE { FILTER(?a<NOW()&&?b>?c) }"
    out = rewrite_now(q)
    assert "NOW()" not in out
    assert _DATETIME_LITERAL in out
