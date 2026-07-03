"""Unit tests for endpoint.py's pure helpers: prefix injection, endpoint
resolution precedence, and basic-auth header construction.

No real HTTP calls are made anywhere in this file -- ``execute()`` itself is
covered indirectly through the generator tests, which monkeypatch it out.
"""

from __future__ import annotations

from ama_kbqa.wikikgqa.endpoint import (
    DEFAULT_ENDPOINT,
    WIKIDATA_PREFIXES,
    resolve_basic_auth,
    resolve_endpoint,
    with_prefixes,
)

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
