"""The SPARQL query timeout that `GraphConfig.timeout_ms` was always meant to be.

The value was plumbed from the adapters through `adapters.resolve` but never
consumed: both MCP servers built a bare `SPARQLWrapper(endpoint)`, leaving
`urlopen` without a timeout, so a query Virtuoso never finished blocked the
tool call forever. These tests pin the wiring and the error path.

No live Virtuoso: the parent class's `query` is monkeypatched.
"""

from __future__ import annotations

import socket
import urllib.error

import pytest
from SPARQLWrapper import SPARQLWrapper as BaseSPARQLWrapper

from ama_kbqa.framework.config import GraphConfig
from ama_kbqa.framework.sparql_client import (
    DEFAULT_TIMEOUT_MS,
    SparqlTimeout,
    TimeoutAwareSPARQLWrapper,
    make_sparql_client,
    timeout_seconds,
)
from ama_kbqa.server import kqapro_server as kqa
from ama_kbqa.server import sciqa_server as sci

ENDPOINT = "http://localhost:8890/sparql"


class _FakeResult:
    def __init__(self, error=None):
        self.response = "raw-response"
        self._error = error

    def convert(self):
        if self._error is not None:
            raise self._error
        return {"results": {"bindings": []}}


def _patch_query(monkeypatch, *, raises=None, returns=None):
    def fake_query(self):
        if raises is not None:
            raise raises
        return returns if returns is not None else _FakeResult()

    monkeypatch.setattr(BaseSPARQLWrapper, "query", fake_query)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestMakeSparqlClient:

    def test_configured_timeout_is_applied_in_seconds(self):
        client = make_sparql_client(ENDPOINT, 45000)

        assert isinstance(client, TimeoutAwareSPARQLWrapper)
        assert client.timeout == 45
        assert client.endpoint == ENDPOINT

    def test_default_is_the_graph_config_default(self):
        assert DEFAULT_TIMEOUT_MS == GraphConfig.timeout_ms
        assert make_sparql_client(ENDPOINT).timeout == timeout_seconds(DEFAULT_TIMEOUT_MS)

    def test_sub_second_budgets_floor_at_one_second(self):
        # setTimeout() int()s its argument, so 0 would silently mean "no timeout".
        assert timeout_seconds(500) == 1
        assert timeout_seconds(0) == 1
        assert timeout_seconds(30000) == 30
        assert timeout_seconds(None) == timeout_seconds(DEFAULT_TIMEOUT_MS)


# ---------------------------------------------------------------------------
# Error path
# ---------------------------------------------------------------------------

class TestTimeoutTranslation:

    def test_request_timeout_becomes_a_sparql_timeout(self, monkeypatch):
        _patch_query(monkeypatch, raises=socket.timeout("timed out"))
        client = make_sparql_client(ENDPOINT, 30000)

        with pytest.raises(SparqlTimeout) as excinfo:
            client.query()

        message = str(excinfo.value)
        assert "30s" in message and "Narrow the query" in message

    def test_timeout_wrapped_in_a_urlerror_is_translated_too(self, monkeypatch):
        _patch_query(
            monkeypatch, raises=urllib.error.URLError(socket.timeout("timed out"))
        )
        client = make_sparql_client(ENDPOINT, 30000)

        with pytest.raises(SparqlTimeout):
            client.query()

    def test_other_transport_errors_keep_their_own_type(self, monkeypatch):
        _patch_query(monkeypatch, raises=urllib.error.URLError(ConnectionRefusedError()))
        client = make_sparql_client(ENDPOINT, 30000)

        with pytest.raises(urllib.error.URLError) as excinfo:
            client.query()

        assert not isinstance(excinfo.value, SparqlTimeout)

    def test_a_timeout_while_reading_the_body_is_translated(self, monkeypatch):
        # The socket timeout applies per read, so it can fire in convert().
        _patch_query(monkeypatch, returns=_FakeResult(error=socket.timeout("timed out")))
        client = make_sparql_client(ENDPOINT, 30000)

        with pytest.raises(SparqlTimeout):
            client.query().convert()

    def test_successful_queries_are_untouched(self, monkeypatch):
        _patch_query(monkeypatch)
        client = make_sparql_client(ENDPOINT, 30000)

        result = client.query()

        assert result.convert() == {"results": {"bindings": []}}
        # The proxy keeps the rest of the QueryResult surface reachable.
        assert result.response == "raw-response"


# ---------------------------------------------------------------------------
# Server wiring
# ---------------------------------------------------------------------------

class TestServersUseTheConfiguredTimeout:

    @pytest.mark.parametrize("server", [kqa, sci], ids=["kqapro", "sciqa"])
    def test_server_client_carries_the_resolved_timeout(self, server):
        client = server._build_sparql_client()

        assert isinstance(client, TimeoutAwareSPARQLWrapper)
        assert client.endpoint == server.VIRTUOSO_ENDPOINT
        assert client.timeout == timeout_seconds(server.SPARQL_TIMEOUT_MS)
        assert client.timeout > 0

    @pytest.mark.parametrize("server", [kqa, sci], ids=["kqapro", "sciqa"])
    def test_timeout_comes_from_the_adapter_resolved_graph_config(self, server):
        assert server.SPARQL_TIMEOUT_MS == server._RESOLVED.graph.timeout_ms
