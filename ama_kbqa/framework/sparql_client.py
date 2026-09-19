"""SPARQL client construction with the configured query timeout applied.

``GraphConfig.timeout_ms`` has been plumbed from the adapters through
``framework.adapters.resolve`` since the adapter refactor, but nothing ever
consumed it: the MCP servers built their client as a bare
``SPARQLWrapper(endpoint)``, which leaves ``urlopen`` with no timeout at all.
A query Virtuoso never finishes therefore blocked the tool call — and with it
the agent's worker thread — forever.

Building clients through :func:`make_sparql_client` applies the resolved
timeout once, at construction, so every ``sparql.query()`` in the servers
inherits it without touching the ~60 call sites. Expiry is translated into
:class:`SparqlTimeout`, whose message names the budget and says what to do
about it: the tool bodies either catch ``Exception`` themselves or let it
reach FastMCP, which turns it into a tool error, so the agent reads a normal,
actionable tool error rather than an opaque socket exception.
"""

from __future__ import annotations

import socket
import urllib.error
from typing import Any, Optional

from SPARQLWrapper import JSON, SPARQLWrapper

from ama_kbqa.framework.config import GraphConfig

# The timeout every adapter starts from (GraphConfig's dataclass default), used
# when a caller has no resolved GraphConfig at hand.
DEFAULT_TIMEOUT_MS: int = GraphConfig.timeout_ms


class SparqlTimeout(Exception):
    """A SPARQL query exceeded the configured timeout."""


def timeout_seconds(timeout_ms: Optional[int] = None) -> int:
    """Whole seconds for ``SPARQLWrapper.setTimeout`` (which int()s its input).

    Sub-second budgets would truncate to 0 (i.e. "no timeout"), so the floor is
    one second.
    """
    ms = DEFAULT_TIMEOUT_MS if timeout_ms is None else int(timeout_ms)
    return max(1, round(ms / 1000))


class _TimeoutAwareResult:
    """Proxy around SPARQLWrapper's ``QueryResult``.

    The socket timeout applies per read, so it can fire while the response body
    is being consumed in ``convert()``, not only while the request is in
    flight. The proxy translates that read and leaves the rest of the
    ``QueryResult`` surface untouched.
    """

    def __init__(self, result: Any, message: str) -> None:
        self._result = result
        self._message = message

    def convert(self) -> Any:
        try:
            return self._result.convert()
        except (socket.timeout, TimeoutError) as exc:
            raise SparqlTimeout(self._message) from exc
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, (socket.timeout, TimeoutError)):
                raise SparqlTimeout(self._message) from exc
            raise

    def __getattr__(self, name: str) -> Any:
        return getattr(self._result, name)


class TimeoutAwareSPARQLWrapper(SPARQLWrapper):
    """``SPARQLWrapper`` that reports timeout expiry as :class:`SparqlTimeout`."""

    def query(self) -> Any:
        try:
            return _TimeoutAwareResult(super().query(), self._timeout_message())
        except (socket.timeout, TimeoutError) as exc:
            raise SparqlTimeout(self._timeout_message()) from exc
        except urllib.error.URLError as exc:
            # urlopen wraps a socket timeout in URLError; other URLErrors
            # (connection refused, DNS) are real transport errors and keep
            # their own type so callers can tell them apart.
            if isinstance(exc.reason, (socket.timeout, TimeoutError)):
                raise SparqlTimeout(self._timeout_message()) from exc
            raise

    def _timeout_message(self) -> str:
        return (
            f"SPARQL query timed out after {self.timeout}s against "
            f"{self.endpoint}. Narrow the query (add a LIMIT, more specific "
            f"filters, or fewer OPTIONAL blocks) and try again."
        )


def make_sparql_client(
    endpoint: str,
    timeout_ms: Optional[int] = None,
    return_format: str = JSON,
) -> TimeoutAwareSPARQLWrapper:
    """Build a SPARQL client with the query timeout and return format applied.

    Args:
        endpoint: SPARQL endpoint URL.
        timeout_ms: Query timeout in milliseconds, normally
            ``adapter.resolved_config.graph.timeout_ms``. Defaults to
            :data:`DEFAULT_TIMEOUT_MS`.
        return_format: SPARQLWrapper return format; JSON everywhere today.
    """
    client = TimeoutAwareSPARQLWrapper(endpoint)
    client.setReturnFormat(return_format)
    client.setTimeout(timeout_seconds(timeout_ms))
    return client
