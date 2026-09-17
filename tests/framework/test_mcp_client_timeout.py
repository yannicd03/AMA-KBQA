"""The bounded wait on MCP tool calls.

`call_tool` used to await a response forever — the only genuinely unbounded
wait in the system. These tests pin the resolution of the timeout, that it
reaches the SDK as `read_timeout_seconds`, and that expiry surfaces as a normal
(clearly-worded) tool error rather than an opaque SDK exception.

No subprocess and no server: a fake ClientSession stands in for the session.
"""

from __future__ import annotations

import asyncio
from datetime import timedelta
from types import SimpleNamespace

import pytest
from mcp.shared.exceptions import McpError
from mcp.types import ErrorData

from ama_kbqa.agents.orchestrator_agent.agent import MCPClient as OrchestratorMCPClient
from ama_kbqa.framework.mcp_client import (
    DEFAULT_TOOL_TIMEOUT_SECONDS,
    TOOL_TIMEOUT_ENV_VAR,
    MCPClient,
    McpToolTimeout,
    resolve_tool_timeout,
)


def _run(coro):
    return asyncio.run(coro)


class _FakeSession:
    """Records the kwargs of the last call, or raises a scripted error."""

    def __init__(self, error=None):
        self._error = error
        self.calls = []

    async def call_tool(self, name, arguments=None, read_timeout_seconds=None):
        self.calls.append({
            "name": name,
            "arguments": arguments,
            "read_timeout_seconds": read_timeout_seconds,
        })
        if self._error is not None:
            raise self._error
        return SimpleNamespace(content=[SimpleNamespace(text="tool output")])


def _client(cls=MCPClient, timeout=7, error=None):
    client = cls("unused/server.py", "test-agent", tool_timeout_seconds=timeout)
    client.session = _FakeSession(error)
    return client


def _timeout_error(message="Timed out while waiting for response"):
    return McpError(ErrorData(code=408, message=message))


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------

class TestResolveToolTimeout:

    def test_explicit_value_wins(self, monkeypatch):
        monkeypatch.setenv(TOOL_TIMEOUT_ENV_VAR, "99")
        assert resolve_tool_timeout(5) == 5.0

    def test_env_var_overrides_config_and_default(self, monkeypatch):
        monkeypatch.setenv(TOOL_TIMEOUT_ENV_VAR, "12.5")
        assert resolve_tool_timeout() == 12.5

    def test_unparseable_env_var_falls_back_to_the_default(self, monkeypatch):
        monkeypatch.setenv(TOOL_TIMEOUT_ENV_VAR, "not-a-number")
        assert resolve_tool_timeout() == DEFAULT_TOOL_TIMEOUT_SECONDS

    def test_default_applies_when_nothing_is_configured(self, monkeypatch):
        monkeypatch.delenv(TOOL_TIMEOUT_ENV_VAR, raising=False)
        assert resolve_tool_timeout() == DEFAULT_TOOL_TIMEOUT_SECONDS

    def test_non_positive_value_disables_the_timeout(self):
        assert resolve_tool_timeout(0) is None
        assert resolve_tool_timeout(-1) is None


# ---------------------------------------------------------------------------
# call_tool
# ---------------------------------------------------------------------------

class TestCallToolTimeout:

    def test_timeout_is_passed_to_the_sdk_as_read_timeout_seconds(self):
        client = _client(timeout=7)

        result = _run(client.call_tool("FindNode", {"name": "Inception"}))

        assert result == "tool output"
        assert client.session.calls[0]["read_timeout_seconds"] == timedelta(seconds=7)

    def test_disabled_timeout_sends_no_read_timeout(self):
        client = _client(timeout=0)

        _run(client.call_tool("FindNode", {}))

        assert client.session.calls[0]["read_timeout_seconds"] is None

    def test_expiry_surfaces_as_a_tool_error_naming_the_tool_and_budget(self):
        client = _client(timeout=7, error=_timeout_error())

        with pytest.raises(McpToolTimeout) as excinfo:
            _run(client.call_tool("RunSPARQLQuery", {}))

        message = str(excinfo.value)
        assert "RunSPARQLQuery" in message and "7s" in message

    def test_a_raw_transport_timeout_is_translated_too(self):
        client = _client(timeout=7, error=TimeoutError("read timed out"))

        with pytest.raises(McpToolTimeout):
            _run(client.call_tool("FindNode", {}))

    def test_other_mcp_errors_are_left_alone(self):
        client = _client(error=McpError(ErrorData(code=500, message="server blew up")))

        with pytest.raises(McpError) as excinfo:
            _run(client.call_tool("FindNode", {}))

        assert not isinstance(excinfo.value, McpToolTimeout)

    def test_orchestrator_client_is_bounded_the_same_way(self):
        """The Orchestrator carries its own MCPClient implementation; its probe
        tool runs on every question, so it must be bounded too."""
        client = _client(cls=OrchestratorMCPClient, timeout=7)

        _run(client.call_tool("analyze_query_recommend_db", {"question": "q"}))
        assert client.session.calls[0]["read_timeout_seconds"] == timedelta(seconds=7)

        timing_out = _client(cls=OrchestratorMCPClient, timeout=7, error=_timeout_error())
        with pytest.raises(McpToolTimeout):
            _run(timing_out.call_tool("analyze_query_recommend_db", {"question": "q"}))
