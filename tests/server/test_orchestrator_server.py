"""Tests for the orchestrator MCP server's graceful degradation.

When Qdrant is unreachable, the server must still boot (so the MCP client
does not see an opaque "Connection closed") and the routing tool must return
a clear KQAPro recommendation instead of crashing. These tests are hermetic:
the degraded path points at a dead port, so no live Qdrant is required.
"""
import asyncio
import json
from types import SimpleNamespace

from openai import OpenAI

from ama_kbqa.server import orchestrator_server as orch


def _run(coro):
    return asyncio.run(coro)


def _tool_fn():
    """The plain callable behind the @mcp.tool() decorator."""
    tool = orch.analyze_query_recommend_db
    return getattr(tool, "fn", tool)


class TestLifespanDegradation:
    """server_lifespan must not crash when Qdrant is unreachable."""

    def test_boots_with_qdrant_none_when_unreachable(self, monkeypatch):
        # Point Qdrant at a port with nothing listening -> connectivity
        # check fails, but the server should still yield a context.
        monkeypatch.setattr(orch, "QDRANT_PORT", 59999)

        async def main():
            async with orch.server_lifespan(orch.mcp) as ctx:
                return ctx

        ctx = _run(main())
        assert ctx.qdrant is None
        assert ctx.openai is not None

    def test_shutdown_is_clean_in_degraded_mode(self, monkeypatch):
        # The finally block guards qdrant.close() against None; entering and
        # exiting the lifespan must not raise even with no Qdrant.
        monkeypatch.setattr(orch, "QDRANT_PORT", 59999)

        async def main():
            async with orch.server_lifespan(orch.mcp):
                pass

        _run(main())  # must not raise


class TestToolDegradation:
    """analyze_query_recommend_db must short-circuit when qdrant is None."""

    def _degraded_context(self):
        # A real OpenAI instance is needed only to satisfy AppContext's
        # pydantic validation; the tool returns before ever using it.
        dummy_openai = OpenAI(api_key="test-not-used", base_url="http://localhost:1")
        return SimpleNamespace(
            request_context=SimpleNamespace(
                lifespan_context=orch.AppContext(qdrant=None, openai=dummy_openai)
            )
        )

    def test_returns_kqapro_recommendation(self):
        out = _tool_fn()(question="Who directed Inception?", context=self._degraded_context())
        data = json.loads(out)
        assert "kqapro" in data["recommendation"].lower()
        assert data["degraded"] is True

    def test_does_not_call_llm_in_degraded_mode(self, monkeypatch):
        # The short-circuit must happen before any semantic extraction /
        # embedding LLM calls.
        def _boom(*args, **kwargs):
            raise AssertionError("extract_semantics must not run when Qdrant is down")

        monkeypatch.setattr(orch, "extract_semantics", _boom)
        out = _tool_fn()(question="anything", context=self._degraded_context())
        assert json.loads(out)["degraded"] is True
