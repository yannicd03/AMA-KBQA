"""Tests for the orchestrator MCP server's graceful degradation.

When Qdrant is unreachable, the server must still boot (so the MCP client
does not see an opaque "Connection closed") and the routing tool must return
a evidence-free degraded payload instead of crashing. These tests are
hermetic: the degraded path points at a dead port, so no live Qdrant or
real LLM is required.

New contract (analyze_query_recommend_db):
  Normal: {"semantics": {...}, "kg_evidence": {"kqapro": {...}, "sciqa": {...}}, "degraded": false}
  Degraded: {"semantics": {}, "kg_evidence": {}, "degraded": true, "note": "<reason>"}
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
        # Dummy provider keys: the lifespan builds an OpenAI client from
        # config, which must not depend on the developer's .env being present.
        monkeypatch.setenv("KIT_API_KEY", "test-not-used")
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-not-used")
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
        monkeypatch.setenv("KIT_API_KEY", "test-not-used")
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-not-used")
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

    def test_returns_evidence_free_degraded_payload_when_qdrant_none(self):
        # Qdrant-None path: must return degraded=True, empty evidence, a note,
        # and must NOT contain the old "recommendation" key.
        out = _tool_fn()(question="Who directed Inception?", context=self._degraded_context())
        data = json.loads(out)
        assert data["degraded"] is True
        assert data["kg_evidence"] == {}
        assert data.get("note"), "degraded payload must include a non-empty note"
        assert "recommendation" not in data, "old 'recommendation' key must not appear in new contract"

    def test_does_not_call_llm_in_degraded_mode(self, monkeypatch):
        # The short-circuit must happen before any semantic extraction /
        # embedding LLM calls.
        def _boom(*args, **kwargs):
            raise AssertionError("extract_semantics must not run when Qdrant is down")

        monkeypatch.setattr(orch, "extract_semantics", _boom)
        out = _tool_fn()(question="anything", context=self._degraded_context())
        assert json.loads(out)["degraded"] is True

    def test_ner_failure_returns_degraded_payload(self, monkeypatch):
        # NER/extract_semantics failure path: even when Qdrant is reachable
        # (non-None), an extraction error must produce a degraded evidence-free
        # payload whose note mentions extraction failure.
        from qdrant_client import QdrantClient

        # Build a QdrantClient pointing at a dead port. We never actually
        # use it (the NER failure short-circuits before any Qdrant call), but
        # AppContext needs an instance to pass pydantic's type check.
        dummy_qdrant = QdrantClient(host="localhost", port=59998)
        dummy_openai = OpenAI(api_key="test-not-used", base_url="http://localhost:1")

        ctx = SimpleNamespace(
            request_context=SimpleNamespace(
                lifespan_context=orch.AppContext(qdrant=dummy_qdrant, openai=dummy_openai)
            )
        )

        # Monkeypatch extract_semantics to simulate NER failure.
        monkeypatch.setattr(orch, "extract_semantics", lambda *a, **kw: {"error": "boom"})

        out = _tool_fn()(question="What is dark matter?", context=ctx)
        data = json.loads(out)

        assert data["degraded"] is True
        assert data["kg_evidence"] == {}
        assert data.get("note"), "degraded NER-failure payload must include a note"
        note_lower = data["note"].lower()
        assert "extract" in note_lower or "entity" in note_lower or "ner" in note_lower, (
            f"note should mention extraction/entity failure, got: {data['note']!r}"
        )
