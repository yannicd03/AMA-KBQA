"""Tests for the structured trace recorder."""

from __future__ import annotations

import asyncio
import json

import pytest

from ama_kbqa.framework.trace import (
    JOURNAL_MUTATING_TOOLS,
    TraceEvent,
    TraceRecorder,
)


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro) if False else asyncio.run(coro)


class TestTraceRecorder:
    def test_root_span_has_no_parent(self):
        rec = TraceRecorder()

        async def main():
            async with rec.span("agent_run", "ask"):
                pass

        _run(main())
        assert len(rec.events) == 1
        evt = rec.events[0]
        assert evt.parent_span_id is None
        assert evt.kind == "agent_run"
        assert evt.name == "ask"
        assert evt.status == "ok"
        assert evt.is_event is False
        assert evt.duration_ms >= 0.0

    def test_nested_spans_parent_correctly(self):
        rec = TraceRecorder()

        async def main():
            async with rec.span("agent_run", "ask"):
                async with rec.span("tool_loop_iter", "iter:1"):
                    async with rec.span("llm_call", "gpt-4o"):
                        pass
                    async with rec.span("tool_call", "FindNode"):
                        pass

        _run(main())

        # Order is exit-order: innermost first, root last.
        assert len(rec.events) == 4
        by_name = {e.name: e for e in rec.events}
        root = by_name["ask"]
        iter1 = by_name["iter:1"]
        llm = by_name["gpt-4o"]
        tool = by_name["FindNode"]

        assert root.parent_span_id is None
        assert iter1.parent_span_id == root.span_id
        assert llm.parent_span_id == iter1.span_id
        assert tool.parent_span_id == iter1.span_id

    def test_exception_inside_span_marks_status_error(self):
        rec = TraceRecorder()

        async def main():
            with pytest.raises(RuntimeError):
                async with rec.span("tool_call", "BadTool"):
                    raise RuntimeError("boom")

        _run(main())
        assert len(rec.events) == 1
        evt = rec.events[0]
        assert evt.status == "error"
        assert evt.error is not None
        assert "boom" in evt.error

    def test_point_in_time_event_uses_current_span_as_parent(self):
        rec = TraceRecorder()

        async def main():
            async with rec.span("agent_run", "ask"):
                rec.event("intervention", "zero-tool-call retry", attributes={"retry": 1})

        _run(main())
        # Order: the event is appended when emitted; the span is appended on exit.
        assert len(rec.events) == 2
        evt_event = next(e for e in rec.events if e.is_event)
        evt_span = next(e for e in rec.events if not e.is_event)

        assert evt_event.parent_span_id == evt_span.span_id
        assert evt_event.duration_ms == 0.0
        assert evt_event.attributes == {"retry": 1}

    def test_span_handle_lets_caller_set_attributes(self):
        rec = TraceRecorder()

        async def main():
            async with rec.span("llm_call", "gpt-4o") as h:
                h.set_attribute("prompt_tokens", 42)
                h.update_attributes({"completion_tokens": 7})
                h.set_payload("messages", [{"role": "user", "content": "hi"}])

        _run(main())
        evt = rec.events[0]
        assert evt.attributes["prompt_tokens"] == 42
        assert evt.attributes["completion_tokens"] == 7
        assert evt.payload["messages"][0]["role"] == "user"

    def test_to_jsonl_roundtrips(self):
        rec = TraceRecorder()

        async def main():
            async with rec.span("agent_run", "ask"):
                async with rec.span("llm_call", "gpt-4o") as h:
                    h.set_attribute("prompt_tokens", 10)

        _run(main())
        lines = rec.to_jsonl().splitlines()
        assert len(lines) == 2
        decoded = [json.loads(line) for line in lines]
        kinds = sorted(d["kind"] for d in decoded)
        assert kinds == ["agent_run", "llm_call"]

    def test_separate_recorders_have_distinct_trace_ids(self):
        a = TraceRecorder()
        b = TraceRecorder()
        assert a.trace_id != b.trace_id

    def test_passing_explicit_trace_id_persists(self):
        rec = TraceRecorder(trace_id="fixed-id")
        assert rec.trace_id == "fixed-id"

        async def main():
            async with rec.span("agent_run", "ask"):
                pass

        _run(main())
        assert rec.events[0].trace_id == "fixed-id"

    def test_reset_clears_events_keeps_trace_id(self):
        rec = TraceRecorder()
        tid = rec.trace_id

        async def main():
            async with rec.span("agent_run", "ask"):
                pass

        _run(main())
        assert len(rec.events) == 1
        rec.reset()
        assert rec.events == []
        assert rec.trace_id == tid

    def test_sync_span_works_and_nests_under_async_span(self):
        rec = TraceRecorder()

        async def main():
            async with rec.span("agent_run", "ask"):
                with rec.span_sync("classify", "gpt-4o"):
                    pass

        _run(main())
        assert len(rec.events) == 2
        sync_evt = next(e for e in rec.events if e.name == "gpt-4o")
        root_evt = next(e for e in rec.events if e.name == "ask")
        assert sync_evt.parent_span_id == root_evt.span_id

    def test_contextvar_does_not_leak_across_recorders(self):
        """Two recorders running back-to-back must not share span context."""
        rec1 = TraceRecorder()
        rec2 = TraceRecorder()

        async def main():
            async with rec1.span("agent_run", "ask1"):
                pass
            async with rec2.span("agent_run", "ask2"):
                pass

        _run(main())
        assert rec1.events[0].parent_span_id is None
        assert rec2.events[0].parent_span_id is None


class TestJournalMutatingTools:
    def test_set_is_non_empty_and_immutable(self):
        assert len(JOURNAL_MUTATING_TOOLS) > 0
        with pytest.raises(AttributeError):
            JOURNAL_MUTATING_TOOLS.add("Foo")  # type: ignore[attr-defined]

    def test_known_kqapro_mutators_present(self):
        for tool in ("FindNode", "GetEdgeQualifiers", "ManageJournal"):
            assert tool in JOURNAL_MUTATING_TOOLS

    def test_known_sciqa_mutators_present(self):
        for tool in ("FindResource", "GetResourceDetails", "ExploreNeighborhood"):
            assert tool in JOURNAL_MUTATING_TOOLS
