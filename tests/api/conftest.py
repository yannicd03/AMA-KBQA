"""Shared fixtures for the API tests.

No real agent, MCP server, LLM or network call is ever made: ``create_agent``
and ``apply_chat_settings`` are replaced in ``ama_kbqa.api.runs`` and the KIT
``/models`` fetch in ``ama_kbqa.api.meta``. The fake agent records into the
real ``TraceRecorder``, so the lifecycle runner, the drain, the SVG renderers
and the graph normalisers all run for real.
"""

from __future__ import annotations

import asyncio
import json
import threading
from typing import Any, Optional

import pytest
from fastapi.testclient import TestClient

from ama_kbqa.api import app as app_module
from ama_kbqa.api import meta, runs
from ama_kbqa.framework.trace import TraceRecorder

MODEL = "kit.gpt-oss-120b"          # in data/model_pricing.json, so it has a price
OTHER_MODEL = "kit.mistral-small-4-119b-a8b"  # no price entry


def catalog_entry(id_, *, name=None, hidden=False, capabilities=None,
                  connection_type="local"):
    return {
        "id": id_,
        "name": name or id_,
        "connection_type": connection_type,
        "preset": False,
        "tags": [],
        "hidden": hidden,
        "capabilities": capabilities if capabilities is not None else {"vision": False},
    }


CATALOG = [
    catalog_entry(MODEL, name="GPT-OSS 120B"),
    catalog_entry(OTHER_MODEL, name="Mistral Small 4"),
    catalog_entry("kit.qwen3-embedding-8b", hidden=True),
    catalog_entry("azure.gpt-5", name="GPT-5", connection_type="external"),
]

LABEL_RESULT = json.dumps({
    "node_id": "Q25191",
    "label": "Christopher Nolan",
    "node_type": "entity",
    "status": "Found label for Q25191",
})


class FakeAgent:
    """Stands in for KQAProAgent / SciQAAgent / Orchestrator.

    Opens the canonical spans on the real recorder, prints ANSI text like the
    real agents do, and returns an answer that names the node its one
    ``GetNodeLabel`` tool call found (so the frozen graph has a highlight).
    """

    def __init__(self, name: str, *, marker: str, fail: bool = False,
                 gate: Optional[threading.Event] = None, big_payload: bool = False,
                 prints: int = 3, delay: float = 0.0) -> None:
        self.name = name
        self.marker = marker
        self.fail = fail
        self.gate = gate
        self.big_payload = big_payload
        self.prints = prints
        self.delay = delay
        self.model = MODEL
        self.recorder = TraceRecorder()
        self.journal_snapshots: list = []
        self.token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self.mcp: Any = object()
        self.reset_calls: list[dict] = []
        self.asked: list[str] = []

    async def reset(self, keep_mcp_open: bool = False, keep_history: bool = False) -> None:
        self.reset_calls.append({"keep_mcp_open": keep_mcp_open, "keep_history": keep_history})
        self.recorder = TraceRecorder()
        self.journal_snapshots = []
        self.token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    async def ask(self, question: str) -> str:
        self.asked.append(question)
        if self.mcp is None:
            self.mcp = object()
        async with self.recorder.span("agent_run", self.name, attributes={"question": question}):
            print(f"\x1b[32m[{self.marker}] start\x1b[0m")
            async with self.recorder.span(
                "llm_call", MODEL,
                attributes={"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
            ):
                await asyncio.sleep(0)
            async with self.recorder.span(
                "tool_call", "GetNodeLabel",
                attributes={"tool_name": "GetNodeLabel"},
                payload={"arguments": {"node_id": "Q25191"}},
            ) as span:
                span.set_payload("result", LABEL_RESULT)
                if self.big_payload:
                    span.set_payload("blob", "x" * 30_000)
                    span.set_payload("nested", {"items": ["y" * 25_000, "short"]})
            for i in range(self.prints):
                print(f"[{self.marker}] line {i}")
                await asyncio.sleep(self.delay)
            while self.gate is not None and not self.gate.is_set():
                await asyncio.sleep(0.01)
        self.token_usage = {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}
        if self.fail:
            raise RuntimeError("boom")
        return "Inception was directed by Christopher Nolan."


class Harness:
    def __init__(self, client: TestClient) -> None:
        self.client = client
        self.created: list[FakeAgent] = []
        self.applied: list[tuple[str, float]] = []
        self.options: dict[str, Any] = {}
        self.create_error: Optional[Exception] = None
        self._gates: list[threading.Event] = []

    def new_gate(self) -> threading.Event:
        gate = threading.Event()
        self._gates.append(gate)
        self.options["gate"] = gate
        return gate

    def release_all(self) -> None:
        for gate in self._gates:
            gate.set()

    def create_agent(self, name: str) -> FakeAgent:
        if self.create_error is not None:
            raise self.create_error
        agent = FakeAgent(name, marker=f"agent{len(self.created)}", **self.options)
        self.created.append(agent)
        return agent

    def post(self, *, session: str = "s1", agent: str = "KQAPro", model: str = MODEL,
             question: str = "Who is the director of Inception?", temperature: float = 1.0):
        return self.client.post("/api/runs", json={
            "session_id": session,
            "question": question,
            "agent": agent,
            "model": model,
            "temperature": temperature,
        })

    def run(self, **kwargs) -> tuple[str, list]:
        """POST a run and follow it to its final event."""
        resp = self.post(**kwargs)
        assert resp.status_code == 201, resp.text
        run_id = resp.json()["run_id"]
        return run_id, read_sse(self.client, run_id)


def read_sse(client: TestClient, run_id: str) -> list[dict]:
    """All frames of a run's event stream as ``{"id", "event", "data"}``;
    keepalive comments are collected as ``{"event": "ping"}``."""
    frames: list[dict] = []
    current: dict = {}
    with client.stream("GET", f"/api/runs/{run_id}/events") as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        assert resp.headers["cache-control"] == "no-cache"
        assert resp.headers["x-accel-buffering"] == "no"
        for line in resp.iter_lines():
            if line.startswith(":"):
                frames.append({"event": "ping"})
            elif line.startswith("id: "):
                current["id"] = int(line[4:])
            elif line.startswith("event: "):
                current["event"] = line[7:]
            elif line.startswith("data: "):
                current["data"] = json.loads(line[6:])
            elif line == "" and current:
                frames.append(current)
                current = {}
    if current:
        frames.append(current)
    return frames


@pytest.fixture
def patch_models(monkeypatch):
    calls: list[str] = []

    def fake_fetch(provider, **_kwargs):
        calls.append(provider)
        return [dict(e) for e in CATALOG]

    monkeypatch.setattr(meta, "fetch_provider_models_meta", fake_fetch)
    meta.reset_caches()
    yield calls
    meta.reset_caches()


@pytest.fixture
def harness(monkeypatch, patch_models):
    for var in ("DEMO_MODE", "DEMO_MAX_QUERIES_PER_SESSION", "DEMO_MIN_SECONDS_BETWEEN_QUERIES"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(runs, "TICK_SECONDS", 0.02)
    monkeypatch.setattr(runs, "HEARTBEAT_SECONDS", 0.05)
    monkeypatch.setattr(runs, "get_live_graph_enabled", lambda: True)

    app = app_module.create_app()
    with TestClient(app) as client:
        h = Harness(client)
        monkeypatch.setattr(runs, "create_agent", h.create_agent)
        monkeypatch.setattr(runs, "apply_chat_settings", lambda m, t: h.applied.append((m, t)))
        try:
            yield h
        finally:
            # Never leave a gated worker thread spinning past the test.
            h.release_all()
