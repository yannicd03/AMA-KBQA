"""Shared fixtures for the API tests.

No real agent, MCP server, LLM or network call is ever made: ``create_agent``
is replaced in ``ama_kbqa.api.runs``, ``apply_chat_settings`` on the
``chat_controls`` module, and the KIT ``/models`` fetch in
``ama_kbqa.api.meta``. The fake agent records into the real
``TraceRecorder``, so the lifecycle runner, the drain, the SVG renderers and
the graph normalisers all run for real.

The model picker has two shapes (see ``ama_kbqa/api/meta.py``). The
``harness`` fixture forces the KIT-only one (it removes
``available_choices`` from ``chat_controls`` for the test, so it also runs on
the provider-aware branches); ``provider_harness`` installs a fake
provider-aware API instead.
"""

from __future__ import annotations

import asyncio
import json
import threading
from dataclasses import dataclass
from typing import Any, Optional

import pytest
from fastapi.testclient import TestClient

from ama_kbqa.api import app as app_module
from ama_kbqa.api import meta, runs
from ama_kbqa.framework.trace import TraceRecorder
from ama_kbqa.frontend.utils import chat_controls

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


# ---------------------------------------------------------------------------
# Fake provider-aware picker (the demo-booth / demo-llamacpp chat_controls API)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FakeChoice:
    """Same fields and ``key`` as chat_controls.ChatModelChoice."""

    provider: str
    model: str
    name: str
    default: bool = False

    @property
    def key(self) -> str:
        return f"{self.provider}:{self.model}"


# The same bare model id behind two providers, like deepseek-v4-pro on
# OpenRouter and DeepSeek direct in the booth config.
CHOICES = [
    FakeChoice("kit", MODEL, "GPT-OSS 120B"),
    FakeChoice("openrouter", "deepseek-v4-pro", "DeepSeek V4 Pro"),
    FakeChoice("deepseek", "deepseek-v4-pro", "DeepSeek V4 Pro", default=True),
    FakeChoice("llamacpp", "local-model", "local-model"),
]
NOTICES = ["OpenRouter model acme/retired is not in the live catalog and was hidden"]
PLACEHOLDER = FakeChoice("kit", "kit.placeholder", "Placeholder")


class FakeChoices:
    def __init__(self) -> None:
        self.calls = 0
        self.error: Optional[Exception] = None

    def available_choices(self):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return list(CHOICES), list(NOTICES)

    @staticmethod
    def default_choice(choices):
        for choice in choices:
            if choice.default:
                return choice
        return choices[0] if choices else PLACEHOLDER

    @staticmethod
    def display_choice(choice) -> str:
        return f"{choice.name} · {choice.provider}"

    @staticmethod
    def price_caption_for(choice) -> Optional[str]:
        # Escaped like the real helper (Streamlit markdown), or plain text.
        return {
            "kit": r"\$0.10 per 1M in and \$0.30 per 1M out",
            "openrouter": r"\$1.00 per 1M in and \$2.00 per 1M out (billed)",
            "llamacpp": "Runs locally: no API cost",
        }.get(choice.provider)


@pytest.fixture
def fake_choices(monkeypatch):
    fake = FakeChoices()
    for name in ("available_choices", "default_choice", "display_choice", "price_caption_for"):
        monkeypatch.setattr(chat_controls, name, getattr(fake, name), raising=False)
    meta.reset_caches()
    yield fake
    meta.reset_caches()


@pytest.fixture
def kit_path(monkeypatch):
    """Force the KIT-only picker, even where chat_controls is provider-aware."""
    monkeypatch.delattr(chat_controls, "available_choices", raising=False)
    meta.reset_caches()
    yield
    meta.reset_caches()


# ---------------------------------------------------------------------------
# Fake agent + HTTP harness
# ---------------------------------------------------------------------------

class FakeAgent:
    """Stands in for KQAProAgent / SciQAAgent / Orchestrator.

    Opens the canonical spans on the real recorder, prints ANSI text like the
    real agents do, and returns an answer that names the node its one
    ``GetNodeLabel`` tool call found (so the frozen graph has a highlight).
    """

    def __init__(self, name: str, *, marker: str, model: str = MODEL, fail: bool = False,
                 gate: Optional[threading.Event] = None, big_payload: bool = False,
                 prints: int = 3, delay: float = 0.0) -> None:
        self.name = name
        self.marker = marker
        self.fail = fail
        self.gate = gate
        self.big_payload = big_payload
        self.prints = prints
        self.delay = delay
        self.model = model
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
        self.applied: list[tuple] = []
        self.current_model: Optional[str] = None
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
        # Like the real agents, it takes the model apply_chat_settings set.
        agent = FakeAgent(
            name, marker=f"agent{len(self.created)}",
            model=self.current_model or MODEL, **self.options,
        )
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


def _run_harness(monkeypatch, *, provider_path: bool):
    for var in ("DEMO_MODE", "DEMO_MAX_QUERIES_PER_SESSION", "DEMO_MIN_SECONDS_BETWEEN_QUERIES"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(runs, "TICK_SECONDS", 0.02)
    monkeypatch.setattr(runs, "HEARTBEAT_SECONDS", 0.05)
    monkeypatch.setattr(runs, "get_live_graph_enabled", lambda: True)

    app = app_module.create_app()
    with TestClient(app) as client:
        h = Harness(client)
        monkeypatch.setattr(runs, "create_agent", h.create_agent)
        if provider_path:
            def apply(model, temperature, provider="kit"):
                h.applied.append((model, temperature, provider))
                h.current_model = model
        else:
            def apply(model, temperature):
                h.applied.append((model, temperature))
                h.current_model = model
        monkeypatch.setattr(chat_controls, "apply_chat_settings", apply)
        try:
            yield h
        finally:
            # Never leave a gated worker thread spinning past the test.
            h.release_all()


@pytest.fixture
def harness(monkeypatch, patch_models, kit_path):
    yield from _run_harness(monkeypatch, provider_path=False)


@pytest.fixture
def provider_harness(monkeypatch, patch_models, fake_choices):
    yield from _run_harness(monkeypatch, provider_path=True)
