"""Tests for synthesis-path resilience guards (B2).

Two independent failure modes must degrade gracefully instead of throwing,
because both `_run_synthesis` and (as of the max-iterations fix) the
tool-loop's cap-out path now depend on synthesis always producing *some*
answer rather than propagating an exception up through `ask()`:

1. `GetJournalSummary` (an MCP tool call) can fail — e.g. a dead subprocess.
   `_run_synthesis_impl` must fall back to the last structured journal
   snapshot, or an empty string if there is none, instead of raising.
2. The synthesis LLM call itself can fail, or return a response with an
   empty `choices` array (some providers do this under load / content
   filtering). `_llm_call_synthesis` must return `""` in both cases instead
   of raising / IndexError-ing.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from ama_kbqa.framework.base_agent import BaseKBQAAgent
from ama_kbqa.framework.trace import TraceRecorder
from chatkit import TransientRetry


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# (a) GetJournalSummary raises during synthesis
# ---------------------------------------------------------------------------


class _FakeMCPRaisesJournalSummary:
    """Simulates a dead/erroring MCP subprocess for the journal-summary call.

    `GetJournalStateJSON` (used by `_snapshot_journal`) returns None, which
    is the normal "nothing to snapshot" no-op path, so it doesn't interfere
    with the fallback logic under test.
    """

    async def call_tool(self, name, args=None):
        if name == "GetJournalSummary":
            raise RuntimeError("MCP subprocess died")
        return None


class _SynthesisImplAgent(BaseKBQAAgent):
    """Minimal double for `_run_synthesis` with the LLM call itself stubbed
    out, so only the journal-summary fallback logic is under test."""

    def __init__(self, mcp, journal_snapshots=None):
        self.recorder = TraceRecorder()
        self.mcp = mcp
        # synthesis_model is a lazily-initialized property; set both backing
        # fields directly (a non-None `_synthesis_client` short-circuits the
        # lazy-init check) so the test doesn't need config/API-key wiring.
        self._synthesis_client = object()
        self._synthesis_model = "test-synth-model"
        self._messages = []
        self.journal_snapshots = journal_snapshots or []
        self.canned_answer = "Christopher Nolan"
        self.last_synthesis_messages = None

    def get_config(self):
        raise NotImplementedError

    def get_mcp_server_path(self) -> str:
        raise NotImplementedError

    def _trace(self, message: str, color: str = "") -> None:
        pass

    def _llm_call_synthesis(self, messages_override=None) -> str:
        self.last_synthesis_messages = messages_override
        return self.canned_answer


def test_synthesis_falls_back_to_empty_journal_when_summary_raises_and_no_snapshots():
    agent = _SynthesisImplAgent(_FakeMCPRaisesJournalSummary())

    # Must not raise, despite GetJournalSummary blowing up.
    result = _run(agent._run_synthesis("Who directed Inception?"))

    assert result == "Christopher Nolan"
    prompt_content = agent.last_synthesis_messages[1]["content"]
    assert prompt_content.startswith("JOURNAL SUMMARY\n\n")


def test_synthesis_falls_back_to_last_snapshot_when_summary_raises():
    snapshot = {
        "ts": 0.0,
        "trigger": "after:FindNode",
        "state": {"director": "Christopher Nolan"},
    }
    agent = _SynthesisImplAgent(
        _FakeMCPRaisesJournalSummary(), journal_snapshots=[snapshot]
    )

    result = _run(agent._run_synthesis("Who directed Inception?"))

    assert result == "Christopher Nolan"
    prompt_content = agent.last_synthesis_messages[1]["content"]
    assert '"director": "Christopher Nolan"' in prompt_content


# ---------------------------------------------------------------------------
# (b) Synthesis LLM call raises, or returns an empty `choices` array
# ---------------------------------------------------------------------------


class _StubSynthesisClient:
    """A stand-in for `self.synthesis_client` that either raises or returns
    a canned response, counting invocations along the way."""

    def __init__(self, response=None, exc: Exception | None = None):
        self._response = response
        self._exc = exc
        self.calls = 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls += 1
        if self._exc is not None:
            raise self._exc
        return self._response


class _SynthAgent(BaseKBQAAgent):
    """Minimal double for `_llm_call_synthesis` with no MCP/tool-loop deps."""

    def __init__(self, synthesis_client):
        self.recorder = TraceRecorder()
        # synthesis_client/synthesis_model are lazily-initialized properties;
        # set the backing fields directly (a non-None `_synthesis_client`
        # short-circuits the lazy-init check).
        self._synthesis_client = synthesis_client
        self._synthesis_model = "test-synth-model"
        self.request_timeout = 30
        self.token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self._retry = TransientRetry()

    def get_config(self):
        raise NotImplementedError

    def get_mcp_server_path(self) -> str:
        raise NotImplementedError

    def _trace(self, message: str, color: str = "") -> None:
        pass


def test_llm_call_synthesis_returns_empty_string_when_client_raises():
    client = _StubSynthesisClient(exc=RuntimeError("provider unreachable"))
    agent = _SynthAgent(client)

    result = agent._llm_call_synthesis(
        messages_override=[{"role": "user", "content": "hi"}]
    )

    assert result == ""
    assert client.calls == 1


def test_llm_call_synthesis_returns_empty_string_when_choices_is_empty():
    response = SimpleNamespace(choices=[], usage=None)
    client = _StubSynthesisClient(response=response)
    agent = _SynthAgent(client)

    result = agent._llm_call_synthesis(
        messages_override=[{"role": "user", "content": "hi"}]
    )

    assert result == ""
    assert client.calls == 1
