"""Cooperative run cancellation in the shared agent layer.

Covers the token itself and the two checkpoints on `BaseKBQAAgent`:

- the token is optional everywhere — with no token nothing changes, which is
  what keeps the Streamlit page and the benchmark runners working;
- a token set before/while the tool loop runs stops it at the NEXT top-of-loop
  checkpoint, without paying for synthesis (the deliberate divergence from the
  "always exit through synthesis" invariant);
- the message stack is left consistent: every assistant message carrying
  `tool_calls` still has its matching `role=tool` results, so a follow-up turn
  can replay the stack without a provider 400.

No LLM, no MCP, no network: the LLM call, the tool execution and synthesis are
all stubbed on a minimal BaseKBQAAgent double.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from ama_kbqa.framework.base_agent import BaseKBQAAgent
from ama_kbqa.framework.cancellation import (
    CANCELLED_ANSWER,
    CancellationToken,
    cancelled_answer,
    is_cancelled,
)
from ama_kbqa.framework.trace import TraceRecorder


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# The token
# ---------------------------------------------------------------------------

class TestCancellationToken:

    def test_fresh_token_is_not_cancelled(self):
        assert CancellationToken().cancelled is False

    def test_cancel_sets_the_flag_and_keeps_it_set(self):
        token = CancellationToken()
        token.cancel()
        token.cancel()
        assert token.cancelled is True

    def test_reason_is_recorded_and_surfaced_in_the_answer(self):
        token = CancellationToken()
        token.cancel("user pressed stop")
        assert token.reason == "user pressed stop"
        assert cancelled_answer(token) == f"{CANCELLED_ANSWER} (user pressed stop)"

    def test_answer_without_a_reason_is_the_bare_text(self):
        token = CancellationToken()
        token.cancel()
        assert cancelled_answer(token) == CANCELLED_ANSWER

    def test_is_cancelled_is_none_safe(self):
        assert is_cancelled(None) is False
        assert is_cancelled(CancellationToken()) is False


# ---------------------------------------------------------------------------
# Tool-loop double
# ---------------------------------------------------------------------------

def _tool_call(call_id: str, name: str, arguments: dict | None = None):
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=json.dumps(arguments or {})),
    )


def _response(content=None, tool_calls=None):
    message = SimpleNamespace(role="assistant", content=content, tool_calls=tool_calls)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason="stop")],
        usage=None,
    )


class _LoopAgent(BaseKBQAAgent):
    """Drives `_run_tool_loop` over a scripted sequence of LLM responses.

    `_execute_tool_calls` is the REAL implementation (only the single-tool
    execution is stubbed) so the message-stack assertions below are meaningful.
    """

    def __init__(self, responses, on_tool=None):
        self.name = "loop-agent"
        self.recorder = TraceRecorder()
        self._messages = []
        self._known_tool_names = {"FindNode"}
        self._text_tool_call_mode = False
        self._max_tool_calls = 0
        # Non-zero so the zero-tool-call guard (a separate intervention) never
        # fires and the loop's exit paths stay the ones under test.
        self.tool_call_counts = {"FindNode": 1}
        self._responses = list(responses)
        self._on_tool = on_tool
        self.llm_calls = 0
        self.synthesis_calls = 0
        self.executed_tools = []

    # --- abstract surface -------------------------------------------------
    def get_config(self):
        raise NotImplementedError

    def get_mcp_server_path(self) -> str:
        raise NotImplementedError

    # --- stubs ------------------------------------------------------------
    def _trace(self, message: str, color: str = "") -> None:
        pass

    def _manage_context_window(self) -> None:
        pass

    def _maybe_inject_raw_sparql_distress(self, iteration_count: int) -> bool:
        return False

    def _detect_loops(self, func_name: str, func_args: dict):
        return False, ""

    def _llm_call(self, tools=None, tool_choice=None, **kwargs):
        self.llm_calls += 1
        return self._responses.pop(0)

    async def _execute_single_tool(self, func_name: str, func_args: dict) -> str:
        self.executed_tools.append(func_name)
        if self._on_tool is not None:
            self._on_tool()
        return '{"matches": []}'

    async def _run_synthesis(self, query: str, qtype: str = "") -> str:
        self.synthesis_calls += 1
        return "SYNTHESIZED"


def _assert_tool_calls_are_paired(messages):
    """Every assistant `tool_calls` batch is followed by all of its results.

    This is the invariant a cancelled run must not break: an assistant message
    whose tool_calls have no matching `role=tool` results makes providers
    reject the replayed stack with a 400.
    """
    pending: list[str] = []
    for message in messages:
        role = message.get("role")
        if role == "tool":
            assert message["tool_call_id"] in pending, (
                f"tool result {message['tool_call_id']} has no open tool call"
            )
            pending.remove(message["tool_call_id"])
            continue
        assert not pending, f"unanswered tool calls {pending} before a {role} message"
        if role == "assistant" and message.get("tool_calls"):
            pending = [tc["id"] for tc in message["tool_calls"]]
    assert not pending, f"unanswered tool calls at end of stack: {pending}"


@pytest.fixture
def _synthesis_on(monkeypatch):
    """Force the synthesis exit on, so "synthesis was skipped" is a real signal
    rather than an artefact of config.toml's synthesis_enabled = false."""
    monkeypatch.setattr(
        "ama_kbqa.framework.base_agent.get_synthesis_enabled", lambda: True
    )


# ---------------------------------------------------------------------------
# No token: nothing changes
# ---------------------------------------------------------------------------

class TestWithoutAToken:

    def test_loop_runs_to_completion_and_exits_through_synthesis(self, _synthesis_on):
        agent = _LoopAgent([
            _response(tool_calls=[_tool_call("c1", "FindNode", {"name": "Inception"})]),
            _response(content="done"),
        ])

        answer = _run(agent._run_tool_loop("q", [], 20, 5, qtype="Query"))

        assert answer == "SYNTHESIZED"
        assert agent.synthesis_calls == 1
        assert agent.executed_tools == ["FindNode"]
        _assert_tool_calls_are_paired(agent._messages)

    def test_explicit_none_token_behaves_like_no_token(self, _synthesis_on):
        agent = _LoopAgent([_response(content="done")])

        answer = _run(
            agent._run_tool_loop("q", [], 20, 5, qtype="Query", cancel_token=None)
        )

        assert answer == "SYNTHESIZED"

    def test_cancel_token_is_optional_on_ask_and_the_tool_loop(self):
        import inspect

        for fn in (BaseKBQAAgent.ask, BaseKBQAAgent._run_tool_loop):
            param = inspect.signature(fn).parameters["cancel_token"]
            assert param.default is None


# ---------------------------------------------------------------------------
# Cancelled runs
# ---------------------------------------------------------------------------

class TestCancelledToolLoop:

    def test_token_set_before_the_loop_stops_it_without_any_llm_call(self, _synthesis_on):
        token = CancellationToken()
        token.cancel("user pressed stop")
        agent = _LoopAgent([_response(content="never reached")])

        answer = _run(
            agent._run_tool_loop("q", [], 20, 5, qtype="Query", cancel_token=token)
        )

        assert answer == cancelled_answer(token)
        assert agent.llm_calls == 0
        assert agent.synthesis_calls == 0

    def test_cancel_during_tool_execution_stops_at_the_next_checkpoint(self, _synthesis_on):
        token = CancellationToken()
        agent = _LoopAgent(
            [
                _response(tool_calls=[_tool_call("c1", "FindNode", {"name": "X"})]),
                _response(content="would be a second iteration"),
            ],
            on_tool=token.cancel,
        )

        answer = _run(
            agent._run_tool_loop("q", [], 20, 5, qtype="Query", cancel_token=token)
        )

        assert answer == CANCELLED_ANSWER
        # Exactly one iteration ran: the second LLM call never happened.
        assert agent.llm_calls == 1
        assert agent.executed_tools == ["FindNode"]

    def test_cancelled_run_skips_synthesis(self, _synthesis_on):
        token = CancellationToken()
        agent = _LoopAgent(
            [_response(tool_calls=[_tool_call("c1", "FindNode")])],
            on_tool=token.cancel,
        )

        _run(agent._run_tool_loop("q", [], 20, 5, qtype="Query", cancel_token=token))

        assert agent.synthesis_calls == 0

    def test_cancelled_run_leaves_a_consistent_message_stack(self, _synthesis_on):
        token = CancellationToken()
        agent = _LoopAgent(
            [
                _response(tool_calls=[
                    _tool_call("c1", "FindNode", {"name": "A"}),
                    _tool_call("c2", "FindNode", {"name": "B"}),
                ]),
            ],
            on_tool=token.cancel,
        )

        _run(agent._run_tool_loop("q", [], 20, 5, qtype="Query", cancel_token=token))

        _assert_tool_calls_are_paired(agent._messages)
        # Both results were appended before the loop noticed the cancellation.
        assert sum(1 for m in agent._messages if m["role"] == "tool") == 2
        # ... and the stack ends answer-terminated, like every other exit.
        assert agent._messages[-1] == {
            "role": "assistant", "content": CANCELLED_ANSWER
        }


# ---------------------------------------------------------------------------
# The ask() checkpoint, before any billed call
# ---------------------------------------------------------------------------

class _AskAgent(BaseKBQAAgent):
    """Double for the pre-classification checkpoint in `_ask_impl`."""

    def __init__(self):
        self.name = "ask-agent"
        self.model = "test-model"
        self.recorder = TraceRecorder()
        self._parent_span_id_override = None
        self._messages = []
        self.mcp = None
        self.text_only_calls = 0

    def get_config(self):
        raise NotImplementedError

    def get_mcp_server_path(self) -> str:
        raise NotImplementedError

    def _trace(self, message: str, color: str = "") -> None:
        pass

    async def _init_mcp(self) -> None:
        pass

    def _llm_call_text_only(self) -> str:
        self.text_only_calls += 1
        return "TEXT"


class TestAskCheckpoint:

    def test_cancelled_before_classification_returns_without_calling_the_llm(self):
        token = CancellationToken()
        token.cancel()
        agent = _AskAgent()

        answer = _run(agent.ask("Who directed Inception?", cancel_token=token))

        assert answer == CANCELLED_ANSWER
        assert agent.text_only_calls == 0

    def test_without_a_token_ask_takes_its_normal_path(self):
        agent = _AskAgent()

        answer = _run(agent.ask("Who directed Inception?"))

        assert answer == "TEXT"
        assert agent.text_only_calls == 1
