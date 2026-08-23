"""Parity test: the SAME scripted 2-round tool-loop transcript, driven once
through the legacy ``while`` loop (``BaseKBQAAgent._run_tool_loop`` with the
default engine) and once through the graph engine
(``ama_kbqa.graph.runner.run_tool_loop_graph``), must produce identical
``agent._messages``, ``agent.token_usage``, and
``agent.get_tool_call_summary()``.

Both runs share one ``_execute_single_tool`` override (tracked on the same
kind of test double, one instance per run) so tool-call bookkeeping is
guaranteed to come from the same code path on both sides — the only things
that differ between the two runs are the LLM-call mechanics (raw OpenAI SDK
dict responses vs LangChain ``AIMessage``s) that this test exists to prove
are equivalent.

Scenario is deliberately free of every Phase-2-only intervention (loop
detection never triggers on 2 distinct calls, journal refresh is disabled via
a large ``refresh_interval``, zero-tool-call retry never triggers since prior
rounds already made tool calls, `max_tool_calls`/wrap-up-nudge/raw-SPARQL
thresholds are all unreached) so the comparison isolates exactly the
core-loop behaviour Phase 1 ports.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

from chatkit import TransientRetry
from langchain_core.messages import AIMessage

import ama_kbqa.graph.runner as runner_module
from ama_kbqa.framework.base_agent import BaseKBQAAgent
from ama_kbqa.framework.trace import TraceRecorder

from ._fakes import ScriptedChatModel


def _run(coro):
    return asyncio.run(coro)


FIND_NODE_ARGS = {"semantic_node_name": "Inception"}
GET_RELATION_ARGS = {"base_node_id": "Q1", "relation_name": "director"}

TOOL_RESULTS = {
    "FindNode": json.dumps({"matches": [{"original_id": "Q1", "name": "Inception"}]}),
    "GetRelationDetails": json.dumps({"triples": [{"related_id": "Q2"}]}),
    "GetJournalSummary": "Discovered: Q1=Inception, director=Christopher Nolan.",
    "RunSPARQL": "SELECT ?x WHERE { ?x ?p ?o }",
}

# Extra synthetic tool names used by tests/graph/test_parity_wrap_up_nudge.py
# (many distinct single-use tools, so no loop-detection layer fires while
# still crossing the iteration-15 wrap-up threshold).
_NUDGE_TOOL_NAMES = {f"Tool{i}" for i in range(1, 21)}

FINAL_CONTENT = "Christopher Nolan directed Inception."

# Same token counts on both sides, per round, so a mismatch in the LangChain
# vs raw-OpenAI usage-accounting path would show up as a token_usage diff.
ROUND_USAGE = [
    {"prompt_tokens": 50, "completion_tokens": 10, "total_tokens": 60},
    {"prompt_tokens": 70, "completion_tokens": 15, "total_tokens": 85},
    {"prompt_tokens": 90, "completion_tokens": 8, "total_tokens": 98},
]


def round_usage(n: int) -> list[dict]:
    """Generate ``n`` distinct per-round usage dicts (see ``ROUND_USAGE``'s
    docstring comment above) for scenarios with more rounds than
    ``ROUND_USAGE`` covers."""
    return [
        {"prompt_tokens": 50 + i, "completion_tokens": 10 + i, "total_tokens": 60 + 2 * i}
        for i in range(n)
    ]


class _FakeMcp:
    """Deterministic stand-in for the real MCP client, used by
    ``_handle_loop_detected``/``_inject_journal_refresh``/``_snapshot_journal``
    (all called for real by both engines via ``BaseKBQAAgent`` methods —
    ``_ParityAgent`` never overrides them). Tool EXECUTION itself never goes
    through this — ``_ParityAgent._execute_single_tool`` is overridden below
    and answers straight from ``TOOL_RESULTS``."""

    async def call_tool(self, name: str, args: dict | None = None) -> str:
        if name == "GetJournalSummary":
            return TOOL_RESULTS["GetJournalSummary"]
        if name == "GetJournalStateJSON":
            return json.dumps({"discovered": True})
        return ""


class _ParityAgent(BaseKBQAAgent):
    """Shared test double for both engines. Overrides only what's needed to
    avoid a real OpenAI/MCP client; loop detection, context management, and
    synthesis-bypass logic all run for real (from BaseKBQAAgent)."""

    def __init__(self, tool_results: dict | None = None):
        self.name = "parity_agent"
        self.model = "test-model"
        self.request_timeout = 60.0
        self.recorder = TraceRecorder()
        self._parent_span_id_override = None
        self._retry = TransientRetry()
        self.token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self.tool_call_counts: dict[str, int] = {}
        self.tool_call_durations: list = []
        self.tool_call_history: list = []
        self.tool_sequence: list = []
        self.journal_snapshots: list = []
        self.last_journal_state = None
        self._tool_results = dict(TOOL_RESULTS)
        if tool_results:
            self._tool_results.update(tool_results)
        self._known_tool_names = set(TOOL_RESULTS) | _NUDGE_TOOL_NAMES
        self._context_limit = 100000
        self._next_trim_trigger = 0.5
        self._find_resource_cap = 8
        self._sparql_cap = 10
        self._raw_sparql_intervention_done = False
        self._text_tool_call_mode = False
        self.mcp = _FakeMcp()
        self._messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "Who directed Inception?"},
        ]

    def get_config(self):  # pragma: no cover - unused
        raise NotImplementedError

    def get_mcp_server_path(self) -> str:  # pragma: no cover - unused
        raise NotImplementedError

    def _trace(self, message: str, color: str = "") -> None:
        pass

    async def _execute_single_tool(self, func_name: str, func_args: dict) -> str:
        self.tool_call_counts[func_name] = self.tool_call_counts.get(func_name, 0) + 1
        self.tool_call_durations.append({
            "tool_name": func_name,
            "duration_seconds": 0.0,
            "success": True,
            "timestamp": "t",
        })
        return self._tool_results.get(func_name, f"result:{func_name}")


def _run_legacy() -> _ParityAgent:
    agent = _ParityAgent()

    def _tc(call_id, name, args):
        return SimpleNamespace(
            id=call_id,
            type="function",
            function=SimpleNamespace(name=name, arguments=json.dumps(args)),
        )

    scripted = [
        SimpleNamespace(
            usage=SimpleNamespace(**ROUND_USAGE[0]),
            choices=[SimpleNamespace(
                finish_reason="tool_calls",
                message=SimpleNamespace(
                    role="assistant", content=None,
                    tool_calls=[_tc("call_1", "FindNode", FIND_NODE_ARGS)],
                ),
            )],
        ),
        SimpleNamespace(
            usage=SimpleNamespace(**ROUND_USAGE[1]),
            choices=[SimpleNamespace(
                finish_reason="tool_calls",
                message=SimpleNamespace(
                    role="assistant", content=None,
                    tool_calls=[_tc("call_2", "GetRelationDetails", GET_RELATION_ARGS)],
                ),
            )],
        ),
        SimpleNamespace(
            usage=SimpleNamespace(**ROUND_USAGE[2]),
            choices=[SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(role="assistant", content=FINAL_CONTENT, tool_calls=None),
            )],
        ),
    ]
    call_index = {"i": 0}

    class _FakeCompletions:
        def create(self, **kwargs):
            resp = scripted[call_index["i"]]
            call_index["i"] += 1
            return resp

    agent.client = SimpleNamespace(chat=SimpleNamespace(completions=_FakeCompletions()))

    _run(agent._run_tool_loop(
        "Who directed Inception?",
        tools=[{"type": "function", "function": {"name": "FindNode"}}],
        max_iterations=10,
        refresh_interval=1000,  # never triggers in a 3-iteration run
        qtype="Query",
    ))
    return agent


def _run_graph(monkeypatch) -> _ParityAgent:
    agent = _ParityAgent()

    responses = [
        AIMessage(
            content="",
            tool_calls=[{"name": "FindNode", "args": FIND_NODE_ARGS, "id": "call_1", "type": "tool_call"}],
            usage_metadata={
                "input_tokens": ROUND_USAGE[0]["prompt_tokens"],
                "output_tokens": ROUND_USAGE[0]["completion_tokens"],
                "total_tokens": ROUND_USAGE[0]["total_tokens"],
            },
        ),
        AIMessage(
            content="",
            tool_calls=[{"name": "GetRelationDetails", "args": GET_RELATION_ARGS, "id": "call_2", "type": "tool_call"}],
            usage_metadata={
                "input_tokens": ROUND_USAGE[1]["prompt_tokens"],
                "output_tokens": ROUND_USAGE[1]["completion_tokens"],
                "total_tokens": ROUND_USAGE[1]["total_tokens"],
            },
        ),
        AIMessage(
            content=FINAL_CONTENT,
            tool_calls=[],
            usage_metadata={
                "input_tokens": ROUND_USAGE[2]["prompt_tokens"],
                "output_tokens": ROUND_USAGE[2]["completion_tokens"],
                "total_tokens": ROUND_USAGE[2]["total_tokens"],
            },
        ),
    ]
    model = ScriptedChatModel(responses)
    monkeypatch.setattr(runner_module, "build_chat_model", lambda *a, **k: model)

    _run(agent._run_tool_loop(
        "Who directed Inception?",
        tools=[{"type": "function", "function": {"name": "FindNode"}}],
        max_iterations=10,
        refresh_interval=1000,
        qtype="Query",
    ))
    return agent


def test_legacy_and_graph_engines_produce_identical_messages_and_usage(monkeypatch):
    monkeypatch.delenv("AMA_AGENT_ENGINE", raising=False)
    legacy_agent = _run_legacy()

    monkeypatch.setenv("AMA_AGENT_ENGINE", "graph")
    graph_agent = _run_graph(monkeypatch)

    assert legacy_agent._messages == graph_agent._messages
    assert legacy_agent.token_usage == graph_agent.token_usage
    assert legacy_agent.get_tool_call_summary()["tool_counts"] == \
        graph_agent.get_tool_call_summary()["tool_counts"]
    assert legacy_agent.get_tool_call_summary()["total_calls"] == \
        graph_agent.get_tool_call_summary()["total_calls"]
