"""Phase 2 parity tests: one scenario per ported behaviour (see the Phase 2
PRD, ``.agent/Tasks/active/langgraph-rewrite.md``), each driving the SAME
scripted transcript through the legacy ``while`` loop
(``BaseKBQAAgent._run_tool_loop``) and the graph engine
(``ama_kbqa.graph.runner.run_tool_loop_graph``) and asserting identical
``agent._messages``, ``agent.token_usage``, ``agent.get_tool_call_summary()``,
the returned answer, and the ``(kind, name)`` sequence of recorder
events/spans.

Builds on the harness and ``_ParityAgent`` test double established in
``test_legacy_graph_parity.py`` (imported, not duplicated) — every scenario
here just supplies a different scripted round sequence and, where needed, a
different ``_ParityAgent`` configuration (a smaller ``_context_limit``, a
``_max_tool_calls`` cap, patched zero-tool-call-retry config, etc).
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

from langchain_core.messages import AIMessage

import ama_kbqa.framework.base_agent as base_agent_module
import ama_kbqa.graph.guards as guards_module
import ama_kbqa.graph.runner as runner_module

from ._fakes import ScriptedChatModel
from .test_legacy_graph_parity import _ParityAgent, round_usage

QUERY = "Who directed Inception?"
DEFAULT_TOOLS = [{"type": "function", "function": {"name": "FindNode"}}]


def _run(coro):
    return asyncio.run(coro)


class _SynthStubAgent(_ParityAgent):
    """``_ParityAgent`` variant for scenarios whose script forces a path
    through ``_run_synthesis`` (the ``max_tool_calls`` cap, and a
    zero-content final turn) — stubs it out instead of needing a working
    synthesis LLM client."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.synthesis_calls: list = []

    async def _run_synthesis(self, query: str, qtype: str = "") -> str:
        self.synthesis_calls.append({"query": query, "qtype": qtype})
        return "SYNTH_SENTINEL"


def _tc_legacy(call_id: str, name: str, args: dict):
    return SimpleNamespace(
        id=call_id, type="function", function=SimpleNamespace(name=name, arguments=json.dumps(args))
    )


def _legacy_round(idx: int, spec: dict, usage: dict):
    if "final" in spec:
        return SimpleNamespace(
            usage=SimpleNamespace(**usage),
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(role="assistant", content=spec["final"], tool_calls=None),
                )
            ],
        )
    tcs = [_tc_legacy(f"call_{idx}_{j}", name, args) for j, (name, args) in enumerate(spec["calls"])]
    return SimpleNamespace(
        usage=SimpleNamespace(**usage),
        choices=[
            SimpleNamespace(
                finish_reason="tool_calls",
                message=SimpleNamespace(role="assistant", content=None, tool_calls=tcs),
            )
        ],
    )


def _graph_round(idx: int, spec: dict, usage: dict):
    um = {
        "input_tokens": usage["prompt_tokens"],
        "output_tokens": usage["completion_tokens"],
        "total_tokens": usage["total_tokens"],
    }
    if "final" in spec:
        return AIMessage(content=spec["final"], tool_calls=[], usage_metadata=um)
    tcs = [
        {"name": name, "args": args, "id": f"call_{idx}_{j}", "type": "tool_call"}
        for j, (name, args) in enumerate(spec["calls"])
    ]
    return AIMessage(content="", tool_calls=tcs, usage_metadata=um)


def _build_scripts(rounds: list[dict]):
    usage_seq = round_usage(len(rounds))
    legacy = [_legacy_round(i, r, u) for i, (r, u) in enumerate(zip(rounds, usage_seq))]
    graph = [_graph_round(i, r, u) for i, (r, u) in enumerate(zip(rounds, usage_seq))]
    return legacy, graph


def _run_legacy_scripted(agent, legacy_script, **loop_kwargs):
    call_index = {"i": 0}

    class _FakeCompletions:
        def create(self, **kwargs):
            resp = legacy_script[call_index["i"]]
            call_index["i"] += 1
            return resp

    agent.client = SimpleNamespace(chat=SimpleNamespace(completions=_FakeCompletions()))
    return _run(agent._run_tool_loop(QUERY, tools=DEFAULT_TOOLS, **loop_kwargs))


def _run_graph_scripted(agent, graph_script, monkeypatch, **loop_kwargs):
    model = ScriptedChatModel(graph_script)
    monkeypatch.setattr(runner_module, "build_chat_model", lambda *a, **k: model)
    return _run(agent._run_tool_loop(QUERY, tools=DEFAULT_TOOLS, **loop_kwargs))


def _event_seq(agent) -> list[tuple[str, str]]:
    return [(e["kind"], e["name"]) for e in agent.recorder.to_dicts()]


def _assert_parity(legacy_agent, graph_agent, legacy_result, graph_result):
    assert legacy_agent._messages == graph_agent._messages
    assert legacy_agent.token_usage == graph_agent.token_usage
    legacy_summary = legacy_agent.get_tool_call_summary()
    graph_summary = graph_agent.get_tool_call_summary()
    assert legacy_summary.get("tool_counts", {}) == graph_summary.get("tool_counts", {})
    assert legacy_summary.get("total_calls", 0) == graph_summary.get("total_calls", 0)
    assert _event_seq(legacy_agent) == _event_seq(graph_agent)
    assert legacy_result == graph_result


def _run_scenario(
    rounds,
    monkeypatch,
    *,
    agent_factory=_ParityAgent,
    configure=None,
    max_iterations=10,
    refresh_interval=1_000_000,
    qtype="Query",
):
    """Shared driver: build both scripts, run legacy then graph (each on a
    FRESH agent instance from ``agent_factory``, optionally tweaked by
    ``configure(agent)``), assert full parity, and return
    ``(legacy_agent, graph_agent, legacy_result)`` for scenario-specific
    extra assertions."""
    legacy_script, graph_script = _build_scripts(rounds)
    loop_kwargs = {"max_iterations": max_iterations, "refresh_interval": refresh_interval, "qtype": qtype}

    monkeypatch.delenv("AMA_AGENT_ENGINE", raising=False)
    legacy_agent = agent_factory()
    if configure:
        configure(legacy_agent)
    legacy_result = _run_legacy_scripted(legacy_agent, legacy_script, **loop_kwargs)

    monkeypatch.setenv("AMA_AGENT_ENGINE", "graph")
    graph_agent = agent_factory()
    if configure:
        configure(graph_agent)
    graph_result = _run_graph_scripted(graph_agent, graph_script, monkeypatch, **loop_kwargs)

    _assert_parity(legacy_agent, graph_agent, legacy_result, graph_result)
    return legacy_agent, graph_agent, legacy_result


# ---------------------------------------------------------------------------
# 1. Loop detection + intervention
# ---------------------------------------------------------------------------


def test_loop_detection_intervenes_on_third_identical_call(monkeypatch):
    args = {"semantic_node_name": "Inception"}
    rounds = [
        {"calls": [("FindNode", args)]},
        {"calls": [("FindNode", args)]},
        {"calls": [("FindNode", args)]},  # 3rd identical call -> loop detected
        {"final": "Christopher Nolan directed Inception."},
    ]
    legacy_agent, graph_agent, _ = _run_scenario(rounds, monkeypatch)

    assert ("loop_detected", "FindNode") in _event_seq(legacy_agent)
    # Only the first 2 calls actually executed; the 3rd was intervened.
    assert legacy_agent.tool_call_counts["FindNode"] == 2
    assert "INFINITE LOOP DETECTED" in legacy_agent._messages[-2]["content"]


# ---------------------------------------------------------------------------
# 2. Periodic journal refresh
# ---------------------------------------------------------------------------


def test_periodic_journal_refresh_appended_every_n_iterations(monkeypatch):
    rounds = [
        {"calls": [("FindNode", {"semantic_node_name": "Inception"})]},
        {"calls": [("GetRelationDetails", {"base_node_id": "Q1"})]},  # refresh fires before this call
        {"final": "Christopher Nolan directed Inception."},
    ]
    legacy_agent, graph_agent, _ = _run_scenario(rounds, monkeypatch, refresh_interval=2)

    assert ("journal_refresh", "iter:2") in _event_seq(legacy_agent)
    refresh_markers = [
        m for m in legacy_agent._messages
        if isinstance(m.get("content"), str) and m["content"].startswith(_ParityAgent._JOURNAL_REFRESH_MARKER)
    ]
    assert len(refresh_markers) == 1
    assert "Iteration 2" in refresh_markers[0]["content"]


# ---------------------------------------------------------------------------
# 3. Context-window compaction
# ---------------------------------------------------------------------------


def _seed_heavy_messages(agent, n=12, size=300):
    agent._messages += [
        {"role": "tool", "tool_call_id": f"pad{i}", "name": "Pad", "content": "x" * size}
        for i in range(n)
    ]
    agent._context_limit = 1000


def test_context_compaction_trims_padding_before_first_llm_call(monkeypatch):
    rounds = [
        {"calls": [("FindNode", {"semantic_node_name": "Inception"})]},
        {"final": "Christopher Nolan directed Inception."},
    ]
    legacy_agent, graph_agent, _ = _run_scenario(rounds, monkeypatch, configure=_seed_heavy_messages)

    assert ("context_trim", "trimmed_4") in _event_seq(legacy_agent) or any(
        k == "context_trim" for k, _ in _event_seq(legacy_agent)
    )
    trimmed = [m for m in legacy_agent._messages if str(m.get("content", "")).endswith("...[trimmed]")]
    assert trimmed, "expected at least one message to have been truncated"


# ---------------------------------------------------------------------------
# 4. GetJournalSummary answer-prompt injection
# ---------------------------------------------------------------------------


def test_get_journal_summary_call_injects_answer_prompt(monkeypatch):
    rounds = [
        {"calls": [("GetJournalSummary", {})]},
        {"final": "Christopher Nolan directed Inception."},
    ]
    legacy_agent, graph_agent, _ = _run_scenario(rounds, monkeypatch)

    prompts = [m for m in legacy_agent._messages if m.get("content") == "Now provide your final answer. Do NOT call more tools."]
    assert len(prompts) == 1


# ---------------------------------------------------------------------------
# 5. Raw-SPARQL distress
# ---------------------------------------------------------------------------


def test_raw_sparql_distress_fires_at_threshold(monkeypatch):
    rounds = [
        {"calls": [("RunSPARQL", {"q": f"query{i}"})]} for i in range(4)
    ] + [{"final": "Christopher Nolan directed Inception."}]
    legacy_agent, graph_agent, _ = _run_scenario(rounds, monkeypatch)

    assert ("intervention", "raw_sparql_distress") in _event_seq(legacy_agent)
    distress = [m for m in legacy_agent._messages if "RAW SPARQL DISTRESS" in str(m.get("content", ""))]
    assert len(distress) == 1


# ---------------------------------------------------------------------------
# 6. max_tool_calls hard cap
# ---------------------------------------------------------------------------


def test_max_tool_calls_cap_exits_through_synthesis(monkeypatch):
    rounds = [
        {"calls": [("FindNode", {"semantic_node_name": "Inception"})]},
        {"calls": [("GetRelationDetails", {"base_node_id": "Q1"})]},  # total=2 -> cap reached
    ]

    def configure(agent):
        agent._max_tool_calls = 2

    legacy_agent, graph_agent, legacy_result = _run_scenario(
        rounds, monkeypatch, agent_factory=_SynthStubAgent, configure=configure
    )

    assert ("intervention", "max_tool_calls_reached") in _event_seq(legacy_agent)
    assert legacy_result == "SYNTH_SENTINEL"
    assert legacy_agent.synthesis_calls == graph_agent.synthesis_calls == [
        {"query": QUERY, "qtype": "Query"}
    ]


# ---------------------------------------------------------------------------
# 7. Wrap-up nudge (iteration >= 15, every 5th)
# ---------------------------------------------------------------------------


def test_wrap_up_nudge_fires_at_iteration_15(monkeypatch):
    rounds = [{"calls": [(f"Tool{i + 1}", {"i": i})]} for i in range(15)] + [
        {"final": "Christopher Nolan directed Inception."}
    ]
    legacy_agent, graph_agent, _ = _run_scenario(rounds, monkeypatch, max_iterations=20)

    nudges = [
        m for m in legacy_agent._messages
        if isinstance(m.get("content"), str) and m["content"].startswith("You are on iteration 15.")
    ]
    assert len(nudges) == 1


# ---------------------------------------------------------------------------
# 8. Zero-tool-call retry then hard stop
# ---------------------------------------------------------------------------


def test_zero_tool_call_retry_then_hard_stop(monkeypatch):
    monkeypatch.setattr(base_agent_module, "get_zero_tool_call_retry_max", lambda: 1)
    monkeypatch.setattr(guards_module, "get_zero_tool_call_retry_max", lambda: 1)

    rounds = [
        {"final": "I recall Nolan directed it."},  # retry (budget 1)
        {"final": "Still no tools needed."},  # retry budget exhausted -> hard stop
    ]
    legacy_agent, graph_agent, legacy_result = _run_scenario(rounds, monkeypatch)

    from ama_kbqa.graph.guards import ZERO_TOOL_HARD_STOP_ANSWER

    assert legacy_result == ZERO_TOOL_HARD_STOP_ANSWER
    assert ("intervention", "zero_tool_call_retry") in _event_seq(legacy_agent)
    assert ("intervention", "zero_tool_call_hard_stop") in _event_seq(legacy_agent)
    # The discarded assistant turns never made it into history — only the
    # retry nudge did.
    assert legacy_agent._messages[-1]["role"] == "user"
    assert "STOP." in legacy_agent._messages[-1]["content"]


# ---------------------------------------------------------------------------
# 9. Final-turn message appended only when it has content
# ---------------------------------------------------------------------------


def test_empty_final_content_not_appended_and_falls_through_to_synthesis(monkeypatch):
    rounds = [
        {"calls": [("FindNode", {"semantic_node_name": "Inception"})]},
        {"final": ""},
    ]
    legacy_agent, graph_agent, legacy_result = _run_scenario(
        rounds, monkeypatch, agent_factory=_SynthStubAgent
    )

    assert legacy_result == "SYNTH_SENTINEL"
    # Only system/user/assistant(tool_call)/tool(result) — no trailing
    # empty-content assistant message.
    assert len(legacy_agent._messages) == 4
    assert legacy_agent._messages[-1]["role"] == "tool"
