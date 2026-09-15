"""Fusion must preserve each specialist's "Reproduce with SPARQL" block.

A fused answer carries two queries written against two different endpoints
(KQAPro's Virtuoso and ORKG's), so they can never be merged, rewritten, or
translated into one another's URI scheme — a booth visitor has to be able to
paste either one unchanged. These tests pin the fusion system prompt's rules
and check that both blocks actually reach the fusion LLM, labelled by graph.

No network calls, no API keys, no MCP subprocess.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from ama_kbqa.agents.orchestrator_agent.agent import AGENT_CONFIG, Orchestrator
from ama_kbqa.framework.trace import TraceRecorder


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)


def _run(coro):
    return asyncio.run(coro)


def _make_orchestrator(client=None):
    """Instantiate Orchestrator without running __init__ (mirrors the pattern
    in test_orchestrator_federation_mode.py). The real _agent_config is kept:
    its graph_label entries are part of what is under test."""
    from chatkit import TransientRetry

    o = Orchestrator.__new__(Orchestrator)
    o.name = "TEST_ORCHESTRATOR"
    o.mcp = None
    o.client = client if client is not None else MagicMock()
    o.model = "test-model"
    o.last_routing_reason = None
    o.last_routing_evidence = None
    o.recorder = TraceRecorder()
    o.journal_snapshots = []
    o.token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    o._agents = {}
    o._retry = TransientRetry()
    # The production table, not a copy with invented labels: the graph_label
    # values are part of what is under test.
    o._agent_config = {k: dict(v) for k, v in AGENT_CONFIG.items()}
    return o


def test_every_specialist_declares_a_graph_label():
    for name, cfg in AGENT_CONFIG.items():
        assert cfg.get("graph_label"), f"{name} has no graph_label for its SPARQL block"


KQAPRO_BLOCK = (
    "```sparql\n"
    "PREFIX ex:   <http://kqapro.org/entity/>\n"
    "PREFIX prop: <http://kqapro.org/property/>\n"
    "SELECT ?answer FROM <http://kqapro.org/kb> WHERE {\n"
    "  ex:Q25188 prop:director ?answer .\n"
    "}\n"
    "```"
)
ORKG_BLOCK = (
    "```sparql\n"
    "PREFIX orkgr: <http://orkg.org/orkg/resource/>\n"
    "PREFIX orkgp: <http://orkg.org/orkg/predicate/>\n"
    "SELECT ?answer WHERE { GRAPH <http://sciqa.org/kg> {\n"
    "  orkgr:R192320 orkgp:P31 ?answer .\n"
    "} }\n"
    "```"
)


def _answers():
    return [
        {
            "agent": "kqapro_agent",
            "answer": (
                "Inception was directed by Christopher Nolan.\n\n"
                "Reproduce with SPARQL:\n" + KQAPRO_BLOCK + "\n"
                "Verified against the knowledge graph."
            ),
        },
        {
            "agent": "sciqa_agent",
            "answer": (
                "Common benchmarks include ImageNet and GLUE.\n\n"
                "Reproduce with SPARQL:\n" + ORKG_BLOCK + "\n"
                "(not executed)"
            ),
        },
    ]


# ---------------------------------------------------------------------------
# The fusion system prompt states the preservation rules
# ---------------------------------------------------------------------------

def test_fusion_prompt_demands_verbatim_preservation():
    prompt = _make_orchestrator()._fusion_system_prompt()
    assert "Reproduce with SPARQL:" in prompt
    assert "character for character" in prompt
    assert "Label each block with the graph it belongs to" in prompt
    # The three ways a model could quietly destroy a block.
    assert "NEVER merge two blocks into one query" in prompt
    assert "rewrite" in prompt
    assert "translate between the two URI schemes" in prompt
    # Agreement is not a licence to drop one.
    assert "drop a block" in prompt
    assert "Two specialists means two blocks." in prompt
    # No invented SPARQL of the orchestrator's own.
    assert "Write no SPARQL of your own" in prompt
    # The verified / not-executed marker rides along with its block.
    assert "(not executed)" in prompt


# ---------------------------------------------------------------------------
# Both blocks reach the fusion LLM, each labelled with its graph
# ---------------------------------------------------------------------------

def test_fuse_answers_sends_both_blocks_and_graph_labels():
    captured = {}

    def _create(**kwargs):
        captured["messages"] = kwargs["messages"]
        message = SimpleNamespace(tool_calls=None, content="fused answer")
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    client = MagicMock()
    client.chat.completions.create.side_effect = _create
    o = _make_orchestrator(client=client)

    assert _run(o._fuse_answers("Q?", _answers())) == "fused answer"

    system, user = captured["messages"]
    assert system["role"] == "system"
    assert "character for character" in system["content"]

    content = user["content"]
    # Both queries arrive intact, fences and all.
    assert KQAPRO_BLOCK in content
    assert ORKG_BLOCK in content
    assert content.count("```sparql") == 2
    # Each is attributed to the graph it must be labelled with downstream.
    assert "Graph name for this specialist's SPARQL block: KQAPro (Wikidata subset)" in content
    assert "Graph name for this specialist's SPARQL block: ORKG" in content
    # The per-block verification markers survive too.
    assert "Verified against the knowledge graph." in content
    assert "(not executed)" in content


def test_fuse_answers_keeps_blocks_when_a_scratchpad_is_attached():
    """The scratchpad is appended after the answer; it must not displace the
    block or break the graph-label association."""
    captured = {}

    def _create(**kwargs):
        captured["messages"] = kwargs["messages"]
        message = SimpleNamespace(tool_calls=None, content="fused")
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    client = MagicMock()
    client.chat.completions.create.side_effect = _create
    o = _make_orchestrator(client=client)

    answers = _answers()
    answers[0]["scratchpad"] = "visited Q25188; found Q25191"
    _run(o._fuse_answers("Q?", answers))

    content = captured["messages"][1]["content"]
    assert KQAPRO_BLOCK in content
    assert ORKG_BLOCK in content
    assert "visited Q25188" in content


def test_fusion_degrades_to_a_specialist_answer_with_its_block_intact():
    """When fusion fails the orchestrator falls back to the first specialist's
    answer — which must still carry its own SPARQL block."""
    client = MagicMock()
    client.chat.completions.create.side_effect = Exception("boom")
    o = _make_orchestrator(client=client)

    answer = _run(o._fuse_answers("Q?", _answers()))
    assert KQAPRO_BLOCK in answer
