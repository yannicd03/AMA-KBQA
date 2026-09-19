"""A fenced ```sparql block must survive everything between the model and the UI.

The frontend renders the answer as Streamlit markdown, so the fence is what
gives the booth visitor a copy button. Anything that touches the answer text on
the way out — think-block stripping, Verify normalization, the fast path, which
has no final LLM turn at all — has to leave the block intact.

No network calls, no API keys, no MCP subprocess.
"""

import asyncio
import json
from types import SimpleNamespace

import pytest

import ama_kbqa.config as config
import ama_kbqa.framework.base_agent as base_agent_module
from ama_kbqa.agents.kqapro_agent.agent import KQAProAgent
from ama_kbqa.framework.base_agent import BaseKBQAAgent
from ama_kbqa.framework.trace import TraceRecorder

SPARQL_BLOCK = (
    "```sparql\n"
    "PREFIX ex:   <http://kqapro.org/entity/>\n"
    "PREFIX prop: <http://kqapro.org/property/>\n"
    "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>\n"
    "SELECT ?answer ?answerLabel\n"
    "FROM <http://kqapro.org/kb>\n"
    "WHERE {\n"
    "  ex:Q25188 prop:director ?answer .\n"
    "  ?answer rdfs:label ?answerLabel .\n"
    "}\n"
    "```"
)


def _agent() -> KQAProAgent:
    return object.__new__(KQAProAgent)


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Answer-cleanup helpers
# ---------------------------------------------------------------------------

def test_think_block_stripping_leaves_the_sparql_fence_intact():
    answer = (
        "<think>weighing the candidates</think>\n"
        "Inception was directed by Christopher Nolan.\n\n"
        "Reproduce with SPARQL:\n" + SPARQL_BLOCK + "\n"
        "Verified against the knowledge graph."
    )

    cleaned = _agent()._finalize_answer_text(answer, "QueryRelation")

    assert "weighing the candidates" not in cleaned
    assert SPARQL_BLOCK in cleaned
    assert cleaned.count("```") == 2
    assert "Verified against the knowledge graph." in cleaned


def test_strip_think_blocks_does_not_touch_fenced_content():
    """Backticks, angle-bracket URIs and braces inside the fence are untouched."""
    assert _agent()._strip_think_blocks(SPARQL_BLOCK) == SPARQL_BLOCK


def test_conversational_verify_answer_keeps_its_sparql_block(monkeypatch):
    """Verify normalization would collapse the answer to "yes" and destroy both
    sections, so it is benchmark-only."""
    monkeypatch.setattr(config, "get_synthesis_mode", lambda: "conversational")
    answer = (
        "Yes, Reno's population is greater than 100.\n\n"
        "How I found this:\n- Looked up Reno.\n\n"
        "Reproduce with SPARQL:\n" + SPARQL_BLOCK + "\n"
        "Verified against the knowledge graph."
    )

    cleaned = _agent()._finalize_answer_text(answer, "Verify")

    assert cleaned != "yes"
    assert SPARQL_BLOCK in cleaned
    assert "How I found this:" in cleaned


def test_benchmark_verify_answer_is_still_normalized(monkeypatch):
    monkeypatch.setattr(config, "get_synthesis_mode", lambda: "benchmark")
    assert _agent()._finalize_answer_text(
        "Reno's population is greater than 100.", "Verify"
    ) == "yes"


# ---------------------------------------------------------------------------
# The fast path has no final LLM turn, so it writes the sections itself
# ---------------------------------------------------------------------------

def test_fast_path_reproduction_query_uses_the_real_uri_scheme():
    built = _agent()._build_fast_path_reproduction_query("Q25188", "director", "relation")

    assert built["tool"] == "RunSPARQL"
    query = built["query"]
    assert "PREFIX ex:   <http://kqapro.org/entity/>" in query
    assert "FROM <http://kqapro.org/kb>" in query
    assert "ex:Q25188 prop:director ?answer ." in query
    assert "?answer rdfs:label ?answerLabel ." in query


def test_fast_path_attribute_query_handles_qualifier_nodes():
    """KQAPro attribute values sit either directly on the edge or on an
    intermediate rdf:value node, so the query must cover both."""
    query = _agent()._build_fast_path_reproduction_query(
        "Q3012", "native_label", "attribute"
    )["query"]

    assert "ex:Q3012 attr:native_label ?node ." in query
    assert "OPTIONAL { ?node rdf:value ?direct . }" in query
    assert "BIND(COALESCE(?direct, ?node) AS ?answer)" in query


@pytest.mark.parametrize(
    "node_id,member,kind",
    [
        ("not-an-id", "director", "relation"),
        ("Q25188", "FIPS_6-4_(US_counties)", "attribute"),  # unsafe prefixed name
        ("Q25188", "", "relation"),
        ("Q25188", "director", "something_else"),
    ],
)
def test_fast_path_declines_queries_it_cannot_express(node_id, member, kind):
    assert _agent()._build_fast_path_reproduction_query(node_id, member, kind) is None


class _FastPathAgent(KQAProAgent):
    """Records fast-path tool calls and replays a canned SPARQL result."""

    def __init__(self, sparql_result):
        self.name = "test_kqapro_agent"  # _trace reads it on the fallback paths
        self.calls = []
        self._sparql_result = sparql_result

    async def _execute_fast_path_tool(self, func_name, func_args):
        self.calls.append((func_name, func_args))
        return self._sparql_result


def test_benchmark_fast_path_answer_stays_bare(monkeypatch):
    monkeypatch.setattr(config, "get_synthesis_mode", lambda: "benchmark")
    agent = _FastPathAgent('{"bindings": [{"answer": "Christopher Nolan"}]}')

    answer = _run(
        agent._finish_fast_path(["Christopher Nolan"], "Inception", "Q25188", "director", "relation")
    )

    assert answer == "Christopher Nolan"
    # Benchmark mode must not pay for the verification round trip either.
    assert agent.calls == []


def test_conversational_fast_path_answer_carries_both_sections(monkeypatch):
    monkeypatch.setattr(config, "get_synthesis_mode", lambda: "conversational")
    agent = _FastPathAgent('{"bindings": [{"answer": "Christopher Nolan"}]}')

    answer = _run(
        agent._finish_fast_path(["Christopher Nolan"], "Inception", "Q25188", "director", "relation")
    )

    assert answer.startswith("Christopher Nolan")
    assert "How I found this:" in answer
    assert "Reproduce with SPARQL:" in answer
    assert "```sparql" in answer
    assert answer.count("```") == 2
    assert "ex:Q25188 prop:director ?answer ." in answer
    assert answer.rstrip().endswith("Verified against the knowledge graph.")
    # Exactly one extra round trip, through the raw SPARQL tool.
    assert [name for name, _ in agent.calls] == ["RunSPARQL"]


def test_conversational_fast_path_falls_back_when_verification_is_empty(monkeypatch):
    """No rows means the query does not actually reproduce the answer, so the
    fast path must hand over to the full loop rather than ship a dud block."""
    monkeypatch.setattr(config, "get_synthesis_mode", lambda: "conversational")
    agent = _FastPathAgent('{"bindings": []}')

    answer = _run(
        agent._finish_fast_path(["Christopher Nolan"], "Inception", "Q25188", "director", "relation")
    )

    assert answer is None


def test_conversational_fast_path_falls_back_when_no_query_can_be_built(monkeypatch):
    monkeypatch.setattr(config, "get_synthesis_mode", lambda: "conversational")
    agent = _FastPathAgent('{"bindings": [{"answer": "x"}]}')

    answer = _run(
        agent._finish_fast_path(["x"], "Some Entity", "not-an-id", "director", "relation")
    )

    assert answer is None
    assert agent.calls == []


# ---------------------------------------------------------------------------
# The loop must not overwrite the real answer with a later filler turn
# ---------------------------------------------------------------------------

CONTRACT_ANSWER = (
    "The director of Inception is Christopher Nolan.\n\n"
    "How I found this:\n"
    "- Looked up Inception (Q25188).\n\n"
    "Reproduce with SPARQL:\n" + SPARQL_BLOCK + "\n"
    "Verified against the knowledge graph."
)


def _tool_call(name, arguments=None):
    return SimpleNamespace(
        id=f"call-{name}",
        type="function",
        function=SimpleNamespace(name=name, arguments=json.dumps(arguments or {})),
    )


def _llm_response(content=None, tool_calls=None):
    message = SimpleNamespace(role="assistant", content=content, tool_calls=tool_calls)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason="stop")],
        usage=None,
    )


class _ScriptedLoopAgent(BaseKBQAAgent):
    """Drives `_run_tool_loop` over a scripted sequence of LLM responses.

    No client, no MCP, no tools: each turn pops the next scripted response and
    records the tool_choice the loop asked for.
    """

    def __init__(self, script):
        self._script = list(script)
        self.tool_choices = []
        self._messages = []
        self.recorder = TraceRecorder()
        self.tool_call_counts = {"FindNode": 1}  # pretend the loop did real work
        self._text_tool_call_mode = False
        self._max_tool_calls = 0

    def get_config(self):
        raise NotImplementedError

    def get_mcp_server_path(self) -> str:
        raise NotImplementedError

    def _trace(self, message: str, color: str = "") -> None:
        pass

    def _manage_context_window(self) -> None:
        pass

    def _maybe_inject_raw_sparql_distress(self, iteration_count: int) -> bool:
        return False

    def _get_journal_summary_answer_prompt(self) -> str:
        return "Now provide your final answer. Do NOT call more tools."

    def _llm_call(self, tools=None, tool_choice=None, messages_override=None):
        self.tool_choices.append(tool_choice)
        return self._script.pop(0)

    async def _execute_tool_calls(self, tool_calls, as_user_messages=False):
        # True only for a bare GetJournalSummary, mirroring production.
        return any(tc.function.name == "GetJournalSummary" for tc in tool_calls)

    async def _run_synthesis(self, query: str, qtype: str = "") -> str:
        return "SYNTHESIS_SENTINEL"


@pytest.fixture
def _loop_env(monkeypatch):
    monkeypatch.setattr(base_agent_module, "get_auto_inject_journal", lambda: True)
    monkeypatch.setattr(base_agent_module, "get_synthesis_enabled", lambda: False)
    monkeypatch.setattr(base_agent_module, "get_zero_tool_call_retry", lambda: False)
    monkeypatch.setattr(base_agent_module, "get_zero_tool_call_retry_max", lambda: 0)


def _script_answer_then_filler():
    """iter 1: real tool call. iter 2: GetJournalSummary (injects the answer
    prompt). iter 3: the real answer plus a throwaway tool call. iter 4: the
    filler turn that used to become the final answer."""
    return [
        _llm_response(tool_calls=[_tool_call("FindNode", {"semantic_node_name": "Inception"})]),
        _llm_response(tool_calls=[_tool_call("GetJournalSummary")]),
        _llm_response(content=CONTRACT_ANSWER, tool_calls=[_tool_call("ManageJournal", {"action": "clear"})]),
        _llm_response(content="Task completed."),
    ]


def test_conversational_loop_stops_forcing_tool_calls_after_the_answer_prompt(
    monkeypatch, _loop_env
):
    monkeypatch.setattr(config, "get_synthesis_mode", lambda: "conversational")
    agent = _ScriptedLoopAgent(_script_answer_then_filler())

    _run(agent._run_tool_loop("Who directed Inception?", [], 20, 5, qtype="QueryRelation"))

    # Iterations 1 and 2 still force a tool call; from iteration 3 (the first
    # call after the answer prompt was injected) the loop must stop forcing,
    # because the prompt itself says "do NOT call more tools".
    assert agent.tool_choices[:2] == ["required", "required"]
    assert agent.tool_choices[2] == "auto"


def test_benchmark_loop_keeps_the_measured_tool_choice_schedule(monkeypatch, _loop_env):
    """Benchmark mode is what the paper's numbers were measured with, so the
    required/auto schedule there must be untouched."""
    monkeypatch.setattr(config, "get_synthesis_mode", lambda: "benchmark")
    agent = _ScriptedLoopAgent(_script_answer_then_filler())

    _run(agent._run_tool_loop("Who directed Inception?", [], 20, 5, qtype="QueryRelation"))

    assert agent.tool_choices[:4] == ["required", "required", "required", "auto"]


def test_conversational_loop_returns_the_contract_answer_not_the_filler(
    monkeypatch, _loop_env
):
    monkeypatch.setattr(config, "get_synthesis_mode", lambda: "conversational")
    agent = _ScriptedLoopAgent(_script_answer_then_filler())

    answer = _run(
        agent._run_tool_loop("Who directed Inception?", [], 20, 5, qtype="QueryRelation")
    )

    assert answer != "Task completed."
    assert "Reproduce with SPARQL:" in answer
    assert "How I found this:" in answer
    assert SPARQL_BLOCK in answer


def test_conversational_loop_keeps_a_later_answer_that_also_satisfies_the_contract(
    monkeypatch, _loop_env
):
    """The preference only kicks in when the LAST message fails the contract;
    a genuine later revision must still win."""
    monkeypatch.setattr(config, "get_synthesis_mode", lambda: "conversational")
    revised = CONTRACT_ANSWER.replace("Christopher Nolan", "Christopher Nolan (revised)")
    agent = _ScriptedLoopAgent([
        _llm_response(tool_calls=[_tool_call("GetJournalSummary")]),
        _llm_response(content=CONTRACT_ANSWER, tool_calls=[_tool_call("ManageJournal", {})]),
        _llm_response(content=revised),
    ])

    answer = _run(
        agent._run_tool_loop("Who directed Inception?", [], 20, 5, qtype="QueryRelation")
    )

    assert "(revised)" in answer


def test_satisfies_conversational_contract_predicate():
    ok = BaseKBQAAgent._satisfies_conversational_contract
    assert ok(CONTRACT_ANSWER)
    assert not ok("Task completed.")
    assert not ok("")
    assert not ok("Christopher Nolan.\n\nHow I found this:\n- looked it up.")
