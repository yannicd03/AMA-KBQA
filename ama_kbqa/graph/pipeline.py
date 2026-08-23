"""Phase 3: the outer per-question pipeline as an explicit ``StateGraph``.

``BaseKBQAAgent._ask_impl`` dispatches here when
``ama_kbqa.config.get_agent_engine() == "graph"`` AND the agent is not in
text-mode tool-call mode (text-mode always uses the legacy ``_ask_impl`` body
unchanged, same fallback rule the tool loop already applies — see the
``ama_kbqa.graph`` package docstring). Everything downstream of the tool loop
was already ported in Phase 1/2 (``ama_kbqa.graph.runner.run_tool_loop_graph``,
which itself decides the synthesis bypass and calls
``BaseKBQAAgent._run_synthesis``/``_finalize_answer_text`` unchanged); this
module lifts what used to be inline ``if``/``while`` control flow in
``_ask_impl`` (MCP init, tool listing, follow-up detection, classification,
prompt assembly, fast path, tool filtering) into graph nodes.

Design: state carries only the DECISIONS made between nodes (qtype, entities,
relations, the filtered tool list, the final answer) — never a duplicate copy
of the message history. Every node that legacy would have mutated
``self._messages`` for does exactly that here too, calling the same
``BaseKBQAAgent`` methods unchanged (``_classify_question``,
``_build_static_qtype_context``, ``_build_question_context``,
``_try_fast_path``, ``_get_allowed_tools_for_qtype``,
``_get_denied_tool_names``, ``_finalize_answer_text``). This is the same
reuse strategy Phase 1/2 established for the tool loop
(``ama_kbqa.graph.builder``/``guards``/``context``): the graph is an explicit
topology bolted onto unchanged agent methods, not a reimplementation of their
logic.

Node list (see ``.agent/Tasks/active/langgraph-rewrite.md`` §4.2/Phase 3):

- ``prepare``: MCP init + tool listing + text-mode catalog injection (dead
  code in practice here since text-mode agents never reach this graph, kept
  for structural parity with ``_ask_impl``) + follow-up detection. Routes to
  ``no_mcp_fallback`` if the MCP connection failed to come up silently (a
  corner case reachable only by test doubles — see ``_ask_impl`` ~:979;
  production ``_init_mcp`` raises on failure, so this branch does not fire
  for ``KQAProAgent``/``SciQAAgent`` today, but the graph mirrors the legacy
  guard rather than dropping it).
- ``classify``: ``_classify_question`` (skipped for follow-ups via the
  conditional edge out of ``prepare``).
- ``assemble_prompt``: the prefix-cache-ordered message assembly — static
  qtype block, raw query, per-question context — for a fresh turn; for a
  follow-up, just appends the query and defaults ``qtype`` to ``"Query"``,
  exactly like ``_ask_impl``'s ``if is_followup: ... else: ...`` branch.
- ``fast_path``: the eligibility gate + ``_try_fast_path`` call for a fresh
  turn (never reached on a follow-up, matching legacy); on success sets the
  final answer, on failure (or ineligibility) computes the qtype-filtered +
  denylisted tool list for ``tool_loop``.
- ``tool_loop``: calls ``agent._run_tool_loop(...)`` — NOT
  ``ama_kbqa.graph.runner.run_tool_loop_graph`` directly. ``_run_tool_loop``
  already contains the engine dispatch (checks
  ``get_agent_engine() == "graph"`` and calls ``run_tool_loop_graph`` itself
  — ``base_agent.py`` ~:1397), which already performs the Phase 1/2 tool loop
  AND the synthesis-bypass/exit-reason routing
  (``max_iterations``/``max_tool_calls`` exit through synthesis,
  ``zero_tool_hard_stop`` returns the literal error string, ``final_answer``
  takes the ``synthesis_enabled`` bypass or falls through to synthesis),
  returning a fully finalized answer string. Going through
  ``agent._run_tool_loop`` rather than importing ``run_tool_loop_graph``
  directly also keeps this node testable through the exact same seam every
  existing test double (`_HookAgent`, `_ParityAgent`, this module's own
  tests) already overrides to script the tool loop. See "Resolved ambiguity"
  below for why this module does not also reimplement the synthesis decision
  as a separate ``synthesis`` graph node.
- ``finalize``: the terminal node every path converges on. Deliberately a
  no-op (the answer is already finalized by whichever upstream node set it —
  ``fast_path``, ``no_mcp_fallback``, or ``tool_loop`` internally via
  ``_finalize_answer_text``/``_run_synthesis``). Present for
  structural parity with the PRD's node list and as a natural place to hang
  future terminal bookkeeping.

``BaseKBQAAgent._finalize_question()`` (MCP journal read-back + tool-summary
trace) is NOT a graph node: it must run even when an earlier node raises, and
LangGraph does not run downstream nodes after a node raises. Instead
``run_pipeline_graph`` (this module's entry point, called from
``_ask_impl``) wraps the whole graph invocation in the SAME try/except/finally
structure as ``_ask_impl`` itself — trace + re-raise on exception, always call
``_finalize_question()`` in the ``finally`` — so the invariant holds
regardless of which node fails.

Resolved ambiguity — no separate ``synthesis`` node
-----------------------------------------------------
The Phase 3 task description offers a choice: keep ``run_tool_loop_graph``'s
existing behaviour (it already decides the synthesis bypass) or fold that
decision out into a dedicated pipeline-level ``synthesis`` node so the
Phase 1/2 subgraph "stops calling ``_run_synthesis`` itself". This module
takes the first option: ``tool_loop`` calls ``run_tool_loop_graph`` UNCHANGED.
Rationale: ``run_tool_loop_graph`` is exercised directly (bypassing
``_ask_impl``/this pipeline entirely) by every Phase 1/2 test in
``tests/graph/test_parity_phase2.py`` and
``tests/graph/test_legacy_graph_parity.py`` via
``agent._run_tool_loop(...)`` under ``AMA_AGENT_ENGINE=graph`` — moving the
synthesis decision out of it would require either duplicating that decision
in two places (this module AND ``runner.py``, an easy place for the two to
drift) or rewriting all of those tests' call surface. Reusing it here instead
keeps Phase 1/2 behaviour and tests completely unchanged and gives the
pipeline graph the answer it needs in one call. The cost is that "synthesis"
is not literally its own ``StateGraph`` node — it is a sub-step inside
``run_tool_loop_graph``, reached through the ``tool_loop`` node — rather than
a separate node in this module's topology. Behaviourally, both options are
identical.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

COLOR_GREEN = "\033[92m"
COLOR_RED = "\033[91m"
COLOR_YELLOW = "\033[93m"
COLOR_CYAN = "\033[96m"
COLOR_END = "\033[0m"

# Mirrors _ask_impl's fast-path eligibility qtype set (~:1080).
_FAST_PATH_TYPES = {"QueryAttr", "QueryRelation", "QueryName"}


class PipelineState(TypedDict, total=False):
    """State threaded through the Phase 3 outer pipeline graph.

    Deliberately does NOT carry ``messages`` — those live only on
    ``agent._messages`` throughout, exactly as in legacy ``_ask_impl``. Every
    field here corresponds to a local variable in ``_ask_impl``'s body.
    """

    query: str
    no_mcp: bool
    is_followup: bool
    openai_tools: List[Dict[str, Any]]
    max_iterations: int
    refresh_interval: int
    qtype: str
    entities: List[str]
    relations: List[str]
    fewshot_examples: str
    answer: Optional[str]


def _filter_tools_for_qtype(agent: Any, qtype: str, openai_tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Mirrors ``_ask_impl``'s tool filtering + denylist gate (~:1110-1130)."""
    allowed_tools = agent._get_allowed_tools_for_qtype(qtype)
    if allowed_tools is not None:
        filtered = [t for t in openai_tools if t["function"]["name"] in allowed_tools]
        agent._trace(
            f"Tool filtering: {len(openai_tools)} → {len(filtered)} tools for {qtype}",
            COLOR_GREEN,
        )
        openai_tools = filtered

    denied = agent._get_denied_tool_names()
    if denied:
        before = len(openai_tools)
        openai_tools = [t for t in openai_tools if t["function"]["name"] not in denied]
        agent._trace(
            f"Tool denylist: removed {sorted(denied)} ({before} → {len(openai_tools)} tools)",
            COLOR_YELLOW,
        )
    return openai_tools


def build_pipeline_graph(agent: Any):
    """Compile the Phase 3 outer pipeline graph, bound to ``agent``.

    Same closure-over-``agent`` convention as
    ``ama_kbqa.graph.builder.build_graph``: nodes call back into the owning
    ``BaseKBQAAgent`` instance for everything (MCP, config, tracing,
    classification, prompt assembly, fast path, and — via
    ``agent._run_tool_loop`` (which itself dispatches to
    ``ama_kbqa.graph.runner.run_tool_loop_graph`` under the graph engine) —
    the tool loop itself).
    """

    async def prepare(state: PipelineState) -> Dict[str, Any]:
        query = state["query"]
        await agent._init_mcp()

        if not agent.mcp:
            agent._trace(f"{COLOR_YELLOW}Tool server not available.{COLOR_END}", COLOR_YELLOW)
            agent._messages.append({"role": "user", "content": query})
            return {"no_mcp": True}

        mcp_tools = await agent.mcp.list_tools()
        openai_tools = agent.mcp.convert_tools_to_openai_format(mcp_tools)
        agent._trace(f"Found {len(openai_tools)} tools.")

        agent._known_tool_names = {t.name for t in mcp_tools}

        # Text-mode catalog injection: dead in practice (this graph is never
        # entered for a text-mode agent — see _ask_impl's engine dispatch),
        # kept verbatim for structural parity with legacy.
        if agent._text_tool_call_mode and not getattr(agent, "_catalog_injected", False):
            from ama_kbqa.framework.text_tool_calls import build_text_mode_tool_catalog

            catalog = build_text_mode_tool_catalog(openai_tools)
            agent._messages.append({"role": "system", "content": catalog})
            agent._catalog_injected = True
            agent._trace("Injected text-mode tool catalog", COLOR_CYAN)

        config = agent.get_config()
        agent._find_resource_cap = config.domain_settings.get("find_resource_cap", 8)
        agent._sparql_cap = config.domain_settings.get("sparql_cap", 10)
        agent._context_limit = config.domain_settings.get("context_limit", 100000)
        agent._max_tool_calls = config.domain_settings.get("max_tool_calls", 0)

        is_followup = any(m.get("role") == "user" for m in agent._messages)

        max_iterations = config.domain_settings.get("max_iterations", 50)
        refresh_interval = config.domain_settings.get("journal_refresh_interval", 5)

        return {
            "no_mcp": False,
            "openai_tools": openai_tools,
            "is_followup": is_followup,
            "max_iterations": max_iterations,
            "refresh_interval": refresh_interval,
        }

    async def no_mcp_fallback(state: PipelineState) -> Dict[str, Any]:
        # Mirrors _ask_impl's `return self._llm_call_text_only()` (~:982).
        return {"answer": agent._llm_call_text_only()}

    async def classify(state: PipelineState) -> Dict[str, Any]:
        query = state["query"]
        agent._trace("Starting pre-agent classification hook", COLOR_CYAN)

        qtype_data = agent._classify_question(query)
        qtype = qtype_data.get("question_type", "Query")
        fewshot_examples = qtype_data.get("fewshot_examples", "")
        entities = qtype_data.get("entities", [])
        relations = qtype_data.get("relations", [])

        agent._trace(
            f"Classification: {qtype} | Entities: {len(entities)}, Relations: {len(relations)}",
            COLOR_GREEN,
        )
        return {
            "qtype": qtype,
            "entities": entities,
            "relations": relations,
            "fewshot_examples": fewshot_examples,
        }

    async def assemble_prompt(state: PipelineState) -> Dict[str, Any]:
        query = state["query"]

        if state.get("is_followup"):
            agent._trace(
                "Follow-up turn: skipping classification/fast-path, "
                "running full loop with all tools",
                COLOR_CYAN,
            )
            agent._messages.append({"role": "user", "content": query})
            return {"qtype": "Query"}

        qtype = state["qtype"]
        fewshot_examples = state.get("fewshot_examples", "")
        entities = state.get("entities", [])
        relations = state.get("relations", [])

        static_context = agent._build_static_qtype_context(qtype, fewshot_examples)
        if static_context:
            agent._messages.append({"role": "user", "content": static_context})

        agent._messages.append({"role": "user", "content": query})
        question_context = agent._build_question_context(qtype, entities, relations, query=query)
        agent._messages.append({"role": "user", "content": question_context})

        agent._trace(f"Pre-agent hook complete - Type: {qtype}", COLOR_GREEN)
        return {}

    async def fast_path(state: PipelineState) -> Dict[str, Any]:
        query = state["query"]
        qtype = state["qtype"]
        entities = state.get("entities", [])
        relations = state.get("relations", [])
        openai_tools = state["openai_tools"]

        config = agent.get_config()
        eligible = (
            qtype in _FAST_PATH_TYPES
            and len(entities) == 1
            and len(relations) <= 1
            and not agent._should_skip_fast_path(query, qtype, entities, relations)
            and config.domain_settings.get("enable_fast_path", True)
        )

        if eligible:
            agent._trace(f"FAST PATH: Simple {qtype} with 1 entity", COLOR_GREEN)
            async with agent.recorder.span(
                "fast_path",
                qtype,
                attributes={
                    "qtype": qtype,
                    "entity": entities[0] if entities else None,
                    "relation": relations[0] if relations else None,
                },
            ) as _fp_span:
                fast_answer = await agent._try_fast_path(query, qtype, entities, relations)
                _fp_span.set_attribute("succeeded", fast_answer is not None)

            if fast_answer is not None:
                agent._trace(f"Fast path succeeded ({len(fast_answer)} chars)", COLOR_GREEN)
                final = agent._finalize_answer_text(fast_answer, qtype)
                agent._messages.append({"role": "assistant", "content": final})
                return {"answer": final}

            agent._trace("Fast path failed - falling back to full loop", COLOR_YELLOW)

        return {"openai_tools": _filter_tools_for_qtype(agent, qtype, openai_tools)}

    async def tool_loop(state: PipelineState) -> Dict[str, Any]:
        # Calls `agent._run_tool_loop(...)` — NOT `run_tool_loop_graph`
        # directly — for two reasons: (1) `_run_tool_loop` already contains
        # the engine dispatch (checks `get_agent_engine() == "graph"` and
        # calls `run_tool_loop_graph` itself — see base_agent.py ~:1397), so
        # calling through it reproduces identical behaviour without
        # duplicating that dispatch here; (2) it is the method every existing
        # test double (`_HookAgent`, `_ParityAgent`, and this module's own
        # tests) overrides to script the tool loop, so keeping this call
        # through the same seam keeps that convention working for the
        # pipeline graph too.
        answer = await agent._run_tool_loop(
            state["query"],
            state["openai_tools"],
            state["max_iterations"],
            state["refresh_interval"],
            qtype=state["qtype"],
        )
        return {"answer": answer}

    async def finalize(state: PipelineState) -> Dict[str, Any]:
        # No-op terminal node — see the module docstring's "Resolved
        # ambiguity" section for why there is nothing left to do here.
        return {}

    def route_after_prepare(state: PipelineState) -> str:
        if state.get("no_mcp"):
            return "no_mcp"
        return "followup" if state.get("is_followup") else "classify"

    def route_after_assemble(state: PipelineState) -> str:
        return "tool_loop" if state.get("is_followup") else "fast_path"

    def route_after_fast_path(state: PipelineState) -> str:
        return "finalize" if state.get("answer") is not None else "tool_loop"

    builder = StateGraph(PipelineState)
    builder.add_node("prepare", prepare)
    builder.add_node("no_mcp_fallback", no_mcp_fallback)
    builder.add_node("classify", classify)
    builder.add_node("assemble_prompt", assemble_prompt)
    builder.add_node("fast_path", fast_path)
    builder.add_node("tool_loop", tool_loop)
    builder.add_node("finalize", finalize)

    builder.add_edge(START, "prepare")
    builder.add_conditional_edges(
        "prepare",
        route_after_prepare,
        {"no_mcp": "no_mcp_fallback", "followup": "assemble_prompt", "classify": "classify"},
    )
    builder.add_edge("no_mcp_fallback", "finalize")
    builder.add_edge("classify", "assemble_prompt")
    builder.add_conditional_edges(
        "assemble_prompt",
        route_after_assemble,
        {"tool_loop": "tool_loop", "fast_path": "fast_path"},
    )
    builder.add_conditional_edges(
        "fast_path", route_after_fast_path, {"finalize": "finalize", "tool_loop": "tool_loop"}
    )
    builder.add_edge("tool_loop", "finalize")
    builder.add_edge("finalize", END)
    return builder.compile()


async def run_pipeline_graph(agent: Any, query: str, _root_span: Any) -> str:
    """Entry point called from ``BaseKBQAAgent._ask_impl`` when the graph
    engine is selected (and the agent is not in text-mode).

    Mirrors ``_ask_impl``'s own try/except/finally structure exactly: on any
    exception, trace it and re-raise (``_ask_impl`` does not swallow
    exceptions into a string — that already happens one layer up in
    whichever caller invokes ``agent.ask()``); ``_finalize_question()`` always
    runs in the ``finally``, regardless of how the graph run ended. See the
    module docstring for why this wrapper — not a graph node — owns that
    guarantee.

    ``_root_span`` is accepted for signature parity with ``_ask_impl`` (which
    also receives, but does not use, the root ``agent_run`` span opened by
    ``ask()``) and is otherwise unused.
    """
    try:
        graph = build_pipeline_graph(agent)
        final_state = await graph.ainvoke({"query": query}, {"recursion_limit": 25})
        return final_state["answer"]
    except Exception as e:
        agent._trace(f"{COLOR_RED}Error in agent loop: {e}{COLOR_END}", COLOR_RED)
        raise
    finally:
        await agent._finalize_question()
