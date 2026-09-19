"""Phase 3b: the orchestrator's routing/delegation topology as an explicit
``StateGraph``.

``Orchestrator.ask()`` dispatches here when
``ama_kbqa.config.get_agent_engine() == "graph"``; the legacy body (built
around ``Orchestrator._route_autonomously``/``_delegate``/``_fallback_kqapro``)
stays completely unchanged and is still the default. This graph models Router
mode only: a Federated orchestrator (``[federation].enabled`` or
``Orchestrator(federation=True)``) always takes the legacy body, and
``Orchestrator.ask()`` enforces that before dispatching here.

Node list (see ``.agent/Tasks/active/langgraph-rewrite.md`` §4.2/Phase 3):

- ``probe``: the deterministic MCP ``analyze_query_recommend_db`` call — no
  LLM involved. Mirrors ``Orchestrator._route_autonomously``'s "Step 1"
  (~:307-324) verbatim: a dead/absent MCP client skips straight to
  ``fallback_kqapro`` (matches legacy's early ``if not self.mcp: return
  None``); a probe exception degrades to a domain-only decision (the same
  ``degraded: true`` evidence payload legacy synthesizes) rather than
  aborting routing.
- ``select_agent``: the single forced ``select_agent`` tool-call decision —
  mirrors "Step 2" (~:326-367) verbatim. An enum-typed tool call keeps the
  decision exact (no substring matching on free text); any failure (no
  tool call, unknown agent name, a raised exception) resolves to "no
  decision", handled the same way as a fully failed probe.
- Routing is a **graph conditional edge** out of ``select_agent`` — not a
  tool returning ``Command`` — so the ``Command.PARENT`` trap the PRD warns
  about (a routing tool call that would resolve *inside* a delegate
  sub-agent's own subgraph instead of the orchestrator's) cannot arise by
  construction: nothing here is a tool call at all.
- ``delegate_kqapro`` / ``delegate_sciqa``: call
  ``Orchestrator._delegate("kqapro_agent"/"sciqa_agent", query)`` UNCHANGED —
  that method already does everything the task requires (shares
  ``self.recorder``, sets ``_parent_span_id_override`` so the sub-agent's own
  ``agent_run`` span nests under this run's ``delegate`` span, hoists journal
  snapshots and token usage, and falls back to ``fallback_kqapro`` itself on
  a load/ask failure). Reusing it directly means zero duplicated delegation
  logic between engines.
- ``fallback_kqapro``: calls ``Orchestrator._fallback_kqapro(query)``
  UNCHANGED — reached when routing produced no decision (dead/absent MCP,
  degraded probe + a routing LLM call that still failed to commit, or an
  unknown/missing tool call).
- There is no ``fallback_llm`` node. The legacy orchestrator used to fall
  back to a knowledge-base-free LLM answer when the KQAPro agent also failed;
  that fallback was removed because it hid real failures (missing key, dead
  MCP server) behind a plausible-looking response. ``_fallback_kqapro`` now
  raises in that case, and the error propagates out of this graph exactly as
  it does out of the legacy body.

Why ``probe``/``select_agent`` duplicate ``_route_autonomously``'s body
--------------------------------------------------------------------------
Unlike ``delegate``/``fallback_kqapro`` (reused verbatim, see above),
``probe`` and ``select_agent`` are copies of ``_route_autonomously``'s two
sequential steps rather than one node calling ``_route_autonomously()``
whole. This is a deliberate exception to the "call the existing method
unchanged" reuse strategy used everywhere else in ``ama_kbqa.graph``: the PRD
explicitly asks for probing and the forced decision to be separate,
independently observable graph nodes with a real edge between them (not one
opaque node), and ``_route_autonomously`` is a single Python function with no
natural split point to call into partially. The two node bodies below are
kept byte-identical to ``_route_autonomously``'s two steps (same trace
messages, same degraded-evidence payload, same error handling) specifically
so ``tests/graph/test_orchestrator_graph.py`` can assert routing-decision and
recorder-event parity against the legacy path token-for-token; if
``_route_autonomously`` changes, both copies need the same edit — call this
out in review.

``classify`` span parity
--------------------------
Legacy wraps the ENTIRE ``_route_autonomously`` call in one
``recorder.span("classify", "route", ...)`` (``Orchestrator.ask()``
~:400-408). To keep that exact span boundary (open before the probe, close
after the decision, with ``selected_agent``/``route_reason`` attributes set
on it) while still giving ``probe``/``select_agent`` genuine node identity,
they are compiled as their own small ``StateGraph`` ("the routing subgraph")
and that subgraph is invoked from inside the span, in the outer graph's
``route`` node. ``route`` is therefore the only node in the OUTER
``OrchestratorGraph`` that does not map 1:1 to a PRD-listed node name; it is
the span-owning wrapper around the ``probe -> select_agent`` subgraph.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

COLOR_RED = "\033[91m"
COLOR_YELLOW = "\033[93m"
COLOR_CYAN = "\033[96m"
COLOR_MAGENTA = "\033[95m"
COLOR_END = "\033[0m"


class RoutingState(TypedDict, total=False):
    query: str
    no_mcp: bool
    tool_result: Optional[str]
    selected_agent: Optional[str]
    routing_reason: Optional[str]


class OrchestratorState(TypedDict, total=False):
    query: str
    selected_agent: Optional[str]
    answer: Optional[str]


def _build_routing_subgraph(agent: Any):
    """``probe -> select_agent``, a byte-identical split of
    ``Orchestrator._route_autonomously``'s two steps — see the module
    docstring for why this is copied rather than called through."""

    async def probe(state: RoutingState) -> Dict[str, Any]:
        query = state["query"]
        if not agent.mcp:
            return {"no_mcp": True, "tool_result": None}

        agent._trace(f"Probing both KGs: {agent.PROBE_TOOL_NAME}", color=COLOR_YELLOW)
        try:
            tool_result = await agent.mcp.call_tool(agent.PROBE_TOOL_NAME, {"question": query})
            agent._log_pretty("Evidence", tool_result, COLOR_MAGENTA)
        except Exception as e:
            agent._trace(
                f"{COLOR_YELLOW}Probe failed ({e}); deciding from domain alone.{COLOR_END}",
                COLOR_YELLOW,
            )
            tool_result = json.dumps(
                {
                    "semantics": {},
                    "kg_evidence": {},
                    "degraded": True,
                    "note": f"probe unavailable: {e}",
                }
            )
        return {"no_mcp": False, "tool_result": tool_result}

    async def select_agent(state: RoutingState) -> Dict[str, Any]:
        query = state["query"]
        tool_result = state["tool_result"]

        messages = [
            {"role": "system", "content": agent._routing_system_prompt()},
            {
                "role": "user",
                "content": (
                    f"Query: {query}\n\n"
                    f"Entity-linking evidence from both knowledge graphs:\n"
                    f"{tool_result}"
                ),
            },
        ]

        try:
            call_params = {
                "model": agent.model,
                "messages": messages,
                "tools": [agent._select_agent_tool()],
                "tool_choice": {"type": "function", "function": {"name": "select_agent"}},
            }
            decision = agent._create_with_retry(agent.client, call_params, label="routing")

            decision_msg = decision.choices[0].message
            if not decision_msg.tool_calls:
                agent._trace(f"{COLOR_YELLOW}LLM did not commit to an agent.{COLOR_END}", COLOR_YELLOW)
                return {"selected_agent": None}

            decision_args = json.loads(decision_msg.tool_calls[0].function.arguments)
            agent_name = decision_args.get("agent")
            reason = decision_args.get("reason", "")

            if agent_name not in agent._agent_config:
                agent._trace(f"{COLOR_YELLOW}Unknown agent '{agent_name}' selected.{COLOR_END}", COLOR_YELLOW)
                return {"selected_agent": None}

            agent.last_routing_reason = reason
            agent._trace(f"Decision: {agent_name} ({reason})", COLOR_CYAN)
            return {"selected_agent": agent_name, "routing_reason": reason}

        except Exception as e:
            agent._trace(f"{COLOR_RED}Error in routing process: {e}{COLOR_END}", COLOR_RED)
            return {"selected_agent": None}

    def route_after_probe(state: RoutingState) -> str:
        return "no_mcp" if state.get("no_mcp") else "select_agent"

    builder = StateGraph(RoutingState)
    builder.add_node("probe", probe)
    builder.add_node("select_agent", select_agent)
    builder.add_edge(START, "probe")
    builder.add_conditional_edges("probe", route_after_probe, {"no_mcp": END, "select_agent": "select_agent"})
    builder.add_edge("select_agent", END)
    return builder.compile()


def build_orchestrator_graph(agent: Any):
    """Compile the Phase 3b orchestrator graph, bound to ``agent``."""

    routing_graph = _build_routing_subgraph(agent)

    async def route(state: OrchestratorState) -> Dict[str, Any]:
        query = state["query"]
        # Mirrors _route_autonomously's own reset (~:303), done here since
        # this node now owns that call's span boundary.
        agent.last_routing_reason = None
        async with agent.recorder.span(
            "classify",
            "route",
            attributes={"model": agent.model},
        ) as _route_span:
            routing_result = await routing_graph.ainvoke({"query": query})
            selected_agent_name = routing_result.get("selected_agent")
            _route_span.set_attribute("selected_agent", selected_agent_name or "<none>")
            if agent.last_routing_reason:
                _route_span.set_attribute("route_reason", agent.last_routing_reason)
        return {"selected_agent": selected_agent_name}

    async def delegate_kqapro(state: OrchestratorState) -> Dict[str, Any]:
        agent._trace("Routing successful -> kqapro_agent", "\033[92m")
        return {"answer": await agent._delegate("kqapro_agent", state["query"])}

    async def delegate_sciqa(state: OrchestratorState) -> Dict[str, Any]:
        agent._trace("Routing successful -> sciqa_agent", "\033[92m")
        return {"answer": await agent._delegate("sciqa_agent", state["query"])}

    async def fallback_kqapro(state: OrchestratorState) -> Dict[str, Any]:
        agent._trace("Routing failed. Fallback to KQAPro agent.", COLOR_YELLOW)
        return {"answer": await agent._fallback_kqapro(state["query"])}

    def route_after_select(state: OrchestratorState) -> str:
        selected = state.get("selected_agent")
        if selected == "kqapro_agent":
            return "delegate_kqapro"
        if selected == "sciqa_agent":
            return "delegate_sciqa"
        return "fallback_kqapro"

    builder = StateGraph(OrchestratorState)
    builder.add_node("route", route)
    builder.add_node("delegate_kqapro", delegate_kqapro)
    builder.add_node("delegate_sciqa", delegate_sciqa)
    builder.add_node("fallback_kqapro", fallback_kqapro)

    builder.add_edge(START, "route")
    builder.add_conditional_edges(
        "route",
        route_after_select,
        {
            "delegate_kqapro": "delegate_kqapro",
            "delegate_sciqa": "delegate_sciqa",
            "fallback_kqapro": "fallback_kqapro",
        },
    )
    builder.add_edge("delegate_kqapro", END)
    builder.add_edge("delegate_sciqa", END)
    builder.add_edge("fallback_kqapro", END)
    return builder.compile()


async def run_orchestrator_graph(agent: Any, query: str) -> str:
    """Entry point called from ``Orchestrator.ask()`` when the graph engine
    is selected.

    Mirrors ``ask()``'s own ``_init_mcp()``/``finally: mcp.close()``
    structure exactly (~:397-424) — only the routing/delegation MIDDLE of
    ``ask()`` is replaced by the graph; the outer ``agent_run`` span is
    already open (opened by ``ask()`` itself, same as
    ``BaseKBQAAgent.ask()``/``_ask_impl``'s dispatch point).
    """
    try:
        await agent._init_mcp()
        graph = build_orchestrator_graph(agent)
        final_state = await graph.ainvoke({"query": query}, {"recursion_limit": 10})
        return final_state["answer"]
    finally:
        if agent.mcp:
            await agent.mcp.close()
            agent._trace("Orchestrator MCP server cleanly terminated")
