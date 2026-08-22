"""The Phase 1 ``StateGraph`` tool loop: ``call_model`` <-> ``execute_tools``.

See the ``ama_kbqa.graph`` package docstring for scope, the deliberately
unported behaviours, and the Phase 2 extension points (``before_model_hooks``/
``after_model_hooks`` below).

``execute_tools`` deliberately reuses ``agent._execute_single_tool`` (the
per-call executor in ``ama_kbqa/framework/base_agent.py``) unchanged, rather
than reimplementing tool execution: that function already opens the
``tool_call`` trace span with the exact attributes the frontend's
``trace_render.py`` reads, updates ``tool_call_counts``/``tool_call_durations``
(what ``get_tool_call_summary()`` reports), and snapshots the journal after
``JOURNAL_MUTATING_TOOLS``. Reusing it is what makes those three things
identical between engines "for free" instead of needing a second
implementation kept in sync.
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any, Callable, Dict, List, Optional

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph import END, START, StateGraph

from ama_kbqa.graph.state import GraphState

COLOR_YELLOW = "\033[93m"

BeforeModelHook = Callable[[GraphState, Any], None]
AfterModelHook = Callable[[GraphState, Any, AIMessage], None]


def _sequence_key(tool_call: Dict[str, Any]) -> Optional[int]:
    """Best-effort ordering key for a LangChain tool-call dict.

    LangChain's OpenAI message parser splits one assistant turn's tool calls
    into two separate lists — ``AIMessage.tool_calls`` (valid JSON args) and
    ``AIMessage.invalid_tool_calls`` (malformed JSON args) — each preserving
    relative order WITHIN itself, but the original interleaving between the
    two lists is lost (see ``langchain_openai.chat_models.base``'s
    ``_convert_dict_to_message``). When every call's ``id`` ends in a run of
    digits (true for OpenAI/KIT-style ``call_<n>`` ids observed in this
    codebase), sort on that suffix to recover the true emission order; the
    fallback below keeps them well-defined but does not undo the interleave
    for models that emit non-numeric-suffixed ids.
    """
    match = re.search(r"(\d+)$", tool_call.get("id") or "")
    return int(match.group(1)) if match else None


def _ordered_tool_calls(message: AIMessage) -> List[Dict[str, Any]]:
    """Merge ``tool_calls`` and ``invalid_tool_calls`` into emission order."""
    combined: List[Dict[str, Any]] = list(message.tool_calls or []) + list(
        message.invalid_tool_calls or []
    )
    if combined and all(_sequence_key(tc) is not None for tc in combined):
        combined.sort(key=_sequence_key)
    return combined


def _track_token_usage_from_message(agent: Any, message: AIMessage) -> Optional[Dict[str, int]]:
    """Record token usage from an ``AIMessage`` into ``agent.token_usage``.

    Prefers the LangChain-standard ``usage_metadata`` (input_tokens/
    output_tokens/total_tokens); falls back to the raw OpenAI-shaped
    ``response_metadata["token_usage"]`` dict some providers only populate
    there. Mirrors ``base_agent.py:_track_token_usage``.
    """
    usage = getattr(message, "usage_metadata", None) or {}
    prompt = usage.get("input_tokens")
    completion = usage.get("output_tokens")
    total = usage.get("total_tokens")

    if prompt is None and completion is None:
        fallback = (getattr(message, "response_metadata", None) or {}).get("token_usage") or {}
        prompt = fallback.get("prompt_tokens")
        completion = fallback.get("completion_tokens")
        total = fallback.get("total_tokens")

    if prompt is None and completion is None and total is None:
        return None

    prompt = prompt or 0
    completion = completion or 0
    total = total if total is not None else prompt + completion

    agent.token_usage["prompt_tokens"] += prompt
    agent.token_usage["completion_tokens"] += completion
    agent.token_usage["total_tokens"] += total
    return {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": total}


def build_graph(
    agent: Any,
    model: Any,
    provider: str,
    tools: List[Dict[str, Any]],
    max_iterations: int,
    *,
    before_model_hooks: Optional[List[BeforeModelHook]] = None,
    after_model_hooks: Optional[List[AfterModelHook]] = None,
):
    """Compile the Phase 1 tool-loop graph.

    Args:
        agent: the owning ``BaseKBQAAgent`` instance. Nodes call back into it
            for tracing (``agent.recorder``, ``agent._trace``), tool
            execution (``agent._execute_single_tool``), tool-name validation
            (``agent._known_tool_names``), and retry (``agent._retry``, used
            for every provider except "kit" — see ``ama_kbqa.graph.model``).
        model: a LangChain ``BaseChatModel`` from
            ``ama_kbqa.graph.model.build_chat_model``.
        provider: the configured chat provider name (drives whether the node
            wraps ``ainvoke`` in ``agent._retry`` itself — "kit" already
            retries internally, see ``ama_kbqa.graph.model``).
        tools: OpenAI-format tool schemas, already qtype/denylist-filtered —
            identical to what ``base_agent.py:_run_tool_loop`` passes to
            ``_llm_call`` today.
        max_iterations: same semantics as the legacy loop's ``max_iterations``
            parameter — checked BEFORE the LLM call is made, so
            ``max_iterations=0`` exits on the very first pass without ever
            calling the model (matches ``tests/framework/test_base_agent_tool_loop.py::test_max_iterations_exits_through_synthesis_not_error_string``).
        before_model_hooks, after_model_hooks: Phase 2 extension points, both
            no-ops today. See the ``ama_kbqa.graph`` package docstring.
    """
    before_model_hooks = list(before_model_hooks or [])
    after_model_hooks = list(after_model_hooks or [])

    async def call_model(state: GraphState) -> Dict[str, Any]:
        iteration = state["iteration"] + 1

        # Iteration cap checked BEFORE the LLM call, exactly like
        # `if iteration_count > max_iterations` in base_agent.py:_run_tool_loop
        # (~:1405) — no LLM call is made on the exiting iteration.
        if iteration > max_iterations:
            return {"iteration": iteration, "exit_reason": "max_iterations"}

        # Iteration-boundary marker, same event the legacy loop emits
        # (base_agent.py:_run_tool_loop ~:1443); the frontend's lifecycle
        # view (frontend/utils/lifecycle_mapping.py) keys on it.
        agent.recorder.event(
            "tool_loop_iter",
            f"iter:{iteration}",
            attributes={"iteration": iteration, "n_messages": len(state["messages"])},
        )

        for hook in before_model_hooks:
            hook(state, agent)

        # tool_choice schedule: matches base_agent.py:_run_tool_loop (~:1446).
        tool_choice = "required" if iteration <= 3 else "auto"
        bound_model = model.bind_tools(tools, tool_choice=tool_choice) if tools else model

        async with agent.recorder.span(
            "llm_call",
            agent.model,
            attributes={
                "model": agent.model,
                "n_messages": len(state["messages"]),
                "n_tools": len(tools) if tools else 0,
                "tool_choice": tool_choice,
            },
        ) as _llm_span:
            if provider == "kit":
                # ChatKIT already wraps _agenerate in the TransientRetry
                # instance passed to build_chat_model(retry=...) — see
                # ama_kbqa.graph.model's docstring. Wrapping again here would
                # double the backoff.
                response = await bound_model.ainvoke(state["messages"])
            else:
                response = await agent._retry.arun(
                    lambda: bound_model.ainvoke(state["messages"])
                )

            usage = _track_token_usage_from_message(agent, response)
            if usage:
                _llm_span.update_attributes(usage)
            if response.tool_calls:
                _llm_span.set_payload(
                    "tool_calls",
                    [
                        {
                            "name": tc["name"],
                            "arguments": json.dumps(tc.get("args") or {}, ensure_ascii=False),
                        }
                        for tc in response.tool_calls
                    ],
                )
            _llm_span.set_payload("assistant_content", (response.content or "")[:4000])

        for hook in after_model_hooks:
            hook(state, agent, response)

        has_calls = bool(response.tool_calls) or bool(response.invalid_tool_calls)
        exit_reason = None if has_calls else "final_answer"
        return {"messages": [response], "iteration": iteration, "exit_reason": exit_reason}

    async def execute_tools(state: GraphState) -> Dict[str, Any]:
        last = state["messages"][-1]
        calls = _ordered_tool_calls(last)
        agent._trace(f"Processing {len(calls)} tool call(s)", COLOR_YELLOW)

        # Phase 1 — validate + parse SEQUENTIALLY (mirrors the "Phase 1" pass
        # in base_agent.py:_execute_tool_calls, minus loop detection — see
        # the ama_kbqa.graph package docstring for why that's deferred).
        planned: List[Dict[str, Any]] = []
        for tc in calls:
            name = tc.get("name") or "invalid_tool"
            tc_id = tc.get("id")

            if tc.get("type") == "invalid_tool_call":
                args_raw = tc.get("args")
                try:
                    json.loads(args_raw) if args_raw else {}
                    # LangChain only classifies a call as invalid when JSON
                    # parsing failed, so this branch is unreachable in
                    # practice; kept so a parse "succeeding" here still
                    # produces a well-defined (empty-args) result rather than
                    # silently executing with the raw string.
                    result, func_args = None, {}
                except json.JSONDecodeError as e:
                    # Same error text as base_agent.py:_execute_tool_calls.
                    result = (
                        f"Error: could not parse arguments as JSON ({e}). "
                        "Re-emit the call with valid JSON arguments on a single line."
                    )
                    func_args = None
                planned.append({"id": tc_id, "name": name, "result": result, "args": func_args})
                continue

            known_tools = getattr(agent, "_known_tool_names", set())
            if known_tools and name not in known_tools:
                result = (
                    f"Error: Tool '{name}' does not exist. "
                    f"Available tools: {', '.join(sorted(known_tools))}"
                )
                planned.append({"id": tc_id, "name": name, "result": result, "args": None})
                continue

            planned.append({"id": tc_id, "name": name, "result": None, "args": tc.get("args") or {}})

        # Phase 2 — execute the remaining calls CONCURRENTLY, same as
        # base_agent.py:_execute_tool_calls.
        pending = [p for p in planned if p["result"] is None]
        if len(pending) == 1:
            p = pending[0]
            p["result"] = await agent._execute_single_tool(p["name"], p["args"])
        elif pending:
            agent._trace(
                f"Executing {len(pending)} independent tool calls concurrently",
                COLOR_YELLOW,
            )
            results = await asyncio.gather(
                *(agent._execute_single_tool(p["name"], p["args"]) for p in pending)
            )
            for p, result in zip(pending, results):
                p["result"] = result

        tool_messages = [
            ToolMessage(content=p["result"], tool_call_id=p["id"], name=p["name"])
            for p in planned
        ]

        history = list(state.get("tool_call_history", []))
        for p in planned:
            args_repr = json.dumps(p["args"], sort_keys=True) if p["args"] is not None else ""
            history.append((p["name"], args_repr))

        return {
            "messages": tool_messages,
            "total_tool_calls": state["total_tool_calls"] + len(planned),
            "tool_call_history": history,
        }

    def route_after_model(state: GraphState) -> str:
        return "end" if state.get("exit_reason") is not None else "tools"

    builder = StateGraph(GraphState)
    builder.add_node("call_model", call_model)
    builder.add_node("execute_tools", execute_tools)
    builder.add_edge(START, "call_model")
    builder.add_conditional_edges(
        "call_model", route_after_model, {"tools": "execute_tools", "end": END}
    )
    builder.add_edge("execute_tools", "call_model")
    return builder.compile()
