"""Graph state schema for the Phase 1 LangGraph tool loop.

Kept as a plain ``TypedDict`` (not a pydantic model) per
``langgraph-rewrite.md``: it is the natural shape for ``StateGraph`` node
return values (partial dict updates merged via each field's reducer) and
needs no validation beyond what the graph runtime already does.
"""

from __future__ import annotations

from typing import Annotated, List, Optional, Tuple, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class GraphState(TypedDict):
    """State threaded through the ``call_model`` <-> ``execute_tools`` loop.

    Attributes:
        messages: LangChain message history for this run. Reduced with
            ``add_messages`` (append-only, matching the legacy loop's
            append-only ``self._messages``). Converted from/to the agent's
            canonical OpenAI-format ``self._messages`` at the graph
            boundary — see :mod:`ama_kbqa.graph.messages`.
        iteration: 1-indexed count of ``call_model`` invocations that
            actually reached the LLM (mirrors ``iteration_count`` in
            ``base_agent.py:_run_tool_loop``). Drives the ``tool_choice``
            schedule and the ``max_iterations`` cap.
        total_tool_calls: Running count of tool calls executed so far,
            seeded from any fast-path tool calls already recorded on the
            agent (mirrors ``total_tool_calls_made`` in the legacy loop).
            Not yet enforced against a cap in Phase 1 (see the package
            docstring — ``max_tool_calls`` is a Phase 2 port).
        exit_reason: ``None`` while the loop is still running (and, as a
            transient value, ``"zero_tool_retry"`` for exactly one
            ``call_model`` return — see ``builder.py:route_after_model`` —
            which self-loops back to ``call_model`` rather than ending the
            graph). Terminal values: ``"final_answer"``, ``"max_iterations"``,
            ``"max_tool_calls"`` (Phase 2 — the ``max_tool_calls`` cap, ported
            from ``base_agent.py:_run_tool_loop`` ~:1633), and
            ``"zero_tool_hard_stop"`` (Phase 2 — the zero-tool-call retry
            budget exhausted, ``base_agent.py`` ~:1596). ``ama_kbqa.graph.runner``
            uses this to decide how to route into synthesis (or, for
            ``zero_tool_hard_stop``, to return the legacy literal error string
            without synthesis at all), replicating the legacy exit paths.
        tool_call_history: ``(tool_name, canonical_json_args)`` pairs
            appended by ``execute_tools`` for every call, in emission order.
            Currently unused by the graph itself (loop detection is reused
            directly from ``BaseKBQAAgent._detect_loops``, which keeps its
            own agent-level history — see ``ama_kbqa.graph.guards``); kept in
            state as a parity aid / for future graph-native loop detection.
        zero_tool_call_retries: Count of zero-tool-call retries used so far
            this run (mirrors the legacy loop's local ``zero_tool_call_retries``
            counter). See ``ama_kbqa.graph.guards.zero_tool_call_decision``.
        final_content: The most recent no-tool-call assistant message's
            ``content``, captured by ``call_model`` regardless of whether that
            message was actually appended to ``messages`` (legacy only
            appends it when non-empty — ``base_agent.py`` ~:1596-1599, item 9
            in the Phase 2 PRD). ``ama_kbqa.graph.runner`` reads this directly
            instead of re-deriving it by scanning ``messages``, so the
            "synthesis bypass" decision matches legacy exactly even when the
            triggering message was never added to history.
    """

    messages: Annotated[List[AnyMessage], add_messages]
    iteration: int
    total_tool_calls: int
    exit_reason: Optional[str]
    tool_call_history: List[Tuple[str, str]]
    zero_tool_call_retries: int
    final_content: Optional[str]
