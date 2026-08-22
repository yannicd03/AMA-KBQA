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
        exit_reason: ``None`` while the loop is still running. Set to
            ``"final_answer"`` when the model responds with no (valid or
            invalid) tool calls, or ``"max_iterations"`` when the iteration
            cap is hit before the next LLM call is made. The graph always
            ends via one of these two reasons; ``ama_kbqa.graph.runner``
            uses this to decide how to route into synthesis, replicating the
            two distinct legacy exit paths (see runner module docstring).
        tool_call_history: ``(tool_name, canonical_json_args)`` pairs
            appended by ``execute_tools`` for every call, in emission order.
            Unused by the Phase 1 graph itself; carried in state so a Phase 2
            loop-detection ``after_model`` hook can inspect the same
            call-history view ``base_agent.py:_detect_loops`` uses today
            without changing the state schema again.
    """

    messages: Annotated[List[AnyMessage], add_messages]
    iteration: int
    total_tool_calls: int
    exit_reason: Optional[str]
    tool_call_history: List[Tuple[str, str]]
