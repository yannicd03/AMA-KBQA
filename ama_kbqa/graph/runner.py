"""Bridge between ``BaseKBQAAgent._run_tool_loop`` and the graph.

``BaseKBQAAgent._run_tool_loop`` dispatches here when
``ama_kbqa.config.get_agent_engine() == "graph"`` (and the agent is not in
text-mode tool-call mode — see the ``ama_kbqa.graph`` package docstring).
Everything before and after the tool loop in ``_ask_impl``/``_run_tool_loop``
(classification, fast path, synthesis, ``_finalize_question``) is completely
unchanged; this function's only job is to run the loop and hand back exactly
what ``_run_tool_loop`` would have returned.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ama_kbqa.config import get_synthesis_enabled, load_config
from ama_kbqa.graph import guards
from ama_kbqa.graph.builder import build_graph
from ama_kbqa.graph.messages import from_lc_messages, to_lc_messages
from ama_kbqa.graph.model import build_chat_model

COLOR_RED = "\033[91m"


async def run_tool_loop_graph(
    agent: Any,
    query: str,
    tools: List[Dict[str, Any]],
    max_iterations: int,
    refresh_interval: int,
    qtype: str = "",
) -> str:
    """Run the graph tool loop and return the final answer string.

    Args:
        agent: the ``BaseKBQAAgent`` instance driving this question.
        query: the original question (passed straight through to
            ``agent._run_synthesis``, same as the legacy loop).
        tools: OpenAI-format tool schemas (already qtype/denylist-filtered).
        max_iterations: iteration cap, same semantics as legacy.
        refresh_interval: iterations between periodic journal refreshes,
            same semantics as the legacy loop's ``refresh_interval`` — wired
            through to :func:`ama_kbqa.graph.builder.build_graph` (Phase 2;
            Phase 1 discarded this parameter).
        qtype: question type, forwarded to ``agent._run_synthesis`` /
            ``agent._finalize_answer_text``.

    Returns:
        The final answer string, via the same exit paths the legacy loop
        uses:

        - ``exit_reason == "max_iterations"``: unconditionally calls
          ``agent._run_synthesis`` (mirrors the early
          ``return await self._run_synthesis(...)`` in
          ``base_agent.py:_run_tool_loop`` on the max-iterations branch,
          which bypasses the ``synthesis_enabled`` short-circuit below).
        - ``exit_reason == "max_tool_calls"`` (Phase 2): same unconditional
          ``agent._run_synthesis`` call, mirroring
          ``base_agent.py:_run_tool_loop``'s ``max_tool_calls`` branch
          (~:1633-1645), which also bypasses the ``synthesis_enabled``
          short-circuit.
        - ``exit_reason == "zero_tool_hard_stop"`` (Phase 2): returns the
          literal ``guards.ZERO_TOOL_HARD_STOP_ANSWER`` WITHOUT running
          synthesis at all, mirroring ``base_agent.py:_run_tool_loop``'s
          immediate ``return "Error: ..."`` (~:1594).
        - ``exit_reason == "final_answer"``: takes the same
          ``get_synthesis_enabled()`` bypass the legacy loop takes after
          the ``while`` loop exits via ``break`` — if synthesis is disabled
          AND the model's last message had non-empty content (read from
          ``final_state["final_content"]``, captured by ``call_model``
          regardless of whether that content was actually appended to
          ``messages`` — see the Phase 2 PRD item 9), that content (cleaned
          via ``_finalize_answer_text``) is returned directly; otherwise
          falls through to ``agent._run_synthesis``.
    """
    config = load_config()
    provider = config["llm"]["chat_provider"]
    chat_model = build_chat_model(provider, purpose="chat", max_retries=0, retry=agent._retry)

    graph = build_graph(agent, chat_model, provider, tools, max_iterations, refresh_interval)

    initial_state: Dict[str, Any] = {
        "messages": to_lc_messages(agent._messages),
        "iteration": 0,
        # Seed with any fast-path tool calls already recorded, same as the
        # legacy loop's `total_tool_calls_made = sum(self.tool_call_counts.values())`.
        "total_tool_calls": sum(agent.tool_call_counts.values()),
        "exit_reason": None,
        "tool_call_history": [],
        "zero_tool_call_retries": 0,
        "final_content": None,
    }
    # Each loop pass is two graph steps (call_model, execute_tools); pad
    # generously so a legitimate max_iterations run never trips LangGraph's
    # own recursion guard before our own cap does. A zero-tool-call retry
    # self-loops call_model -> call_model (one step per retried iteration),
    # so this padding is generous for that path too.
    recursion_limit = max(2 * max_iterations + 10, 50)
    final_state = await graph.ainvoke(initial_state, {"recursion_limit": recursion_limit})

    agent._messages = from_lc_messages(final_state["messages"])

    exit_reason = final_state.get("exit_reason")
    if exit_reason == "max_iterations":
        agent._trace(f"WARNING: Reached max iterations ({max_iterations})", COLOR_RED)
        agent.recorder.event(
            "intervention",
            "max_iterations_reached",
            attributes={"iteration_count": final_state["iteration"], "max": max_iterations},
        )
        return await agent._run_synthesis(query, qtype=qtype)

    if exit_reason == "max_tool_calls":
        return await agent._run_synthesis(query, qtype=qtype)

    if exit_reason == "zero_tool_hard_stop":
        return guards.ZERO_TOOL_HARD_STOP_ANSWER

    final_agent_content: Optional[str] = final_state.get("final_content")

    if not get_synthesis_enabled():
        if final_agent_content and final_agent_content.strip():
            return agent._finalize_answer_text(final_agent_content, qtype)

    return await agent._run_synthesis(query, qtype=qtype)
