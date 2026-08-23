"""Phase 2 guard functions for the graph tool loop.

Covers: loop detection + intervention, the zero-tool-call retry/hard-stop
budget, the ``max_tool_calls`` cap, and the iteration-15+ wrap-up nudge. See
``ama_kbqa/graph/context.py`` for journal refresh, context compaction, the
``GetJournalSummary`` answer prompt, and raw-SPARQL distress — those live in
a separate module because they share one reuse strategy (the "swap
``agent._messages`` for a temporary mirror, call the legacy method
unchanged, diff the result" pattern) that the functions here don't need.

Reuse strategy
---------------
``apply_loop_detection`` calls ``BaseKBQAAgent._detect_loops`` /
``_handle_loop_detected`` UNCHANGED. Both read/mutate only agent-level
attributes (``tool_call_history``, ``tool_sequence``, ``tool_call_counts``)
and never touch ``self._messages``, so there is nothing to swap or diff —
the graph's ``execute_tools`` node can call them directly against the real
``agent`` and get byte-identical behavior for free.

The zero-tool-call retry/hard-stop, the ``max_tool_calls`` cap, and the
wrap-up nudge have no equivalent extracted method on ``BaseKBQAAgent`` — in
``base_agent.py:_run_tool_loop`` they are inline code inside the ``while
True:`` loop body (~:1496-1600, ~:1633, ~:1672). There was nothing to reuse,
so they are reimplemented here as small pure functions that mirror that
inline logic exactly, including verbatim copies of the literal strings the
legacy loop appends (cross-referenced by approximate line number below) so
the two engines produce byte-identical messages.
"""

from __future__ import annotations

from typing import Any, Dict, Literal, Optional

from ama_kbqa.config import get_zero_tool_call_retry, get_zero_tool_call_retry_max


async def apply_loop_detection(agent: Any, name: str, args: Dict[str, Any]) -> Optional[str]:
    """Run legacy loop detection for one proposed tool call.

    Returns the intervention text (``BaseKBQAAgent._handle_loop_detected``'s
    return value) if a loop was detected — the caller should use this as the
    call's synthetic result instead of executing it — else ``None``.
    """
    loop_detected, reason = agent._detect_loops(name, args)
    if not loop_detected:
        return None
    return await agent._handle_loop_detected(name, reason)


def zero_tool_call_retry_max() -> int:
    """Mirrors ``base_agent.py:_run_tool_loop``'s
    ``zero_tool_call_retry_max = get_zero_tool_call_retry_max() if get_zero_tool_call_retry() else 0``
    (~:1416)."""
    return get_zero_tool_call_retry_max() if get_zero_tool_call_retry() else 0


def zero_tool_call_decision(
    total_tool_calls_made: int, retries_used: int, retry_max: int
) -> Literal["retry", "hard_stop", "accept"]:
    """Mirrors the branch structure of ``base_agent.py:_run_tool_loop``
    (~:1539-1600) for a turn where the model returned no tool calls.

    Call this ONLY when the model's response carried no (valid or invalid)
    tool calls — mirrors the ``if not message.tool_calls:`` guard the legacy
    branch lives inside.

    Returns:
        ``"accept"`` if evidence already exists (``total_tool_calls_made >
        0``) — the normal "break to synthesis" case, unaffected by the
        zero-tool-call machinery.
        ``"retry"`` if there is no evidence yet but retry budget remains —
        the assistant turn must be DISCARDED (never appended) and
        :data:`ZERO_TOOL_RETRY_NUDGE` appended instead.
        ``"hard_stop"`` if there is no evidence and the retry budget is
        exhausted — the caller must return :data:`ZERO_TOOL_HARD_STOP_ANSWER`
        literally, WITHOUT running synthesis.
    """
    if total_tool_calls_made > 0:
        return "accept"
    if retries_used < retry_max:
        return "retry"
    return "hard_stop"


# Verbatim copy of the user nudge appended in base_agent.py:_run_tool_loop
# (~:1565-1578) on a zero-tool-call retry.
ZERO_TOOL_RETRY_NUDGE: Dict[str, str] = {
    "role": "user",
    "content": (
        "STOP. You produced a final answer without calling any tools, "
        "which violates RULE 0 (MANDATORY TOOL USE). The knowledge graph "
        "almost certainly has the answer; you have not yet looked. "
        "Discard your previous response. "
        "Now: pick one entity from the question and call FindNode "
        "(semantic name) or FindByAttribute (exact code/URL/ID/ISNI). "
        "Then proceed with normal lookup. Do NOT answer in prose until "
        "you have queried the KG."
    ),
}

# Verbatim copy of the literal returned by base_agent.py:_run_tool_loop
# (~:1594) on the zero-tool-call hard stop — no synthesis is run.
ZERO_TOOL_HARD_STOP_ANSWER = "Error: Agent attempted final answer with zero tool calls."


def max_tool_calls_reached(total_tool_calls_made: int, max_tool_calls: int) -> bool:
    """Mirrors ``base_agent.py:_run_tool_loop``'s
    ``max_tool_calls = getattr(self, "_max_tool_calls", 0) or 0; if max_tool_calls and total_tool_calls_made >= max_tool_calls``
    (~:1633)."""
    return bool(max_tool_calls) and total_tool_calls_made >= max_tool_calls


def wrap_up_nudge_message(iteration: int) -> Optional[Dict[str, str]]:
    """Mirrors ``base_agent.py:_run_tool_loop``'s wrap-up nudge (~:1672-1679):
    fires at iteration >= 15 and every 5th iteration after that."""
    if iteration >= 15 and iteration % 5 == 0:
        return {
            "role": "user",
            "content": (
                f"You are on iteration {iteration}. If you have found relevant data, "
                "call GetJournalSummary and provide your answer now. "
                "Only continue if you have a concrete next step that will yield new information."
            ),
        }
    return None
