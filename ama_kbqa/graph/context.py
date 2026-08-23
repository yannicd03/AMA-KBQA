"""Phase 2 context-management hooks for the graph tool loop.

Covers: periodic journal refresh (+ no-progress template), hysteresis-based
context-window compaction, the ``GetJournalSummary`` answer-prompt
injection, and the raw-SPARQL distress intervention. See
``ama_kbqa/graph/guards.py`` for loop detection, the zero-tool-call retry
budget, the ``max_tool_calls`` cap, and the wrap-up nudge — those live in a
separate module because they don't touch ``self._messages`` and so don't
need the reuse strategy below.

Reuse strategy: swap-and-diff
------------------------------
``BaseKBQAAgent._manage_context_window``, ``_inject_journal_refresh``, and
``_maybe_inject_raw_sparql_distress`` all read and mutate ``self._messages``
directly (in place for compaction's stub/truncate passes; append-only for
refresh, the journal-summary prompt, and distress). None of them can be
called against the real agent mid-graph-run, because the graph — not
``self._messages`` — owns the authoritative message list while a run is in
flight (``self._messages`` is only resynced once, at the very end, by
``ama_kbqa.graph.runner``).

Re-deriving this logic as pure ``list[dict] -> list[dict]`` functions was
considered and rejected: compaction alone encodes nontrivial, easy-to-drift
behavior (hysteresis trigger schedule, superseded-refresh stubbing keyed off
the journal-refresh marker, a two-tier aggressive/moderate truncation
policy). Instead, every hook here uses the same trick: temporarily point
``agent._messages`` at a plain OpenAI-dict mirror of the relevant slice of
graph state, call the legacy method UNCHANGED (it mutates the mirror as if
it were the real thing), then diff the mutated mirror against the original
to produce a LangGraph message-list update:

- In-place edits (compaction can rewrite an EXISTING message's content) are
  turned into a replacement LangChain message carrying the SAME ``id`` as
  the message it replaces, so ``add_messages`` overwrites it in place
  instead of duplicating it (this is why ``ama_kbqa.graph.messages.to_lc_messages``
  guarantees every message an ``id``).
- Pure appends (refresh, the journal-summary prompt, distress) become new
  LangChain messages with fresh ids, appended via the same reducer.

This keeps every one of these behaviors byte-for-byte identical to legacy
(same thresholds, same templates, same recorder events — all fire naturally
because the swapped-in call still runs against the REAL ``agent.recorder``/
``agent.mcp``/config getters, only ``agent._messages`` is temporarily
redirected) while adding zero duplicated logic.
"""

from __future__ import annotations

from typing import Any, Dict, List

from langchain_core.messages import AnyMessage

from ama_kbqa.config import get_auto_inject_journal
from ama_kbqa.graph.messages import from_lc_messages, to_lc_messages

COLOR_CYAN = "\033[96m"


def _diff_to_lc_updates(
    original_lc: List[AnyMessage], mirror_dicts: List[Dict[str, Any]]
) -> List[AnyMessage]:
    """Turn a (possibly in-place-mutated, possibly grown) dict mirror back
    into a LangGraph ``messages`` update: replacements-by-id for entries
    whose content changed in place, plus fresh messages for any appended
    tail. Never called with a mirror shorter than ``original_lc`` — none of
    the legacy methods reused here ever remove a message."""
    original_dicts = from_lc_messages(original_lc)
    updates: List[AnyMessage] = []
    for i, orig_lc in enumerate(original_lc):
        if i < len(mirror_dicts) and mirror_dicts[i] != original_dicts[i]:
            replacement = to_lc_messages([mirror_dicts[i]])[0]
            replacement.id = orig_lc.id
            updates.append(replacement)
    if len(mirror_dicts) > len(original_lc):
        updates.extend(to_lc_messages(mirror_dicts[len(original_lc) :]))
    return updates


async def run_before_model_mutations(
    agent: Any, state: Dict[str, Any], iteration: int, refresh_interval: int
) -> List[AnyMessage]:
    """``call_model``'s before-LLM-call hook.

    Reuses ``BaseKBQAAgent._manage_context_window()`` then, if due,
    ``_inject_journal_refresh(iteration)`` UNCHANGED and in the SAME order as
    ``base_agent.py:_run_tool_loop`` (~:1417-1429) — compaction first, so a
    freshly-appended refresh is never itself truncated on the same pass.
    """
    original = agent._messages
    mirror = from_lc_messages(state["messages"])
    agent._messages = mirror
    try:
        agent._manage_context_window()
        if (
            get_auto_inject_journal()
            and iteration % refresh_interval == 0
            and iteration > 0
        ):
            await agent._inject_journal_refresh(iteration)
        updated_mirror = agent._messages
    finally:
        agent._messages = original
    return _diff_to_lc_updates(state["messages"], updated_mirror)


def run_after_tools_mutations(
    agent: Any,
    base_lc_messages: List[AnyMessage],
    iteration: int,
    called_get_journal_summary: bool,
) -> List[AnyMessage]:
    """``execute_tools``'s after-execution hook.

    ``base_lc_messages`` is the message list INCLUDING this round's freshly
    produced tool-result messages (i.e. ``add_messages(state["messages"],
    tool_messages)``) — matches ``base_agent.py:_run_tool_loop``, which runs
    this logic right after ``_execute_tool_calls`` has already appended the
    tool results to ``self._messages`` (~:1638-1650). Both mutations here are
    pure appends (never in-place edits), so the diff is just the mirror's
    new tail.
    """
    original = agent._messages
    mirror = from_lc_messages(base_lc_messages)
    agent._messages = mirror
    try:
        if called_get_journal_summary and get_auto_inject_journal():
            agent._trace("GetJournalSummary called - injecting answer prompt", COLOR_CYAN)
            agent._messages.append(
                {"role": "user", "content": agent._get_journal_summary_answer_prompt()}
            )
        agent._maybe_inject_raw_sparql_distress(iteration)
        updated_mirror = agent._messages
    finally:
        agent._messages = original
    return to_lc_messages(updated_mirror[len(base_lc_messages) :])
