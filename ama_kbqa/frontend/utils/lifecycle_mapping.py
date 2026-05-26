"""Map TraceRecorder spans/events onto lifecycle-figure node IDs.

Pure-Python, Streamlit-free. The mapping is consumed by
``lifecycle_runner.drain_into`` to decide which boxes to light up as the
agent runs.

The figure has a single *Tool Call* box, so both tool families below light the
same node; the split is kept for documentation and possible future use.
``TOOLS_A`` is graph-traversal ("FindNode / GetAttr…"); ``TOOLS_B`` is
summary/verify ("GetSumm / Verify").
"""

from __future__ import annotations

from typing import Optional


# Graph-traversal tools (Tools A in the figure).
TOOLS_A: frozenset[str] = frozenset({
    # KQAPro / Wikidata-style server
    "FindNode",
    "BatchGetNodeLabels",
    "GetEdgeQualifiers",
    "GetQualifierValue",
    "GetQualifiersByPredicate",
    "FindByAttribute",
    # SciQA / ORKG server
    "FindResource",
    "FindPredicate",
    "GetResourceDetails",
    "ExploreNeighborhood",
    "FindEntitiesByRelationPath",
})

# Summary / verification tools (Tools B in the figure).
TOOLS_B: frozenset[str] = frozenset({
    "GetNodeSummary",
    "VerifyFact",
    "GetJournalSummary",
    "ManageJournal",
})


def span_to_node_ids(
    kind: str,
    name: str,
    phase: str = "close",
    attributes: Optional[dict] = None,
) -> list[str]:
    """Return lifecycle-figure node ids that should light up for a span/event.

    ``phase`` is one of ``"open"``, ``"close"``, ``"event"``. The same span
    can light different nodes on open vs close (e.g. ``classify`` lights the
    classifier/extractor on open and the strategy-injection box on close).
    """
    if kind == "agent_run":
        if phase == "open":
            return ["agent_invocation"]
        if phase == "close":
            # Final settle-down: the Post-Agent Hook boxes get a brief flash.
            return ["post_evaluate", "post_lessons"]
        return []

    if kind == "classify":
        if phase == "open":
            return ["pre_classifier", "pre_extractor"]
        if phase == "close":
            return ["pre_strategy_inject"]
        return []

    if kind == "synthesis":
        return ["post_synthesis"]

    if kind == "llm_call":
        return ["main_llm_reason"]

    if kind == "tool_call":
        if name == "GetJournalStateJSON":
            return ["main_scratchpad"]
        # All KG tools (traversal + summary/verify) light the single Tool Call
        # box; the scratchpad reflects journal state separately.
        if name in TOOLS_A or name in TOOLS_B:
            return ["main_tool_call"]
        # Unknown tool: still light the Tool Call box so the viz never goes
        # blank during a tool call.
        return ["main_tool_call"]

    if kind == "tool_loop_iter":
        return ["main_llm_reason"]

    if kind == "loop_detected":
        return ["main_done"]

    if kind in ("intervention", "context_trim"):
        return ["main_done"]

    if kind == "journal_refresh":
        return ["main_scratchpad"]

    if kind == "fast_path":
        # Fast-path bypasses the loop and goes straight to synthesis.
        return ["main_llm_reason"] if phase == "open" else ["post_synthesis"]

    if kind == "delegate":
        # The child sub-agent's spans drive the figure.
        return []

    return []
