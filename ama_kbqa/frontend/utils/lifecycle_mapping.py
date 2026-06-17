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
            # Final settle-down: the whole Post-Agent Hook band flashes as the
            # answer is finalized. Answer Synthesis is included so the box
            # lights once *at the end* even when the synthesis LLM step is
            # bypassed (the demo default), instead of never lighting.
            return ["post_synthesis", "post_evaluate", "post_lessons"]
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
        # The synthesis step runs its own ``llm_call`` (``mode="synthesis"``)
        # *inside* the wrapping ``synthesis`` span. Mapping it to the main-loop
        # reasoning box would light ``main_llm_reason`` simultaneously with
        # ``post_synthesis`` — making it look like a stray loop step fired
        # together with answer synthesis. Keep it in the Post-Agent band.
        if (attributes or {}).get("mode") == "synthesis":
            return ["post_synthesis"]
        return ["main_llm_reason"]

    if kind == "tool_call":
        # Journal/scratchpad tools are scratchpad operations end to end — light
        # the Scratchpad box for both open and close.
        if name in ("GetJournalStateJSON", "GetJournalSummary", "ManageJournal"):
            return ["main_scratchpad"]
        # Every other (KG) tool: light Tool Call while it runs, then Scratchpad
        # as its result is recorded into the journal. The agent's working memory
        # is updated server-side as a side effect of each KG tool, with no
        # client-side span of its own, so the tool-call *close* is the only
        # reliable per-iteration scratchpad signal. This also gives the ReAct
        # "Tool Call → Scratchpad" beat after every tool call.
        if phase == "close":
            return ["main_scratchpad"]
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
        # A deterministic shortcut that runs tool calls without the LLM loop.
        # Light Reasoning while it runs, but do NOT light Answer Synthesis on
        # close: the shortcut frequently fails and falls back to the full loop,
        # so a synthesis flash mid-run is misleading ("synthesis randomly lights
        # up"). The inner tool_call spans drive Tool Call/Scratchpad, and
        # agent_run close handles the final post-band flash.
        return ["main_llm_reason"] if phase == "open" else []

    if kind == "delegate":
        # The child sub-agent's spans drive the figure.
        return []

    return []


# ---------------------------------------------------------------------------
# Orchestrator figure
# ---------------------------------------------------------------------------

# The orchestrator dispatches to a fixed set of specialist sub-agents. Each maps
# to a sub-agent *container* node id in the orchestrator figure (see
# ``orchestrator_svg.py``). Identifiers are the sub-agent names carried on the
# ``delegate`` span's ``sub_agent`` attribute (and the sub-agent's own
# ``agent_run`` span name). ``display`` is the human label shown in the figure.
ORCH_SUBAGENTS: dict[str, tuple[str, str]] = {
    # agent_id        -> (display label, container node id)
    "kqapro_agent": ("KQAPro", "sub_kqapro"),
    "sciqa_agent":  ("SciQA",  "sub_sciqa"),
}


def orchestrator_span_to_node_ids(
    kind: str,
    name: str,
    phase: str = "close",
    attributes: Optional[dict] = None,
) -> list[str]:
    """Return *orchestrator-figure* node ids that should light for a span/event.

    Only orchestrator-LEVEL spans map here (the orchestrator's own
    ``agent_run``, its ``route`` classification, and each ``delegate``). The
    dispatched sub-agent's own internal lifecycle spans are attributed to that
    sub-agent and drive its separate detail figure instead — see
    ``lifecycle_runner``.
    """
    attributes = attributes or {}

    # Orchestrator root: the User Query box lights for the whole run; on close
    # the Answer Combination box flashes as the merged answer is returned.
    if kind == "agent_run" and name == "ORCHESTRATOR":
        if phase == "open":
            return ["orch_user"]
        if phase == "close":
            return ["orch_combine"]
        return []

    # Autonomous routing == datasource probing + async dispatch decision.
    if kind == "classify" and name == "route":
        if phase == "open":
            return ["orch_probe", "orch_dispatch"]
        return []

    # Dispatch to a specialist: light its container box while the delegate span
    # is open (the runner keeps it active until the span closes).
    if kind == "delegate":
        if phase == "open":
            sub = attributes.get("sub_agent") or ""
            mapped = ORCH_SUBAGENTS.get(sub)
            if mapped:
                return [mapped[1]]
        return []

    return []


# Phase a single agent-lifecycle (Fig. 1) node belongs to. Used to drive the
# coarse Pre/Main/Post mini-boxes inside an orchestrator sub-agent container
# from that sub-agent's detailed lifecycle state. ``agent_invocation`` (the
# sub-agent's entry) folds into "pre".
_NODE_PHASE: dict[str, str] = {
    "agent_invocation": "pre",
    "pre_classifier": "pre",
    "pre_extractor": "pre",
    "pre_strategy_inject": "pre",
    "main_llm_reason": "main",
    "main_tool_call": "main",
    "main_scratchpad": "main",
    "main_done": "main",
    "post_synthesis": "post",
    "post_evaluate": "post",
    "post_lessons": "post",
}


def lifecycle_node_phase(node_id: str) -> Optional[str]:
    """Map an agent-lifecycle node id to its phase (``pre``/``main``/``post``)."""
    return _NODE_PHASE.get(node_id)


# Edge id of the ReAct loop-back arrow (Done? → LLM Reasoning) in the figure.
# Must match the ``id=`` set on that edge in ``lifecycle_svg.py``.
LOOP_BACK_EDGE_ID = "loop_back"


def span_to_edge_ids(
    kind: str,
    name: str,
    phase: str = "close",
    attributes: Optional[dict] = None,
) -> list[str]:
    """Return lifecycle-figure *edge* ids that should light up for a span/event.

    Today only the ReAct loop-back arrow (Done? → LLM Reasoning) is
    addressable. It lights when the agent starts a *new* loop iteration — the
    ``tool_loop_iter`` event with ``iteration >= 2``. Iteration 1 is the
    initial entry into the loop (via the Strategy Injection bracket), not a
    loop-back, so it doesn't light the arrow.
    """
    attributes = attributes or {}
    if kind == "tool_loop_iter":
        try:
            iteration = int(attributes.get("iteration", 0))
        except (TypeError, ValueError):
            iteration = 0
        if iteration >= 2:
            return [LOOP_BACK_EDGE_ID]
    return []
