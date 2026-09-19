"""LangGraph tool-loop engine for AMA-KBQA.

See ``.agent/Tasks/active/langgraph-rewrite.md`` for the full PRD. This
package is the "graph" engine selected by ``[agent].engine = "graph"`` in
config.toml (or the ``AMA_AGENT_ENGINE`` env var); the default is
``"legacy"``, which keeps running the hand-written ``while`` loop in
``ama_kbqa.framework.base_agent.BaseKBQAAgent._run_tool_loop`` byte-for-byte
unchanged.

Topology
--------
The graph in :mod:`ama_kbqa.graph.builder` implements the core tool loop:

    START -> call_model -(tool_calls?)-> execute_tools -(cap reached?)-> END
                |    \\-(no tool_calls, evidence exists)-> END               |
                |    \\-(no tool_calls, zero-tool retry)-> call_model         |
                \\-(max_iterations reached)-> END                            v
                                                                       call_model
                                                                       (loop back)

plus the ``max_iterations`` cap and the ``tool_choice`` schedule
(``"required"`` for iteration <= 3, then ``"auto"``), matching
``base_agent.py`` around :1405 and :1446. ``ama_kbqa.graph.runner`` is the
bridge back into ``BaseKBQAAgent``: it converts ``self._messages`` to/from
LangChain messages, drives the graph to completion, and then calls the
agent's existing ``_run_synthesis`` exactly as the legacy loop does — nothing
downstream of the loop (synthesis, ``_finalize_question``) changes.

Phase 2 — every remaining ``_run_tool_loop`` behaviour, ported
-----------------------------------------------------------------
Every behaviour below now has parity with ``base_agent.py:_run_tool_loop``.
See :mod:`ama_kbqa.graph.guards` and :mod:`ama_kbqa.graph.context` module
docstrings for the reuse-vs-reimplementation strategy for each:

- Loop detection (``base_agent.py:_detect_loops``, 5 layers) and its
  intervention message — ``ama_kbqa.graph.guards.apply_loop_detection``,
  run in ``execute_tools`` during tool-call validation.
- Periodic journal refresh / no-progress template
  (``base_agent.py:_inject_journal_refresh``), and the "answer now" prompt
  auto-injected after ``GetJournalSummary`` (``get_auto_inject_journal()``)
  — ``ama_kbqa.graph.context``.
- Context-window compaction with hysteresis
  (``base_agent.py:_manage_context_window``) — ``ama_kbqa.graph.context``.
- Zero-tool-call retry and hard stop (``base_agent.py:_run_tool_loop``
  ~:1539-1600) — ``ama_kbqa.graph.guards.zero_tool_call_decision``, wired
  into ``call_model``'s self-loop.
- Raw-SPARQL distress intervention
  (``base_agent.py:_maybe_inject_raw_sparql_distress``) — ``ama_kbqa.graph.context``.
- The iteration-15+ wrap-up nudge — ``ama_kbqa.graph.guards.wrap_up_nudge_message``.
- The ``max_tool_calls`` hard cap (``config.domain_settings["max_tool_calls"]``)
  — ``ama_kbqa.graph.guards.max_tool_calls_reached``, exits straight to
  synthesis via a new ``exit_reason == "max_tool_calls"``.

Remaining gap (unchanged from Phase 1, out of scope per the PRD)
-------------------------------------------------------------------
- Text-mode tool calls (``framework/text_tool_calls.py``) and the
  text-mode-only truncated-tool-call retry
  (``base_agent.py:_run_tool_loop`` ~:1496-1517): the graph engine is not
  used at all for text-mode agents — ``BaseKBQAAgent._run_tool_loop`` falls
  back to the legacy loop and logs a warning when
  ``self._text_tool_call_mode`` is true, regardless of the configured
  engine.
"""

from __future__ import annotations
