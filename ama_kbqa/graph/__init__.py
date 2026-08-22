"""Phase 1 LangGraph tool-loop engine for AMA-KBQA.

See ``.agent/Tasks/active/langgraph-rewrite.md`` for the full PRD. This
package is the "graph" engine selected by ``[agent].engine = "graph"`` in
config.toml (or the ``AMA_AGENT_ENGINE`` env var); the default is
``"legacy"``, which keeps running the hand-written ``while`` loop in
``ama_kbqa.framework.base_agent.BaseKBQAAgent._run_tool_loop`` byte-for-byte
unchanged.

Scope (Phase 1 — foundation only)
----------------------------------
The graph in :mod:`ama_kbqa.graph.builder` covers ONLY the core tool loop:

    START -> call_model -(tool_calls?)-> execute_tools -> call_model -> ...
                        \\-(no tool_calls)-> END

plus the ``max_iterations`` cap and the ``tool_choice`` schedule
(``"required"`` for iteration <= 3, then ``"auto"``), matching
``base_agent.py`` around :1405 and :1446. ``ama_kbqa.graph.runner`` is the
bridge back into ``BaseKBQAAgent``: it converts ``self._messages`` to/from
LangChain messages, drives the graph to completion, and then calls the
agent's existing ``_run_synthesis`` exactly as the legacy loop does — nothing
downstream of the loop (synthesis, ``_finalize_question``) changes.

Explicitly NOT ported in Phase 1 (deliberate scope cut, not a bug)
-------------------------------------------------------------------
Every one of these is a real behavioural difference from the legacy engine
today. They are Phase 2 work:

- Loop detection (``base_agent.py:_detect_loops``, 5 layers) and its
  intervention message.
- Periodic journal refresh / no-progress template
  (``base_agent.py:_inject_journal_refresh``), and the "answer now" prompt
  auto-injected after ``GetJournalSummary`` (``get_auto_inject_journal()``).
- Context-window compaction with hysteresis (``base_agent.py:_manage_context_window``).
- Zero-tool-call retry and truncated-tool-call retry
  (``base_agent.py:_run_tool_loop`` around :1496-1584).
- Raw-SPARQL distress intervention (``base_agent.py:_maybe_inject_raw_sparql_distress``).
- The iteration-15+ wrap-up nudge.
- The ``max_tool_calls`` hard cap (``config.domain_settings["max_tool_calls"]``).
- Text-mode tool calls (``framework/text_tool_calls.py``): the graph engine
  is not used at all for text-mode agents — ``BaseKBQAAgent._run_tool_loop``
  falls back to the legacy loop and logs a warning when
  ``self._text_tool_call_mode`` is true, regardless of the configured engine.

Extension points left for Phase 2
-----------------------------------
:func:`ama_kbqa.graph.builder.build_graph` accepts two optional hook lists,
both no-ops (``None`` -> ``[]``) today:

- ``before_model_hooks: list[Callable[[GraphState, BaseKBQAAgent], None]]``
  — called at the top of the ``call_model`` node, after the max-iterations
  check and before the LLM call. Intended home for journal refresh and the
  wrap-up nudge (both currently message-stack mutations keyed off
  ``state["iteration"]``).
- ``after_model_hooks: list[Callable[[GraphState, BaseKBQAAgent, AIMessage], None]]``
  — called after the LLM call, before the conditional edge routes on
  ``exit_reason``. Intended home for loop detection, zero-tool-call retry,
  truncated-tool-call retry, and raw-SPARQL distress — all of which, in the
  legacy loop, inspect the proposed message/tool_calls against history and
  either rewrite the outcome or append an intervention message.

The state schema (:mod:`ama_kbqa.graph.state`) already carries
``tool_call_history: list[tuple[str, str]]``, appended by ``execute_tools``
on every call, specifically so a Phase 2 loop-detection hook has the same
view ``base_agent.py:_detect_loops`` has today without any state-schema
change.
"""

from __future__ import annotations
