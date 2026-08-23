"""Unit tests for ``ama_kbqa.graph.context``'s swap-and-diff hooks.

Uses a minimal fake agent (not a full ``BaseKBQAAgent``) whose
``_manage_context_window``/``_inject_journal_refresh``/
``_maybe_inject_raw_sparql_distress`` are simple scripted mutators, so these
tests isolate the swap-and-diff MECHANISM (id-preserving in-place replacement
vs plain append) from the legacy methods' own logic (already covered against
the real ``BaseKBQAAgent`` methods by ``tests/framework/test_prefix_cache_history.py``,
``tests/framework/test_raw_sparql_distress.py``, and the full-loop parity
scenarios in ``tests/graph/test_parity_phase2.py``).
"""

from __future__ import annotations

import asyncio

from ama_kbqa.graph import context
from ama_kbqa.graph.messages import to_lc_messages


def _run(coro):
    return asyncio.run(coro)


class _FakeContextAgent:
    def __init__(self, mutate_compaction=None, mutate_refresh=None, distress=None, journal_prompt="ANSWER NOW"):
        self._messages = []
        self._mutate_compaction = mutate_compaction or (lambda msgs: None)
        self._mutate_refresh = mutate_refresh
        self._distress = distress or (lambda msgs, iteration: None)
        self._journal_prompt = journal_prompt

    def _manage_context_window(self):
        self._mutate_compaction(self._messages)

    async def _inject_journal_refresh(self, iteration):
        if self._mutate_refresh:
            self._mutate_refresh(self._messages, iteration)

    def _maybe_inject_raw_sparql_distress(self, iteration):
        self._distress(self._messages, iteration)

    def _get_journal_summary_answer_prompt(self):
        return self._journal_prompt


# ---------------------------------------------------------------------------
# run_before_model_mutations: compaction (in-place edit, id preserved)
# ---------------------------------------------------------------------------


def test_run_before_model_mutations_preserves_id_on_in_place_edit(monkeypatch):
    monkeypatch.setattr(context, "get_auto_inject_journal", lambda: False)

    def mutate(msgs):
        msgs[1]["content"] = "trimmed"

    agent = _FakeContextAgent(mutate_compaction=mutate)
    original = to_lc_messages(
        [{"role": "system", "content": "sys"}, {"role": "user", "content": "long content"}]
    )
    state = {"messages": original}

    updates = _run(context.run_before_model_mutations(agent, state, iteration=1, refresh_interval=1000))

    assert len(updates) == 1
    assert updates[0].id == original[1].id
    assert updates[0].content == "trimmed"


def test_run_before_model_mutations_no_op_when_nothing_mutated(monkeypatch):
    monkeypatch.setattr(context, "get_auto_inject_journal", lambda: False)
    agent = _FakeContextAgent()
    original = to_lc_messages([{"role": "system", "content": "sys"}])
    state = {"messages": original}

    updates = _run(context.run_before_model_mutations(agent, state, iteration=1, refresh_interval=1000))

    assert updates == []


# ---------------------------------------------------------------------------
# run_before_model_mutations: journal refresh (pure append, fresh id)
# ---------------------------------------------------------------------------


def test_run_before_model_mutations_appends_journal_refresh_when_due(monkeypatch):
    monkeypatch.setattr(context, "get_auto_inject_journal", lambda: True)

    def refresh(msgs, iteration):
        msgs.append({"role": "user", "content": f"REFRESH {iteration}"})

    agent = _FakeContextAgent(mutate_refresh=refresh)
    original = to_lc_messages([{"role": "system", "content": "sys"}])
    state = {"messages": original}

    updates = _run(context.run_before_model_mutations(agent, state, iteration=2, refresh_interval=2))

    assert len(updates) == 1
    assert updates[0].content == "REFRESH 2"
    assert updates[0].id not in {m.id for m in original}


def test_run_before_model_mutations_skips_refresh_when_not_due(monkeypatch):
    monkeypatch.setattr(context, "get_auto_inject_journal", lambda: True)
    calls = []

    def refresh(msgs, iteration):
        calls.append(iteration)

    agent = _FakeContextAgent(mutate_refresh=refresh)
    original = to_lc_messages([{"role": "system", "content": "sys"}])
    state = {"messages": original}

    _run(context.run_before_model_mutations(agent, state, iteration=1, refresh_interval=2))

    assert calls == []


def test_run_before_model_mutations_skips_refresh_when_disabled(monkeypatch):
    monkeypatch.setattr(context, "get_auto_inject_journal", lambda: False)
    calls = []

    def refresh(msgs, iteration):
        calls.append(iteration)

    agent = _FakeContextAgent(mutate_refresh=refresh)
    original = to_lc_messages([{"role": "system", "content": "sys"}])
    state = {"messages": original}

    _run(context.run_before_model_mutations(agent, state, iteration=2, refresh_interval=2))

    assert calls == []


# ---------------------------------------------------------------------------
# run_after_tools_mutations: journal-summary prompt + raw-SPARQL distress
# ---------------------------------------------------------------------------


def test_run_after_tools_mutations_appends_prompt_then_distress_in_order(monkeypatch):
    monkeypatch.setattr(context, "get_auto_inject_journal", lambda: True)

    def distress(msgs, iteration):
        msgs.append({"role": "user", "content": "DISTRESS"})

    agent = _FakeContextAgent(distress=distress, journal_prompt="ANSWER NOW")
    base = to_lc_messages([{"role": "system", "content": "sys"}])

    updates = context.run_after_tools_mutations(
        agent, base, iteration=1, called_get_journal_summary=True
    )

    assert [m.content for m in updates] == ["ANSWER NOW", "DISTRESS"]
    assert all(m.id not in {b.id for b in base} for m in updates)


def test_run_after_tools_mutations_skips_prompt_when_journal_summary_not_called(monkeypatch):
    monkeypatch.setattr(context, "get_auto_inject_journal", lambda: True)
    agent = _FakeContextAgent(journal_prompt="ANSWER NOW")
    base = to_lc_messages([{"role": "system", "content": "sys"}])

    updates = context.run_after_tools_mutations(
        agent, base, iteration=1, called_get_journal_summary=False
    )

    assert updates == []


def test_run_after_tools_mutations_skips_prompt_when_auto_inject_disabled(monkeypatch):
    monkeypatch.setattr(context, "get_auto_inject_journal", lambda: False)
    agent = _FakeContextAgent(journal_prompt="ANSWER NOW")
    base = to_lc_messages([{"role": "system", "content": "sys"}])

    updates = context.run_after_tools_mutations(
        agent, base, iteration=1, called_get_journal_summary=True
    )

    assert updates == []
