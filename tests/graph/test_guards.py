"""Unit tests for the pure helpers in ``ama_kbqa.graph.guards``.

See ``tests/graph/test_parity_phase2.py`` for the full-loop parity coverage
of the same behaviours; these tests exercise the helpers directly, in
isolation, without driving a whole tool loop.
"""

from __future__ import annotations

import asyncio

from ama_kbqa.graph import guards


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# zero_tool_call_decision
# ---------------------------------------------------------------------------


def test_zero_tool_call_decision_accepts_when_evidence_exists():
    assert guards.zero_tool_call_decision(total_tool_calls_made=3, retries_used=0, retry_max=1) == "accept"
    # Evidence trumps retry budget even if it's exhausted.
    assert guards.zero_tool_call_decision(total_tool_calls_made=1, retries_used=5, retry_max=1) == "accept"


def test_zero_tool_call_decision_retries_within_budget():
    assert guards.zero_tool_call_decision(total_tool_calls_made=0, retries_used=0, retry_max=1) == "retry"
    assert guards.zero_tool_call_decision(total_tool_calls_made=0, retries_used=2, retry_max=3) == "retry"


def test_zero_tool_call_decision_hard_stops_once_budget_exhausted():
    assert guards.zero_tool_call_decision(total_tool_calls_made=0, retries_used=1, retry_max=1) == "hard_stop"
    assert guards.zero_tool_call_decision(total_tool_calls_made=0, retries_used=0, retry_max=0) == "hard_stop"


# ---------------------------------------------------------------------------
# zero_tool_call_retry_max
# ---------------------------------------------------------------------------


def test_zero_tool_call_retry_max_zero_when_feature_disabled(monkeypatch):
    monkeypatch.setattr(guards, "get_zero_tool_call_retry", lambda: False)
    monkeypatch.setattr(guards, "get_zero_tool_call_retry_max", lambda: 5)
    assert guards.zero_tool_call_retry_max() == 0


def test_zero_tool_call_retry_max_reads_config_when_enabled(monkeypatch):
    monkeypatch.setattr(guards, "get_zero_tool_call_retry", lambda: True)
    monkeypatch.setattr(guards, "get_zero_tool_call_retry_max", lambda: 5)
    assert guards.zero_tool_call_retry_max() == 5


# ---------------------------------------------------------------------------
# max_tool_calls_reached
# ---------------------------------------------------------------------------


def test_max_tool_calls_reached():
    assert guards.max_tool_calls_reached(5, 5) is True
    assert guards.max_tool_calls_reached(6, 5) is True
    assert guards.max_tool_calls_reached(4, 5) is False


def test_max_tool_calls_disabled_when_cap_is_zero():
    assert guards.max_tool_calls_reached(1000, 0) is False


# ---------------------------------------------------------------------------
# wrap_up_nudge_message
# ---------------------------------------------------------------------------


def test_wrap_up_nudge_fires_at_15_and_every_5th_after():
    assert guards.wrap_up_nudge_message(14) is None
    msg15 = guards.wrap_up_nudge_message(15)
    assert msg15 is not None
    assert msg15["role"] == "user"
    assert "iteration 15" in msg15["content"]
    assert guards.wrap_up_nudge_message(16) is None
    assert guards.wrap_up_nudge_message(19) is None
    assert guards.wrap_up_nudge_message(20) is not None


# ---------------------------------------------------------------------------
# apply_loop_detection
# ---------------------------------------------------------------------------


class _LoopAgent:
    def __init__(self, detected: bool, reason: str = "reason"):
        self._detected = detected
        self._reason = reason
        self.handled: list[tuple[str, str]] = []

    def _detect_loops(self, name, args):
        return self._detected, self._reason

    async def _handle_loop_detected(self, name, reason):
        self.handled.append((name, reason))
        return f"INTERVENTION:{reason}"


def test_apply_loop_detection_returns_none_when_no_loop():
    agent = _LoopAgent(detected=False)
    result = _run(guards.apply_loop_detection(agent, "FindNode", {}))
    assert result is None
    assert agent.handled == []


def test_apply_loop_detection_returns_intervention_text_when_looping():
    agent = _LoopAgent(detected=True, reason="Identical call repeated 3 times")
    result = _run(guards.apply_loop_detection(agent, "FindNode", {}))
    assert result == "INTERVENTION:Identical call repeated 3 times"
    assert agent.handled == [("FindNode", "Identical call repeated 3 times")]
