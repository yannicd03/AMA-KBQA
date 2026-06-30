"""Tests for the WikiKGQA Macro QALD F1 scorer and loader.

The headline guarantee is that scoring the gold answers against themselves
yields a perfect macro F1 of 1.0 across the whole training set (both SELECT and
ASK questions). The remaining cases pin down the QALD edge-case conventions.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ama_kbqa.wikikgqa.dataset import WikiKGQADataset
from ama_kbqa.wikikgqa.scoring import answer_set, macro_qald_f1, score_question

# Training file lives in the repo's data/ dir; skip gracefully if absent.
_TRAIN = Path(__file__).resolve().parents[2] / "data" / "wikikgqa" / "wikikgqa.json"


def _select(values):
    return {
        "head": {"vars": ["x"]},
        "results": {"bindings": [{"x": {"type": "uri", "value": v}} for v in values]},
    }


def test_gold_vs_gold_is_perfect():
    if not _TRAIN.exists():
        pytest.skip(f"training data not present at {_TRAIN}")
    ds = WikiKGQADataset.load(_TRAIN)
    pairs = [(q.id, q.gold_answer, q.gold_answer) for q in ds if q.has_gold]
    assert pairs, "expected gold-bearing questions"
    result = macro_qald_f1(pairs)
    assert result.macro_f1 == pytest.approx(1.0)
    assert result.macro_precision == pytest.approx(1.0)
    assert result.macro_recall == pytest.approx(1.0)


def test_both_empty_is_perfect():
    s = score_question(1, _select([]), _select([]))
    assert s.f1 == pytest.approx(1.0)


def test_one_empty_is_zero():
    assert score_question(1, _select(["A"]), _select([])).f1 == pytest.approx(0.0)
    assert score_question(1, _select([]), _select(["A"])).f1 == pytest.approx(0.0)


def test_partial_overlap():
    s = score_question(1, _select(["A", "B"]), _select(["A", "C"]))
    assert s.precision == pytest.approx(0.5)
    assert s.recall == pytest.approx(0.5)
    assert s.f1 == pytest.approx(0.5)


def test_perfect_select_overlap():
    s = score_question(1, _select(["A", "B"]), _select(["B", "A"]))
    assert s.f1 == pytest.approx(1.0)


def test_ask_match_and_mismatch():
    yes = {"head": {"vars": []}, "boolean": True}
    no = {"head": {"vars": []}, "boolean": False}
    assert score_question(1, yes, yes).f1 == pytest.approx(1.0)
    assert score_question(1, yes, no).f1 == pytest.approx(0.0)


def test_ask_gold_with_select_system_is_zero():
    yes = {"head": {"vars": []}, "boolean": True}
    assert score_question(1, yes, _select(["A"])).f1 == pytest.approx(0.0)


def test_literal_typing_distinguishes_values():
    gold = {
        "head": {"vars": ["x"]},
        "results": {"bindings": [{"x": {"type": "literal", "value": "5", "datatype": "int"}}]},
    }
    system = {
        "head": {"vars": ["x"]},
        "results": {"bindings": [{"x": {"type": "literal", "value": "5"}}]},
    }
    # Same string but different datatype -> not a match.
    assert score_question(1, gold, system).f1 == pytest.approx(0.0)


def test_answer_set_multivar_uses_tuples():
    res = {
        "head": {"vars": ["a", "b"]},
        "results": {
            "bindings": [
                {"a": {"type": "uri", "value": "X"}, "b": {"type": "uri", "value": "Y"}}
            ]
        },
    }
    assert answer_set(res) == {("X", "Y")}
