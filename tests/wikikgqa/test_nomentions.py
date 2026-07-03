"""Tests for the without-mentions split loader/scorer (nomentions.py).

The without-mentions gold format is a bare list of values, not W3C SPARQL-JSON,
so gold/system normalization and the macro F1 empty-set conventions have to be
pinned down separately from the QALD scorer (see test_scoring.py).
"""

from __future__ import annotations

import json

import pytest

from ama_kbqa.wikikgqa.nomentions import (
    Score,
    gold_answer_values,
    load_nomentions,
    macro_f1,
    score_one,
    system_answer_values,
)


def _select(*cells, var="x"):
    return {
        "head": {"vars": [var]},
        "results": {"bindings": [{var: c} for c in cells]},
    }


def _uri(value):
    return {"type": "uri", "value": value}


def _literal(value):
    return {"type": "literal", "value": value}


# --- gold_answer_values ---


def test_gold_answer_values_strips_entity_and_property_uris():
    gold = [
        "http://www.wikidata.org/entity/Q42",
        "http://www.wikidata.org/prop/direct/P31",
    ]
    assert gold_answer_values(gold) == {"Q42", "P31"}


def test_gold_answer_values_lowercases_booleans():
    assert gold_answer_values(["True", "False"]) == {"true", "false"}


def test_gold_answer_values_keeps_bare_literals_unchanged():
    assert gold_answer_values(["3", "hello world"]) == {"3", "hello world"}


def test_gold_answer_values_strips_generic_uris_to_last_segment():
    assert gold_answer_values(["https://example.org/some/path/Q99"]) == {"Q99"}


# --- system_answer_values ---


def test_system_answer_values_boolean_true():
    assert system_answer_values({"boolean": True}) == {"true"}


def test_system_answer_values_boolean_false():
    assert system_answer_values({"boolean": False}) == {"false"}


def test_system_answer_values_select_strips_entity_uris():
    result = _select(_uri("http://www.wikidata.org/entity/Q1"), _uri("http://www.wikidata.org/entity/Q2"))
    assert system_answer_values(result) == {"Q1", "Q2"}


def test_system_answer_values_select_keeps_literal_values():
    result = _select(_literal("3"), _literal("hello"))
    assert system_answer_values(result) == {"3", "hello"}


def test_system_answer_values_empty_bindings_is_empty_set():
    assert system_answer_values(_select()) == set()


def test_system_answer_values_none_or_non_dict_is_empty_set():
    assert system_answer_values(None) == set()
    assert system_answer_values("not a dict") == set()


def test_system_answer_values_falls_back_to_first_binding_key_without_head():
    # No head.vars: derive the projected variable from the first row's keys.
    result = {"results": {"bindings": [{"y": _uri("http://www.wikidata.org/entity/Q7")}]}}
    assert system_answer_values(result) == {"Q7"}


# --- score_one (per-question macro F1 empty-set conventions) ---


def test_score_one_both_empty_is_perfect():
    s = score_one(1, [], {"head": {"vars": ["x"]}, "results": {"bindings": []}})
    assert (s.precision, s.recall, s.f1) == (1.0, 1.0, 1.0)
    assert s.n_gold == 0 and s.n_sys == 0


def test_score_one_gold_nonempty_system_empty_is_zero():
    s = score_one(1, ["Q1"], {"head": {"vars": ["x"]}, "results": {"bindings": []}})
    assert (s.precision, s.recall, s.f1) == (0.0, 0.0, 0.0)


def test_score_one_gold_empty_system_nonempty_is_zero():
    result = _select(_uri("http://www.wikidata.org/entity/Q1"))
    s = score_one(1, [], result)
    assert (s.precision, s.recall, s.f1) == (0.0, 0.0, 0.0)


def test_score_one_perfect_match():
    result = _select(_uri("http://www.wikidata.org/entity/Q1"), _uri("http://www.wikidata.org/entity/Q2"))
    s = score_one(1, ["Q1", "Q2"], result)
    assert (s.precision, s.recall, s.f1) == (1.0, 1.0, 1.0)


def test_score_one_partial_overlap():
    result = _select(_uri("http://www.wikidata.org/entity/Q1"), _uri("http://www.wikidata.org/entity/Q3"))
    s = score_one(1, ["Q1", "Q2"], result)
    assert s.precision == pytest.approx(0.5)
    assert s.recall == pytest.approx(0.5)
    assert s.f1 == pytest.approx(0.5)


def test_score_one_boolean_gold_and_system_match():
    s = score_one(1, ["true"], {"boolean": True})
    assert (s.precision, s.recall, s.f1) == (1.0, 1.0, 1.0)


def test_score_one_boolean_gold_and_system_mismatch():
    s = score_one(1, ["true"], {"boolean": False})
    assert s.f1 == pytest.approx(0.0)


# --- macro_f1 ---


def test_macro_f1_averages_over_scores():
    scores = [
        Score(qid=1, precision=1.0, recall=1.0, f1=1.0, n_gold=1, n_sys=1),
        Score(qid=2, precision=0.0, recall=0.0, f1=0.0, n_gold=1, n_sys=0),
    ]
    m = macro_f1(scores)
    assert m.n == 2
    assert m.macro_precision == pytest.approx(0.5)
    assert m.macro_recall == pytest.approx(0.5)
    assert m.macro_f1 == pytest.approx(0.5)
    assert m.per_question == scores


def test_macro_f1_empty_scores_does_not_divide_by_zero():
    m = macro_f1([])
    assert m.n == 0
    assert m.macro_precision == 0.0
    assert m.macro_recall == 0.0
    assert m.macro_f1 == 0.0


# --- load_nomentions ---


def test_load_nomentions_parses_questions_and_dict_wrapper(tmp_path):
    data = {
        "questions": [
            {
                "id": 5,
                "question": [{"string": "Who is X?", "language": "en"}, {"string": "Quien es X?", "language": "es"}],
                "sparql": "SELECT ?x WHERE {}",
                "answers": ["Q1", "Q2"],
            }
        ]
    }
    path = tmp_path / "nomentions.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    qs = load_nomentions(path)
    assert len(qs) == 1
    q = qs[0]
    assert q.id == 5
    assert q.question("en") == "Who is X?"
    assert q.question("es") == "Quien es X?"
    assert q.gold_sparql == "SELECT ?x WHERE {}"
    assert q.gold_answers == ["Q1", "Q2"]
    assert q.mentions_for("en") == []
    assert q.has_gold is True


def test_load_nomentions_accepts_bare_list_wrapper(tmp_path):
    data = [
        {
            "id": 1,
            "question": [{"string": "Q?", "language": "en"}],
            "answers": ["true"],
        }
    ]
    path = tmp_path / "nomentions_list.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    qs = load_nomentions(path)
    assert len(qs) == 1
    assert qs[0].gold_answers == ["true"]
    assert qs[0].gold_sparql == ""  # missing sparql key defaults to ""
