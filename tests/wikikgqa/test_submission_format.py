"""Codabench submission-format tests: `answers` must be a BARE array (Q/P ids, literal
strings, JSON booleans), exact-string-matched. Regression guard for the format the
official sample (github.com/debayan/wikikgqa-public 2026/samples/res) requires.
"""

from __future__ import annotations

from ama_kbqa.wikikgqa.submission import to_codabench_answers


def _select(*cells):
    return {"head": {"vars": ["x"]}, "results": {"bindings": [{"x": c} for c in cells]}}


def test_ask_returns_json_boolean():
    assert to_codabench_answers({"boolean": True, "head": {}}) == [True]
    assert to_codabench_answers({"boolean": False, "head": {}}) == [False]


def test_entity_uris_reduced_to_bare_ids():
    r = _select(
        {"type": "uri", "value": "http://www.wikidata.org/entity/Q42"},
        {"type": "uri", "value": "http://www.wikidata.org/entity/Q1761"},
    )
    assert to_codabench_answers(r) == ["Q42", "Q1761"]


def test_literals_kept_as_raw_value_strings():
    r = _select({"type": "literal", "datatype": "http://www.w3.org/2001/XMLSchema#int", "value": "3"})
    assert to_codabench_answers(r) == ["3"]
    r2 = _select({"type": "literal", "datatype": "...#dateTime", "value": "1230-01-01T00:00:00Z"})
    assert to_codabench_answers(r2) == ["1230-01-01T00:00:00Z"]


def test_empty_and_nondict():
    assert to_codabench_answers({"head": {"vars": []}, "results": {"bindings": []}}) == []
    assert to_codabench_answers({}) == []
    assert to_codabench_answers(None) == []


def test_dedup_preserves_order():
    r = _select(
        {"type": "uri", "value": "http://www.wikidata.org/entity/Q1"},
        {"type": "uri", "value": "http://www.wikidata.org/entity/Q1"},
        {"type": "uri", "value": "http://www.wikidata.org/entity/Q2"},
    )
    assert to_codabench_answers(r) == ["Q1", "Q2"]


def test_build_submission_shape():
    from ama_kbqa.wikikgqa.dataset import WikiKGQADataset, WikiKGQAQuestion
    from ama_kbqa.wikikgqa.submission import QuestionOutcome, build_submission

    q = WikiKGQAQuestion(id=7, questions={"en": "who?"}, raw={"id": 7, "question": [{"string": "who?", "language": "en"}]})
    ds = WikiKGQADataset("test", [q])
    outcome = QuestionOutcome(7, "ASK {}", {"boolean": True, "head": {}}, True)
    sub = build_submission(ds, {7: outcome})
    assert list(sub.keys()) == ["questions"]  # no top-level dataset
    item = sub["questions"][0]
    assert set(item.keys()) == {"id", "question", "sparql", "answers"}
    assert item["answers"] == [True]
