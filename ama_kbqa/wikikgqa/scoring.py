"""Macro QALD F1 scoring over SPARQL-JSON answer sets.

QALD evaluates a system answer against the gold answer set per question, then
macro-averages the per-question F1. This mirrors the GERBIL-QA / QALD definition:

* Gold and system answers are W3C SPARQL-JSON results (``head.vars`` +
  ``results.bindings``) for SELECT queries, or ``{"boolean": true/false}`` for ASK.
* For SELECT, the answer set is the set of binding *values* (one column in
  WikiKGQA; multi-column falls back to per-row value-tuples).
* precision = |sys ∩ gold| / |sys|, recall = |sys ∩ gold| / |gold|,
  F1 = 2·P·R / (P + R).
* Empty-set conventions: both empty -> F1 = 1.0; exactly one empty -> F1 = 0.0.
* For ASK, F1 = 1.0 iff the booleans match, else 0.0.

NOTE: this is our reference implementation of the metric for local development.
It must be cross-checked against the official Codabench scorer before trusting
absolute numbers; see the WikiKGQA ADR.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Hashable

# A SPARQL-JSON results object, or None when a system produced no answer at all.
SparqlJson = dict[str, Any] | None


def _binding_value_key(binding_cell: dict[str, Any]) -> Hashable:
    """Hashable identity for one binding cell.

    URIs compare on their value; literals compare on (value, datatype, lang) so
    ``"5"^^xsd:integer`` does not collide with the plain string ``"5"``.
    """
    btype = binding_cell.get("type", "")
    value = binding_cell.get("value", "")
    if btype in ("literal", "typed-literal"):
        return (value, binding_cell.get("datatype"), binding_cell.get("xml:lang"))
    return value


def answer_set(result: SparqlJson) -> set[Hashable]:
    """Extract the comparable answer set from a SPARQL-JSON SELECT result.

    Single-variable results (the WikiKGQA norm) yield a flat set of cell keys.
    Multi-variable results yield a set of per-row tuples ordered by ``head.vars``.
    """
    if not result:
        return set()
    head_vars = result.get("head", {}).get("vars", [])
    bindings = result.get("results", {}).get("bindings", [])
    if not head_vars:
        # Infer variables from the rows if head is missing.
        head_vars = sorted({k for row in bindings for k in row})

    out: set[Hashable] = set()
    if len(head_vars) == 1:
        var = head_vars[0]
        for row in bindings:
            if var in row:
                out.add(_binding_value_key(row[var]))
    else:
        for row in bindings:
            out.add(tuple(_binding_value_key(row[v]) if v in row else None for v in head_vars))
    return out


def _is_boolean(result: SparqlJson) -> bool:
    return bool(result) and "boolean" in result


@dataclass
class QuestionScore:
    qid: int
    precision: float
    recall: float
    f1: float
    kind: str  # "select" | "ask" | "mismatch"


def score_question(qid: int, gold: SparqlJson, system: SparqlJson) -> QuestionScore:
    """Precision/recall/F1 for a single question following QALD conventions."""
    gold_bool = _is_boolean(gold)
    sys_bool = _is_boolean(system)

    # ASK / boolean questions.
    if gold_bool:
        if not sys_bool:
            return QuestionScore(qid, 0.0, 0.0, 0.0, "mismatch")
        match = gold.get("boolean") == system.get("boolean")
        v = 1.0 if match else 0.0
        return QuestionScore(qid, v, v, v, "ask")

    # SELECT questions.
    gold_set = answer_set(gold)
    sys_set = answer_set(system)

    if not gold_set and not sys_set:
        return QuestionScore(qid, 1.0, 1.0, 1.0, "select")
    if not gold_set or not sys_set:
        return QuestionScore(qid, 0.0, 0.0, 0.0, "select")

    tp = len(gold_set & sys_set)
    precision = tp / len(sys_set)
    recall = tp / len(gold_set)
    f1 = 0.0 if (precision + recall) == 0 else (2 * precision * recall) / (precision + recall)
    return QuestionScore(qid, precision, recall, f1, "select")


@dataclass
class MacroScore:
    macro_precision: float
    macro_recall: float
    macro_f1: float
    n: int
    per_question: list[QuestionScore]


def macro_qald_f1(
    pairs: list[tuple[int, SparqlJson, SparqlJson]],
) -> MacroScore:
    """Macro-average P/R/F1 over ``(qid, gold, system)`` triples."""
    scores = [score_question(qid, gold, system) for qid, gold, system in pairs]
    n = len(scores) or 1
    return MacroScore(
        macro_precision=sum(s.precision for s in scores) / n,
        macro_recall=sum(s.recall for s in scores) / n,
        macro_f1=sum(s.f1 for s in scores) / n,
        n=len(scores),
        per_question=scores,
    )
