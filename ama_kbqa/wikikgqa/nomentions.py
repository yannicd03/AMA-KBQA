"""Loader + scorer for the without-mentions splits (en_no_mentions_train.json,
dev_ref_no_mentions.json).

These files use a DIFFERENT schema than the main QALD-JSON set:
  {"questions": [{"id", "sparql", "answers": [<bare value>, ...], "question": [...]}]}
The gold ``answers`` are bare strings — entity ids ("Q25936414"), literals ("3"),
or booleans ("true"/"false") — not W3C SPARQL-JSON. So we can't reuse the QALD
scorer directly: we normalize BOTH sides to bare comparable values and compute
macro P/R/F1 with the standard QALD empty-set conventions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_ENTITY_PREFIXES = (
    "http://www.wikidata.org/entity/",
    "http://www.wikidata.org/prop/direct/",
    "http://www.wikidata.org/prop/",
)


@dataclass
class NoMentionQuestion:
    """A without-mentions question: only the text + gold (no mentions provided)."""

    id: int
    questions: dict[str, str]           # language -> string
    gold_sparql: str
    gold_answers: list[str]             # bare values
    mentions: list = field(default_factory=list)  # always empty (kept for generator API)

    def question(self, language: str = "en") -> str:
        return self.questions.get(language) or next(iter(self.questions.values()), "")

    def mentions_for(self, language: str) -> list:
        return []

    @property
    def has_gold(self) -> bool:
        return True


def load_nomentions(path: str | Path) -> list[NoMentionQuestion]:
    """Load a without-mentions split into NoMentionQuestion objects."""
    raw = json.load(open(path, encoding="utf-8"))
    qs = raw["questions"] if isinstance(raw, dict) else raw
    out: list[NoMentionQuestion] = []
    for q in qs:
        langs = {s["language"]: s["string"] for s in q.get("question", []) if "string" in s}
        answers = [str(a) for a in (q.get("answers") or [])]
        out.append(
            NoMentionQuestion(
                id=q.get("id"),
                questions=langs,
                gold_sparql=(q.get("sparql") or ""),
                gold_answers=answers,
            )
        )
    return out


def _strip(value: str) -> str:
    """Reduce a URI to its bare id (Q.../P...); leave literals unchanged."""
    for pre in _ENTITY_PREFIXES:
        if value.startswith(pre):
            return value[len(pre):]
    if value.startswith("http://") or value.startswith("https://"):
        return value.rsplit("/", 1)[-1]
    return value


def system_answer_values(sparql_json: dict[str, Any] | None) -> set[str]:
    """Bare comparable values from our executed SPARQL-JSON result.

    Booleans -> {"true"}/{"false"}; SELECT -> the (single) projected column's values,
    URIs reduced to bare ids and literals kept as their raw value string (so a typed
    "3"^^xsd:int compares equal to the gold "3").
    """
    if not isinstance(sparql_json, dict):
        return set()
    if "boolean" in sparql_json:
        return {"true" if sparql_json["boolean"] else "false"}
    head = sparql_json.get("head", {}).get("vars", [])
    bindings = sparql_json.get("results", {}).get("bindings", [])
    if not bindings:
        return set()
    var = head[0] if head else next(iter(bindings[0]))  # single-column answers
    out: set[str] = set()
    for row in bindings:
        cell = row.get(var)
        if cell is not None:
            out.add(_strip(cell.get("value", "")))
    return out


def gold_answer_values(answers: list[str]) -> set[str]:
    """Normalize the gold bare list the same way (booleans lowercased, URIs stripped)."""
    out: set[str] = set()
    for a in answers:
        a = str(a).strip()
        if a.lower() in ("true", "false"):
            out.add(a.lower())
        else:
            out.add(_strip(a))
    return out


@dataclass
class Score:
    qid: int
    precision: float
    recall: float
    f1: float
    n_gold: int
    n_sys: int


def score_one(qid: int, gold: list[str], sparql_json: dict[str, Any] | None) -> Score:
    g = gold_answer_values(gold)
    s = system_answer_values(sparql_json)
    if not g and not s:
        return Score(qid, 1.0, 1.0, 1.0, 0, 0)
    if not g or not s:
        return Score(qid, 0.0, 0.0, 0.0, len(g), len(s))
    tp = len(g & s)
    p = tp / len(s)
    r = tp / len(g)
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return Score(qid, p, r, f1, len(g), len(s))


@dataclass
class MacroScore:
    n: int
    macro_precision: float
    macro_recall: float
    macro_f1: float
    per_question: list[Score]


def macro_f1(scores: list[Score]) -> MacroScore:
    n = len(scores) or 1
    return MacroScore(
        n=len(scores),
        macro_precision=sum(s.precision for s in scores) / n,
        macro_recall=sum(s.recall for s in scores) / n,
        macro_f1=sum(s.f1 for s in scores) / n,
        per_question=scores,
    )
