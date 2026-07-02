"""Build WikiKGQA submission files in the Codabench-required format.

Per the competition (#15358) the scoring uses ONLY ``id`` + ``answers`` (SPARQL
queries are ignored), with exact string match, and ``answers`` is a BARE array:

* entity/property ids as bare strings: ``["Q23", "P453"]`` (NOT full URIs),
* literals/numbers/dates as strings: ``["250681000000.0", "1230-01-01T00:00:00Z"]``,
* ASK results as JSON booleans: ``[true]`` / ``[false]``,
* ``[]`` when there is no answer.

The upload file must be named ``submission.json`` and zipped so the json sits at
the zip root. Output shape mirrors the official sample:
``{"questions": [{"id", "question", "sparql", "answers"}]}``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from ama_kbqa.wikikgqa.dataset import WikiKGQADataset, WikiKGQAQuestion
from ama_kbqa.wikikgqa.endpoint import SparqlResult, execute

# A SPARQL generator turns a question into a query string (or None if it can't).
SparqlGenerator = Callable[[WikiKGQAQuestion], str | None]

_EMPTY_SELECT = {"head": {"vars": []}, "results": {"bindings": []}}

_WD_URI_PREFIXES = (
    "http://www.wikidata.org/entity/",
    "http://www.wikidata.org/prop/direct/",
    "http://www.wikidata.org/prop/",
)


def _bare_id(value: str) -> str:
    """Reduce a Wikidata URI to its bare id (Q.../P...); leave literals untouched."""
    for pre in _WD_URI_PREFIXES:
        if value.startswith(pre):
            return value[len(pre):]
    return value


def to_codabench_answers(result_json: Any) -> list:
    """Convert a SPARQL-JSON result to Codabench's bare ``answers`` array.

    ASK -> [true]/[false] (JSON booleans); SELECT -> the single projected column's
    values with entity URIs reduced to bare ids and literals kept as their raw value
    string; empty/failed -> []. Deduped, order preserved (scoring is set-based).
    """
    if not isinstance(result_json, dict):
        return []
    if "boolean" in result_json:
        return [bool(result_json["boolean"])]
    head = result_json.get("head", {}).get("vars", [])
    bindings = result_json.get("results", {}).get("bindings", [])
    if not bindings:
        return []
    var = head[0] if head else next(iter(bindings[0]), None)
    out: list = []
    seen: set = set()
    for row in bindings:
        cell = row.get(var)
        if not cell:
            continue
        value = cell.get("value", "")
        if cell.get("type") == "uri":
            value = _bare_id(value)
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


@dataclass
class QuestionOutcome:
    qid: int
    sparql: str | None
    answer: dict[str, Any]
    exec_ok: bool
    error: str | None = None
    elapsed_s: float = 0.0


def run_question(
    question: WikiKGQAQuestion,
    generate: SparqlGenerator,
    endpoint: str | None = None,
    timeout: int = 120,
) -> QuestionOutcome:
    """Generate a query for one question, execute it, and capture the answer."""
    sparql = generate(question)
    if not sparql:
        return QuestionOutcome(question.id, None, dict(_EMPTY_SELECT), False, "no query generated")
    result: SparqlResult = execute(sparql, endpoint=endpoint, timeout=timeout)
    if not result.ok or result.json is None:
        return QuestionOutcome(question.id, sparql, dict(_EMPTY_SELECT), False, result.error, result.elapsed_s)
    return QuestionOutcome(question.id, sparql, result.json, True, None, result.elapsed_s)


def build_submission(
    dataset: WikiKGQADataset,
    outcomes: dict[int, QuestionOutcome],
) -> dict[str, Any]:
    """Assemble a Codabench submission: ``{"questions": [{id, question, sparql, answers}]}``.

    Only ``id`` + ``answers`` are scored; ``answers`` is the BARE array (see
    to_codabench_answers). ``sparql``/``question`` are included for readability but ignored.
    """
    out_questions: list[dict[str, Any]] = []
    for q in dataset.questions:
        raw = getattr(q, "raw", {}) or {}
        outcome = outcomes.get(q.id)
        out_questions.append(
            {
                "id": raw.get("id", q.id),
                "question": raw.get("question")
                or [{"string": s, "language": lang} for lang, s in q.questions.items()],
                "sparql": (outcome.sparql or "") if outcome is not None else "",
                "answers": to_codabench_answers(outcome.answer) if outcome is not None else [],
            }
        )
    return {"questions": out_questions}


def write_submission(submission: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(submission, fh, ensure_ascii=False, indent=2)
    return path


def generate_submission(
    dataset: WikiKGQADataset,
    generate: SparqlGenerator,
    out_path: str | Path,
    endpoint: str | None = None,
    timeout: int = 120,
    progress: Callable[[QuestionOutcome], None] | None = None,
) -> tuple[Path, list[QuestionOutcome]]:
    """End-to-end: generate + execute every question, write the submission file."""
    outcomes: dict[int, QuestionOutcome] = {}
    ordered: list[QuestionOutcome] = []
    for q in dataset.questions:
        outcome = run_question(q, generate, endpoint=endpoint, timeout=timeout)
        outcomes[q.id] = outcome
        ordered.append(outcome)
        if progress:
            progress(outcome)
    submission = build_submission(dataset, outcomes)
    written = write_submission(submission, out_path)
    return written, ordered
