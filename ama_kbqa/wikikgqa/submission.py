"""Build WikiKGQA submission files.

A submission is the challenge's QALD-JSON with two fields filled per question:

* ``query.sparql`` — the generated query (optional but recommended).
* ``answers`` — MANDATORY: the SPARQL-JSON results of executing that query
  against the evaluation endpoint.

The original entry structure (id, question, mentions) is preserved verbatim so
the output validates against the organisers' reader. A question whose query
fails to execute is written with an empty SELECT result rather than dropped, so
the submission stays aligned with the test set.
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
    """Assemble a submission dict mirroring the input file, with answers filled."""
    out_questions: list[dict[str, Any]] = []
    for q in dataset.questions:
        entry = json.loads(json.dumps(q.raw))  # deep copy of the original entry
        outcome = outcomes.get(q.id)
        if outcome is not None:
            entry["query"] = {"sparql": outcome.sparql or ""}
            entry["answers"] = [outcome.answer]
        out_questions.append(entry)
    return {"dataset": {"id": dataset.dataset_id}, "questions": out_questions}


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
