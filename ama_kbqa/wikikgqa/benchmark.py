"""WikiKGQA benchmark runner.

Loads a WikiKGQA dataset, runs a SPARQL generator over (a sample of) it,
executes each query against the endpoint, writes a valid submission file, and —
when the dataset carries gold answers (the training set) — reports Macro QALD F1.

Kept separate from the KQAPro/SciQA harness (``benchmark_agents.py``) because the
scoring semantics differ: exact answer-set F1 versus LLM-judged correctness.

CLI::

    python -m ama_kbqa.wikikgqa.benchmark \
        --data data/wikikgqa/wikikgqa.json --limit 25 --language en \
        --out-dir benchmark_results/wikikgqa

The challenge endpoint is taken from ``--endpoint`` or ``WIKIKGQA_ENDPOINT``;
it defaults to public WDQS for development.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from ama_kbqa.wikikgqa.dataset import WikiKGQADataset
from ama_kbqa.wikikgqa.generator import (
    AgentSparqlGenerator,
    Generator,
    MentionSparqlGenerator,
)
from ama_kbqa.wikikgqa.scoring import macro_qald_f1
from ama_kbqa.wikikgqa.submission import (
    QuestionOutcome,
    build_submission,
    write_submission,
)


def _macro_score(pairs):
    """Score with the right metric for the data: bare-value (no-mentions splits, where
    gold is a list of strings) vs QALD SPARQL-JSON (gold is a dict). Both return an
    object with .n/.macro_f1/.macro_precision/.macro_recall/.per_question."""
    if pairs and isinstance(pairs[0][1], list):
        from ama_kbqa.wikikgqa.nomentions import macro_f1 as _nm_macro
        from ama_kbqa.wikikgqa.nomentions import score_one

        return _nm_macro([score_one(qid, gold, sys) for qid, gold, sys in pairs])
    return macro_qald_f1(pairs)


def run_benchmark(
    dataset: WikiKGQADataset,
    generator: Generator,
    out_dir: str | Path,
    limit: int | None = None,
    verbose: bool = True,
) -> dict:
    """Run the generator over the dataset; write submission + summary; score."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    questions = dataset.questions[:limit] if limit else dataset.questions
    outcomes: dict[int, QuestionOutcome] = {}
    score_pairs: list[tuple[int, dict, dict]] = []

    sub_dataset = WikiKGQADataset(dataset.dataset_id, questions)

    def _checkpoint() -> None:
        """Persist submission + summary after each question so an interruption
        (hang, kill, crash) never loses completed work."""
        write_submission(build_submission(sub_dataset, outcomes), out_dir / "submission.json")
        if score_pairs:
            sc = _macro_score(score_pairs)
            with open(out_dir / "summary.json", "w", encoding="utf-8") as fh:
                json.dump(
                    {
                        "dataset": dataset.dataset_id,
                        "n_questions": len(questions),
                        "n_done": len(outcomes),
                        "n_exec_ok": sum(o.exec_ok for o in outcomes.values()),
                        "scored": sc.n,
                        "macro_f1": round(sc.macro_f1, 4),
                        "macro_precision": round(sc.macro_precision, 4),
                        "macro_recall": round(sc.macro_recall, 4),
                        "per_question": [asdict(s) for s in sc.per_question],
                    },
                    fh,
                    ensure_ascii=False,
                    indent=2,
                )

    for i, q in enumerate(questions, 1):
        try:
            gen = generator.generate(q)
        except Exception as exc:  # one question must never kill the whole run
            from ama_kbqa.wikikgqa.generator import GeneratedQuery

            print(f"[{i}/{len(questions)}] q{q.id} GENERATOR ERROR: {exc}", flush=True)
            gen = GeneratedQuery(qid=q.id, sparql=None, result=None)
        outcomes[q.id] = QuestionOutcome(
            qid=q.id,
            sparql=gen.sparql,
            answer=gen.answer,
            exec_ok=gen.ok,
            error=(gen.result.error if gen.result else "no result"),
            elapsed_s=(gen.result.elapsed_s if gen.result else 0.0),
        )
        if q.has_gold:
            # gold_answer (QALD dict) or gold_values (bare list) — _macro_score dispatches.
            score_pairs.append((q.id, q.gold_answer if q.gold_answer is not None else q.gold_values, gen.answer))
        if verbose:
            n_rows = len(gen.answer.get("results", {}).get("bindings", []))
            n_vars = len(gen.answer.get("head", {}).get("vars", []))
            # Gold always projects a single column; >1 var can never match (formality).
            multivar = f" [WARN: {n_vars} projected vars, gold is single-column]" if n_vars > 1 else ""
            print(
                f"[{i}/{len(questions)}] q{q.id} ok={gen.ok} "
                f"attempts={gen.attempts} rows={n_rows} :: {q.question('en')[:60]}{multivar}",
                flush=True,
            )
        _checkpoint()  # durable after every question

    # Final submission file (mirrors the input, answers filled).
    submission = build_submission(sub_dataset, outcomes)
    sub_path = write_submission(submission, out_dir / "submission.json")

    summary: dict = {
        "dataset": dataset.dataset_id,
        "n_questions": len(questions),
        "n_exec_ok": sum(o.exec_ok for o in outcomes.values()),
        "submission": str(sub_path),
    }
    if score_pairs:
        score = _macro_score(score_pairs)
        summary["scored"] = score.n
        summary["macro_f1"] = round(score.macro_f1, 4)
        summary["macro_precision"] = round(score.macro_precision, 4)
        summary["macro_recall"] = round(score.macro_recall, 4)
        summary["per_question"] = [asdict(s) for s in score.per_question]

    with open(out_dir / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    if verbose:
        print(f"\nwrote {sub_path}")
        if "macro_f1" in summary:
            print(
                f"Macro QALD F1 = {summary['macro_f1']} "
                f"(P={summary['macro_precision']} R={summary['macro_recall']}) "
                f"over {summary['scored']} scored questions"
            )
        print(f"exec ok: {summary['n_exec_ok']}/{summary['n_questions']}")
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="WikiKGQA benchmark runner")
    parser.add_argument("--data", required=True, help="path to a WikiKGQA QALD-JSON file")
    parser.add_argument("--limit", type=int, default=None, help="only run the first N questions")
    parser.add_argument("--language", default="en", choices=["en", "es"])
    parser.add_argument("--endpoint", default=None, help="SPARQL endpoint (default: WIKIKGQA_ENDPOINT env or WDQS)")
    parser.add_argument("--max-repairs", type=int, default=2)
    parser.add_argument("--provider", default=None, help="override LLM provider (e.g. kit, openrouter)")
    parser.add_argument("--model", default=None, help="override chat model id (e.g. kit.gemma4-31b-it)")
    parser.add_argument(
        "--generator", default="mention", choices=["mention", "agent"],
        help="'mention' = blind LLM-to-SPARQL baseline; 'agent' = exploring BaseKBQAAgent",
    )
    parser.add_argument(
        "--conventions", default="full", choices=["full", "minimal"],
        help="agent only: 'full' = R1-R10 modeling conventions; 'minimal' = R1-R6 (held-out A/B)",
    )
    parser.add_argument(
        "--entity-search", action="store_true",
        help="agent only: enable the SearchEntities linker (without-mentions track)",
    )
    parser.add_argument(
        "--minimal-synthesis", action="store_true",
        help="agent only: use minimal (journal-only) synthesis context instead of the full "
             "exploration transcript (for the full-vs-minimal synthesis A/B).",
    )
    parser.add_argument(
        "--tool-budget", type=int, default=20,
        help="agent only: per-question tool-call budget (default 20; raise for the "
             "linking-heavy without-mentions track, e.g. 30).",
    )
    parser.add_argument(
        "--agent-timeout", type=float, default=None,
        help="agent only: per-question wall-clock cap in seconds (default: none). A stuck "
             "question is cancelled and its best validated query recovered from the journal. "
             "Use a generous value (e.g. 600) as a safety net for unattended submission runs.",
    )
    parser.add_argument("--out-dir", default="benchmark_results/wikikgqa")
    parser.add_argument(
        "--sample", type=int, default=None,
        help="randomly sample N questions (seeded) instead of taking the first N",
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for --sample")
    args = parser.parse_args(argv)

    # Formality guard: the final answer's literal datatypes (xsd:int vs xsd:integer),
    # date serialization, and entity URIs must match the gold, which was produced on the
    # challenge QLever endpoint. Running answers against public WDQS silently breaks these
    # even for a correct query, so make the resolved endpoint visible and warn on WDQS.
    from ama_kbqa.wikikgqa.endpoint import DEFAULT_ENDPOINT, resolve_endpoint

    resolved = resolve_endpoint(args.endpoint)
    print(f"[endpoint] final answers execute against: {resolved}", flush=True)
    if resolved == DEFAULT_ENDPOINT:
        print(
            "[endpoint] WARNING: this is public WDQS (the default), NOT the challenge "
            "endpoint. Answer datatypes/date formats may not match the gold. Set "
            "WIKIKGQA_ENDPOINT (or pass --endpoint) to the challenge QLever backend for a "
            "real submission.",
            flush=True,
        )

    if args.minimal_synthesis:
        import os as _os
        _os.environ["WIKIKGQA_FULL_SYNTHESIS"] = "0"

    dataset = WikiKGQADataset.load(args.data)
    if args.sample:
        import random

        rng = random.Random(args.seed)
        chosen = rng.sample(list(dataset.questions), min(args.sample, len(dataset.questions)))
        chosen.sort(key=lambda q: q.id)  # stable id order for readable output
        dataset = WikiKGQADataset(dataset.dataset_id, chosen)
        args.limit = None  # use the whole sampled set
    if args.generator == "agent":
        generator: Generator = AgentSparqlGenerator(
            endpoint=args.endpoint,
            language=args.language,
            provider=args.provider,
            model=args.model,
            conventions=args.conventions,
            entity_search=args.entity_search,
            tool_budget=args.tool_budget,
            agent_timeout=args.agent_timeout,
        )
    else:
        generator = MentionSparqlGenerator(
            endpoint=args.endpoint,
            language=args.language,
            max_repairs=args.max_repairs,
            provider=args.provider,
            model=args.model,
        )
    run_benchmark(dataset, generator, out_dir=args.out_dir, limit=args.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
