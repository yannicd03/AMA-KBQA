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
    to_codabench_answers,
    write_submission,
)

# Escalation raises the per-question tool budget by this much over whatever the
# main pass used (see _escalate_empty_answers). Fixed rather than a CLI knob: the
# 2026-07-13 EN-WITHOUT-MENTIONS incident (one manually-escalated empty answer,
# q3) used +15 by hand and it was enough; keep it simple until evidence says
# otherwise.
_ESCALATION_BUDGET_BUMP = 15


def _macro_score(pairs):
    """Score with the right metric for the data: bare-value (no-mentions splits, where
    gold is a list of strings) vs QALD SPARQL-JSON (gold is a dict). Both return an
    object with .n/.macro_f1/.macro_precision/.macro_recall/.per_question."""
    if pairs and isinstance(pairs[0][1], list):
        from ama_kbqa.wikikgqa.nomentions import macro_f1 as _nm_macro
        from ama_kbqa.wikikgqa.nomentions import score_one

        return _nm_macro([score_one(qid, gold, sys) for qid, gold, sys in pairs])
    return macro_qald_f1(pairs)


def _codabench_macro(pairs):
    """Leaderboard-estimate score (bare-value exact match), or None when the gold
    is already bare (no-mentions splits, where _macro_score compares bare values)."""
    if pairs and isinstance(pairs[0][1], list):
        return None
    from ama_kbqa.wikikgqa.scoring import macro_qald_f1_codabench

    return macro_qald_f1_codabench(pairs)


def _write_run_manifest(out_dir: str | Path, args, resolved_endpoint: str) -> None:
    """Persist the run's full configuration so results map to code and settings.

    Mirrors benchmark_agents._write_run_manifest: without this, a summary.json
    cannot be attributed to a model/conventions/seed/commit after the fact (the
    exact gap that made earlier failure-mode analysis unreliable).
    """
    import subprocess

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            cwd=Path(__file__).resolve().parents[2],
        ).stdout.strip() or None
    except Exception:
        commit = None
    manifest = {
        "timestamp": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
        "git_commit": commit,
        "endpoint": resolved_endpoint,
        "args": {k: v for k, v in vars(args).items()},
    }
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        with open(out_dir / "run_manifest.json", "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2, ensure_ascii=False, default=str)
    except OSError as exc:
        print(f"WARNING: could not write run_manifest.json: {exc}")


def _escalate_empty_answers(
    generator: Generator,
    questions: list,
    outcomes: dict[int, QuestionOutcome],
    score_pairs: list[tuple[int, dict, dict]],
    verbose: bool = True,
) -> dict[str, list[int]]:
    """Re-run empty-answer questions once with a raised tool budget (agent generator only).

    Every gold answer in this benchmark is non-empty, so a submission answer that
    ends up empty is known-wrong (the premise behind
    scripts/assemble_voted_submission.py's ``--patch`` escalation flow, which this
    mirrors and automates: replace ONLY empty answers, and only with a non-empty
    escalation result -- a still-empty or errored escalation never overwrites the
    original outcome, and a question that already has a non-empty answer is never
    re-run).

    No-op (returns empty lists, never touches ``generator``) unless ``generator``
    is an :class:`AgentSparqlGenerator` with at least one empty-answer question.
    Mutates ``outcomes`` and ``score_pairs`` in place for the questions that get
    filled, so callers only need to rebuild/write the submission afterwards.
    """
    result: dict[str, list[int]] = {"escalated_qids": [], "filled_qids": [], "still_empty_qids": []}
    if not isinstance(generator, AgentSparqlGenerator):
        return result

    empty_qids = [
        q.id for q in questions if q.id in outcomes and not to_codabench_answers(outcomes[q.id].answer)
    ]
    if not empty_qids:
        return result

    result["escalated_qids"] = list(empty_qids)
    qid_to_question = {q.id: q for q in questions}
    score_index = {qid: i for i, (qid, _gold, _sys) in enumerate(score_pairs)}
    original_budget = generator.tool_budget
    raised_budget = original_budget + _ESCALATION_BUDGET_BUMP
    if verbose:
        print(
            f"\n[escalate] {len(empty_qids)} empty answer(s) after the main pass: "
            f"{empty_qids} -- re-running with tool_budget={raised_budget} "
            f"(was {original_budget})",
            flush=True,
        )
    generator.tool_budget = raised_budget
    try:
        for qid in empty_qids:
            q = qid_to_question[qid]
            try:
                gen = generator.generate(q)
            except Exception as exc:  # escalation must never kill the run either
                if verbose:
                    print(f"[escalate] q{qid} GENERATOR ERROR: {exc} -- still empty", flush=True)
                result["still_empty_qids"].append(qid)
                continue
            new_answer = gen.answer
            if to_codabench_answers(new_answer):
                outcomes[qid] = QuestionOutcome(
                    qid=qid,
                    sparql=gen.sparql,
                    answer=new_answer,
                    exec_ok=gen.ok,
                    error=(gen.result.error if gen.result else "no result"),
                    elapsed_s=(gen.result.elapsed_s if gen.result else 0.0),
                )
                result["filled_qids"].append(qid)
                if qid in score_index:  # keep score_pairs in sync with the patched submission
                    gold = score_pairs[score_index[qid]][1]
                    score_pairs[score_index[qid]] = (qid, gold, new_answer)
                if verbose:
                    print(f"[escalate] q{qid} FILLED on re-run", flush=True)
            else:
                result["still_empty_qids"].append(qid)
                if verbose:
                    print(f"[escalate] q{qid} still empty after re-run", flush=True)
    finally:
        generator.tool_budget = original_budget
    if verbose:
        print(
            f"[escalate] done: {len(result['filled_qids'])} filled, "
            f"{len(result['still_empty_qids'])} still empty "
            f"(of {len(empty_qids)} escalated)\n",
            flush=True,
        )
    return result


def run_benchmark(
    dataset: WikiKGQADataset,
    generator: Generator,
    out_dir: str | Path,
    limit: int | None = None,
    verbose: bool = True,
    auto_escalate: bool = True,
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
            payload = {
                "dataset": dataset.dataset_id,
                "n_questions": len(questions),
                "n_done": len(outcomes),
                "n_exec_ok": sum(o.exec_ok for o in outcomes.values()),
                "scored": sc.n,
                "macro_f1": round(sc.macro_f1, 4),
                "macro_precision": round(sc.macro_precision, 4),
                "macro_recall": round(sc.macro_recall, 4),
                "per_question": [asdict(s) for s in sc.per_question],
            }
            cb = _codabench_macro(score_pairs)
            if cb is not None:
                payload["macro_f1_codabench"] = round(cb.macro_f1, 4)
                payload["macro_precision_codabench"] = round(cb.macro_precision, 4)
                payload["macro_recall_codabench"] = round(cb.macro_recall, 4)
            with open(out_dir / "summary.json", "w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False, indent=2)

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

    # Every gold answer in this benchmark is non-empty, so an empty submission
    # answer is known-wrong. Give agent runs one more shot at exactly those
    # questions with a raised tool budget before writing the final submission
    # (automates the manual escalation done by hand for q3 on 2026-07-13).
    escalation = {"escalated_qids": [], "filled_qids": [], "still_empty_qids": []}
    if auto_escalate:
        escalation = _escalate_empty_answers(generator, questions, outcomes, score_pairs, verbose=verbose)
        if escalation["escalated_qids"]:
            _checkpoint()  # persist the patched outcomes durably before the final write

    # Final submission file (mirrors the input, answers filled).
    submission = build_submission(sub_dataset, outcomes)
    sub_path = write_submission(submission, out_dir / "submission.json")

    summary: dict = {
        "dataset": dataset.dataset_id,
        "n_questions": len(questions),
        "n_exec_ok": sum(o.exec_ok for o in outcomes.values()),
        "submission": str(sub_path),
    }
    if escalation["escalated_qids"]:
        # Additive-only: absent entirely on runs with no empty answers (or
        # auto_escalate=False / a non-agent generator), so summary.json's shape
        # is unchanged in the common case.
        summary["escalated_qids"] = escalation["escalated_qids"]
        summary["escalated_filled_qids"] = escalation["filled_qids"]
        summary["escalated_still_empty_qids"] = escalation["still_empty_qids"]
    if score_pairs:
        score = _macro_score(score_pairs)
        summary["scored"] = score.n
        summary["macro_f1"] = round(score.macro_f1, 4)
        summary["macro_precision"] = round(score.macro_precision, 4)
        summary["macro_recall"] = round(score.macro_recall, 4)
        cb = _codabench_macro(score_pairs)
        if cb is not None:
            summary["macro_f1_codabench"] = round(cb.macro_f1, 4)
            summary["macro_precision_codabench"] = round(cb.macro_precision, 4)
            summary["macro_recall_codabench"] = round(cb.macro_recall, 4)
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
            if "macro_f1_codabench" in summary:
                print(
                    f"Codabench-estimate F1 = {summary['macro_f1_codabench']} "
                    f"(bare-value exact match, leaderboard semantics)"
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
        "--conventions", default="minimal", choices=["full", "minimal"],
        help="agent only: 'minimal' = R1-R6 (default; the 2026-07-03 held-out seed-99 A/B "
             "found no benefit from the extended rules: 0.7311 minimal vs 0.6833 full, "
             "noise-dominated); 'full' = R1-R10 (opt-in).",
    )
    parser.add_argument(
        "--entity-search", action="store_true",
        help="agent only: enable the SearchEntities linker (without-mentions track)",
    )
    parser.add_argument(
        "--full-synthesis", action="store_true",
        help="agent only: feed the full exploration transcript to synthesis instead of the "
             "default minimal journal-only context. (The 2026-07-02 A/B found this does not "
             "help and costs more tokens; kept as an opt-in.)",
    )
    parser.add_argument(
        "--tool-budget", type=int, default=20,
        help="agent only: per-question tool-call budget (default 20; raise for the "
             "linking-heavy without-mentions track, e.g. 30).",
    )
    parser.add_argument(
        "--no-auto-escalate", action="store_true",
        help="agent only: disable the automatic empty-answer escalation pass (default: ON). "
             "Every gold answer in this benchmark is non-empty, so a submission answer that "
             "ends up empty is known-wrong; by default such questions get one more attempt "
             "at generation with the tool budget raised by "
             f"{_ESCALATION_BUDGET_BUMP} before the final submission is written.",
    )
    parser.add_argument(
        "--votes", type=int, default=1,
        help="agent only: self-consistency votes over executed answer sets (all "
             "question types). Runs the full generation up to N times, stopping "
             "early when two runs agree, and returns the modal answer (ties keep "
             "the first run's). Attacks run-to-run trajectory divergence, the "
             "dominant residual error source (seed-99 A/B). 1 = off (default).",
    )
    parser.add_argument(
        "--no-closure-expansion", action="store_true",
        help="agent only: disable the commit-time class-closure repair (widening bare "
             "wdt:P31 membership to the P279* closure when the wider answer set is a "
             "strict superset; ANSWER_CONVENTIONS.md rule 10 applied mechanically).",
    )
    parser.add_argument(
        "--no-pin-now", action="store_true",
        help="send NOW() as written instead of pinning it to the frozen gold reference "
             "instant (2026-04-08, endpoint.REFERENCE_TIME). Pinning is on by default "
             "because gold answers were computed at that instant.",
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

    if args.full_synthesis:
        import os as _os
        _os.environ["WIKIKGQA_FULL_SYNTHESIS"] = "1"
    if args.no_pin_now:
        import os as _os
        _os.environ["WIKIKGQA_PIN_NOW"] = "0"

    _write_run_manifest(args.out_dir, args, resolved)

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
            votes=args.votes,
            closure_expansion=not args.no_closure_expansion,
        )
    else:
        generator = MentionSparqlGenerator(
            endpoint=args.endpoint,
            language=args.language,
            max_repairs=args.max_repairs,
            provider=args.provider,
            model=args.model,
        )
    run_benchmark(
        dataset, generator, out_dir=args.out_dir, limit=args.limit,
        auto_escalate=not args.no_auto_escalate,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
