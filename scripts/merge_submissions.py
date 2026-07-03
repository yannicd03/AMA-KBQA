"""Merge multiple Codabench submission.json files by per-question answer voting.

Ensemble strategy for independent runs of the same test set (run-to-run
trajectory divergence is the dominant residual error source, seed-99 A/B
2026-07-03): for each question, identical answer sets vote together, the
modal non-empty answer set wins, and an empty answer never beats a non-empty
one (every gold answer in this benchmark is non-empty). Ties resolve by the
order the files are given on the command line (put the most-trusted run
first).

Usage:
    uv run python scripts/merge_submissions.py OUT.json IN1.json IN2.json [IN3.json ...]

The first input also provides the output skeleton (ids, question, sparql),
so pass the newest-code run first.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path


def _key(answers: list) -> frozenset | None:
    """Comparable identity of an answers array; None for empty (can never win)."""
    if not answers:
        return None
    return frozenset(
        ("true" if a else "false") if isinstance(a, bool) else str(a) for a in answers
    )


def merge(paths: list[Path]) -> tuple[dict, dict]:
    subs = [json.load(open(p, encoding="utf-8")) for p in paths]
    by_id = [{q["id"]: q for q in s["questions"]} for s in subs]

    stats = {"agree": 0, "filled_empty": 0, "vote_changed": 0, "still_empty": 0}
    out_questions = []
    for q in subs[0]["questions"]:
        qid = q["id"]
        candidates = [b[qid] for b in by_id if qid in b]
        keys = [_key(c.get("answers") or []) for c in candidates]

        counts = Counter(k for k in keys if k is not None)
        merged = dict(candidates[0])  # skeleton: first (most-trusted) run
        if not counts:
            stats["still_empty"] += 1  # every run empty; nothing to salvage
        else:
            # most_common ties resolve by insertion order = file order.
            best = counts.most_common(1)[0][0]
            winner = next(c for c, k in zip(candidates, keys) if k == best)
            if keys[0] == best:
                stats["agree"] += 1
            elif keys[0] is None:
                stats["filled_empty"] += 1
            else:
                stats["vote_changed"] += 1
            merged = dict(winner)
        out_questions.append(merged)

    return {"questions": out_questions}, stats


def main() -> int:
    if len(sys.argv) < 4:
        print(__doc__)
        return 2
    out_path = Path(sys.argv[1])
    in_paths = [Path(p) for p in sys.argv[2:]]
    merged, stats = merge(in_paths)
    n = len(merged["questions"])
    n_answered = sum(1 for q in merged["questions"] if q.get("answers"))
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(merged, fh, ensure_ascii=False, indent=2)
    print(f"merged {len(in_paths)} submissions -> {out_path}")
    print(f"questions: {n}, answered: {n_answered}, stats: {stats}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
