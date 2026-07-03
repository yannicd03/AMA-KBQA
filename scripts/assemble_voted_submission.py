"""Assemble a sharded benchmark run back into one Codabench submission.

The without-mentions voted run splits the test set into shards purely for
wall-clock parallelism; each question is answered by ONE system instance
(with internal self-consistency voting). This tool reassembles the shard
submissions in the original test-file order — no cross-run voting.

It also supports the empty-answer escalation policy: questions whose answers
are empty after the voted run are known-wrong (every gold answer in this
benchmark is non-empty), so a follow-up run with a raised tool budget may be
executed on exactly those questions and patched in via --patch. Patching
replaces ONLY empty answers, and only with non-empty escalation results.

Usage:
    # reassemble shards (prints any still-empty question ids):
    uv run python scripts/assemble_voted_submission.py OUT.json \
        --test-file data/wikikgqa/wikikgqa2026_test_without_mentions_questions.json \
        --shards dir1/submission.json dir2/submission.json ...

    # patch empties from an escalation run:
    uv run python scripts/assemble_voted_submission.py OUT.json \
        --test-file ... --shards ... --patch escalation/submission.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--test-file", required=True, help="original test question file (defines id order)")
    ap.add_argument("--shards", nargs="+", required=True, help="shard submission.json paths")
    ap.add_argument("--patch", default=None, help="escalation submission.json to fill empty answers from")
    args = ap.parse_args()

    test = json.load(open(args.test_file, encoding="utf-8"))
    order = [q["id"] for q in test["questions"]]

    by_id: dict = {}
    for p in args.shards:
        for q in json.load(open(p, encoding="utf-8"))["questions"]:
            if q["id"] in by_id:
                raise SystemExit(f"duplicate question id {q['id']} across shards")
            by_id[q["id"]] = q

    missing = [i for i in order if i not in by_id]
    if missing:
        raise SystemExit(f"shards do not cover all questions; missing ids: {missing}")

    patched = 0
    if args.patch:
        patch_by_id = {q["id"]: q for q in json.load(open(args.patch, encoding="utf-8"))["questions"]}
        for qid, q in by_id.items():
            if not q.get("answers") and patch_by_id.get(qid, {}).get("answers"):
                by_id[qid] = patch_by_id[qid]
                patched += 1

    questions = [by_id[i] for i in order]
    empty = [q["id"] for q in questions if not q.get("answers")]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"questions": questions}, fh, ensure_ascii=False, indent=2)

    print(f"wrote {out}: {len(questions)} questions, {len(questions) - len(empty)} answered, {patched} patched")
    if empty:
        print(f"EMPTY answers ({len(empty)}): {empty}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
