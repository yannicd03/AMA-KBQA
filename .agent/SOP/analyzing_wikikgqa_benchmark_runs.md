# SOP: Analyzing WikiKGQA Benchmark Runs and Scored Submissions

## Related Docs
- [Decisions/wikikgqa-answer-sanity-guard-2026-07-15.md](../Decisions/wikikgqa-answer-sanity-guard-2026-07-15.md) — the sanity-guard ADR whose 2026-07-25 correction note this SOP's methodology produced
- [Decisions/wikikgqa-commit-time-recovery-2026-07-03.md](../Decisions/wikikgqa-commit-time-recovery-2026-07-03.md) — introduced `run_manifest.json` per run
- [System/wikikgqa_agent.md](../System/wikikgqa_agent.md) — generator/submission pipeline this SOP inspects the output of

---

## The pitfall: run directory names and dates are misleading

`benchmark_results/wikikgqa/<run-dir>/` directory names and their embedded
dates do **not** reliably identify which scored challenge submission a run
corresponds to. Example: `TEST-wm-votes3-20260714` is dated 07-14 but its
`run_manifest.json` timestamp is `2026-07-14T21:20:42` and it was the run
**uploaded as the 07-15 submission** — the directory name reflects when the
run was kicked off locally, not the leaderboard submission it became. Never
infer run→submission mapping from directory names, dates in the name, or
file mtimes.

## The fix: map through `run_manifest.json` git commits

Every benchmark run writes `run_manifest.json` at its root (see
`wikikgqa-commit-time-recovery-2026-07-03.md`) containing the exact `git_commit`
the code was at when the run executed, plus a `timestamp` and `endpoint`.
Cross-reference that commit hash against `git log` (or your own memory of
which commit was pushed for which submission deadline) to get an
unambiguous run→submission mapping:

```bash
for d in benchmark_results/wikikgqa/*/; do
  m="$d/run_manifest.json"
  [ -f "$m" ] && python3 -c "
import json
d = json.load(open('$m'))
print('$d', d.get('git_commit'), d.get('timestamp'))
"
done
```

Then confirm each commit against `git log --oneline | grep <short-hash>` to
attach it to a submission date/round.

## Scanning submissions for invalid answer values

`benchmark_results/wikikgqa/<run-dir>/submission.json` has the shape
`{"questions": [{"id": int, "question": [...], "sparql": str, "answers": [str, ...]}, ...]}`
— **not** a flat `{qid: [...]}` dict. Iterate `data["questions"]`, not
`data.items()`.

A value is invalid (would be rejected by `generator.py::_result_is_sane`) if
it contains `://`, contains `wikidata.org`, contains `statement/`, or starts
with `_:` (blank node). Minimal scan:

```python
import json

def is_invalid(v):
    return isinstance(v, str) and (
        "://" in v or "wikidata.org" in v or "statement/" in v or v.startswith("_:")
    )

data = json.load(open("benchmark_results/wikikgqa/<run-dir>/submission.json"))
for q in data["questions"]:
    bad = [a for a in q.get("answers", []) if is_invalid(a)]
    if bad:
        print(q["id"], len(bad), bad[:3])
```

## Diffing answer sets between two submissions

Gold answers are held out, so per-question correctness can never be verified
locally — only answer **shape** (invalid-value scan above) and **changes
between runs** are checkable. When diffing two submissions to attribute a
score delta:

- **Compare as sets, not ordered lists.** `AgentSparqlGenerator` does not
  guarantee stable answer ordering across runs (votes, generation order),
  so a plain list `!=` comparison overcounts "changed" questions — it flags
  pure reorderings (same answer set, different order) as changes even
  though they cannot affect a set-based F1 score. Confirmed empirically on
  the 2026-07-14→07-15 with-mentions diff: list-diff found 13 changed
  questions, set-diff found 9 real content changes (4 were reorderings only:
  q72, q83, q93, q103).
- Always report both the largest single contributor *and* the full changed
  set — do not attribute a net score delta entirely to the single largest
  regression without diffing every question. That overattribution happened
  once already (see the sanity-guard ADR's 2026-07-25 correction: q113 was
  documented as "the entire" regression when it was actually the dominant
  one among 9 real changes).

```python
s1 = {q["id"]: q["answers"] for q in json.load(open(path1))["questions"]}
s2 = {q["id"]: q["answers"] for q in json.load(open(path2))["questions"]}
changed = [qid for qid in s1 if set(s1[qid]) != set(s2.get(qid, []))]
```
