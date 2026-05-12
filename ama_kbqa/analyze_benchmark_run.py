"""Summarize benchmark run health and tool adoption.

Usage:
  python -m ama_kbqa.analyze_benchmark_run benchmark_results/<run>/<agent>/<model>
  python -m ama_kbqa.analyze_benchmark_run /tmp/amakbqa-runs/gemma-sciqa.../sciqa/gemma...
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any


TARGET_TOOLS = {
    "GetQualifierValue",
    "GetEdgeQualifiers",
    "GetQualifiersByPredicate",
    "GetAttributeWithQualifiers",
    "CountEntities",
    "CountUnion",
    "GetRelationBetween",
    "FindFrequentValues",
    "AggregateComparisonValues",
    "RunORKGSPARQL",
    "RunSPARQL",
    "GetRelationTargets",
}


def _load_results(run_dir: Path) -> list[dict[str, Any]]:
    for name in ("results.json", "detailed_results.json"):
        path = run_dir / name
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    raise FileNotFoundError(f"No results.json or detailed_results.json found in {run_dir}")


def _tool_counts(row: dict[str, Any]) -> dict[str, int]:
    summary = row.get("tool_call_summary") or {}
    counts = summary.get("tool_counts")
    if isinstance(counts, dict):
        return counts

    breakdown = summary.get("tool_breakdown")
    if not isinstance(breakdown, dict):
        return {}

    normalized: dict[str, int] = {}
    for tool, value in breakdown.items():
        if isinstance(value, dict):
            count = value.get("count", value.get("calls", 0))
        else:
            count = value
        try:
            normalized[tool] = int(count)
        except (TypeError, ValueError):
            normalized[tool] = 0
    return normalized


def analyze(run_dir: Path) -> dict[str, Any]:
    results = _load_results(run_dir)
    wrong = [(i, r) for i, r in enumerate(results) if not r.get("accuracy")]

    qtype_wrong = collections.Counter(
        (r.get("q_type") or r.get("qtype") or "Unknown")
        for _, r in wrong
    )
    zero_tool_wrong = [
        i for i, r in wrong
        if (r.get("tool_call_summary") or {}).get("total_calls") == 0
    ]
    context_errors = [
        i for i, r in enumerate(results)
        if "contextwindow" in str(r.get("error") or "").lower()
        or "context window" in str(r.get("error") or "").lower()
    ]
    max_iter_errors = [
        i for i, r in enumerate(results)
        if "maximum iteration" in str(r.get("error") or r.get("predicted_answer") or "").lower()
    ]

    correct_tools: collections.Counter[str] = collections.Counter()
    wrong_tools: collections.Counter[str] = collections.Counter()
    total_tools: collections.Counter[str] = collections.Counter()
    for row in results:
        bucket = correct_tools if row.get("accuracy") else wrong_tools
        for tool, count in _tool_counts(row).items():
            bucket[tool] += count
            total_tools[tool] += count

    return {
        "run_dir": str(run_dir),
        "total": len(results),
        "correct": len(results) - len(wrong),
        "accuracy": (len(results) - len(wrong)) / len(results) if results else 0.0,
        "wrong_indices": [i for i, _ in wrong],
        "wrong_by_qtype": qtype_wrong.most_common(),
        "zero_tool_wrong": zero_tool_wrong,
        "context_errors": context_errors,
        "max_iteration_errors": max_iter_errors,
        "target_tool_adoption": {
            tool: {
                "total": total_tools[tool],
                "correct": correct_tools[tool],
                "wrong": wrong_tools[tool],
            }
            for tool in sorted(TARGET_TOOLS)
            if total_tools[tool]
        },
        "top_tools": total_tools.most_common(20),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    args = parser.parse_args()

    report = analyze(args.run_dir)
    if args.json:
        print(json.dumps(report, indent=2))
        return

    print(f"Run: {report['run_dir']}")
    print(f"Accuracy: {report['correct']}/{report['total']} = {report['accuracy']:.3f}")
    print(f"Wrong indices: {report['wrong_indices']}")
    print(f"Wrong by qtype: {report['wrong_by_qtype']}")
    print(f"Zero-tool wrong: {report['zero_tool_wrong']}")
    print(f"Context errors: {report['context_errors']}")
    print(f"Max-iteration errors: {report['max_iteration_errors']}")
    print("Target tool adoption:")
    for tool, data in report["target_tool_adoption"].items():
        print(f"  {tool}: total={data['total']} correct={data['correct']} wrong={data['wrong']}")
    print(f"Top tools: {report['top_tools']}")


if __name__ == "__main__":
    main()
