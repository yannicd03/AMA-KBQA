"""Route-only benchmark: measure the orchestrator's routing accuracy.

Samples labeled questions from the KQAPro and SciQA benchmark datasets, runs
ONLY the routing step (no answering, no synthesis), and reports a confusion
matrix plus per-class accuracy. Each question's ground-truth agent is the
dataset it came from: KQAPro questions should route to kqapro_agent, SciQA
questions to sciqa_agent.

This exists to validate routing changes with a number instead of anecdotes
(lesson from the fewshot-classifier fix: "classifier fixed" != "system
improved"). Run it before and after a routing change and compare.

Requirements: a reachable Qdrant with the kqapro-entities and sciqa-entities
collections, and a configured LLM provider key (config.toml / .env). Routing
costs roughly two small LLM calls plus embeddings per question.

Run from the repo root:
    uv run python scripts/benchmark_routing.py --n 25
    uv run python scripts/benchmark_routing.py --n 50 --seed 7 --out routing_results.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
from collections import Counter
from pathlib import Path

from ama_kbqa.agents.orchestrator_agent.agent import Orchestrator
from ama_kbqa.benchmark_agents import load_raw_dataset


def sample_questions(n: int, seed: int, dataset_type: str) -> list[dict]:
    """Return up to n labeled questions per agent, shuffled together."""
    rng = random.Random(seed)
    samples = []
    for source, expected in (("kqapro", "kqapro_agent"), ("sciqa", "sciqa_agent")):
        rows = load_raw_dataset(source, dataset_type)
        rows = [r for r in rows if r.get("question")]
        for row in rng.sample(rows, min(n, len(rows))):
            samples.append({"question": row["question"], "expected": expected})
    rng.shuffle(samples)
    return samples


async def run(samples: list[dict]) -> list[dict]:
    """Route every sample through one shared Orchestrator/MCP instance."""
    orch = Orchestrator(session_id="routing-benchmark")
    records = []
    await orch._init_mcp()
    if not orch.mcp:
        sys.exit("MCP server failed to start; cannot benchmark routing.")
    try:
        for i, sample in enumerate(samples, 1):
            routed_names = await orch._route_autonomously(sample["question"])
            # _route_autonomously returns a list (federated dispatch support);
            # the benchmark scores single-dispatch routing, so a multi-agent
            # selection is recorded as a joined label and counts as a miss.
            routed = "+".join(routed_names) if routed_names else None
            records.append({
                **sample,
                "routed": routed,  # None => router failed, runtime falls back to KQAPro
                "reason": orch.last_routing_reason,
                "correct": routed == sample["expected"],
            })
            print(f"[{i}/{len(samples)}] {'OK ' if records[-1]['correct'] else 'MISS'} "
                  f"expected={sample['expected']} routed={routed}")
    finally:
        await orch.mcp.close()
    return records


def report(records: list[dict]) -> dict:
    matrix = Counter((r["expected"], r["routed"] or "<route_failed>") for r in records)
    per_class = {}
    for expected in ("kqapro_agent", "sciqa_agent"):
        subset = [r for r in records if r["expected"] == expected]
        if subset:
            per_class[expected] = sum(r["correct"] for r in subset) / len(subset)

    print("\nConfusion matrix (expected -> routed):")
    for (expected, routed), count in sorted(matrix.items()):
        print(f"  {expected:>13} -> {routed:<15} {count}")
    for expected, acc in per_class.items():
        print(f"Accuracy {expected}: {acc:.2%}")
    overall = sum(r["correct"] for r in records) / len(records)
    print(f"Overall routing accuracy: {overall:.2%} ({len(records)} questions)")

    return {
        "overall_accuracy": overall,
        "per_class_accuracy": per_class,
        "confusion": {f"{e}->{r}": c for (e, r), c in matrix.items()},
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n", type=int, default=25, help="questions per agent (default: 25)")
    parser.add_argument("--seed", type=int, default=42, help="sampling seed (default: 42)")
    parser.add_argument("--dataset", default="handcrafted", choices=["handcrafted", "auto"],
                        help="SciQA dataset type (default: handcrafted)")
    parser.add_argument("--out", type=Path, default=None, help="write full results JSON here")
    args = parser.parse_args()

    samples = sample_questions(args.n, args.seed, args.dataset)
    records = asyncio.run(run(samples))
    results = report(records)

    if args.out:
        args.out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Results written to {args.out}")


if __name__ == "__main__":
    main()
