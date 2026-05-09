"""Run the fewshot generator post-hoc on a saved benchmark result directory.

Usage:
    uv run python -m ama_kbqa.run_fewshot_generator \
        --result-dir benchmark_results/minimax-kqapro-2026-05-09-classifier-fix \
        --agent kqapro \
        --output-dir db/datasets/kqapro/fewshot-examples.generated-2026-05-09

This loads results.json + tool_traces/question_NNN.json from the result dir,
reconstructs the inputs `generate_and_save_fewshot_examples` expects, and
redirects FEWSHOT_DIR to a shadow path so live fewshot files are never touched.

The generator's settings (model, temperature, max_tokens, full-conversation
toggle, tool-catalog toggle) come from the [fewshot_generator] section of
config.toml.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from ama_kbqa import fewshot_generator as fg


@dataclass
class ReplayResult:
    """Stand-in for QuestionResult — only needs `.question` and `.full_messages`."""
    question: str
    full_messages: List[Dict[str, Any]] = field(default_factory=list)


def _load_results(result_dir: Path, agent_name: str, model_name: Optional[str]) -> tuple[Path, List[Dict]]:
    """Locate the per-model subdir and load results.json."""
    agent_dir = result_dir / agent_name
    if not agent_dir.exists():
        raise FileNotFoundError(f"agent dir not found: {agent_dir}")

    if model_name:
        model_dir = agent_dir / model_name
    else:
        model_dirs = [p for p in agent_dir.iterdir() if p.is_dir()]
        if len(model_dirs) != 1:
            raise ValueError(
                f"--model required: found {len(model_dirs)} model dirs in {agent_dir}: "
                f"{[p.name for p in model_dirs]}"
            )
        model_dir = model_dirs[0]

    results_path = model_dir / "results.json"
    if not results_path.exists():
        raise FileNotFoundError(f"results.json not found at {results_path}")
    with open(results_path) as f:
        results_data = json.load(f)
    return model_dir, results_data


def _load_full_messages(model_dir: Path, n_results: int) -> Dict[int, List[Dict]]:
    """Load full_messages from tool_traces/question_NNN.json, keyed by question text."""
    traces_dir = model_dir / "tool_traces"
    by_question: Dict[str, List[Dict]] = {}
    if not traces_dir.exists():
        return by_question
    for i in range(n_results):
        path = traces_dir / f"question_{i:03d}.json"
        if not path.exists():
            continue
        with open(path) as f:
            trace = json.load(f)
        q = trace.get("question", "")
        msgs = trace.get("messages", [])
        if q and msgs:
            by_question[q] = msgs
    return by_question


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", required=True, type=Path,
                        help="Top-level benchmark output dir (e.g. benchmark_results/<run-name>)")
    parser.add_argument("--agent", default="kqapro",
                        help="Agent name (kqapro|sciqa) — picks the MCP tool catalog")
    parser.add_argument("--model", default=None,
                        help="Model subdir name (auto-detected if only one)")
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="Shadow directory to write generated fewshot files into. "
                             "MUST NOT be the live db/datasets/.../fewshot-examples dir.")
    parser.add_argument("--audit-name", default="generated_fewshot_posthoc.json",
                        help="Filename for the audit log written into --result-dir")
    args = parser.parse_args()

    output_dir: Path = args.output_dir.resolve()
    live_dir = fg.FEWSHOT_DIR.resolve()
    if output_dir == live_dir:
        raise SystemExit(
            f"refusing to write to live fewshot dir {live_dir}. "
            f"Pass a different --output-dir for shadow output."
        )

    model_dir, results_data = _load_results(args.result_dir, args.agent, args.model)
    print(f"[Replay] loaded {len(results_data)} results from {model_dir}")

    msgs_by_question = _load_full_messages(model_dir, len(results_data))
    print(f"[Replay] reconstructed full_messages for {len(msgs_by_question)} questions")

    full_results = [
        ReplayResult(question=q, full_messages=msgs)
        for q, msgs in msgs_by_question.items()
    ]

    # Redirect writes to the shadow dir. The save helpers in fewshot_generator
    # resolve FEWSHOT_DIR at call time, so monkey-patching the module attr is
    # sufficient and avoids touching live training data.
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Replay] redirecting FEWSHOT_DIR -> {output_dir}")
    original_dir = fg.FEWSHOT_DIR
    fg.FEWSHOT_DIR = output_dir
    try:
        # Override audit-log filename via the same trick: pass result_dir but
        # the generator hardcodes "generated_fewshot.json" — we'll move it after.
        counts = fg.generate_and_save_fewshot_examples(
            results_data=results_data,
            full_results=full_results,
            agent_name=args.agent,
            result_dir=args.result_dir,
        )
    finally:
        fg.FEWSHOT_DIR = original_dir

    # Rename the generator's audit log so it doesn't clobber any prior one.
    default_audit = args.result_dir / "generated_fewshot.json"
    if default_audit.exists() and args.audit_name != "generated_fewshot.json":
        target = args.result_dir / args.audit_name
        default_audit.rename(target)
        print(f"[Replay] audit log -> {target}")

    print(f"[Replay] done. counts={counts}")


if __name__ == "__main__":
    main()
