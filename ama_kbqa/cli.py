"""
CLI entrypoint for AMA-KBQA system.

Usage:
    ama-kbqa ask "query"                           # Use orchestrator (default)
    ama-kbqa ask --subagent kqapro "query"         # Use KQAPro subagent
    ama-kbqa ask --subagent sciqa "query"          # Use SciQA subagent
    ama-kbqa benchmark --subagent kqapro -n 10 --seed 42  # Run KQAPro benchmarking
    ama-kbqa benchmark --subagent sciqa -n 10 --seed 42 --dataset handcrafted  # Run SciQA benchmarking
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from typing import Optional


def run_ask(query: str, subagent: Optional[str] = None) -> None:
    """
    Run an ask query against the specified agent.

    Args:
        query: The question to ask
        subagent: Optional subagent name (e.g., 'kqapro', 'sciqa'). If None, uses orchestrator.
    """
    if subagent == "kqapro":
        from ama_kbqa.agents.kqapro_agent.agent import KQAProAgent

        async def _ask():
            agent = KQAProAgent()
            try:
                answer = await agent.ask(query)
                print(f"\n{'='*60}")
                print("ANSWER")
                print(f"{'='*60}")
                print(answer)
                print(f"{'='*60}")

                # Print token usage summary
                print(f"\nTokens used: {agent.token_usage.get('total_tokens', 0)}")
            finally:
                await agent.close()

        asyncio.run(_ask())

    elif subagent == "sciqa":
        from ama_kbqa.agents.sciqa_agent.agent import SciQAAgent

        async def _ask():
            agent = SciQAAgent()
            try:
                answer = await agent.ask(query)
                print(f"\n{'='*60}")
                print("ANSWER")
                print(f"{'='*60}")
                print(answer)
                print(f"{'='*60}")

                # Print token usage summary
                print(f"\nTokens used: {agent.token_usage.get('total_tokens', 0)}")
            finally:
                await agent.close()

        asyncio.run(_ask())

    elif subagent is None:
        # Use orchestrator by default
        from ama_kbqa.agents.orchestrator_agent.agent import Orchestrator

        async def _ask():
            orchestrator = Orchestrator()
            answer = await orchestrator.ask(query)
            print(f"\n{'='*60}")
            print("ANSWER")
            print(f"{'='*60}")
            print(answer)
            print(f"{'='*60}")

        asyncio.run(_ask())

    else:
        print(f"Error: Unknown subagent '{subagent}'. Available: kqapro, sciqa", file=sys.stderr)
        sys.exit(1)


def run_benchmark(
    subagent: str,
    n_questions: int = 10,
    seed: int = 42,
    postprocessing_mode: str = "llm_judge",
    dataset: Optional[str] = None
) -> None:
    """
    Run benchmarking for the specified agent.

    Args:
        subagent: The subagent to benchmark (e.g., 'kqapro', 'sciqa')
        n_questions: Number of questions to sample
        seed: Random seed for reproducibility
        postprocessing_mode: Postprocessing method (choice, sparql, llm_judge)
        dataset: Dataset type for SciQA (handcrafted, auto)
    """
    if subagent == "kqapro":
        from ama_kbqa.agents.kqapro_agent.batch_runner import run_batch

        asyncio.run(run_batch(
            n_questions=n_questions,
            seed=seed,
            postprocessing_mode=postprocessing_mode
        ))

    elif subagent == "sciqa":
        from ama_kbqa.agents.sciqa_agent.batch_runner import run_batch

        # Default to handcrafted if not specified
        dataset_type = dataset if dataset else "handcrafted"

        asyncio.run(run_batch(
            n_questions=n_questions,
            seed=seed,
            dataset_type=dataset_type,
            postprocessing_mode=postprocessing_mode
        ))

    else:
        print(f"Error: Unknown subagent '{subagent}'. Available: kqapro, sciqa", file=sys.stderr)
        sys.exit(1)


def main():
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(
        prog="ama-kbqa",
        description="AMA-KBQA: Knowledge Base Question Answering System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Ask questions
  ama-kbqa ask "Who directed Inception?"
  ama-kbqa ask --subagent kqapro "Who directed Inception?"
  ama-kbqa ask --subagent sciqa "What papers address text classification?"

  # Run benchmarks
  ama-kbqa benchmark --subagent kqapro -n 10 --seed 42
  ama-kbqa benchmark --subagent kqapro -n 50 --seed 123 --postprocessing llm_judge
  ama-kbqa benchmark --subagent sciqa -n 10 --seed 42 --dataset handcrafted
  ama-kbqa benchmark --subagent sciqa -n 50 --dataset auto --postprocessing simple
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- ASK command ---
    ask_parser = subparsers.add_parser(
        "ask",
        help="Ask a question to the KBQA system"
    )
    ask_parser.add_argument(
        "query",
        type=str,
        help="The question to ask"
    )
    ask_parser.add_argument(
        "--subagent", "-s",
        type=str,
        choices=["kqapro", "sciqa"],
        default=None,
        help="Specific subagent to use. If not specified, uses the orchestrator."
    )

    # --- BENCHMARK command ---
    benchmark_parser = subparsers.add_parser(
        "benchmark",
        help="Run benchmarking on validation dataset"
    )
    benchmark_parser.add_argument(
        "--subagent", "-s",
        type=str,
        choices=["kqapro", "sciqa"],
        required=True,
        help="The subagent to benchmark"
    )
    benchmark_parser.add_argument(
        "-n", "--n_questions",
        type=int,
        default=10,
        help="Number of questions to sample (default: 10)"
    )
    benchmark_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    benchmark_parser.add_argument(
        "--postprocessing", "-p",
        type=str,
        choices=["choice", "sparql", "llm_judge", "simple"],
        default="llm_judge",
        help="Postprocessing method (default: llm_judge)"
    )
    benchmark_parser.add_argument(
        "--dataset", "-d",
        type=str,
        choices=["handcrafted", "auto", "autogenerated"],
        default=None,
        help="Dataset type for SciQA benchmark (default: handcrafted). Ignored for KQAPro."
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    if args.command == "ask":
        run_ask(query=args.query, subagent=args.subagent)

    elif args.command == "benchmark":
        run_benchmark(
            subagent=args.subagent,
            n_questions=args.n_questions,
            seed=args.seed,
            postprocessing_mode=args.postprocessing,
            dataset=args.dataset
        )

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
