"""
CLI entrypoint for AMA-KBQA system.

Usage:
    ama-kbqa ask "query"                           # Use orchestrator (default)
    ama-kbqa ask --subagent kqapro "query"         # Use specific subagent
    ama-kbqa benchmark --subagent kqapro -n 10 --seed 42  # Run benchmarking
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
        subagent: Optional subagent name (e.g., 'kqapro'). If None, uses orchestrator.
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
        print(f"Error: Unknown subagent '{subagent}'. Available: kqapro", file=sys.stderr)
        sys.exit(1)


def run_benchmark(
    subagent: str,
    n_questions: int = 10,
    seed: int = 42,
    postprocessing_mode: str = "llm_judge"
) -> None:
    """
    Run benchmarking for the specified agent.

    Args:
        subagent: The subagent to benchmark (e.g., 'kqapro')
        n_questions: Number of questions to sample
        seed: Random seed for reproducibility
        postprocessing_mode: Postprocessing method (choice, sparql, llm_judge)
    """
    if subagent == "kqapro":
        from ama_kbqa.agents.kqapro_agent.batch_runner import run_batch

        asyncio.run(run_batch(
            n_questions=n_questions,
            seed=seed,
            postprocessing_mode=postprocessing_mode
        ))
    else:
        print(f"Error: Unknown subagent '{subagent}'. Available: kqapro", file=sys.stderr)
        sys.exit(1)


def main():
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(
        prog="ama-kbqa",
        description="AMA-KBQA: Knowledge Base Question Answering System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  ama-kbqa ask "Who directed Inception?"
  ama-kbqa ask --subagent kqapro "Who directed Inception?"
  ama-kbqa benchmark --subagent kqapro -n 10 --seed 42
  ama-kbqa benchmark --subagent kqapro -n 50 --seed 123 --postprocessing llm_judge
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
        choices=["kqapro"],
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
        choices=["kqapro"],
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
        choices=["choice", "sparql", "llm_judge"],
        default="llm_judge",
        help="Postprocessing method (default: llm_judge)"
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
            postprocessing_mode=args.postprocessing
        )

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
