"""
Batch Runner for KQAPro Benchmark

This script processes a random sample of questions from the validation dataset
and saves detailed results including metadata for analysis.

Usage:
    python batch_runner.py --n_questions 10 --seed 42
"""

from __future__ import annotations
import os
import sys
import asyncio
import json
import random
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv(override=True)

# Add parent directory to path for imports
current_file = Path(__file__).resolve()
ama_kbqa_root = current_file.parents[2]
project_root = current_file.parents[3]  # Go up one more level to project root
sys.path.insert(0, str(ama_kbqa_root))

from ama_kbqa.agents.kqapro_agent.benchmark import KQAProAgent


# ============================================================================
# CONFIGURATION
# ============================================================================

VALIDATION_DATASET_PATH = project_root / "db" / "datasets" / "kqapro" / "val.json"
BATCH_RESULTS_BASE_DIR = project_root / "batch_results"

# OpenRouter configuration for answer selection
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "minimax/minimax-m2")

# Ensure batch_results directory exists
BATCH_RESULTS_BASE_DIR.mkdir(exist_ok=True)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_next_batch_folder() -> Path:
    """
    Find the next available batch folder (Batch001, Batch002, etc.)

    Returns:
        Path to the next batch folder
    """
    batch_num = 1
    while True:
        batch_folder = BATCH_RESULTS_BASE_DIR / f"Batch{batch_num:03d}"
        if not batch_folder.exists():
            batch_folder.mkdir(parents=True, exist_ok=True)
            return batch_folder
        batch_num += 1


def load_validation_dataset() -> List[Dict[str, Any]]:
    """
    Load the validation dataset from the JSON file.

    Returns:
        List of validation questions with metadata
    """
    with open(VALIDATION_DATASET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"[OK] Loaded {len(data)} questions from validation dataset")
    return data


def sample_questions(
    data: List[Dict[str, Any]],
    n: int,
    seed: int = 42
) -> List[Dict[str, Any]]:
    """
    Randomly sample n questions from the dataset with a fixed seed.

    Args:
        data: Full validation dataset
        n: Number of questions to sample
        seed: Random seed for reproducibility

    Returns:
        Sampled questions
    """
    random.seed(seed)
    n_sample = min(n, len(data))
    sampled = random.sample(data, n_sample)

    print(f"[OK] Sampled {n_sample} questions with seed={seed}")
    return sampled


def classify_question_type(question: str, program: List[Dict] = None) -> str:
    """
    Attempt to classify the question type based on the question text and program.

    Args:
        question: The question text
        program: The program structure (optional)

    Returns:
        Estimated question type
    """
    question_lower = question.lower()

    # Check program if available
    if program and len(program) > 0:
        last_function = program[-1].get("function", "")

        if last_function == "Count":
            return "Count"
        elif last_function in ["VerifyQuery", "Verify"]:
            return "Verify"
        elif last_function in ["SelectBetween", "SelectAmong"]:
            return "Select"
        elif last_function == "QueryAttr":
            return "Query"
        elif last_function == "QueryRelation":
            return "QueryRelation"
        elif last_function == "QueryAttrQualifier":
            return "QueryAttrQualifier"
        elif last_function == "QueryRelationQualifier":
            return "QueryRelationQualifier"

    # Fallback to question text analysis
    if question_lower.startswith("how many"):
        return "Count"
    elif question_lower.startswith(("is ", "does ", "did ", "was ", "were ", "are ")):
        return "Verify"
    elif question_lower.startswith(("what ", "which ", "who ", "when ", "where ", "whose ")):
        return "Query"

    return "Unknown"


def select_answer_from_choices(
    question: str,
    predicted_answer: str,
    choices: List[str],
    client: OpenAI
) -> str:
    """
    Use the LLM to select the most suitable answer from the available choices
    based on the question and the agent's predicted answer.

    This is a non-agentic, single-call approach that maps the verbose agent
    response to one of the predefined answer choices.

    Args:
        question: The original question
        predicted_answer: The verbose answer from the agent
        choices: List of available answer choices
        client: OpenAI client instance

    Returns:
        The selected answer from the choices list
    """
    # Filter out "unknown" choices to only show valid options
    valid_choices = [c for c in choices if c.lower() != "unknown"]

    # If no valid choices, return the first choice as fallback
    if not valid_choices:
        return choices[0] if choices else "unknown"

    # Build the prompt for answer selection
    prompt = f"""Given the following question and detailed answer, select the most appropriate choice from the available options.

Question: {question}

Detailed Answer: {predicted_answer}

Available Choices:
{chr(10).join(f"- {choice}" for choice in valid_choices)}

Instructions:
1. Carefully read the detailed answer
2. Determine which of the available choices best matches the meaning of the detailed answer
3. Respond with ONLY the exact choice text, nothing else

Your selected choice:"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a precise answer selector. You must respond with only one of the provided choices, exactly as written."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,  # Deterministic selection
            max_tokens=50  # Short response expected
        )

        selected = response.choices[0].message.content.strip()

        # Verify the selected answer is in the valid choices (case-insensitive)
        selected_lower = selected.lower()
        for choice in valid_choices:
            if choice.lower() == selected_lower:
                return choice

        # If exact match not found, try to find partial match
        for choice in valid_choices:
            if choice.lower() in selected_lower or selected_lower in choice.lower():
                return choice

        # If no match found, return the first valid choice as fallback
        print(f"[WARNING] LLM selected '{selected}' which is not in choices. Using first choice.")
        return valid_choices[0]

    except Exception as e:
        print(f"[ERROR] Failed to select answer from choices: {e}")
        # Return first valid choice as fallback
        return valid_choices[0] if valid_choices else (choices[0] if choices else "unknown")


async def process_question(
    agent: KQAProAgent,
    item: Dict[str, Any],
    question_idx: int,
    total_questions: int,
    client: OpenAI
) -> Dict[str, Any]:
    """
    Process a single question through the agent and collect metadata.

    Args:
        agent: The KQAProAgent instance
        item: The question item from the validation set
        question_idx: Current question index (0-based)
        total_questions: Total number of questions being processed
        client: OpenAI client for answer selection

    Returns:
        Dictionary with results and metadata
    """
    question = item["question"]
    gold_answer = item.get("answer", None)
    gold_sparql = item.get("sparql", None)
    program = item.get("program", None)
    choices = item.get("choices", [])

    qtype = classify_question_type(question, program)

    print(f"\n{'='*80}")
    print(f"[{question_idx + 1}/{total_questions}] Processing question")
    print(f"{'='*80}")
    print(f"Q: {question}")
    print(f"Type: {qtype}")
    print(f"Gold Answer: {gold_answer}")

    start_time = time.time()

    try:
        # Process the question
        predicted_answer = await agent.ask(question)

        end_time = time.time()
        duration = end_time - start_time

        # Post-processing: Select answer from choices using LLM
        selected_answer = None
        accuracy = False

        if choices and len(choices) > 0:
            selected_answer = select_answer_from_choices(
                question=question,
                predicted_answer=predicted_answer,
                choices=choices,
                client=client
            )
            # Calculate accuracy by comparing selected answer to gold answer (case-insensitive)
            if gold_answer and selected_answer:
                accuracy = selected_answer.lower().strip() == gold_answer.lower().strip()

        # Collect metadata
        result = {
            "question": question,
            "answer": gold_answer,
            "predicted_answer": predicted_answer,
            "selected_answer": selected_answer,
            "accuracy": accuracy,
            "qtype": qtype,
            "predicted_qtype": qtype,  # Could be different if agent determines it
            "duration": f"{duration:.2f}s",
            "tokens_used": agent.token_usage.get("total_tokens", 0),
            "prompt_tokens": agent.token_usage.get("prompt_tokens", 0),
            "completion_tokens": agent.token_usage.get("completion_tokens", 0),
            "number_of_turns_used": len([m for m in agent._messages if (isinstance(m, dict) and m.get("role") == "assistant") or (hasattr(m, "role") and m.role == "assistant")]),
            "gold_sparql_query": gold_sparql,
            "extracted_sparql_query": None,  # Would need to parse from agent output
            "program": program,
            "choices": choices,
            "success": True,
            "error": None
        }

        print(f"[OK] Predicted: {predicted_answer[:100]}...")
        print(f"[OK] Selected Answer: {selected_answer} | Accuracy: {'[OK]' if accuracy else '[ERROR]'}")
        print(f"[OK] Duration: {duration:.2f}s | Tokens: {agent.token_usage.get('total_tokens', 0)}")

    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time

        result = {
            "question": question,
            "answer": gold_answer,
            "predicted_answer": None,
            "selected_answer": None,
            "accuracy": False,
            "qtype": qtype,
            "predicted_qtype": None,
            "duration": f"{duration:.2f}s",
            "tokens_used": agent.token_usage.get("total_tokens", 0),
            "prompt_tokens": agent.token_usage.get("prompt_tokens", 0),
            "completion_tokens": agent.token_usage.get("completion_tokens", 0),
            "number_of_turns_used": 0,
            "gold_sparql_query": gold_sparql,
            "extracted_sparql_query": None,
            "program": program,
            "choices": choices,
            "success": False,
            "error": str(e)
        }

        print(f"[ERROR] {e}")

    # Reset agent for next question
    agent.reset()

    return result


def save_batch_results(
    batch_folder: Path,
    sampled_questions: List[Dict[str, Any]],
    results: List[Dict[str, Any]],
    config: Dict[str, Any]
):
    """
    Save all batch results to files.

    Args:
        batch_folder: Path to the batch folder
        sampled_questions: The sampled questions
        results: Processing results
        config: Configuration used for this batch
    """
    # Save sampled questions
    sampled_file = batch_folder / "sampled_questions.json"
    with open(sampled_file, "w", encoding="utf-8") as f:
        json.dump(sampled_questions, f, indent=2, ensure_ascii=False)

    print(f"\n[OK] Saved sampled questions to: {sampled_file}")

    # Save individual results
    results_file = batch_folder / "results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"[OK] Saved detailed results to: {results_file}")

    # Calculate summary statistics
    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    # Calculate accuracy metrics
    accurate = [r for r in results if r.get("accuracy", False)]
    accuracy_rate = len(accurate) / len(results) if results else 0

    total_tokens = sum(r.get("tokens_used", 0) for r in results)
    total_duration = sum(float(r["duration"].replace("s", "")) for r in results)
    avg_duration = total_duration / len(results) if results else 0

    # Count by question type
    qtype_counts = {}
    qtype_accuracy = {}
    for r in results:
        qtype = r.get("qtype", "Unknown")
        qtype_counts[qtype] = qtype_counts.get(qtype, 0) + 1

        # Track accuracy per question type
        if qtype not in qtype_accuracy:
            qtype_accuracy[qtype] = {"total": 0, "accurate": 0}
        qtype_accuracy[qtype]["total"] += 1
        if r.get("accuracy", False):
            qtype_accuracy[qtype]["accurate"] += 1

    # Calculate accuracy rate per question type
    qtype_accuracy_rates = {}
    for qtype, counts in qtype_accuracy.items():
        qtype_accuracy_rates[qtype] = counts["accurate"] / counts["total"] if counts["total"] > 0 else 0

    summary = {
        "config": config,
        "timestamp": datetime.now().isoformat(),
        "statistics": {
            "total_questions": len(results),
            "successful": len(successful),
            "failed": len(failed),
            "success_rate": len(successful) / len(results) if results else 0,
            "accurate": len(accurate),
            "accuracy_rate": accuracy_rate,
            "total_tokens_used": total_tokens,
            "total_duration_seconds": round(total_duration, 2),
            "average_duration_seconds": round(avg_duration, 2),
            "question_type_distribution": qtype_counts,
            "accuracy_by_question_type": qtype_accuracy_rates
        },
        "failed_questions": [
            {
                "question": r["question"],
                "error": r["error"]
            }
            for r in failed
        ]
    }

    # Save summary
    summary_file = batch_folder / "summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"[OK] Saved summary to: {summary_file}")

    # Print summary to console
    print(f"\n{'='*80}")
    print("BATCH SUMMARY")
    print(f"{'='*80}")
    print(f"Total Questions:     {len(results)}")
    print(f"Successful:          {len(successful)}")
    print(f"Failed:              {len(failed)}")
    print(f"Success Rate:        {summary['statistics']['success_rate']:.1%}")
    print(f"Accurate:            {len(accurate)}")
    print(f"Accuracy Rate:       {accuracy_rate:.1%}")
    print(f"Total Tokens:        {total_tokens:,}")
    print(f"Total Duration:      {total_duration:.2f}s")
    print(f"Average Duration:    {avg_duration:.2f}s")
    print(f"\nQuestion Type Distribution:")
    for qtype, count in qtype_counts.items():
        acc_rate = qtype_accuracy_rates.get(qtype, 0)
        print(f"  {qtype:20s}: {count:3d} (Accuracy: {acc_rate:.1%})")

    if failed:
        print(f"\nFailed Questions ({len(failed)}):")
        for i, fail in enumerate(failed[:5], 1):  # Show first 5
            print(f"  {i}. {fail['question'][:60]}...")
            print(f"     Error: {fail['error']}")


# ============================================================================
# MAIN BATCH PROCESSING
# ============================================================================

async def run_batch(n_questions: int = 10, seed: int = 42):
    """
    Run a complete batch processing job.

    Args:
        n_questions: Number of questions to sample and process
        seed: Random seed for reproducibility
    """
    print(f"\n{'='*80}")
    print("KQAPro Batch Runner")
    print(f"{'='*80}")
    print(f"Configuration:")
    print(f"  Questions:  {n_questions}")
    print(f"  Seed:       {seed}")
    print(f"  Dataset:    {VALIDATION_DATASET_PATH}")
    print(f"{'='*80}\n")

    # Create batch folder
    batch_folder = get_next_batch_folder()
    print(f"[OK] Created batch folder: {batch_folder}\n")

    # Load and sample questions
    validation_data = load_validation_dataset()
    sampled_questions = sample_questions(validation_data, n_questions, seed)

    # Create agent
    agent = KQAProAgent(name="batch_runner")

    # Create OpenAI client for answer selection
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY missing in .env")

    client = OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=OPENROUTER_API_KEY
    )

    # Process all questions
    results = []
    for i, item in enumerate(sampled_questions):
        result = await process_question(agent, item, i, len(sampled_questions), client)
        results.append(result)

    # Save results
    config = {
        "n_questions": n_questions,
        "seed": seed,
        "dataset_path": str(VALIDATION_DATASET_PATH),
        "total_available": len(validation_data)
    }

    save_batch_results(batch_folder, sampled_questions, results, config)

    print(f"\n{'='*80}")
    print(f"[OK] Batch processing complete!")
    print(f"[OK] Results saved to: {batch_folder}")
    print(f"{'='*80}\n")


# ============================================================================
# CLI ENTRY POINT
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run KQAPro benchmark on a sample of validation questions"
    )
    parser.add_argument(
        "--n_questions",
        type=int,
        default=10,
        help="Number of questions to sample (default: 10)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )

    args = parser.parse_args()

    # Run the batch
    asyncio.run(run_batch(n_questions=args.n_questions, seed=args.seed))


if __name__ == "__main__":
    main()
