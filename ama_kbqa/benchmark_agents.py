"""
Multi-Model Benchmarking Script for KBQA Agents
================================================

This script benchmarks multiple LLM models against KQAPro and SciQA agents
using pre-generated questionnaires. It provides comprehensive evaluation
with leaderboards, detailed metrics, cost estimation, and LLM-as-judge
accuracy evaluation using DeepSeek v3.2.

Prerequisites
-------------
1. Questionnaire files must exist in db/:
   - db/kqapro_questionnaire.json (for KQAPro agent)
   - db/sciqa_questionnaire.json (for SciQA agent)

   Generate these using: python -m ama_kbqa.generate_questionnaire

2. Required environment variables in .env:
   - OPENROUTER_API_KEY (for all models via OpenRouter)

3. Databases must be running:
   - Qdrant (vector database)
   - Virtuoso (SPARQL endpoint)

Available Models
----------------
The following models are benchmarked by default (all via OpenRouter):

  Name                  Model ID
  ----                  --------
  minimax-m2.1          minimax/minimax-m2.1
  glm-4.7               zhipu-ai/glm-4.7
  kimi-k2.5             moonshotai/kimi-k2.5
  deepseek-v3.2         deepseek/deepseek-chat-v3-0324
  gpt-oss-120b          openai/gpt-oss-120b:nitro
  qwen3-32b             qwen/qwen3-32b:nitro
  nemotron-3-nano-30b   nvidia/nemotron-3-nano-30b-a3b:nitro

Available Agents
----------------
  kqapro    KQAPro knowledge base (Wikidata-derived)
  sciqa     SciQA scientific knowledge base

CLI Arguments
-------------
  --agents AGENTS       Which agents to test: kqapro, sciqa, or both (default: both)
  --models MODELS       Filter to specific models by name (default: all)
  --n-questions N       Limit questions per agent (default: all questions)
  --output-dir DIR      Custom output directory (default: benchmark_results/<timestamp>)
  --timeout SECONDS     Timeout per question in seconds (default: 300)
  --resume              Skip model/agent combinations that already have results
  --dry-run             Preview what would run without executing
  --export-csv          Export results to CSV file

Output Structure
----------------
Results are saved to benchmark_results/<timestamp>/ with the following structure:

  benchmark_results/<timestamp>/
  ├── overview.json              # Leaderboards and cross-model comparison
  ├── benchmark_results.csv      # CSV export (if --export-csv)
  ├── kqapro/
  │   ├── minimax-m2.1/
  │   │   ├── results.json       # Per-question results
  │   │   ├── summary.json       # Aggregate statistics
  │   │   └── console_output.txt # Full console log
  │   ├── deepseek-v3.2/
  │   │   └── ...
  │   └── ...
  └── sciqa/
      └── ...

Summary JSON Fields
-------------------
Each summary.json contains:
  - total_questions: Number of questions processed
  - correct: Number of correct answers
  - errors: Number of errors/timeouts
  - accuracy: Correct / Total ratio
  - total_time_seconds: Total processing time
  - avg_time_seconds: Average time per question
  - total_tokens: Total tokens used
  - avg_tokens: Average tokens per question
  - prompt_tokens: Total input tokens
  - completion_tokens: Total output tokens
  - estimated_cost_usd: Estimated API cost
  - accuracy_by_type: Breakdown by question type

Usage Examples
--------------
# Show help and all options
python -m ama_kbqa.benchmark_agents --help

# Preview what would run (no execution)
python -m ama_kbqa.benchmark_agents --dry-run

# Quick test: single model, single agent, 3 questions
python -m ama_kbqa.benchmark_agents --models minimax-m2.1 --agents kqapro --n-questions 3

# Test multiple specific models
python -m ama_kbqa.benchmark_agents --models deepseek-v3.2 qwen3-32b --agents kqapro

# Full benchmark on KQAPro only with CSV export
python -m ama_kbqa.benchmark_agents --agents kqapro --export-csv

# Resume an interrupted benchmark (skips completed runs)
python -m ama_kbqa.benchmark_agents --resume

# Resume into a specific output directory
python -m ama_kbqa.benchmark_agents --resume --output-dir benchmark_results/2024-01-15_10-30-00

# Run with longer timeout for complex questions
python -m ama_kbqa.benchmark_agents --timeout 600

# Full benchmark: all models, all agents, with CSV
python -m ama_kbqa.benchmark_agents --export-csv
"""

from __future__ import annotations
import argparse
import asyncio
import csv
import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import AsyncOpenAI
from tqdm import tqdm

# Load environment variables
load_dotenv(override=True)

# Project root for file paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class ModelConfig:
    """Configuration for an LLM model to benchmark."""
    name: str              # Human-readable name (e.g., "minimax-m2.1")
    provider: str          # Provider type (e.g., "openrouter")
    model_id: str          # Full model ID (e.g., "minimax/minimax-m2.1")
    base_url: str          # API endpoint
    api_key_env: str       # Environment variable name for API key


@dataclass
class QuestionResult:
    """Result of processing a single question."""
    question_id: int
    question: str
    gold_answer: str
    predicted_answer: Optional[str]
    accuracy: bool
    elapsed_time: float
    token_usage: Dict[str, int] = field(default_factory=dict)
    tool_summary: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    q_type: str = "Unknown"


# ============================================================================
# MODEL CONFIGURATIONS
# ============================================================================

BENCHMARK_MODELS: List[ModelConfig] = [
    ModelConfig(
        name="minimax-m2.1",
        provider="openrouter",
        model_id="minimax/minimax-m2.1",
        base_url="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY"
    ),
    ModelConfig(
        name="glm-4.7",
        provider="openrouter",
        model_id="zhipu-ai/glm-4.7",
        base_url="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY"
    ),
    ModelConfig(
        name="kimi-k2.5",
        provider="openrouter",
        model_id="moonshotai/kimi-k2.5",
        base_url="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY"
    ),
    ModelConfig(
        name="deepseek-v3.2",
        provider="openrouter",
        model_id="deepseek/deepseek-chat-v3-0324",
        base_url="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY"
    ),
    ModelConfig(
        name="gpt-oss-120b",
        provider="openrouter",
        model_id="openai/gpt-oss-120b:nitro",
        base_url="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY"
    ),
    ModelConfig(
        name="qwen3-32b",
        provider="openrouter",
        model_id="qwen/qwen3-32b:nitro",
        base_url="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY"
    ),
    ModelConfig(
        name="nemotron-3-nano-30b",
        provider="openrouter",
        model_id="nvidia/nemotron-3-nano-30b-a3b:nitro",
        base_url="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY"
    ),
]

# LLM Judge configuration (DeepSeek v3.2)
JUDGE_MODEL_CONFIG = ModelConfig(
    name="deepseek-v3.2-judge",
    provider="openrouter",
    model_id="deepseek/deepseek-chat-v3-0324",
    base_url="https://openrouter.ai/api/v1",
    api_key_env="OPENROUTER_API_KEY"
)

# Approximate costs per 1M tokens (input/output) for cost estimation
# These are estimates and may change - update as needed
MODEL_COSTS = {
    "minimax-m2.1": {"input": 0.5, "output": 1.5},
    "glm-4.7": {"input": 0.5, "output": 1.5},
    "kimi-k2.5": {"input": 0.5, "output": 1.5},
    "deepseek-v3.2": {"input": 0.27, "output": 1.10},
    "gpt-oss-120b": {"input": 2.0, "output": 8.0},
    "qwen3-32b": {"input": 0.12, "output": 0.30},
    "nemotron-3-nano-30b": {"input": 0.10, "output": 0.20},
}


# ============================================================================
# DUAL OUTPUT LOGGER (from batch_runner.py)
# ============================================================================

class DualOutputLogger:
    """Captures console output to both stdout and a file buffer."""

    def __init__(self):
        self.terminal = sys.stdout
        self.log_buffer = StringIO()
        # Support for tqdm compatibility
        self.encoding = getattr(sys.stdout, 'encoding', 'utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log_buffer.write(message)

    def flush(self):
        self.terminal.flush()

    def get_log(self) -> str:
        return self.log_buffer.getvalue()

    def clear(self):
        self.log_buffer = StringIO()

    def isatty(self) -> bool:
        """Return whether the underlying terminal is a tty."""
        return hasattr(self.terminal, 'isatty') and self.terminal.isatty()

    def fileno(self):
        """Return the file descriptor of the underlying terminal."""
        return self.terminal.fileno()


# ============================================================================
# CONFIG OVERRIDE
# ============================================================================

def override_model_config(model: ModelConfig) -> None:
    """
    Override the config cache at runtime to switch models.

    This modifies the cached configuration so that subsequent calls
    to get_chat_client() and get_chat_model_name() use the new model.

    Args:
        model: The model configuration to switch to
    """
    import ama_kbqa.config as config_module

    # Clear the cache to force reload
    config_module._config_cache = None

    # Load fresh config
    config = config_module.load_config()

    # Override LLM settings
    config["llm"]["chat_provider"] = model.provider

    # Ensure the provider section exists
    if model.provider not in config:
        config[model.provider] = {}

    # Set the model-specific settings
    config[model.provider]["base_url"] = model.base_url
    config[model.provider]["chat_model"] = model.model_id

    # Clear any provider preference to avoid routing issues
    if "chat_model_provider" in config.get(model.provider, {}):
        del config[model.provider]["chat_model_provider"]

    # Update the cache
    config_module._config_cache = config


def check_api_key(model: ModelConfig) -> bool:
    """Check if the API key for a model is available."""
    api_key = os.getenv(model.api_key_env)
    return api_key is not None and len(api_key) > 0


# ============================================================================
# QUESTIONNAIRE LOADING
# ============================================================================

def load_kqapro_questionnaire(
    path: Path,
    n_questions: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Load questions from the KQAPro questionnaire.

    Args:
        path: Path to the questionnaire JSON file
        n_questions: Optional limit on number of questions

    Returns:
        List of question dictionaries
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    questions = data.get("questions", [])

    if n_questions and n_questions < len(questions):
        questions = questions[:n_questions]

    return questions


def load_sciqa_questionnaire(
    path: Path,
    n_questions: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Load questions from the SciQA questionnaire.

    Args:
        path: Path to the questionnaire JSON file
        n_questions: Optional limit on number of questions

    Returns:
        List of question dictionaries
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    questions = data.get("questions", [])

    if n_questions and n_questions < len(questions):
        questions = questions[:n_questions]

    return questions


def get_question_type(question: Dict[str, Any], agent_name: str) -> str:
    """Extract question type from a question dict."""
    if agent_name == "kqapro":
        # KQAPro: Get from program[0]["function"] or infer from question
        program = question.get("program", [])
        if program and len(program) > 0:
            last_func = program[-1].get("function", "")
            if last_func in ["Count"]:
                return "Count"
            elif last_func in ["VerifyQuery", "Verify", "VerifyStr", "VerifyNum", "VerifyDate", "VerifyYear"]:
                return "Verify"
            elif last_func in ["SelectBetween", "SelectAmong"]:
                return "Select"
            elif last_func == "QueryAttr":
                return "Query"
            elif last_func == "QueryRelation":
                return "QueryRelation"
            elif last_func == "QueryAttrQualifier":
                return "QueryAttrQualifier"
            elif last_func == "QueryRelationQualifier":
                return "QueryRelationQualifier"
        return "Query"
    else:
        # SciQA: Get from q_type field
        return question.get("q_type", "General")


# ============================================================================
# ANSWER EVALUATION
# ============================================================================

def normalize_answer(answer: str) -> str:
    """Normalize an answer for comparison."""
    if answer is None:
        return ""
    return str(answer).lower().strip()


def evaluate_accuracy_string(predicted: Optional[str], gold: str, q_type: str = "") -> bool:
    """
    Evaluate if predicted answer matches gold answer using string matching.

    This is the fallback method when LLM judge is unavailable.

    Args:
        predicted: The predicted answer
        gold: The gold/correct answer
        q_type: Question type for special handling

    Returns:
        True if answers match (accounting for normalization)
    """
    if predicted is None:
        return False

    pred_norm = normalize_answer(predicted)
    gold_norm = normalize_answer(gold)

    # Exact match
    if pred_norm == gold_norm:
        return True

    # Substring match (for verbose answers)
    if gold_norm in pred_norm:
        return True

    # For yes/no questions
    if gold_norm in ["yes", "no", "true", "false"]:
        # Map true/false to yes/no
        pred_mapped = pred_norm.replace("true", "yes").replace("false", "no")
        gold_mapped = gold_norm.replace("true", "yes").replace("false", "no")
        if pred_mapped == gold_mapped:
            return True
        # Check if answer starts with yes/no
        if pred_norm.startswith(gold_norm):
            return True

    # For count questions, try numeric comparison
    if q_type.lower() == "count":
        try:
            pred_num = int(pred_norm.split()[0])
            gold_num = int(gold_norm)
            return pred_num == gold_num
        except (ValueError, IndexError):
            pass

    return False


async def evaluate_accuracy_llm(
    question: str,
    predicted: Optional[str],
    gold: str,
    q_type: str = ""
) -> bool:
    """
    Evaluate if predicted answer matches gold answer using LLM as a judge.

    Uses DeepSeek v3.2 as the judge model. Token usage from the judge
    is NOT counted towards the benchmarked model's token count.

    Args:
        question: The original question
        predicted: The predicted answer
        gold: The gold/correct answer
        q_type: Question type for context

    Returns:
        True if the judge determines answers are semantically equivalent
    """
    if predicted is None:
        return False

    # Get API key for judge
    api_key = os.getenv(JUDGE_MODEL_CONFIG.api_key_env)
    if not api_key:
        # Fall back to string matching if no API key
        print("  [Judge] No API key, falling back to string matching")
        return evaluate_accuracy_string(predicted, gold, q_type)

    client = AsyncOpenAI(
        api_key=api_key,
        base_url=JUDGE_MODEL_CONFIG.base_url
    )

    prompt = f"""You are a strict judge evaluating answers to knowledge base questions.
Your task is to determine if the predicted answer correctly answers the question.

Question Type: {q_type}
Question: {question}
Gold Answer: {gold}
Predicted Answer: {predicted}

EVALUATION RULES (apply strictly):

1. INCORRECT if the predicted answer:
   - Says "I don't know", "unable to find", "no results", or similar failure phrases
   - Provides a different entity, date, number, or fact than the gold answer
   - Is empty, None, or contains only filler text
   - Asks a clarifying question instead of answering

2. CORRECT only if the predicted answer:
   - Provides the same factual information as the gold answer
   - For yes/no questions: "true"="yes", "false"="no" (must match the gold)
   - For counts: the number must match exactly
   - For entities: must refer to the same entity (alternate names OK, e.g., "USA"="United States")
   - For dates: must match (different formats OK, e.g., "1990-01-15"="January 15, 1990")

3. BE STRICT: When in doubt, mark as INCORRECT. The predicted answer must clearly and directly answer the question with the correct information.

Respond with JSON: {{"correct": true}} or {{"correct": false}}"""

    async def call_judge(use_json_mode: bool) -> Optional[bool]:
        """Call the judge API with optional JSON mode."""
        kwargs = {
            "model": JUDGE_MODEL_CONFIG.model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 20,
            "temperature": 0
        }
        if use_json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        response = await client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content.strip()

        # Try to parse as JSON first
        try:
            result = json.loads(content)
            if isinstance(result.get("correct"), bool):
                return result["correct"]
        except json.JSONDecodeError:
            pass

        # Fall back to text parsing
        content_upper = content.upper()
        if content_upper.startswith("CORRECT") or '"correct": true' in content.lower() or '"correct":true' in content.lower():
            return True
        if content_upper.startswith("INCORRECT") or '"correct": false' in content.lower() or '"correct":false' in content.lower():
            return False

        return None  # Could not parse

    try:
        # Try with JSON mode first (most reliable)
        result = await call_judge(use_json_mode=True)
        if result is not None:
            return result

        # If JSON mode returned unparseable result, try without
        result = await call_judge(use_json_mode=False)
        if result is not None:
            return result

        # Could not parse response, default to string matching
        print("  [Judge] Could not parse response, falling back to string matching")
        return evaluate_accuracy_string(predicted, gold, q_type)

    except Exception as e:
        error_msg = str(e).lower()
        # If JSON mode not supported, retry without it
        if "json" in error_msg or "response_format" in error_msg:
            try:
                result = await call_judge(use_json_mode=False)
                if result is not None:
                    return result
            except Exception as e2:
                print(f"  [Judge Error: {e2}] Falling back to string matching")
                return evaluate_accuracy_string(predicted, gold, q_type)

        print(f"  [Judge Error: {e}] Falling back to string matching")
        return evaluate_accuracy_string(predicted, gold, q_type)


# ============================================================================
# COST ESTIMATION
# ============================================================================

def estimate_cost(
    model_name: str,
    prompt_tokens: int,
    completion_tokens: int
) -> float:
    """
    Estimate the cost of API usage.

    Args:
        model_name: Name of the model
        prompt_tokens: Number of input tokens
        completion_tokens: Number of output tokens

    Returns:
        Estimated cost in USD
    """
    costs = MODEL_COSTS.get(model_name, {"input": 1.0, "output": 3.0})

    input_cost = (prompt_tokens / 1_000_000) * costs["input"]
    output_cost = (completion_tokens / 1_000_000) * costs["output"]

    return input_cost + output_cost


# ============================================================================
# AGENT CREATION
# ============================================================================

def create_agent(agent_name: str):
    """
    Create a fresh agent instance.

    Args:
        agent_name: "kqapro" or "sciqa"

    Returns:
        Agent instance
    """
    if agent_name == "kqapro":
        from ama_kbqa.agents.kqapro_agent.agent import KQAProAgent
        return KQAProAgent(name="benchmark_kqapro")
    elif agent_name == "sciqa":
        from ama_kbqa.agents.sciqa_agent.agent import SciQAAgent
        return SciQAAgent(name="benchmark_sciqa")
    else:
        raise ValueError(f"Unknown agent: {agent_name}")


# ============================================================================
# BENCHMARK RUNNER
# ============================================================================

async def process_single_question(
    agent,
    question: Dict[str, Any],
    agent_name: str,
    timeout: int
) -> QuestionResult:
    """
    Process a single question through the agent.

    Args:
        agent: The KBQA agent instance
        question: Question dictionary
        agent_name: Name of the agent ("kqapro" or "sciqa")
        timeout: Timeout in seconds

    Returns:
        QuestionResult with the evaluation
    """
    q_text = question.get("question", "")
    gold_answer = question.get("answer", "")
    q_id = question.get("id", 0)
    q_type = get_question_type(question, agent_name)

    start_time = time.time()
    predicted_answer = None
    error = None

    try:
        # Process with timeout
        predicted_answer = await asyncio.wait_for(
            agent.ask(q_text),
            timeout=timeout
        )
    except asyncio.TimeoutError:
        error = f"Timeout after {timeout}s"
    except Exception as e:
        error = str(e)

    elapsed_time = time.time() - start_time

    # Get token usage
    token_usage = {
        "prompt_tokens": agent.token_usage.get("prompt_tokens", 0),
        "completion_tokens": agent.token_usage.get("completion_tokens", 0),
        "total_tokens": agent.token_usage.get("total_tokens", 0)
    }

    # Get tool summary
    tool_summary = agent.get_tool_call_summary()

    # Evaluate accuracy using LLM judge (tokens not counted towards model)
    accuracy = await evaluate_accuracy_llm(q_text, predicted_answer, gold_answer, q_type)

    # Soft reset for next question
    await agent.soft_reset()

    return QuestionResult(
        question_id=q_id,
        question=q_text,
        gold_answer=gold_answer,
        predicted_answer=predicted_answer,
        accuracy=accuracy,
        elapsed_time=elapsed_time,
        token_usage=token_usage,
        tool_summary=tool_summary,
        error=error,
        q_type=q_type
    )


def save_results_to_disk(
    results: List[QuestionResult],
    model: ModelConfig,
    agent_name: str,
    result_dir: Path,
    console_log: StringIO,
    is_complete: bool = False
) -> Dict[str, Any]:
    """
    Save current results to disk (intermediate or final).

    Args:
        results: List of question results so far
        model: Model configuration
        agent_name: Agent name
        result_dir: Directory to save to
        console_log: Console log buffer
        is_complete: Whether this is the final save

    Returns:
        Summary dictionary
    """
    if not results:
        return {}

    # Calculate summary statistics
    total = len(results)
    correct = sum(1 for r in results if r.accuracy)
    errors = sum(1 for r in results if r.error)

    total_time = sum(r.elapsed_time for r in results)
    avg_time = total_time / total if total > 0 else 0

    total_prompt_tokens = sum(r.token_usage.get("prompt_tokens", 0) for r in results)
    total_completion_tokens = sum(r.token_usage.get("completion_tokens", 0) for r in results)
    total_tokens = total_prompt_tokens + total_completion_tokens
    avg_tokens = total_tokens / total if total > 0 else 0

    estimated_cost = estimate_cost(model.name, total_prompt_tokens, total_completion_tokens)

    # Accuracy by question type
    type_stats = {}
    for r in results:
        if r.q_type not in type_stats:
            type_stats[r.q_type] = {"total": 0, "correct": 0}
        type_stats[r.q_type]["total"] += 1
        if r.accuracy:
            type_stats[r.q_type]["correct"] += 1

    type_accuracy = {
        k: v["correct"] / v["total"] if v["total"] > 0 else 0
        for k, v in type_stats.items()
    }

    summary = {
        "model": model.name,
        "agent": agent_name,
        "timestamp": datetime.now().isoformat(),
        "is_complete": is_complete,
        "statistics": {
            "total_questions": total,
            "correct": correct,
            "errors": errors,
            "accuracy": correct / total if total > 0 else 0,
            "total_time_seconds": round(total_time, 2),
            "avg_time_seconds": round(avg_time, 2),
            "total_tokens": total_tokens,
            "avg_tokens": round(avg_tokens, 2),
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "estimated_cost_usd": round(estimated_cost, 4),
            "accuracy_by_type": type_accuracy
        }
    }

    # Save results
    results_data = [
        {
            "question_id": r.question_id,
            "question": r.question,
            "gold_answer": r.gold_answer,
            "predicted_answer": r.predicted_answer,
            "accuracy": r.accuracy,
            "elapsed_time": round(r.elapsed_time, 2),
            "token_usage": r.token_usage,
            "tool_summary": r.tool_summary,
            "error": r.error,
            "q_type": r.q_type
        }
        for r in results
    ]

    with open(result_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)

    with open(result_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Save console output
    with open(result_dir / "console_output.txt", "w", encoding="utf-8") as f:
        f.write(console_log.getvalue())

    return summary


async def run_benchmark_for_model_agent(
    model: ModelConfig,
    agent_name: str,
    questions: List[Dict[str, Any]],
    timeout: int,
    output_dir: Path
) -> Dict[str, Any]:
    """
    Run benchmark for a specific model/agent combination.

    Args:
        model: Model configuration
        agent_name: Agent name ("kqapro" or "sciqa")
        questions: List of questions to process
        timeout: Timeout per question
        output_dir: Directory to save results

    Returns:
        Summary dictionary with results
    """
    # Create output directory
    result_dir = output_dir / agent_name / model.name
    result_dir.mkdir(parents=True, exist_ok=True)

    # Set up log capture (we'll capture via StringIO, not stdout redirect)
    console_log = StringIO()

    def log_print(msg: str):
        """Print to both console and log buffer."""
        print(msg)
        console_log.write(msg + "\n")

    log_print(f"\n{'='*80}")
    log_print(f"Benchmarking: {model.name} on {agent_name}")
    log_print(f"Questions: {len(questions)}")
    log_print(f"Timeout: {timeout}s per question")
    log_print(f"{'='*80}\n")

    # Override config for this model
    override_model_config(model)

    # Create fresh agent
    agent = create_agent(agent_name)

    results: List[QuestionResult] = []
    summary = {}

    try:
        # Process questions with progress bar
        pbar = tqdm(questions, desc=f"{model.name}/{agent_name}", unit="q")

        for question in pbar:
            result = await process_single_question(
                agent=agent,
                question=question,
                agent_name=agent_name,
                timeout=timeout
            )
            results.append(result)

            # Update progress bar with current accuracy
            correct = sum(1 for r in results if r.accuracy)
            pbar.set_postfix({
                "acc": f"{correct}/{len(results)}",
                "time": f"{result.elapsed_time:.1f}s"
            })

            # Log result
            status = "CORRECT" if result.accuracy else "INCORRECT"
            if result.error:
                status = f"ERROR: {result.error[:50]}"
            log_print(f"  [{result.question_id}] {status} | {result.elapsed_time:.1f}s")
            log_print(f"      Gold: {result.gold_answer}")
            log_print(f"      Pred: {result.predicted_answer}")

            # Save intermediate results after each question
            summary = save_results_to_disk(
                results=results,
                model=model,
                agent_name=agent_name,
                result_dir=result_dir,
                console_log=console_log,
                is_complete=False
            )

        pbar.close()

    finally:
        # Always close the agent
        await agent.close()

        # Save final results (even if interrupted)
        if results:
            summary = save_results_to_disk(
                results=results,
                model=model,
                agent_name=agent_name,
                result_dir=result_dir,
                console_log=console_log,
                is_complete=True
            )

    # Print completion message
    if results and summary:
        stats = summary.get("statistics", {})
        total = stats.get("total_questions", len(results))
        correct = stats.get("correct", 0)
        accuracy_pct = stats.get("accuracy", 0) * 100
        print(f"\nCompleted {model.name}/{agent_name}: {correct}/{total} ({accuracy_pct:.1f}%)")

    return summary


def is_run_completed(output_dir: Path, agent_name: str, model_name: str) -> bool:
    """Check if a benchmark run has already been completed."""
    result_dir = output_dir / agent_name / model_name
    summary_file = result_dir / "summary.json"
    return summary_file.exists()


async def run_full_benchmark(
    models: List[ModelConfig],
    agents: List[str],
    n_questions: Optional[int],
    output_dir: Path,
    timeout: int,
    resume: bool,
    dry_run: bool,
    export_csv: bool
):
    """
    Run the full benchmark across all models and agents.

    Args:
        models: List of model configurations
        agents: List of agent names to test
        n_questions: Optional limit on questions per agent
        output_dir: Base output directory
        timeout: Timeout per question
        resume: Skip completed runs
        dry_run: Preview without executing
        export_csv: Export results to CSV
    """
    # Load questionnaires
    questionnaires = {}

    if "kqapro" in agents:
        kqapro_path = PROJECT_ROOT / "db" / "kqapro_questionnaire.json"
        if kqapro_path.exists():
            questionnaires["kqapro"] = load_kqapro_questionnaire(kqapro_path, n_questions)
            print(f"Loaded {len(questionnaires['kqapro'])} KQAPro questions")
        else:
            print(f"WARNING: KQAPro questionnaire not found at {kqapro_path}")
            agents.remove("kqapro")

    if "sciqa" in agents:
        sciqa_path = PROJECT_ROOT / "db" / "sciqa_questionnaire.json"
        if sciqa_path.exists():
            questionnaires["sciqa"] = load_sciqa_questionnaire(sciqa_path, n_questions)
            print(f"Loaded {len(questionnaires['sciqa'])} SciQA questions")
        else:
            print(f"WARNING: SciQA questionnaire not found at {sciqa_path}")
            agents.remove("sciqa")

    if not agents:
        print("ERROR: No agents available to benchmark")
        return

    # Build list of runs
    runs = []
    for model in models:
        # Check API key
        if not check_api_key(model):
            print(f"WARNING: Skipping {model.name} - API key {model.api_key_env} not set")
            continue

        for agent_name in agents:
            if agent_name not in questionnaires:
                continue

            # Check if already completed (resume mode)
            if resume and is_run_completed(output_dir, agent_name, model.name):
                print(f"SKIP: {model.name}/{agent_name} (already completed)")
                continue

            runs.append((model, agent_name, questionnaires[agent_name]))

    # Dry run mode
    if dry_run:
        print(f"\n{'='*80}")
        print("DRY RUN - The following benchmarks would be executed:")
        print(f"{'='*80}\n")

        for model, agent_name, questions in runs:
            print(f"  - {model.name} on {agent_name}: {len(questions)} questions")

        print(f"\nTotal runs: {len(runs)}")
        print(f"Output directory: {output_dir}")
        return

    # Execute benchmarks
    print(f"\n{'='*80}")
    print(f"Starting benchmark: {len(runs)} runs")
    print(f"Output directory: {output_dir}")
    print(f"{'='*80}\n")

    all_summaries = []

    for i, (model, agent_name, questions) in enumerate(runs):
        print(f"\n[{i+1}/{len(runs)}] {model.name} on {agent_name}")

        try:
            summary = await run_benchmark_for_model_agent(
                model=model,
                agent_name=agent_name,
                questions=questions,
                timeout=timeout,
                output_dir=output_dir
            )
            all_summaries.append(summary)
        except Exception as e:
            print(f"ERROR: {model.name}/{agent_name} failed: {e}")
            all_summaries.append({
                "model": model.name,
                "agent": agent_name,
                "error": str(e)
            })

    # Generate overview
    generate_overview(all_summaries, output_dir, export_csv)


def generate_overview(
    summaries: List[Dict[str, Any]],
    output_dir: Path,
    export_csv: bool
):
    """
    Generate the overview.json with leaderboards.

    Args:
        summaries: List of summary dictionaries
        output_dir: Output directory
        export_csv: Whether to also export to CSV
    """
    # Filter out failed runs
    valid_summaries = [s for s in summaries if "error" not in s]

    if not valid_summaries:
        print("No valid results to generate overview")
        return

    # Build leaderboard by accuracy
    by_accuracy = sorted(
        valid_summaries,
        key=lambda x: x["statistics"]["accuracy"],
        reverse=True
    )

    # Build leaderboard by speed (avg time per question)
    by_speed = sorted(
        valid_summaries,
        key=lambda x: x["statistics"]["avg_time_seconds"]
    )

    # Build leaderboard by efficiency (accuracy per dollar)
    def efficiency_score(s):
        cost = s["statistics"]["estimated_cost_usd"]
        acc = s["statistics"]["accuracy"]
        if cost == 0:
            return float('inf') if acc > 0 else 0
        return acc / cost

    by_efficiency = sorted(
        valid_summaries,
        key=efficiency_score,
        reverse=True
    )

    # Build results matrix
    results_matrix = {}
    for s in valid_summaries:
        agent = s["agent"]
        model = s["model"]

        if agent not in results_matrix:
            results_matrix[agent] = {}

        results_matrix[agent][model] = {
            "accuracy": s["statistics"]["accuracy"],
            "avg_time": s["statistics"]["avg_time_seconds"],
            "total_tokens": s["statistics"]["total_tokens"],
            "estimated_cost_usd": s["statistics"]["estimated_cost_usd"]
        }

    overview = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "models_tested": len(set(s["model"] for s in valid_summaries)),
            "agents_tested": len(set(s["agent"] for s in valid_summaries)),
            "total_runs": len(valid_summaries)
        },
        "leaderboard": {
            "by_accuracy": [
                {
                    "rank": i + 1,
                    "model": s["model"],
                    "agent": s["agent"],
                    "accuracy": round(s["statistics"]["accuracy"], 4)
                }
                for i, s in enumerate(by_accuracy)
            ],
            "by_speed": [
                {
                    "rank": i + 1,
                    "model": s["model"],
                    "agent": s["agent"],
                    "avg_time_seconds": round(s["statistics"]["avg_time_seconds"], 2)
                }
                for i, s in enumerate(by_speed)
            ],
            "by_efficiency": [
                {
                    "rank": i + 1,
                    "model": s["model"],
                    "agent": s["agent"],
                    "accuracy_per_dollar": round(efficiency_score(s), 4)
                }
                for i, s in enumerate(by_efficiency)
            ]
        },
        "results_matrix": results_matrix
    }

    with open(output_dir / "overview.json", "w", encoding="utf-8") as f:
        json.dump(overview, f, indent=2, ensure_ascii=False)

    print(f"\nOverview saved to: {output_dir / 'overview.json'}")

    # Print leaderboard
    print(f"\n{'='*80}")
    print("LEADERBOARD - By Accuracy")
    print(f"{'='*80}")
    for entry in overview["leaderboard"]["by_accuracy"][:10]:
        print(f"  {entry['rank']}. {entry['model']}/{entry['agent']}: {entry['accuracy']*100:.1f}%")

    # Export to CSV if requested
    if export_csv:
        export_results_to_csv(valid_summaries, output_dir)


def export_results_to_csv(
    summaries: List[Dict[str, Any]],
    output_dir: Path
):
    """Export results to CSV files."""
    # Main results CSV
    csv_path = output_dir / "benchmark_results.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Model", "Agent", "Accuracy", "Correct", "Total",
            "Avg Time (s)", "Total Tokens", "Est. Cost ($)"
        ])

        for s in summaries:
            stats = s["statistics"]
            writer.writerow([
                s["model"],
                s["agent"],
                f"{stats['accuracy']*100:.1f}%",
                stats["correct"],
                stats["total_questions"],
                f"{stats['avg_time_seconds']:.2f}",
                stats["total_tokens"],
                f"${stats['estimated_cost_usd']:.4f}"
            ])

    print(f"CSV exported to: {csv_path}")


# ============================================================================
# CLI
# ============================================================================

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Benchmark multiple LLM models against KBQA agents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available Models:
  minimax-m2.1, glm-4.7, kimi-k2.5, deepseek-v3.2, gpt-oss-120b,
  qwen3-32b, nemotron-3-nano-30b

Examples:
  # Preview what would run (no execution)
  python -m ama_kbqa.benchmark_agents --dry-run

  # Quick test: single model, single agent, 3 questions
  python -m ama_kbqa.benchmark_agents --models minimax-m2.1 --agents kqapro --n-questions 3

  # Test multiple specific models on KQAPro
  python -m ama_kbqa.benchmark_agents --models deepseek-v3.2 qwen3-32b --agents kqapro

  # Full benchmark with CSV export
  python -m ama_kbqa.benchmark_agents --export-csv

  # Resume an interrupted benchmark
  python -m ama_kbqa.benchmark_agents --resume

  # Resume into specific output directory
  python -m ama_kbqa.benchmark_agents --resume --output-dir benchmark_results/2024-01-15_10-30-00

Output:
  Results saved to benchmark_results/<timestamp>/ containing:
  - overview.json: Leaderboards by accuracy, speed, efficiency
  - <agent>/<model>/results.json: Per-question results with gold/predicted answers
  - <agent>/<model>/summary.json: Aggregate statistics and token usage
  - benchmark_results.csv: CSV export (with --export-csv)

Notes:
  - Requires OPENROUTER_API_KEY in .env
  - Questionnaires must exist in db/ (generate with generate_questionnaire.py)
  - Uses DeepSeek v3.2 as LLM judge for semantic answer evaluation
"""
    )

    parser.add_argument(
        "--agents",
        nargs="+",
        choices=["kqapro", "sciqa"],
        default=["kqapro", "sciqa"],
        help="Which agents to test (default: both)"
    )

    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Filter to specific models by name (default: all)"
    )

    parser.add_argument(
        "--n-questions",
        type=int,
        default=None,
        help="Limit questions per agent (default: all)"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: benchmark_results/<timestamp>)"
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Timeout per question in seconds (default: 300)"
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip completed model/agent combinations"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview what would run without executing"
    )

    parser.add_argument(
        "--export-csv",
        action="store_true",
        help="Export results to CSV"
    )

    args = parser.parse_args()

    # Filter models if specified
    if args.models:
        models = [m for m in BENCHMARK_MODELS if m.name in args.models]
        if not models:
            print(f"ERROR: No matching models found. Available: {[m.name for m in BENCHMARK_MODELS]}")
            sys.exit(1)
    else:
        models = BENCHMARK_MODELS

    # Set up output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = PROJECT_ROOT / "benchmark_results" / timestamp

    output_dir.mkdir(parents=True, exist_ok=True)

    # Run the benchmark
    asyncio.run(run_full_benchmark(
        models=models,
        agents=args.agents,
        n_questions=args.n_questions,
        output_dir=output_dir,
        timeout=args.timeout,
        resume=args.resume,
        dry_run=args.dry_run,
        export_csv=args.export_csv
    ))


if __name__ == "__main__":
    main()
