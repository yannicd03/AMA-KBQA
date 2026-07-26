"""
Unified Benchmarking Script for KBQA Agents
=============================================

Supports both single-model and multi-model benchmarking with rich postprocessing.

Modes
-----
- **Single-model mode**: Run one agent with the model from config.toml.
  Uses --seed/--questionnaire for question source, --postprocessing for eval.
- **Multi-model mode**: Supply --models to iterate over several LLMs.
  Uses questionnaires by default for cross-model comparability.

CLI Arguments
-------------
  --agents AGENTS         Which agents to test: kqapro, sciqa, or both (default: both)
  --models MODELS         Filter to specific models by name (default: config model)
  --n-questions N         Limit questions per agent (default: all)
  --output-dir DIR        Custom output directory (default: YYYY-MM-DD-N)
  --timeout SECONDS       Timeout per question (default: 300)
  --resume                Skip completed runs
  --dry-run               Preview what would run
  --export-csv            Export results to CSV
  --no-fewshot            Disable few-shot examples
  --postprocessing / -p   choice|sparql|llm_judge|simple (default: llm_judge)
  --seed                  Random seed for on-the-fly sampling
  --questionnaire         Path to pre-generated questionnaire JSON
  --dataset / -d          SciQA dataset type: handcrafted|auto (default: handcrafted)
  --stratified            Stratified sampling by question type

Output Structure
----------------
  benchmark_results/<YYYY-MM-DD-N>/
    overview.json                  # Multi-model leaderboard (multi-model only)
    benchmark_results.csv          # CSV export (if --export-csv)
    kqapro/
      minimax-m2.1/                # or "default" for single-model mode
        results.json
        summary.json
        console_output.txt
        detailed_log.txt
        judgments.json             # if llm_judge mode
        tool_traces/               # full conversation per question
          question_000.json
          question_001.json
"""

from __future__ import annotations
import argparse
import asyncio
import copy
import csv as csv_module
import json
import os
import random
import re
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

from ama_kbqa.postprocessing import PostProcessor, PostProcessingResult
from ama_kbqa.utils.trace_utils import (
    extract_tool_trace,
    export_fewshot_examples_from_traces,
)

# Load environment variables
load_dotenv(override=True)

# Project root for file paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _git_commit_sha() -> Optional[str]:
    """Return short git HEAD sha, or None if unavailable.

    Resolution order (first hit wins):
      1. $GIT_COMMIT / $GIT_COMMIT_SHA — set by CI or baked into a Docker image
         at build time (the container has no .git/ so `git` below returns None).
      2. `git rev-parse --short HEAD` in PROJECT_ROOT.
    """
    for env_var in ("GIT_COMMIT", "GIT_COMMIT_SHA"):
        val = os.environ.get(env_var)
        if val:
            return val.strip()[:12] or None

    try:
        import subprocess
        out = subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=2,
        )
        if out.returncode == 0:
            return out.stdout.strip() or None
    except Exception:
        pass
    return None


def _write_run_manifest(output_dir: Path, args, resolved_models) -> None:
    """Persist a full reproducibility snapshot of this batch run.

    Writes <output_dir>/run_manifest.json containing:
      - timestamp, git SHA, python version, cwd, command line
      - parsed CLI args (argparse Namespace → dict)
      - resolved ModelConfig list (name, provider, model_id, base_url, api_key_env)
      - full config.toml contents
    """
    try:
        import platform
        import tomllib
        config_path = PROJECT_ROOT / "config.toml"
        config_snapshot: Any = None
        if config_path.exists():
            with open(config_path, "rb") as f:
                config_snapshot = tomllib.load(f)

        manifest = {
            "timestamp": datetime.now().isoformat(),
            "git_commit": _git_commit_sha(),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "cwd": str(Path.cwd()),
            "command_line": sys.argv,
            "cli_args": {k: v for k, v in vars(args).items()},
            "resolved_models": [
                {
                    "name": m.name,
                    "provider": m.provider,
                    "model_id": m.model_id,
                    "base_url": m.base_url,
                    "api_key_env": m.api_key_env,
                }
                for m in resolved_models
            ],
            "config_toml": config_snapshot,
        }

        with open(output_dir / "run_manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False, default=str)
    except Exception as e:
        # Manifest is diagnostic — never fail the run if it can't be written.
        print(f"WARNING: could not write run_manifest.json: {e}")


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
    # Extended fields from batch_runner merge
    selected_answer: Optional[str] = None
    judgment: Optional[Dict] = None
    tool_trace: List[Dict] = field(default_factory=list)
    full_messages: Optional[List[Dict]] = None
    tool_call_summary: Dict = field(default_factory=dict)
    intermediate_thinking: str = ""
    synthesized_sparql: Optional[str] = None
    postprocessing_mode: str = ""
    program: Optional[List] = None
    choices: Optional[List] = None


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
        name="minimax-m2.5",
        provider="openrouter",
        model_id="minimax/minimax-m2.5",
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
        model_id="openai/gpt-oss-120b",
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
        model_id="nvidia/nemotron-3-nano-30b-a3b",
        base_url="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY"
    ),
    ModelConfig(
        name="gpt-oss-120b-kit",
        provider="kit",
        model_id="kit.gpt-oss-120b",
        base_url="https://ki-toolbox.scc.kit.edu/api/v1",
        api_key_env="KIT_API_KEY"
    ),
    ModelConfig(
        name="qwen3-vl-235b-kit",
        provider="kit",
        model_id="kit.qwen3-vl-235b-a22b-instruct",
        base_url="https://ki-toolbox.scc.kit.edu/api/v1",
        api_key_env="KIT_API_KEY"
    ),
    ModelConfig(
        name="gpt-4.1-mini-kit",
        provider="kit",
        model_id="azure.gpt-4.1-mini",
        base_url="https://ki-toolbox.scc.kit.edu/api/v1",
        api_key_env="KIT_API_KEY"
    ),
    ModelConfig(
        name="o4-mini-kit",
        provider="kit",
        model_id="azure.o4-mini",
        base_url="https://ki-toolbox.scc.kit.edu/api/v1",
        api_key_env="KIT_API_KEY"
    ),
    ModelConfig(
        name="mixtral-8x22b-kit",
        provider="kit",
        model_id="kit.mixtral-8x22b-instruct",
        base_url="https://ki-toolbox.scc.kit.edu/api/v1",
        api_key_env="KIT_API_KEY"
    ),
    ModelConfig(
        name="minimax-m2.1-kit",
        provider="kit",
        model_id="kit.minimax-m2.1-229b",
        base_url="https://ki-toolbox.scc.kit.edu/api/v1",
        api_key_env="KIT_API_KEY"
    ),
    ModelConfig(
        name="minimax-m2.7-kit",
        provider="kit",
        model_id="kit.minimax-m2.7-229b",
        base_url="https://ki-toolbox.scc.kit.edu/api/v1",
        api_key_env="KIT_API_KEY"
    ),
    ModelConfig(
        name="gemma-4-31b-kit",
        provider="kit",
        model_id="kit.gemma4-31b-it",
        base_url="https://ki-toolbox.scc.kit.edu/api/v1",
        api_key_env="KIT_API_KEY"
    ),
    ModelConfig(
        name="qwen3.5-397b-kit",
        provider="kit",
        model_id="kit.qwen3.5-397b-A17b",
        base_url="https://ki-toolbox.scc.kit.edu/api/v1",
        api_key_env="KIT_API_KEY"
    ),
    ModelConfig(
        name="gemma-4-26b",
        provider="openrouter",
        model_id="google/gemma-4-26b-a4b-it",
        base_url="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY"
    ),
    ModelConfig(
        name="gemma-4-31b",
        provider="openrouter",
        model_id="google/gemma-4-31b-it",
        base_url="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY"
    ),
    ModelConfig(
        name="qwen3.5-35b",
        provider="openrouter",
        model_id="qwen/qwen3.5-35b-a3b",
        base_url="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY"
    ),
    ModelConfig(
        name="qwen3.5-27b",
        provider="openrouter",
        model_id="qwen/qwen3.5-27b",
        base_url="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY"
    ),
]

# LLM Judge configuration for multi-model mode (fast, cheap judge)
JUDGE_MODEL_CONFIG = ModelConfig(
    name="deepseek-v3.2-judge",
    provider="openrouter",
    model_id="deepseek/deepseek-v3.2",
    base_url="https://openrouter.ai/api/v1",
    api_key_env="OPENROUTER_API_KEY"
)

# Approximate costs per 1M tokens (input/output) for cost estimation
MODEL_COSTS = {
    "minimax-m2.1": {"input": 0.5, "output": 1.5},
    "minimax-m2.5": {"input": 0.30, "output": 1.20},
    "glm-4.7": {"input": 0.5, "output": 1.5},
    "kimi-k2.5": {"input": 0.5, "output": 1.5},
    "deepseek-v3.2": {"input": 0.27, "output": 1.10},
    "gpt-oss-120b": {"input": 2.0, "output": 8.0},
    "qwen3-32b": {"input": 0.12, "output": 0.30},
    "nemotron-3-nano-30b": {"input": 0.10, "output": 0.20},
    "gpt-oss-120b-kit": {"input": 0.0, "output": 0.0},
    "qwen3-vl-235b-kit": {"input": 0.0, "output": 0.0},
    "gpt-4.1-mini-kit": {"input": 0.0, "output": 0.0},
    "minimax-m2.7-kit": {"input": 0.0, "output": 0.0},
    "gemma-4-31b-kit": {"input": 0.0, "output": 0.0},
    "qwen3.5-397b-kit": {"input": 0.0, "output": 0.0},
}


# ============================================================================
# DUAL OUTPUT LOGGER
# ============================================================================

class DualOutputLogger:
    """Captures console output to both stdout and a file buffer."""

    def __init__(self):
        self.terminal = sys.stdout
        self.log_buffer = StringIO()
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
        return hasattr(self.terminal, 'isatty') and self.terminal.isatty()

    def fileno(self):
        return self.terminal.fileno()


# ============================================================================
# CONFIG OVERRIDE (for multi-model mode)
# ============================================================================

def override_model_config(model: ModelConfig) -> None:
    """Override the config cache at runtime to switch models."""
    import ama_kbqa.config as config_module
    config_module._config_cache = None
    config = config_module.load_config()
    config["llm"]["chat_provider"] = model.provider
    if model.provider not in config:
        config[model.provider] = {}
    config[model.provider]["base_url"] = model.base_url
    config[model.provider]["chat_model"] = model.model_id
    if "chat_model_provider" in config.get(model.provider, {}):
        del config[model.provider]["chat_model_provider"]
    config_module._config_cache = config


def check_api_key(model: ModelConfig) -> bool:
    """Check if the API key for a model is available."""
    api_key = os.getenv(model.api_key_env)
    return api_key is not None and len(api_key) > 0


# ============================================================================
# QUESTIONNAIRE / DATASET LOADING
# ============================================================================

def load_kqapro_questionnaire(
    path: Path,
    n_questions: Optional[int] = None
) -> List[Dict[str, Any]]:
    """Load questions from the KQAPro questionnaire."""
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
    """Load questions from the SciQA questionnaire."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    questions = data.get("questions", [])
    questions = [normalize_sciqa_question(q) for q in questions]
    if n_questions and n_questions < len(questions):
        questions = questions[:n_questions]
    return questions


def normalize_sciqa_question(question: Dict[str, Any]) -> Dict[str, Any]:
    """Apply deterministic SciQA question text normalizations.

    Some SciQA rows contain the literal placeholder "paper title" while another
    field carries the actual title. Replace the placeholder before the agent sees
    the question. Rows with a comparison-scoped gold query also carry a
    context-rich question variant; use it so the agent sees the Comparison title
    required by the gold SPARQL. The CSV also uses "no comparison used" as a
    sentinel, not as a question.
    """
    text = question.get("question", "")
    raw = question.get("raw") or {}
    normalized = dict(question)

    if isinstance(raw, dict):
        contextual = raw.get("Question with context (comparison)", "")
        question_without_context = raw.get("Question without context (comparison)", "")
        related_comparison = raw.get("Related to Comparison", "")
        contextual_clean = contextual.strip()
        contextual_is_usable = (
            contextual_clean
            and contextual_clean.upper() != "#N/A"
            and contextual_clean.lower() != "no comparison used"
        )
        if contextual_is_usable and related_comparison:
            contextual = contextual.strip()
            if contextual != text:
                normalized["question"] = contextual
                normalized["normalized_question"] = True
                normalized.setdefault("original_question", text)
                normalized["comparison_id_hint"] = related_comparison
                text = contextual
        elif question_without_context and (
            "paper title" in text.lower()
            or "\n" in text
            or text.lstrip().startswith("-")
        ):
            normalized["question"] = question_without_context.strip()
            normalized["normalized_question"] = True
            normalized.setdefault("original_question", text)
            text = normalized["question"]

    if "paper title" not in text.lower():
        return normalized

    sparql_query = question.get("sparql_query", "")
    if not sparql_query and isinstance(raw, dict):
        sparql_query = raw.get("Machine-readable query", "")

    title = None
    match = re.search(r'REGEX\(\s*STR\(\?paper_title\)\s*,\s*"([^"]+)"', sparql_query, flags=re.I)
    if match:
        title = match.group(1)
    elif isinstance(raw, dict):
        for key in ("Paper title", "paper title", "Title"):
            if raw.get(key):
                title = raw[key]
                break
        if not title:
            question_without_context = raw.get("Question without context (comparison)", "")
            title_match = re.search(r'"([^"]+)"', question_without_context)
            if title_match:
                title = title_match.group(1)

    if not title:
        return normalized

    normalized["question"] = re.sub(
        r'"paper title"',
        f'"{title}"',
        text,
        flags=re.I,
    )
    normalized["normalized_question"] = True
    normalized.setdefault("original_question", text)
    return normalized


def load_raw_dataset(agent_name: str, dataset_type: str = "handcrafted") -> List[Dict[str, Any]]:
    """
    Load raw dataset for on-the-fly sampling.

    Args:
        agent_name: "kqapro" or "sciqa"
        dataset_type: For SciQA: "handcrafted" or "auto"

    Returns:
        List of question dictionaries
    """
    if agent_name == "kqapro":
        path = PROJECT_ROOT / "db" / "datasets" / "kqapro" / "val.json"
        if not path.exists():
            raise FileNotFoundError(f"KQAPro validation dataset not found at {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"[OK] Loaded {len(data)} questions from KQAPro val.json")
        return data

    elif agent_name == "sciqa":
        import csv as csv_reader
        from ama_kbqa.config import get_sciqa_dataset_path

        dataset_path = get_sciqa_dataset_path(dataset_type)
        full_path = PROJECT_ROOT / dataset_path / "full dataset.csv"

        if not full_path.exists():
            full_path = PROJECT_ROOT / dataset_path / "test.csv"

        if not full_path.exists():
            raise FileNotFoundError(f"SciQA dataset not found at {full_path}")

        data = []
        with open(full_path, "r", encoding="utf-8") as f:
            reader = csv_reader.DictReader(f)
            for row in reader:
                question = row.get("Paraphrase", "").strip()
                if not question:
                    question = row.get("Question without context (comparison)", "").strip()
                answer = row.get("Result", "").strip()
                sparql_query = row.get("Machine-readable query", "").strip()
                q_type = row.get("Q Content", row.get("ORKG-based type", "")).strip()
                research_field = row.get("Research field", "").strip()

                if question and answer:
                    data.append(normalize_sciqa_question({
                        "question": question,
                        "answer": answer,
                        "sparql_query": sparql_query,
                        "q_type": q_type,
                        "research_field": research_field,
                        "raw": row
                    }))

        print(f"[OK] Loaded {len(data)} questions from SciQA {dataset_type} dataset")
        return data
    else:
        raise ValueError(f"Unknown agent: {agent_name}")


def sample_questions(
    data: List[Dict[str, Any]],
    n: int,
    seed: int
) -> List[Dict[str, Any]]:
    """Randomly sample n questions from the dataset."""
    random.seed(seed)
    n_sample = min(n, len(data))
    sampled = random.sample(data, n_sample)
    if n_sample >= len(data):
        print(
            f"[WARN] Requested {n} of {len(data)} available questions: the whole "
            f"dataset is used, so seed={seed} only changes ORDER, not which "
            f"questions are drawn. Question membership is identical across seeds; "
            f"run-to-run differences come from agent LLM sampling (now seeded "
            f"via AMA_LLM_SEED). Use --n-questions < {len(data)} to vary the set."
        )
    else:
        print(f"[OK] Sampled {n_sample} of {len(data)} questions with seed={seed}")
    return sampled


def stratified_sample(
    data: List[Dict[str, Any]],
    n: int,
    seed: int,
    agent_name: str
) -> List[Dict[str, Any]]:
    """
    Stratified sample: groups by question type, allocates proportionally.

    Args:
        data: Full dataset
        n: Total number to sample
        seed: Random seed
        agent_name: Agent name for type extraction

    Returns:
        Stratified sample
    """
    random.seed(seed)

    # Group by type
    groups: Dict[str, List[Dict]] = {}
    for item in data:
        qtype = get_question_type(item, agent_name)
        groups.setdefault(qtype, []).append(item)

    total = len(data)
    n_sample = min(n, total)

    # Allocate proportionally
    result = []
    remaining = n_sample
    sorted_types = sorted(groups.keys())

    for i, qtype in enumerate(sorted_types):
        items = groups[qtype]
        if i == len(sorted_types) - 1:
            # Last type gets remaining quota
            quota = remaining
        else:
            quota = max(1, round(n_sample * len(items) / total))
            quota = min(quota, remaining, len(items))

        sampled = random.sample(items, min(quota, len(items)))
        result.extend(sampled)
        remaining -= len(sampled)

    # Do NOT shuffle across qtypes here. The benchmark's LLM calls share a
    # leading system-prompt + qtype-tool-schema byte sequence for every
    # question of the same qtype; running same-qtype questions back-to-back
    # lets the KIT endpoint's position-anchored prefix cache reuse that
    # shared prefix across consecutive questions instead of invalidating it
    # on every question (see .agent/Tasks/active/prompt-cache-utilization.md,
    # item 1). This does NOT change which questions are sampled -- only
    # their execution order -- and stays deterministic/seed-reproducible:
    # groups are ordered by qtype name (`sorted_types`, computed above), and
    # within a group `random.sample`'s selection order already came from the
    # RNG seeded at the top of this function, so no fresh randomness is
    # introduced. The explicit stable sort below (rather than relying on the
    # accumulation loop's incidental ordering) makes this grouping invariant
    # robust to future refactors of that loop.
    result.sort(key=lambda item: get_question_type(item, agent_name))

    # Print distribution
    type_counts: Dict[str, int] = {}
    for item in result:
        qtype = get_question_type(item, agent_name)
        type_counts[qtype] = type_counts.get(qtype, 0) + 1

    print(f"[OK] Stratified sample: {len(result)} questions")
    for qtype, count in sorted(type_counts.items()):
        print(f"     {qtype}: {count}")

    return result


def get_question_type(question: Dict[str, Any], agent_name: str) -> str:
    """Extract question type from a question dict."""
    if agent_name == "kqapro":
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
        return question.get("q_type", "General")


# ============================================================================
# ANSWER EVALUATION (for multi-model mode without PostProcessor)
# ============================================================================

def normalize_answer(answer: str) -> str:
    if answer is None:
        return ""
    return str(answer).lower().strip()


def evaluate_accuracy_string(predicted: Optional[str], gold: str, q_type: str = "") -> bool:
    """Evaluate via string matching (fallback)."""
    if predicted is None:
        return False
    pred_norm = normalize_answer(predicted)
    gold_norm = normalize_answer(gold)
    if pred_norm == gold_norm:
        return True
    if gold_norm in pred_norm:
        return True
    if gold_norm in ["yes", "no", "true", "false"]:
        pred_mapped = pred_norm.replace("true", "yes").replace("false", "no")
        gold_mapped = gold_norm.replace("true", "yes").replace("false", "no")
        if pred_mapped == gold_mapped:
            return True
        if pred_norm.startswith(gold_norm):
            return True
    if q_type.lower() == "count":
        try:
            return int(pred_norm.split()[0]) == int(gold_norm)
        except (ValueError, IndexError):
            pass
    return False


async def evaluate_accuracy_llm(
    question: str,
    predicted: Optional[str],
    gold: str,
    q_type: str = ""
) -> bool:
    """Evaluate via LLM judge (for multi-model mode)."""
    if predicted is None:
        return False

    api_key = os.getenv(JUDGE_MODEL_CONFIG.api_key_env)
    if not api_key:
        print("  [Judge] No API key, falling back to string matching")
        return evaluate_accuracy_string(predicted, gold, q_type)

    client = AsyncOpenAI(api_key=api_key, base_url=JUDGE_MODEL_CONFIG.base_url)

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

2. CORRECT only if the predicted answer:
   - Provides the same factual information as the gold answer
   - For yes/no questions: "true"="yes", "false"="no" (must match the gold)
   - For counts: the number must match exactly
   - For entities: must refer to the same entity (alternate names OK)

3. BE STRICT: When in doubt, mark as INCORRECT.

Respond with JSON: {{"correct": true}} or {{"correct": false}}"""

    async def call_judge(use_json_mode: bool) -> Optional[bool]:
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
        try:
            result = json.loads(content)
            if isinstance(result.get("correct"), bool):
                return result["correct"]
        except json.JSONDecodeError:
            pass
        content_upper = content.upper()
        if content_upper.startswith("CORRECT") or '"correct": true' in content.lower() or '"correct":true' in content.lower():
            return True
        if content_upper.startswith("INCORRECT") or '"correct": false' in content.lower() or '"correct":false' in content.lower():
            return False
        return None

    try:
        result = await call_judge(use_json_mode=True)
        if result is not None:
            return result
        result = await call_judge(use_json_mode=False)
        if result is not None:
            return result
        print("  [Judge] Could not parse response, falling back to string matching")
        return evaluate_accuracy_string(predicted, gold, q_type)
    except Exception as e:
        error_msg = str(e).lower()
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

def estimate_cost(model_name: str, prompt_tokens: int, completion_tokens: int) -> float:
    costs = MODEL_COSTS.get(model_name, {"input": 1.0, "output": 3.0})
    return (prompt_tokens / 1_000_000) * costs["input"] + (completion_tokens / 1_000_000) * costs["output"]


# ============================================================================
# AGENT CREATION
# ============================================================================

def create_agent(agent_name: str, use_fewshot: bool = True):
    if agent_name == "kqapro":
        from ama_kbqa.agents.kqapro_agent.agent import KQAProAgent
        return KQAProAgent(name="benchmark_kqapro", use_fewshot=use_fewshot)
    elif agent_name == "sciqa":
        from ama_kbqa.agents.sciqa_agent.agent import SciQAAgent
        return SciQAAgent(name="benchmark_sciqa", use_fewshot=use_fewshot)
    else:
        raise ValueError(f"Unknown agent: {agent_name}")


# ============================================================================
# PROCESS SINGLE QUESTION (unified)
# ============================================================================

async def process_single_question(
    agent,
    question: Dict[str, Any],
    agent_name: str,
    timeout: int,
    postprocessor: Optional[PostProcessor] = None,
    question_index: Optional[int] = None,
) -> QuestionResult:
    """
    Process a single question through the agent.

    When postprocessor is provided, uses it for evaluation (single-model mode
    and multi-model with --postprocessing). Otherwise falls back to the
    built-in LLM judge (legacy multi-model mode).

    question_index is the 0-based position in the run; it becomes the
    question_id when the question dict has no explicit "id" (sampled CSV rows
    do not), so per-question logs are distinguishable instead of all "[0]".
    """
    q_text = question.get("question", "")
    gold_answer = question.get("answer", "")
    q_id = question.get("id", question_index if question_index is not None else 0)
    q_type = get_question_type(question, agent_name)
    choices = question.get("choices", [])
    program = question.get("program", None)

    start_time = time.time()
    predicted_answer = None
    error = None

    try:
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

    # Get tool summaries
    tool_summary = agent.get_tool_call_summary()

    # Extract tool trace
    tool_trace = extract_tool_trace(agent._messages)

    # Evaluate accuracy
    pp_result = PostProcessingResult()
    pp_mode = ""

    if postprocessor and not error:
        pp_result = postprocessor.evaluate(
            question=q_text,
            predicted=predicted_answer,
            gold=gold_answer,
            agent_messages=agent._messages,
            choices=choices,
            program=program,
            q_type=q_type,
        )
        pp_mode = postprocessor.mode
        accuracy = pp_result.accuracy
    elif not error:
        accuracy = await evaluate_accuracy_llm(q_text, predicted_answer, gold_answer, q_type)
    else:
        accuracy = False

    # Capture full messages before reset clears them
    full_messages = copy.deepcopy(agent._messages)

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
        q_type=q_type,
        selected_answer=pp_result.selected_answer,
        judgment=pp_result.judgment,
        tool_trace=tool_trace,
        full_messages=full_messages,
        tool_call_summary=tool_summary,
        intermediate_thinking=pp_result.intermediate_thinking,
        synthesized_sparql=pp_result.synthesized_sparql,
        postprocessing_mode=pp_mode,
        program=program,
        choices=choices,
    )


# ============================================================================
# SAVE RESULTS (extended with rich output)
# ============================================================================

def save_results_to_disk(
    results: List[QuestionResult],
    model: ModelConfig,
    agent_name: str,
    result_dir: Path,
    console_log: StringIO,
    is_complete: bool = False,
    postprocessing_mode: str = "",
    generate_fewshot: bool = False,
) -> Dict[str, Any]:
    """Save current results to disk (intermediate or final)."""
    if not results:
        return {}

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
    type_stats: Dict[str, Dict[str, int]] = {}
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

    # Question type distribution
    qtype_distribution = {k: v["total"] for k, v in type_stats.items()}

    # Tool call aggregation
    total_tool_calls = 0
    total_tool_duration = 0.0
    tool_breakdown: Dict[str, Dict] = {}

    for r in results:
        ts = r.tool_call_summary
        total_tool_calls += ts.get("total_calls", 0)
        total_tool_duration += ts.get("total_duration_seconds", 0)
        for tool_name, stats in ts.get("tool_breakdown", {}).items():
            if tool_name not in tool_breakdown:
                tool_breakdown[tool_name] = {"count": 0, "total_duration": 0, "success_count": 0, "failure_count": 0}
            tool_breakdown[tool_name]["count"] += stats.get("count", 0)
            tool_breakdown[tool_name]["total_duration"] += stats.get("total_duration", 0)
            tool_breakdown[tool_name]["success_count"] += stats.get("success_count", 0)
            tool_breakdown[tool_name]["failure_count"] += stats.get("failure_count", 0)

    for tn in tool_breakdown:
        c = tool_breakdown[tn]["count"]
        if c > 0:
            tool_breakdown[tn]["avg_duration"] = round(tool_breakdown[tn]["total_duration"] / c, 3)
        tool_breakdown[tn]["total_duration"] = round(tool_breakdown[tn]["total_duration"], 3)

    accuracy_rate = correct / total if total > 0 else 0

    # Superset summary that works for both old batch_runner consumers and benchmark consumers
    summary = {
        "model": model.name,
        "agent": agent_name,
        "timestamp": datetime.now().isoformat(),
        "is_complete": is_complete,
        "config": {
            "postprocessing_mode": postprocessing_mode,
        },
        "statistics": {
            # Benchmark-style keys
            "total_questions": total,
            "correct": correct,
            "errors": errors,
            "accuracy": accuracy_rate,
            "total_time_seconds": round(total_time, 2),
            "avg_time_seconds": round(avg_time, 2),
            "total_tokens": total_tokens,
            "avg_tokens": round(avg_tokens, 2),
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "estimated_cost_usd": round(estimated_cost, 4),
            "accuracy_by_type": type_accuracy,
            # Batch-runner-style aliases
            "accurate": correct,
            "accuracy_rate": accuracy_rate,
            "total_tokens_used": total_tokens,
            "total_duration_seconds": round(total_time, 2),
            "average_duration_seconds": round(avg_time, 2),
            "question_type_distribution": qtype_distribution,
            "accuracy_by_question_type": type_accuracy,
            # Tool call statistics
            "total_tool_calls": total_tool_calls,
            "total_tool_duration_seconds": round(total_tool_duration, 3),
            "avg_tool_calls_per_question": round(total_tool_calls / total, 2) if total else 0,
            "tool_breakdown": tool_breakdown,
        }
    }

    # Save results.json (exclude intermediate_thinking to avoid bloat)
    results_data = []
    for r in results:
        rd = {
            "question_id": r.question_id,
            "question": r.question,
            "gold_answer": r.gold_answer,
            "answer": r.gold_answer,  # alias for batch_runner compat
            "predicted_answer": r.predicted_answer,
            "selected_answer": r.selected_answer,
            "accuracy": r.accuracy,
            "elapsed_time": round(r.elapsed_time, 2),
            "duration": f"{r.elapsed_time:.2f}s",
            "token_usage": r.token_usage,
            "tokens_used": r.token_usage.get("total_tokens", 0),
            "tool_summary": r.tool_summary,
            "tool_trace": r.tool_trace,
            "tool_call_summary": r.tool_call_summary,
            "error": r.error,
            "q_type": r.q_type,
            "qtype": r.q_type,  # alias
            "judgment": r.judgment,
            "synthesized_sparql": r.synthesized_sparql,
            "postprocessing_mode": r.postprocessing_mode,
            "program": r.program,
            "choices": r.choices,
            "success": r.error is None,
        }
        results_data.append(rd)

    with open(result_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)

    with open(result_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    with open(result_dir / "console_output.txt", "w", encoding="utf-8") as f:
        f.write(console_log.getvalue())

    # Save detailed_log.txt with intermediate thinking
    if is_complete:
        _save_detailed_log(result_dir, results)
        _save_tool_traces(result_dir, results)

    # Save judgments.json if llm_judge mode
    if is_complete and postprocessing_mode == "llm_judge":
        _save_judgments(result_dir, results, accuracy_rate, summary.get("config", {}))

        # Export few-shot examples from correct answers
        print("\n[INFO] Exporting tool-trace few-shot examples...")
        export_counts = export_fewshot_examples_from_traces(
            results=results_data,
            max_tool_count=15,
            max_per_type=5,
            # Without this the export defaults to the KQAPro directory for every
            # agent, so SciQA-derived examples land in (and contaminate) KQAPro's
            # fewshot bank. See resolve_fewshot_dir() in utils/trace_utils.py.
            agent_name=agent_name,
        )
        total_exported = sum(export_counts.values())
        if total_exported > 0:
            print(f"[OK] Exported {total_exported} new tool-trace examples")
        else:
            print("[INFO] No new tool-trace examples to export")

        # LLM-based fewshot generation
        if generate_fewshot:
            from ama_kbqa.fewshot_generator import generate_and_save_fewshot_examples
            print("\n[INFO] Running LLM-based fewshot example generation...")
            generate_and_save_fewshot_examples(
                results_data=results_data,
                full_results=results,
                agent_name=agent_name,
                result_dir=result_dir,
            )

    return summary


def _save_detailed_log(result_dir: Path, results: List[QuestionResult]):
    """Save detailed_log.txt with full intermediate thinking."""
    detailed_log_file = result_dir / "detailed_log.txt"
    with open(detailed_log_file, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("DETAILED BATCH PROCESSING LOG\n")
        f.write("=" * 80 + "\n\n")

        for i, r in enumerate(results, 1):
            f.write(f"\n{'=' * 80}\n")
            f.write(f"QUESTION {i}/{len(results)}\n")
            f.write(f"{'=' * 80}\n\n")
            f.write(f"Question: {r.question}\n")
            f.write(f"Type: {r.q_type}\n")
            f.write(f"Gold Answer: {r.gold_answer}\n\n")

            f.write("--- Predicted Answer ---\n")
            f.write(f"{r.predicted_answer or 'N/A'}\n\n")

            f.write("--- Accuracy ---\n")
            f.write(f"{'CORRECT' if r.accuracy else 'INCORRECT'}\n\n")

            f.write("--- Intermediate Thinking Process ---\n")
            f.write(f"{r.intermediate_thinking or 'No thinking recorded'}\n\n")

            if r.judgment:
                j = r.judgment
                f.write("--- LLM Judge Evaluation ---\n")
                f.write(f"Correctness: {'CORRECT' if j.get('is_correct', False) else 'INCORRECT'}\n")
                f.write(f"Argumentation Score: {j.get('argumentation_score', 'N/A')}/5\n\n")
                f.write(f"Correctness Reasoning:\n{j.get('correctness_reasoning', 'N/A')}\n\n")
                f.write(f"Argumentation Quality:\n{j.get('argumentation_quality', 'N/A')}\n\n")
                f.write(f"Suggested Improvement:\n{j.get('suggested_improvement', 'N/A')}\n\n")

            ts = r.tool_call_summary
            if ts.get('total_calls', 0) > 0:
                f.write("--- Tool Call Summary ---\n")
                f.write(f"Total Calls: {ts.get('total_calls', 0)}\n")
                f.write(f"Total Duration: {ts.get('total_duration_seconds', 0)}s\n")
                for tool_name, stats in ts.get('tool_breakdown', {}).items():
                    f.write(f"  {tool_name}: {stats.get('count', 0)}x, "
                            f"total {stats.get('total_duration', 0)}s, "
                            f"avg {stats.get('avg_duration', 0)}s\n")
                f.write("\n")

            f.write("--- Metadata ---\n")
            f.write(f"Duration: {r.elapsed_time:.2f}s\n")
            f.write(f"Tokens Used: {r.token_usage.get('total_tokens', 0)}\n")
            f.write(f"Success: {r.error is None}\n")
            if r.error:
                f.write(f"Error: {r.error}\n")
            f.write("\n")

    print(f"[OK] Saved detailed log to: {detailed_log_file}")


def _save_tool_traces(result_dir: Path, results: List[QuestionResult]):
    """Save full tool traces as separate JSON files per question.

    Filenames are keyed by r.question_id, not by position in `results`. The
    concurrent runner (run_benchmark_for_model_agent_parallel) appends results
    in completion order, not launch order, so the list position of a result no
    longer matches the question it belongs to. Keying by question_id keeps
    tool_traces/question_NNN.json joinable with the question_id column in
    results.json regardless of processing order.
    """
    traces_dir = result_dir / "tool_traces"
    traces_dir.mkdir(exist_ok=True)

    for r in results:
        if not r.full_messages:
            continue

        # Process messages: parse tool_calls arguments from JSON strings
        processed_messages = []
        for msg in r.full_messages:
            msg_copy = dict(msg)
            if msg_copy.get("tool_calls"):
                parsed_calls = []
                for tc in msg_copy["tool_calls"]:
                    tc_copy = dict(tc)
                    args = tc_copy.get("arguments", tc_copy.get("function", {}).get("arguments", ""))
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except (json.JSONDecodeError, TypeError):
                            pass
                    tc_copy["arguments"] = args
                    parsed_calls.append(tc_copy)
                msg_copy["tool_calls"] = parsed_calls
            processed_messages.append(msg_copy)

        trace_data = {
            "question_id": r.question_id,
            "question": r.question,
            "gold_answer": r.gold_answer,
            "predicted_answer": r.predicted_answer,
            "accuracy": r.accuracy,
            "messages": processed_messages,
        }

        try:
            qid_suffix = f"{int(r.question_id):03d}"
        except (TypeError, ValueError):
            # Non-numeric ids (unexpected, but keep this from crashing the run).
            qid_suffix = str(r.question_id)
        trace_file = traces_dir / f"question_{qid_suffix}.json"
        with open(trace_file, "w", encoding="utf-8") as f:
            json.dump(trace_data, f, indent=2, ensure_ascii=False, default=str)

    print(f"[OK] Saved tool traces to: {traces_dir}")


def _save_judgments(
    result_dir: Path,
    results: List[QuestionResult],
    accuracy_rate: float,
    config: Dict
):
    """Save judgments.json for llm_judge mode."""
    judgments_with_results = []
    avg_arg_score = 0
    judgment_count = 0

    for r in results:
        if r.judgment:
            judgments_with_results.append({
                "question": r.question,
                "gold_answer": r.gold_answer,
                "predicted_answer": r.predicted_answer,
                "judgment": r.judgment,
                "accuracy": r.accuracy,
                "qtype": r.q_type
            })
            if r.judgment.get("argumentation_score"):
                avg_arg_score += r.judgment["argumentation_score"]
                judgment_count += 1

    avg_arg_score = avg_arg_score / judgment_count if judgment_count > 0 else 0

    judgments_file = result_dir / "judgments.json"
    with open(judgments_file, "w", encoding="utf-8") as f:
        json.dump({
            "config": config,
            "timestamp": datetime.now().isoformat(),
            "statistics": {
                "total_judgments": len(judgments_with_results),
                "average_argumentation_score": round(avg_arg_score, 2),
                "accuracy_rate": accuracy_rate
            },
            "judgments": judgments_with_results
        }, f, indent=2, ensure_ascii=False)

    print(f"[OK] Saved LLM judgments to: {judgments_file}")
    print(f"[OK] Average Argumentation Score: {avg_arg_score:.2f}/5")


# ============================================================================
# BENCHMARK RUNNER (for multi-model / single-model)
# ============================================================================

async def run_benchmark_for_model_agent(
    model: ModelConfig,
    agent_name: str,
    questions: List[Dict[str, Any]],
    timeout: int,
    output_dir: Path,
    use_fewshot: bool = True,
    postprocessing_mode: str = "",
    generate_fewshot: bool = False,
    concurrency: int = 1,
) -> Dict[str, Any]:
    """Run benchmark for a specific model/agent combination.

    concurrency=1 keeps the original strictly-serial path. concurrency>1 runs a
    pool of that many isolated agents (each its own MCP subprocess) over a shared
    question queue — see run_benchmark_for_model_agent_parallel.
    """
    result_dir = output_dir / agent_name / model.name
    result_dir.mkdir(parents=True, exist_ok=True)

    console_log = StringIO()

    def log_print(msg: str):
        print(msg)
        console_log.write(msg + "\n")

    log_print(f"\n{'='*80}")
    log_print(f"Benchmarking: {model.name} on {agent_name}")
    log_print(f"Questions: {len(questions)}")
    log_print(f"Timeout: {timeout}s per question")
    if postprocessing_mode:
        log_print(f"Postprocessing: {postprocessing_mode}")
    if not use_fewshot:
        log_print("[ABLATION] Few-shot examples DISABLED")
    log_print(f"{'='*80}\n")

    # Override config for this model (skip for "default" which uses config.toml as-is)
    if model.name != "default":
        override_model_config(model)

    # Create PostProcessor if mode specified
    postprocessor = None
    if postprocessing_mode:
        try:
            postprocessor = PostProcessor(mode=postprocessing_mode, agent_name=agent_name)
        except Exception as e:
            log_print(f"[WARNING] Could not init PostProcessor: {e}, using built-in judge")

    # C1: opt-in parallel path. Default (concurrency<=1) falls through to the
    # original serial loop below, unchanged.
    if concurrency and concurrency > 1:
        return await run_benchmark_for_model_agent_parallel(
            model=model,
            agent_name=agent_name,
            questions=questions,
            timeout=timeout,
            result_dir=result_dir,
            console_log=console_log,
            log_print=log_print,
            postprocessor=postprocessor,
            postprocessing_mode=postprocessing_mode,
            generate_fewshot=generate_fewshot,
            use_fewshot=use_fewshot,
            concurrency=concurrency,
        )

    agent = create_agent(agent_name, use_fewshot=use_fewshot)
    results: List[QuestionResult] = []
    summary = {}

    # Fail-fast MCP startup check: if the tool server can't come up, abort the
    # whole run rather than writing N identical "Failed to start MCP server"
    # rows. Prior runs (2026-04-20-1, -3) silently failed this way.
    try:
        await agent._init_mcp()
    except Exception as e:
        log_print(f"[FATAL] MCP server failed to start for {model.name}/{agent_name}: {e}")
        log_print("[FATAL] Aborting this model/agent run; no questions will be processed.")
        try:
            await agent.close()
        except Exception:
            pass
        # Persist the failure context so the manifest + console_output
        # capture *why* the run is empty, instead of looking like it was never attempted.
        console_out_path = result_dir / "console_output.txt"
        console_out_path.write_text(console_log.getvalue(), encoding="utf-8")
        return {
            "error": f"MCP startup failed: {e}",
            "model": model.name,
            "agent": agent_name,
        }

    # Consecutive-infrastructure-failure cutoff. An LLM endpoint that's down
    # (KIT bursts 404s for ~10 min, or hangs at the socket level) will emit
    # the same error class on every question it touches. We've seen a 24-row
    # KIT outage corrupt seed=43 (2026-05-06) and a single hang lock up
    # gemma seed=44 (2026-05-08). Aborting after N consecutive infra
    # failures preserves the partial run and surfaces a clear cause instead
    # of writing dozens of fake "INCORRECT" rows.
    INFRA_ERROR_PATTERNS = (
        "NotFoundError", "404",  # KIT bursty 404s
        "NoneType' object has no attribute 'choices'",  # response None / litellm wrapper failure
        "litellm.NotFoundError", "Hosted_vllmException",
        "ConnectError", "ReadTimeout", "WriteTimeout", "PoolTimeout",
        "ConnectTimeout", "RemoteProtocolError",
    )
    INFRA_FAILURE_ABORT_THRESHOLD = 5
    consecutive_infra_failures = 0

    def _is_infra_error(err: Optional[str]) -> bool:
        if not err:
            return False
        return any(p in err for p in INFRA_ERROR_PATTERNS)

    try:
        total_q = len(questions)
        pbar = tqdm(questions, desc=f"{model.name}/{agent_name}", unit="q")

        for i, question in enumerate(pbar):
            result = await process_single_question(
                agent=agent,
                question=question,
                agent_name=agent_name,
                timeout=timeout,
                postprocessor=postprocessor,
                question_index=i,
            )
            results.append(result)

            answered = len(results)
            correct = sum(1 for r in results if r.accuracy)
            running_acc = 100.0 * correct / answered if answered else 0.0
            # Distinct labels: 'done' is completion progress, 'acc' is accuracy.
            # The previous postfix showed "acc: 77/100", which reads like a
            # progress counter and was routinely misread as 77-of-100-answered.
            pbar.set_postfix({
                "done": f"{answered}/{total_q}",
                "acc": f"{running_acc:.0f}%",
                "time": f"{result.elapsed_time:.1f}s",
            })

            status = "CORRECT" if result.accuracy else "INCORRECT"
            if result.error:
                status = f"ERROR: {result.error[:50]}"
            # Explicit answered-count makes progress legible in nohup logs where
            # the tqdm bar does not animate.
            log_print(
                f"  [answered {answered}/{total_q}] q#{result.question_id} "
                f"{status} | running acc {correct}/{answered} ({running_acc:.0f}%) "
                f"| {result.elapsed_time:.1f}s"
            )
            log_print(f"      Gold: {result.gold_answer}")
            log_print(f"      Pred: {result.predicted_answer}")

            summary = save_results_to_disk(
                results=results,
                model=model,
                agent_name=agent_name,
                result_dir=result_dir,
                console_log=console_log,
                is_complete=False,
                postprocessing_mode=postprocessing_mode,
            )

            # Track consecutive infrastructure failures and abort if the
            # endpoint is clearly unhealthy.
            if _is_infra_error(result.error):
                consecutive_infra_failures += 1
                log_print(
                    f"  [infra-watchdog] consecutive infra failures: "
                    f"{consecutive_infra_failures}/{INFRA_FAILURE_ABORT_THRESHOLD}"
                )
                if consecutive_infra_failures >= INFRA_FAILURE_ABORT_THRESHOLD:
                    log_print(
                        f"\n[FATAL] Endpoint unhealthy: "
                        f"{consecutive_infra_failures} consecutive infrastructure errors "
                        f"(model={model.name}, agent={agent_name}). "
                        f"Last error: {result.error[:120]}\n"
                        f"Aborting run after {len(results)} questions to avoid logging "
                        f"a wall of fake INCORRECT rows. Re-run when the endpoint is back."
                    )
                    break
            else:
                consecutive_infra_failures = 0

        pbar.close()

    finally:
        await agent.close()

        if results:
            log_print(f"[POST] {model.name}/{agent_name}: finalizing results and exporting traces...")
            summary = save_results_to_disk(
                results=results,
                model=model,
                agent_name=agent_name,
                result_dir=result_dir,
                console_log=console_log,
                is_complete=True,
                postprocessing_mode=postprocessing_mode,
                generate_fewshot=generate_fewshot,
            )

    if results and summary:
        stats = summary.get("statistics", {})
        total = stats.get("total_questions", len(results))
        answered = len(results)
        correct = stats.get("correct", 0)
        accuracy_pct = stats.get("accuracy", 0) * 100
        # Two clearly-separated facts: how many were answered (completion) and
        # how many were correct (accuracy). The old single "{correct}/{total}
        # ({pct}%)" was read as a progress counter.
        print(
            f"\n[DONE] {model.name}/{agent_name}: answered {answered}/{total} questions "
            f"| accuracy {correct}/{total} correct ({accuracy_pct:.1f}%)"
        )

    return summary


async def run_benchmark_for_model_agent_parallel(
    model: ModelConfig,
    agent_name: str,
    questions: List[Dict[str, Any]],
    timeout: int,
    result_dir: Path,
    console_log: "StringIO",
    log_print,
    postprocessor: Optional[PostProcessor],
    postprocessing_mode: str,
    generate_fewshot: bool,
    use_fewshot: bool,
    concurrency: int,
) -> Dict[str, Any]:
    """Concurrent variant of the per-model/agent run (C1).

    Spins up `concurrency` fully isolated agents (each its own MCP subprocess)
    and drains a shared question queue. Each agent processes one question at a
    time and soft-resets between, exactly like the serial loop — only several
    run at once. Results are appended and persisted incrementally under a lock.

    The infra-failure watchdog is preserved but adapted: it counts infra errors
    in completion order and aborts (stops scheduling new questions) once
    INFRA_FAILURE_ABORT_THRESHOLD accumulate consecutively, matching the serial
    intent (a genuinely-down endpoint won't yield that many in a row while
    healthy work also completes).
    """
    log_print(f"[PARALLEL] Running with concurrency={concurrency}")

    # Same infra patterns/threshold as the serial path.
    INFRA_ERROR_PATTERNS = (
        "NotFoundError", "404",
        "NoneType' object has no attribute 'choices'",
        "litellm.NotFoundError", "Hosted_vllmException",
        "ConnectError", "ReadTimeout", "WriteTimeout", "PoolTimeout",
        "ConnectTimeout", "RemoteProtocolError",
    )
    INFRA_FAILURE_ABORT_THRESHOLD = 5

    def _is_infra_error(err: Optional[str]) -> bool:
        return bool(err) and any(p in err for p in INFRA_ERROR_PATTERNS)

    # Build the worker pool. Each agent gets its own MCP subprocess.
    n_workers = min(concurrency, len(questions)) or 1
    agents = [create_agent(agent_name, use_fewshot=use_fewshot) for _ in range(n_workers)]

    # Fail-fast: if any worker's MCP can't start, abort the whole run cleanly.
    try:
        await asyncio.gather(*(a._init_mcp() for a in agents))
    except Exception as e:
        log_print(f"[FATAL] MCP server failed to start for {model.name}/{agent_name}: {e}")
        log_print("[FATAL] Aborting this model/agent run; no questions will be processed.")
        for a in agents:
            try:
                await a.close()
            except Exception:
                pass
        (result_dir / "console_output.txt").write_text(console_log.getvalue(), encoding="utf-8")
        return {"error": f"MCP startup failed: {e}", "model": model.name, "agent": agent_name}

    results: List[QuestionResult] = []
    summary_holder: Dict[str, Any] = {"summary": {}}
    q_queue: "asyncio.Queue" = asyncio.Queue()
    for idx, question in enumerate(questions):
        q_queue.put_nowait((idx, question))

    lock = asyncio.Lock()
    stop_event = asyncio.Event()
    watchdog = {"consecutive_infra_failures": 0, "aborted_for_infra": False}
    pbar = tqdm(total=len(questions), desc=f"{model.name}/{agent_name}", unit="q")

    async def worker(agent) -> None:
        while not stop_event.is_set():
            try:
                q_index, question = q_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            result = await process_single_question(
                agent=agent,
                question=question,
                agent_name=agent_name,
                timeout=timeout,
                postprocessor=postprocessor,
                question_index=q_index,
            )
            async with lock:
                results.append(result)
                correct = sum(1 for r in results if r.accuracy)
                pbar.update(1)
                pbar.set_postfix({"acc": f"{correct}/{len(results)}", "time": f"{result.elapsed_time:.1f}s"})

                status = "CORRECT" if result.accuracy else "INCORRECT"
                if result.error:
                    status = f"ERROR: {result.error[:50]}"
                log_print(f"  [{result.question_id}] {status} | {result.elapsed_time:.1f}s")
                log_print(f"      Gold: {result.gold_answer}")
                log_print(f"      Pred: {result.predicted_answer}")

                summary_holder["summary"] = save_results_to_disk(
                    results=results,
                    model=model,
                    agent_name=agent_name,
                    result_dir=result_dir,
                    console_log=console_log,
                    is_complete=False,
                    postprocessing_mode=postprocessing_mode,
                )

                if _is_infra_error(result.error):
                    watchdog["consecutive_infra_failures"] += 1
                    log_print(
                        f"  [infra-watchdog] consecutive infra failures: "
                        f"{watchdog['consecutive_infra_failures']}/{INFRA_FAILURE_ABORT_THRESHOLD}"
                    )
                    if watchdog["consecutive_infra_failures"] >= INFRA_FAILURE_ABORT_THRESHOLD:
                        log_print(
                            f"\n[FATAL] Endpoint unhealthy: "
                            f"{watchdog['consecutive_infra_failures']} consecutive infrastructure errors "
                            f"(model={model.name}, agent={agent_name}). "
                            f"Last error: {result.error[:120]}\n"
                            f"Aborting run after {len(results)} questions to avoid logging "
                            f"a wall of fake INCORRECT rows. Re-run when the endpoint is back."
                        )
                        watchdog["aborted_for_infra"] = True
                        stop_event.set()
                else:
                    watchdog["consecutive_infra_failures"] = 0

    try:
        await asyncio.gather(*(worker(a) for a in agents))
    finally:
        pbar.close()
        for a in agents:
            try:
                await a.close()
            except Exception:
                pass
        if results:
            summary_holder["summary"] = save_results_to_disk(
                results=results,
                model=model,
                agent_name=agent_name,
                result_dir=result_dir,
                console_log=console_log,
                is_complete=True,
                postprocessing_mode=postprocessing_mode,
                generate_fewshot=generate_fewshot,
            )

    summary = summary_holder["summary"]
    if results and summary:
        stats = summary.get("statistics", {})
        total = stats.get("total_questions", len(results))
        correct = stats.get("correct", 0)
        accuracy_pct = stats.get("accuracy", 0) * 100
        print(f"\nCompleted {model.name}/{agent_name}: {correct}/{total} ({accuracy_pct:.1f}%)")

    return summary


def is_run_completed(output_dir: Path, agent_name: str, model_name: str) -> bool:
    """Return True only if the run's summary.json reports is_complete: true.

    save_results_to_disk() writes summary.json after *every* question, with
    is_complete: false until the run's finally-block writes the last, complete
    summary. Treating mere existence as "done" (the old behavior) meant a
    --resume launch would treat a run that crashed at question 5/100 as
    finished and skip it forever. Missing, unreadable, or malformed summaries
    are treated conservatively as not completed so the combination gets
    retried rather than silently skipped.
    """
    summary_path = output_dir / agent_name / model_name / "summary.json"
    if not summary_path.exists():
        return False
    try:
        with open(summary_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return False
    if not isinstance(data, dict):
        return False
    return data.get("is_complete") is True


async def run_full_benchmark(
    models: List[ModelConfig],
    agents: List[str],
    n_questions: Optional[int],
    output_dir: Path,
    timeout: int,
    resume: bool,
    dry_run: bool,
    export_csv: bool,
    use_fewshot: bool = True,
    postprocessing_mode: str = "",
    seed: Optional[int] = None,
    questionnaire_path: Optional[str] = None,
    dataset_type: str = "handcrafted",
    stratified: bool = False,
    generate_fewshot: bool = False,
    question_indices: Optional[List[int]] = None,
    concurrency: int = 1,
):
    """Run the full benchmark across models and agents."""
    is_single_model = len(models) == 1 and models[0].name == "default"

    # Load questions for each agent
    questionnaires: Dict[str, List[Dict[str, Any]]] = {}

    for agent_name in list(agents):
        questions = _load_questions_for_agent(
            agent_name=agent_name,
            questionnaire_path=questionnaire_path,
            seed=seed,
            n_questions=n_questions,
            dataset_type=dataset_type,
            stratified=stratified,
            is_single_model=is_single_model,
            question_indices=question_indices,
        )
        if questions is None:
            print(f"WARNING: No questions available for {agent_name}, skipping")
            agents.remove(agent_name)
        else:
            questionnaires[agent_name] = questions
            print(f"Loaded {len(questions)} {agent_name} questions")

    if not agents:
        print("ERROR: No agents available to benchmark")
        return

    # Build list of runs
    runs = []
    for model in models:
        if model.name != "default" and not check_api_key(model):
            print(f"WARNING: Skipping {model.name} - API key {model.api_key_env} not set")
            continue
        for agent_name in agents:
            if agent_name not in questionnaires:
                continue
            if resume and is_run_completed(output_dir, agent_name, model.name):
                print(f"SKIP: {model.name}/{agent_name} (already completed)")
                continue
            runs.append((model, agent_name, questionnaires[agent_name]))

    if dry_run:
        print(f"\n{'='*80}")
        print("DRY RUN - The following benchmarks would be executed:")
        print(f"{'='*80}\n")
        for model, agent_name, questions in runs:
            print(f"  - {model.name} on {agent_name}: {len(questions)} questions (postprocessing: {postprocessing_mode or 'built-in judge'})")
        print(f"\nTotal runs: {len(runs)}")
        print(f"Output directory: {output_dir}")
        return

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
                output_dir=output_dir,
                use_fewshot=use_fewshot,
                postprocessing_mode=postprocessing_mode,
                generate_fewshot=generate_fewshot,
                concurrency=concurrency,
            )
            all_summaries.append(summary)
        except Exception as e:
            print(f"ERROR: {model.name}/{agent_name} failed: {e}")
            all_summaries.append({"model": model.name, "agent": agent_name, "error": str(e)})

    # Generate overview only for multi-model mode
    if not is_single_model:
        generate_overview(all_summaries, output_dir, export_csv)
    elif export_csv:
        valid = [s for s in all_summaries if "error" not in s]
        if valid:
            export_results_to_csv(valid, output_dir)

    # Return True if at least one run succeeded
    return any("error" not in s for s in all_summaries) if all_summaries else False


def _load_questions_for_agent(
    agent_name: str,
    questionnaire_path: Optional[str],
    seed: Optional[int],
    n_questions: Optional[int],
    dataset_type: str,
    stratified: bool,
    is_single_model: bool,
    question_indices: Optional[List[int]] = None,
) -> Optional[List[Dict[str, Any]]]:
    """Load questions for an agent based on CLI args."""
    def _apply_question_indices(questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not question_indices:
            return questions
        selected = []
        for idx in question_indices:
            if 0 <= idx < len(questions):
                selected.append(questions[idx])
            else:
                print(f"WARNING: question index {idx} out of range for {agent_name} ({len(questions)} questions)")
        print(f"[OK] Selected {len(selected)} explicit question indices for {agent_name}: {question_indices}")
        return selected

    # Priority 1: explicit questionnaire path
    if questionnaire_path:
        path = Path(questionnaire_path)
        if not path.exists():
            print(f"ERROR: Questionnaire file not found: {questionnaire_path}")
            return None
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        questions = data.get("questions", [])
        if n_questions and n_questions < len(questions):
            questions = questions[:n_questions]
        print(f"[OK] Loaded questionnaire from: {questionnaire_path} ({len(questions)} questions)")
        return _apply_question_indices(questions)

    # Priority 2: seed-based on-the-fly sampling
    if seed is not None:
        if n_questions is None:
            n_questions = 10  # default for seed-based sampling
        raw_data = load_raw_dataset(agent_name, dataset_type)
        if stratified:
            return _apply_question_indices(stratified_sample(raw_data, n_questions, seed, agent_name))
        else:
            return _apply_question_indices(sample_questions(raw_data, n_questions, seed))

    # Priority 3: default questionnaire files (multi-model mode) or error (single-model)
    if is_single_model:
        # Single-model mode without seed or questionnaire: try default questionnaire
        pass

    # Try default questionnaire
    if agent_name == "kqapro":
        path = PROJECT_ROOT / "db" / "kqapro_questionnaire.json"
        if path.exists():
            return _apply_question_indices(load_kqapro_questionnaire(path, n_questions))
        print(f"WARNING: KQAPro questionnaire not found at {path}")
        return None
    elif agent_name == "sciqa":
        path = PROJECT_ROOT / "db" / "sciqa_questionnaire.json"
        if path.exists():
            return _apply_question_indices(load_sciqa_questionnaire(path, n_questions))
        print(f"WARNING: SciQA questionnaire not found at {path}")
        return None

    return None


# ============================================================================
# OVERVIEW & CSV EXPORT
# ============================================================================

def generate_overview(summaries: List[Dict[str, Any]], output_dir: Path, export_csv: bool):
    valid_summaries = [s for s in summaries if "error" not in s]
    if not valid_summaries:
        print("No valid results to generate overview")
        return

    by_accuracy = sorted(valid_summaries, key=lambda x: x["statistics"]["accuracy"], reverse=True)
    by_speed = sorted(valid_summaries, key=lambda x: x["statistics"]["avg_time_seconds"])

    def efficiency_score(s):
        cost = s["statistics"]["estimated_cost_usd"]
        acc = s["statistics"]["accuracy"]
        if cost == 0:
            return float('inf') if acc > 0 else 0
        return acc / cost

    by_efficiency = sorted(valid_summaries, key=efficiency_score, reverse=True)

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
                {"rank": i+1, "model": s["model"], "agent": s["agent"], "accuracy": round(s["statistics"]["accuracy"], 4)}
                for i, s in enumerate(by_accuracy)
            ],
            "by_speed": [
                {"rank": i+1, "model": s["model"], "agent": s["agent"], "avg_time_seconds": round(s["statistics"]["avg_time_seconds"], 2)}
                for i, s in enumerate(by_speed)
            ],
            "by_efficiency": [
                {"rank": i+1, "model": s["model"], "agent": s["agent"], "accuracy_per_dollar": round(efficiency_score(s), 4)}
                for i, s in enumerate(by_efficiency)
            ]
        },
        "results_matrix": results_matrix
    }

    with open(output_dir / "overview.json", "w", encoding="utf-8") as f:
        json.dump(overview, f, indent=2, ensure_ascii=False)

    print(f"\nOverview saved to: {output_dir / 'overview.json'}")
    print(f"\n{'='*80}")
    print("LEADERBOARD - By Accuracy")
    print(f"{'='*80}")
    for entry in overview["leaderboard"]["by_accuracy"][:10]:
        print(f"  {entry['rank']}. {entry['model']}/{entry['agent']}: {entry['accuracy']*100:.1f}%")

    if export_csv:
        export_results_to_csv(valid_summaries, output_dir)


def export_results_to_csv(summaries: List[Dict[str, Any]], output_dir: Path):
    csv_path = output_dir / "benchmark_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv_module.writer(f)
        writer.writerow(["Model", "Agent", "Accuracy", "Correct", "Total", "Avg Time (s)", "Total Tokens", "Est. Cost ($)"])
        for s in summaries:
            stats = s["statistics"]
            writer.writerow([
                s["model"], s["agent"],
                f"{stats['accuracy']*100:.1f}%", stats["correct"], stats["total_questions"],
                f"{stats['avg_time_seconds']:.2f}", stats["total_tokens"],
                f"${stats['estimated_cost_usd']:.4f}"
            ])
    print(f"CSV exported to: {csv_path}")


# ============================================================================
# CLI
# ============================================================================

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Unified KBQA Benchmark - single-model and multi-model modes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single-model: sample from raw dataset, use llm_judge
  python -m ama_kbqa.benchmark_agents --agents kqapro --n-questions 10 --seed 42 --postprocessing llm_judge

  # Single-model: stratified sampling
  python -m ama_kbqa.benchmark_agents --agents kqapro --n-questions 20 --seed 42 --stratified --postprocessing llm_judge

  # Single-model: use questionnaire
  python -m ama_kbqa.benchmark_agents --agents kqapro --questionnaire db/kqapro_questionnaire.json --n-questions 5

  # Multi-model: benchmark several LLMs
  python -m ama_kbqa.benchmark_agents --models minimax-m2.1 deepseek-v3.2 --agents kqapro --n-questions 3

  # Dry run
  python -m ama_kbqa.benchmark_agents --agents kqapro --n-questions 3 --seed 42 --dry-run
"""
    )

    parser.add_argument("--agents", nargs="+", choices=["kqapro", "sciqa"], default=["kqapro", "sciqa"],
                        help="Which agents to test (default: both)")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Filter to specific models by name (default: use config model)")
    parser.add_argument("--n-questions", type=int, default=None,
                        help="Limit questions per agent (default: all)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: benchmark_results/<YYYY-MM-DD-N>)")
    parser.add_argument("--timeout", type=int, default=300,
                        help="Timeout per question in seconds (default: 300)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip completed model/agent combinations")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview what would run without executing")
    parser.add_argument("--export-csv", action="store_true",
                        help="Export results to CSV")
    parser.add_argument("--no-fewshot", action="store_true", default=False,
                        help="Disable few-shot example injection (for ablation study)")

    # New args from batch_runner merge
    parser.add_argument("--postprocessing", "-p", type=str, default="llm_judge",
                        choices=["choice", "sparql", "llm_judge", "simple"],
                        help="Postprocessing/evaluation mode (default: llm_judge)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for on-the-fly sampling from raw dataset")
    parser.add_argument("--questionnaire", type=str, default=None,
                        help="Path to pre-generated questionnaire JSON")
    parser.add_argument("--dataset", "-d", type=str, default="handcrafted",
                        choices=["handcrafted", "auto", "autogenerated"],
                        help="SciQA dataset type (default: handcrafted)")
    parser.add_argument("--stratified", action="store_true",
                        help="Stratified sampling by question type")
    parser.add_argument("--question-indices", type=str, default=None,
                        help="Comma-separated 0-based indices after questionnaire/sampling selection, e.g. 0,8,15")
    parser.add_argument("--generate-fewshot", action="store_true", default=None,
                        help="Generate LLM-based fewshot examples from results (default: from config.toml)")
    parser.add_argument("--concurrency", type=int, default=None,
                        help="Number of questions to process concurrently (default: from "
                             "config.toml [benchmark] concurrency, which defaults to 1 = serial). "
                             "Values > 1 use a pool of isolated agents; mind provider rate limits.")

    args = parser.parse_args()

    # Propagate the run seed to the agent and the MCP server subprocesses via the
    # environment so LLM sampling is reproducible (same seed) and independently
    # re-rollable (new seed). The agent/orchestrator read AMA_LLM_SEED at each
    # chat-completion call; subprocesses inherit os.environ. Without this, --seed
    # only reshuffled question selection and never touched the (dominant) source
    # of run-to-run variation, the agent's own sampling.
    if args.seed is not None:
        os.environ["AMA_LLM_SEED"] = str(args.seed)
        print(f"[OK] Seeding agent LLM sampling with AMA_LLM_SEED={args.seed}")

    question_indices = None
    if args.question_indices:
        try:
            question_indices = [
                int(part.strip())
                for part in args.question_indices.split(",")
                if part.strip()
            ]
        except ValueError as e:
            print(f"ERROR: --question-indices must be comma-separated integers: {e}", file=sys.stderr)
            sys.exit(1)

    # Timeout sanity check. kqapro/sciqa agents routinely need 30–60 s per
    # question (FindNode + several tool calls); run 2026-04-20-4 hit --timeout 60
    # and killed its only question mid-reasoning after 108k tokens.
    MIN_SAFE_TIMEOUT = 120
    if args.timeout < MIN_SAFE_TIMEOUT:
        print(
            f"WARNING: --timeout {args.timeout}s is below the recommended floor of "
            f"{MIN_SAFE_TIMEOUT}s. Agents typically need 30–60s per question, so "
            f"low timeouts produce false-positive error rows. Consider --timeout 300.",
            file=sys.stderr,
        )

    # Determine mode: multi-model or single-model
    if args.models:
        by_name = {m.name: m for m in BENCHMARK_MODELS}
        provider_defaults = {
            "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
            "kit": ("https://ki-toolbox.scc.kit.edu/api/v1", "KIT_API_KEY"),
        }
        models = []
        for entry in args.models:
            if entry in by_name:
                models.append(by_name[entry])
                continue
            # Free-form entry: either "provider:model_id" or bare model_id
            if ":" in entry and entry.split(":", 1)[0] in provider_defaults:
                provider, model_id = entry.split(":", 1)
            else:
                provider, model_id = "openrouter", entry
            base_url, api_key_env = provider_defaults[provider]
            models.append(ModelConfig(
                name=entry.replace("/", "_").replace(":", "_"),
                provider=provider,
                model_id=model_id,
                base_url=base_url,
                api_key_env=api_key_env,
            ))
        if not models:
            print(f"ERROR: No models resolved. Available presets: {[m.name for m in BENCHMARK_MODELS]}")
            sys.exit(1)
    else:
        # Single-model mode: use model from config.toml
        from ama_kbqa.config import get_chat_model_name
        try:
            config_model_name = get_chat_model_name()
        except Exception:
            config_model_name = "default"

        # Create a "default" ModelConfig that just uses whatever config.toml says
        models = [ModelConfig(
            name="default",
            provider="config",
            model_id=config_model_name,
            base_url="",
            api_key_env=""
        )]

    # Set up output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        base_dir = PROJECT_ROOT / "benchmark_results"
        date_prefix = datetime.now().strftime("%Y-%m-%d")
        # Find the next available run number for today
        n = 1
        if base_dir.exists():
            for d in base_dir.iterdir():
                if d.is_dir() and d.name.startswith(date_prefix + "-"):
                    try:
                        existing_n = int(d.name[len(date_prefix) + 1:])
                        n = max(n, existing_n + 1)
                    except ValueError:
                        pass
        output_dir = base_dir / f"{date_prefix}-{n}"

    output_dir.mkdir(parents=True, exist_ok=True)

    _write_run_manifest(output_dir, args, models)

    # Resolve generate_fewshot: CLI flag > config.toml > False
    generate_fewshot = args.generate_fewshot
    if generate_fewshot is None:
        try:
            import toml as _toml
            _cfg = _toml.load(PROJECT_ROOT / "config.toml")
            generate_fewshot = _cfg.get("postprocessing", {}).get("generate_fewshot", False)
        except Exception:
            generate_fewshot = False

    # Resolve concurrency: CLI flag > config.toml [benchmark] concurrency > 1.
    if args.concurrency is not None:
        concurrency = max(1, args.concurrency)
    else:
        try:
            from ama_kbqa.config import get_benchmark_concurrency
            concurrency = get_benchmark_concurrency()
        except Exception:
            concurrency = 1
    if concurrency > 1:
        print(f"[PARALLEL] Benchmark concurrency = {concurrency}")

    result = asyncio.run(run_full_benchmark(
        models=models,
        agents=list(args.agents),
        n_questions=args.n_questions,
        output_dir=output_dir,
        timeout=args.timeout,
        resume=args.resume,
        dry_run=args.dry_run,
        export_csv=args.export_csv,
        use_fewshot=not args.no_fewshot,
        postprocessing_mode=args.postprocessing,
        seed=args.seed,
        questionnaire_path=args.questionnaire,
        dataset_type=args.dataset,
        stratified=args.stratified,
        generate_fewshot=generate_fewshot,
        question_indices=question_indices,
        concurrency=concurrency,
    ))

    # Exit with non-zero code if all runs failed
    if result is not None and not result:
        sys.exit(1)


if __name__ == "__main__":
    main()
