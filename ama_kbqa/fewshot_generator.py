"""LLM-based fewshot example generator.

After the LLM judge evaluates benchmark results, this module analyzes the full
conversation traces to generate three types of learning material:

1. Per-qtype examples  — saved into existing ``db/datasets/kqapro/fewshot-examples/<QType>.json``
2. General guidance     — saved into ``_general.json``
3. Tool tips            — saved into ``_tool_tips.json``

An audit log of every generated example (pre-dedup) is written to the
benchmark result directory as ``generated_fewshot.json``.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI
from pydantic import BaseModel, Field

from ama_kbqa.postprocessing import load_judge_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEWSHOT_DIR = PROJECT_ROOT / "db" / "datasets" / "kqapro" / "fewshot-examples"

# Limits
MAX_PER_QTYPE = 5
MAX_GENERAL = 10
MAX_TOOL_TIPS = 20


# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class FewshotQTypeExample(BaseModel):
    question: str
    answer: str
    qtype: str
    trace: List[Dict[str, Any]]
    lesson: str
    pitfall: str
    tool_count: int
    was_correct: bool
    source: str = "auto_generator"
    collected_at: str


class FewshotGeneralExample(BaseModel):
    title: str
    guidance: str
    applies_to: List[str]
    derived_from_qtype: str
    was_correct: bool
    source: str = "auto_generator"
    collected_at: str


class ToolTip(BaseModel):
    tool_name: str
    problem_pattern: str
    guidance: str
    example_args: Optional[str] = None
    derived_from_question: str
    source: str = "auto_generator"
    collected_at: str


class FewshotGeneratorOutput(BaseModel):
    qtype_example: Optional[FewshotQTypeExample] = None
    general_example: Optional[FewshotGeneralExample] = None
    tool_tip: Optional[ToolTip] = None
    reasoning: str = ""


# ============================================================================
# QUALIFICATION FILTER
# ============================================================================

def qualifies_for_generation(result: Dict[str, Any]) -> Optional[str]:
    """Return 'correct' or 'incorrect' if this result is worth learning from, else None."""
    if result.get("error"):
        return None
    judgment = result.get("judgment")
    if not judgment:
        return None
    is_correct = judgment.get("is_correct", False)
    arg_score = judgment.get("argumentation_score", 0)
    if is_correct and arg_score >= 4:
        return "correct"
    if not is_correct:
        return "incorrect"
    return None


# ============================================================================
# MESSAGE TRUNCATION
# ============================================================================

def _truncate_messages(messages: List[Dict], max_messages: int = 20, max_result_chars: int = 500) -> List[Dict]:
    """Truncate conversation messages to fit token budget."""
    recent = messages[-max_messages:] if len(messages) > max_messages else messages
    truncated = []
    for msg in recent:
        msg_copy = dict(msg)
        if msg_copy.get("role") == "tool":
            content = msg_copy.get("content", "")
            if isinstance(content, str) and len(content) > max_result_chars:
                msg_copy["content"] = content[:max_result_chars] + "... [truncated]"
        truncated.append(msg_copy)
    return truncated


# ============================================================================
# LLM PROMPT
# ============================================================================

GENERATOR_SYSTEM_PROMPT = """You are a fewshot example generator for a Knowledge Base Question Answering (KBQA) system.

You analyze completed question-answering traces (the full conversation between an AI agent and its tools) to extract reusable learning material.

You produce up to 3 optional outputs — ONLY if genuinely useful. Do not force output.

## Output Schema (JSON)
{
  "reasoning": "Your chain-of-thought analysis (always required)",
  "qtype_example": {                    // OPTIONAL — only if a clear strategy pattern exists
    "question": "...",
    "answer": "...",                     // Always the GOLD answer
    "qtype": "...",
    "trace": [{"tool": "...", "args": "...", "result": "..."}],
    "lesson": "1-2 sentence strategy summary",
    "pitfall": "What to avoid",
    "tool_count": <int>,
    "was_correct": <bool>,
    "source": "auto_generator",
    "collected_at": "<ISO timestamp>"
  },
  "general_example": {                  // OPTIONAL — only for cross-type insights
    "title": "Short descriptive title",
    "guidance": "The actual guidance text",
    "applies_to": ["Count", "Query"] or ["all"],
    "derived_from_qtype": "...",
    "was_correct": <bool>,
    "source": "auto_generator",
    "collected_at": "<ISO timestamp>"
  },
  "tool_tip": {                         // OPTIONAL — only for tool-specific gotchas
    "tool_name": "RunSPARQL",
    "problem_pattern": "When this tip applies",
    "guidance": "What to do",
    "example_args": "optional example",
    "derived_from_question": "...",
    "source": "auto_generator",
    "collected_at": "<ISO timestamp>"
  }
}

## Rules
- For CORRECT answers: extract the successful strategy as a reusable pattern. Note inefficiencies that could be trimmed.
- For INCORRECT answers: diagnose what went wrong AND propose what the correct tool trace SHOULD have been. The trace field should contain your proposed corrected trace. Set was_correct=false.
- The "answer" field in qtype_example must ALWAYS be the gold answer.
- Only produce an output if it provides genuine value. An empty optional field is better than a forced one.
- Keep lessons and guidance concise and actionable.
"""


def _build_user_prompt(
    question: str,
    gold_answer: str,
    predicted_answer: Optional[str],
    q_type: str,
    tool_trace: List[Dict],
    messages: List[Dict],
    judgment: Dict,
    accuracy: bool,
) -> str:
    timestamp = datetime.now().isoformat()

    trace_overview = ""
    for i, step in enumerate(tool_trace, 1):
        trace_overview += f"  {i}. {step.get('tool', '?')}({step.get('args', '')}) -> {step.get('result', '')}\n"

    msgs_str = ""
    for msg in messages:
        role = msg.get("role", "?")
        if role == "assistant":
            content = msg.get("content", "")
            tool_calls = msg.get("tool_calls", [])
            if tool_calls:
                for tc in tool_calls:
                    name = tc.get("function", {}).get("name", tc.get("name", "?"))
                    args = tc.get("function", {}).get("arguments", tc.get("arguments", ""))
                    if isinstance(args, str) and len(args) > 300:
                        args = args[:300] + "..."
                    msgs_str += f"[ASSISTANT tool_call] {name}({args})\n"
            if content:
                msgs_str += f"[ASSISTANT] {content[:500]}\n"
        elif role == "tool":
            content = msg.get("content", "")[:500]
            msgs_str += f"[TOOL RESULT] {content}\n"
        elif role == "user":
            content = msg.get("content", "")
            if len(content) > 200:
                content = content[:200] + "..."
            msgs_str += f"[USER] {content}\n"

    return f"""## Question Analysis Request
Timestamp: {timestamp}
Question Type: {q_type}
Question: {question}
Gold Answer: {gold_answer}
Predicted Answer: {predicted_answer}
Accuracy: {"CORRECT" if accuracy else "INCORRECT"}

## Judge Verdict
- is_correct: {judgment.get('is_correct', 'N/A')}
- correctness_reasoning: {judgment.get('correctness_reasoning', 'N/A')}
- argumentation_quality: {judgment.get('argumentation_quality', 'N/A')}
- argumentation_score: {judgment.get('argumentation_score', 'N/A')}/5
- suggested_improvement: {judgment.get('suggested_improvement', 'N/A')}

## Tool Trace Overview (abbreviated)
{trace_overview}

## Full Conversation (last messages, truncated)
{msgs_str}

Please analyze this trace and generate fewshot learning material. Return valid JSON only."""


# ============================================================================
# CORE GENERATION FUNCTION
# ============================================================================

def generate_fewshot_from_result(
    question: str,
    gold_answer: str,
    predicted_answer: Optional[str],
    q_type: str,
    tool_trace: List[Dict],
    full_messages: List[Dict],
    judgment: Dict,
    accuracy: bool,
    client: OpenAI,
    model_name: str,
) -> Optional[FewshotGeneratorOutput]:
    """Generate fewshot examples from a single benchmark result."""
    truncated_msgs = _truncate_messages(full_messages)

    user_prompt = _build_user_prompt(
        question=question,
        gold_answer=gold_answer,
        predicted_answer=predicted_answer,
        q_type=q_type,
        tool_trace=tool_trace,
        messages=truncated_msgs,
        judgment=judgment,
        accuracy=accuracy,
    )

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": GENERATOR_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=4000,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content.strip()
        data = json.loads(content)
        return FewshotGeneratorOutput(**data)
    except Exception as e:
        print(f"  [Fewshot Generator] Error: {e}")
        return None


# ============================================================================
# SAVE HELPERS (with deduplication)
# ============================================================================

def _load_json_file(path: Path) -> List[Dict]:
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, Exception):
            return []
    return []


def _save_json_file(path: Path, data: List[Dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _save_qtype_example(example: FewshotQTypeExample) -> bool:
    """Save a qtype example, deduplicating by question text. Returns True if saved."""
    file_path = FEWSHOT_DIR / f"{example.qtype}.json"
    existing = _load_json_file(file_path)
    existing_questions = {ex.get("question", "") for ex in existing}

    if example.question in existing_questions:
        return False

    entry = example.model_dump()
    existing.append(entry)

    # Sort: correct first, then by tool_count (efficiency)
    existing.sort(key=lambda x: (not x.get("was_correct", True), x.get("tool_count", 999)))
    existing = existing[:MAX_PER_QTYPE]

    _save_json_file(file_path, existing)
    return True


def _save_general_example(example: FewshotGeneralExample) -> bool:
    """Save a general example, deduplicating by title. Returns True if saved."""
    file_path = FEWSHOT_DIR / "_general.json"
    existing = _load_json_file(file_path)
    existing_titles = {ex.get("title", "") for ex in existing}

    if example.title in existing_titles:
        return False

    entry = example.model_dump()
    existing.append(entry)

    # Sort: correct first
    existing.sort(key=lambda x: not x.get("was_correct", True))
    existing = existing[:MAX_GENERAL]

    _save_json_file(file_path, existing)
    return True


def _save_tool_tip(tip: ToolTip) -> bool:
    """Save a tool tip, deduplicating by tool_name+problem_pattern. Returns True if saved."""
    file_path = FEWSHOT_DIR / "_tool_tips.json"
    existing = _load_json_file(file_path)
    existing_keys = {
        (ex.get("tool_name", ""), ex.get("problem_pattern", ""))
        for ex in existing
    }

    if (tip.tool_name, tip.problem_pattern) in existing_keys:
        return False

    entry = tip.model_dump()
    existing.append(entry)
    existing = existing[:MAX_TOOL_TIPS]

    _save_json_file(file_path, existing)
    return True


# ============================================================================
# BATCH GENERATION
# ============================================================================

def generate_and_save_fewshot_examples(
    results_data: List[Dict[str, Any]],
    full_results: List[Any],
    agent_name: str,
    result_dir: Path,
) -> Dict[str, int]:
    """Generate fewshot examples from all qualifying benchmark results.

    Args:
        results_data: List of result dicts (from save_results_to_disk serialization)
        full_results: List of QuestionResult objects (for full_messages access)
        agent_name: Agent name (e.g. "kqapro")
        result_dir: Benchmark result directory for audit log

    Returns:
        Dict with counts of generated examples by type
    """
    # Set up LLM client using judge config
    judge_config = load_judge_config()
    api_key = os.getenv(judge_config["api_key_env"], "")
    if not api_key:
        print("[Fewshot Generator] No API key available, skipping generation")
        return {}

    client = OpenAI(api_key=api_key, base_url=judge_config["base_url"])
    model_name = "deepseek/deepseek-v3.2-speciale"

    # Build mapping from question text to full_messages
    messages_by_question: Dict[str, List[Dict]] = {}
    for r in full_results:
        if hasattr(r, "full_messages") and r.full_messages:
            messages_by_question[r.question] = r.full_messages

    # Filter qualifying results
    qualifying = []
    for rd in results_data:
        qual_type = qualifies_for_generation(rd)
        if qual_type:
            qualifying.append((rd, qual_type))

    if not qualifying:
        print("[Fewshot Generator] No qualifying results to generate from")
        return {}

    print(f"\n[Fewshot Generator] Generating examples from {len(qualifying)} qualifying results...")

    counts = {"qtype": 0, "general": 0, "tool_tip": 0}
    audit_log: List[Dict] = []

    for rd, qual_type in qualifying:
        question = rd["question"]
        full_msgs = messages_by_question.get(question, [])
        if not full_msgs:
            continue

        print(f"  Analyzing: {question[:60]}... ({qual_type})")

        output = generate_fewshot_from_result(
            question=question,
            gold_answer=rd.get("gold_answer", rd.get("answer", "")),
            predicted_answer=rd.get("predicted_answer"),
            q_type=rd.get("q_type", rd.get("qtype", "Query")),
            tool_trace=rd.get("tool_trace", []),
            full_messages=full_msgs,
            judgment=rd.get("judgment", {}),
            accuracy=rd.get("accuracy", False),
            client=client,
            model_name=model_name,
        )

        if not output:
            continue

        # Audit log entry
        audit_entry = {
            "question": question,
            "qualification": qual_type,
            "reasoning": output.reasoning,
        }

        if output.qtype_example:
            saved = _save_qtype_example(output.qtype_example)
            if saved:
                counts["qtype"] += 1
            audit_entry["qtype_example"] = output.qtype_example.model_dump()

        if output.general_example:
            saved = _save_general_example(output.general_example)
            if saved:
                counts["general"] += 1
            audit_entry["general_example"] = output.general_example.model_dump()

        if output.tool_tip:
            saved = _save_tool_tip(output.tool_tip)
            if saved:
                counts["tool_tip"] += 1
            audit_entry["tool_tip"] = output.tool_tip.model_dump()

        audit_log.append(audit_entry)

    # Save audit log
    audit_path = result_dir / "generated_fewshot.json"
    _save_json_file(audit_path, audit_log)

    total = sum(counts.values())
    print(f"[Fewshot Generator] Generated {total} examples: "
          f"{counts['qtype']} qtype, {counts['general']} general, {counts['tool_tip']} tool tips")
    print(f"[Fewshot Generator] Audit log: {audit_path}")

    return counts
