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

import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import toml
from openai import OpenAI
from pydantic import BaseModel, Field

from ama_kbqa.postprocessing import load_judge_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEWSHOT_DIR = PROJECT_ROOT / "db" / "datasets" / "kqapro" / "fewshot-examples"
CONFIG_PATH = PROJECT_ROOT / "config.toml"

# Limits
MAX_PER_QTYPE = 5
MAX_GENERAL = 10
MAX_TOOL_TIPS = 20


# ============================================================================
# CONFIG
# ============================================================================

def load_generator_config() -> Dict[str, Any]:
    """Load [fewshot_generator] settings from config.toml with sane defaults."""
    defaults = {
        "provider": "openrouter",
        "model": "deepseek/deepseek-v4-pro",
        "temperature": 1.0,
        "max_tokens": 16000,
        "include_tool_descriptions": True,
        "include_full_conversation": True,
        "max_messages": 20,
        "max_result_chars": 500,
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
    }
    if not CONFIG_PATH.exists():
        return defaults
    try:
        cfg = toml.load(CONFIG_PATH)
        gen = cfg.get("fewshot_generator", {}) or {}
        provider = gen.get("provider", defaults["provider"])
        provider_cfg = cfg.get(provider, {}) or {}
        return {
            "provider": provider,
            "model": gen.get("model", defaults["model"]),
            "temperature": float(gen.get("temperature", defaults["temperature"])),
            "max_tokens": int(gen.get("max_tokens", defaults["max_tokens"])),
            "include_tool_descriptions": bool(gen.get("include_tool_descriptions", True)),
            "include_full_conversation": bool(gen.get("include_full_conversation", True)),
            "max_messages": int(gen.get("max_messages", defaults["max_messages"])),
            "max_result_chars": int(gen.get("max_result_chars", defaults["max_result_chars"])),
            "base_url": provider_cfg.get("base_url", defaults["base_url"]),
            "api_key_env": (
                "OPENROUTER_API_KEY" if provider == "openrouter"
                else f"{provider.upper()}_API_KEY"
            ),
        }
    except Exception as e:
        print(f"[Fewshot Generator] Config load failed ({e}); using defaults")
        return defaults


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

def _truncate_messages(
    messages: List[Dict],
    max_messages: int = 20,
    max_result_chars: int = 500,
    full: bool = False,
) -> List[Dict]:
    """Optionally truncate conversation messages.

    When ``full=True`` returns the original message list unmodified — the
    generator gets the entire trace (full tool results, every turn) so it can
    judge whether the chosen path was the most direct one available.

    When ``full=False`` the tail ``max_messages`` are returned with each tool
    result clipped to ``max_result_chars`` characters.
    """
    if full:
        return list(messages)
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
# TOOL CATALOG LOADER
# ============================================================================

_TOOL_CATALOG_CACHE: Dict[str, str] = {}


def _resolve_mcp_server_path(agent_name: str) -> Optional[str]:
    """Map agent name → absolute MCP server script path."""
    server_dir = PROJECT_ROOT / "ama_kbqa" / "server"
    candidates = {
        "kqapro": server_dir / "kqapro_server.py",
        "sciqa": server_dir / "sciqa_server.py",
    }
    p = candidates.get(agent_name)
    return str(p) if p and p.exists() else None


async def _fetch_tool_catalog_async(server_path: str, agent_name: str) -> str:
    """Spawn the MCP server, list its tools, return a plain-text catalog."""
    from ama_kbqa.framework.mcp_client import MCPClient
    from ama_kbqa.framework.text_tool_calls import build_text_mode_tool_catalog

    client = MCPClient(server_path, agent_name)
    try:
        await client.start()
        mcp_tools = await client.list_tools()
        openai_tools = client.convert_tools_to_openai_format(mcp_tools)
        return build_text_mode_tool_catalog(openai_tools)
    finally:
        try:
            await client.close()
        except Exception:
            pass


def _extract_catalog_from_messages(messages: List[Dict]) -> Optional[str]:
    """Some text-mode runs already inject the catalog as a system message."""
    for m in messages or []:
        if m.get("role") != "system":
            continue
        content = m.get("content", "")
        if isinstance(content, str) and content.startswith("AVAILABLE TOOLS"):
            return content
    return None


def load_tool_catalog(agent_name: str, sample_messages: Optional[List[Dict]] = None) -> str:
    """Return the agent's tool catalog. Cached per agent.

    Tries (1) the cache, (2) the MCP server, (3) extraction from a saved trace.
    Returns "" if no source works — the generator just gets fewer hints.
    """
    if agent_name in _TOOL_CATALOG_CACHE:
        return _TOOL_CATALOG_CACHE[agent_name]

    server_path = _resolve_mcp_server_path(agent_name)
    if server_path:
        try:
            catalog = asyncio.run(_fetch_tool_catalog_async(server_path, agent_name))
            if catalog:
                _TOOL_CATALOG_CACHE[agent_name] = catalog
                return catalog
        except Exception as e:
            print(f"[Fewshot Generator] MCP catalog fetch failed for {agent_name}: {e}")

    if sample_messages:
        from_msgs = _extract_catalog_from_messages(sample_messages)
        if from_msgs:
            _TOOL_CATALOG_CACHE[agent_name] = from_msgs
            return from_msgs

    _TOOL_CATALOG_CACHE[agent_name] = ""
    return ""


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
- For CORRECT answers: extract the successful strategy as a reusable pattern. Critically assess path optimality — if the agent reached the right answer via a roundabout route, the lesson should describe the SHORTER path the agent should have taken, and the trace should reflect that shorter path (not the actual one). Note inefficiencies (redundant lookups, repeated FindNode calls, exploratory SPARQL that could have been skipped given the available tools).
- For INCORRECT answers: diagnose what went wrong AND propose what the correct tool trace SHOULD have been. The trace field should contain your proposed corrected trace. Set was_correct=false.
- Use the AVAILABLE TOOLS catalog (when provided) to judge whether a more direct tool existed than the one the agent picked. If it did, that belongs in the `pitfall` field of the qtype_example or as a `tool_tip`.
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
    tool_catalog: str = "",
    full_conversation: bool = True,
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
                    if isinstance(args, str) and not full_conversation and len(args) > 300:
                        args = args[:300] + "..."
                    msgs_str += f"[ASSISTANT tool_call] {name}({args})\n"
            if content:
                rendered = content if full_conversation else content[:500]
                msgs_str += f"[ASSISTANT] {rendered}\n"
        elif role == "tool":
            content = msg.get("content", "")
            if not full_conversation:
                content = content[:500]
            msgs_str += f"[TOOL RESULT] {content}\n"
        elif role == "user":
            content = msg.get("content", "")
            if not full_conversation and len(content) > 200:
                content = content[:200] + "..."
            msgs_str += f"[USER] {content}\n"
        elif role == "system":
            content = msg.get("content", "")
            if isinstance(content, str) and content.startswith("AVAILABLE TOOLS"):
                # Skip — catalog rendered separately above.
                continue
            if full_conversation:
                msgs_str += f"[SYSTEM] {content}\n"

    catalog_block = ""
    if tool_catalog:
        catalog_block = f"## Available Tools (the agent had access to all of these)\n{tool_catalog}\n\n"

    convo_header = (
        "## Full Conversation (entire trace)"
        if full_conversation
        else "## Full Conversation (last messages, truncated)"
    )

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

{catalog_block}## Tool Trace Overview
{trace_overview}

{convo_header}
{msgs_str}

Please analyze this trace and generate fewshot learning material. Critically assess path optimality given the AVAILABLE TOOLS list. Return valid JSON only."""


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
    *,
    temperature: float = 1.0,
    max_tokens: int = 16000,
    tool_catalog: str = "",
    full_conversation: bool = True,
    max_messages: int = 20,
    max_result_chars: int = 500,
) -> Optional[FewshotGeneratorOutput]:
    """Generate fewshot examples from a single benchmark result."""
    msgs = _truncate_messages(
        full_messages,
        max_messages=max_messages,
        max_result_chars=max_result_chars,
        full=full_conversation,
    )

    user_prompt = _build_user_prompt(
        question=question,
        gold_answer=gold_answer,
        predicted_answer=predicted_answer,
        q_type=q_type,
        tool_trace=tool_trace,
        messages=msgs,
        judgment=judgment,
        accuracy=accuracy,
        tool_catalog=tool_catalog,
        full_conversation=full_conversation,
    )

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": GENERATOR_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
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
    # Load generator settings from config.toml ([fewshot_generator] section).
    gen_cfg = load_generator_config()
    api_key = os.getenv(gen_cfg["api_key_env"], "")
    if not api_key:
        # Fall back to the judge's API key — same provider in most setups.
        judge_config = load_judge_config()
        api_key = os.getenv(judge_config["api_key_env"], "")
    if not api_key:
        print("[Fewshot Generator] No API key available, skipping generation")
        return {}

    client = OpenAI(api_key=api_key, base_url=gen_cfg["base_url"])
    model_name = gen_cfg["model"]
    print(
        f"[Fewshot Generator] model={model_name} temp={gen_cfg['temperature']} "
        f"max_tokens={gen_cfg['max_tokens']} "
        f"tool_descriptions={gen_cfg['include_tool_descriptions']} "
        f"full_conversation={gen_cfg['include_full_conversation']}"
    )

    # Load tool catalog once for this agent (kqapro/sciqa) — shared across results.
    tool_catalog = ""
    if gen_cfg["include_tool_descriptions"]:
        sample_msgs = next(
            (r.full_messages for r in full_results
             if hasattr(r, "full_messages") and r.full_messages),
            None,
        )
        tool_catalog = load_tool_catalog(agent_name, sample_messages=sample_msgs)
        if tool_catalog:
            print(f"[Fewshot Generator] Loaded tool catalog ({len(tool_catalog)} chars)")
        else:
            print(f"[Fewshot Generator] No tool catalog available for {agent_name}")

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
            temperature=gen_cfg["temperature"],
            max_tokens=gen_cfg["max_tokens"],
            tool_catalog=tool_catalog,
            full_conversation=gen_cfg["include_full_conversation"],
            max_messages=gen_cfg["max_messages"],
            max_result_chars=gen_cfg["max_result_chars"],
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
