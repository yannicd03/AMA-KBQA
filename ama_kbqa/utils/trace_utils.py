"""Tool trace extraction and few-shot example export utilities.

Extracted from kqapro_agent/batch_runner.py for reuse across all agents.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


# Project root for default paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Legacy default: kept byte-for-byte for callers (and the module-level
# ``fewshot_generator.FEWSHOT_DIR`` constant) that don't specify an agent.
_DEFAULT_FEWSHOT_DIR = PROJECT_ROOT / "db" / "datasets" / "kqapro" / "fewshot-examples"


# ============================================================================
# FILENAME SANITISATION
# ============================================================================

# Any run of whitespace — including embedded newlines. Some datasets (e.g.
# SciQA) emit compound question-type labels joined by a literal "\n"
# (observed: "Factoid\nSuperlative"), which previously produced filenames
# containing raw newline bytes.
_WHITESPACE_RUN_RE = re.compile(r"\s+")

# Path separators, the NUL byte, other control characters (incl. the
# newlines/tabs already collapsed above, kept here as a second net), and the
# handful of characters that are invalid in filenames on common filesystems.
_UNSAFE_FILENAME_CHARS_RE = re.compile(r'[\\/:*?"<>|\x00-\x1f\x7f]')


def sanitize_qtype_for_filename(qtype: str) -> str:
    """Turn a question-type label into a single filesystem-safe path segment.

    Collapses any run of whitespace (including embedded newlines/tabs) to a
    single "_" separator, then replaces path separators and other
    filesystem-hostile characters with "_" as well, so the result is always
    safe to use as a filename stem (e.g. ``f"{sanitize_qtype_for_filename(qtype)}.json"``).

    Plain single-word qtypes (e.g. "Count", "Query") are returned unchanged —
    this keeps existing KQAPro output filenames byte-for-byte identical.
    """
    if not qtype:
        return "Unknown"
    collapsed = _WHITESPACE_RUN_RE.sub("_", qtype.strip())
    safe = _UNSAFE_FILENAME_CHARS_RE.sub("_", collapsed)
    safe = safe.strip("._")
    return safe or "Unknown"


# ============================================================================
# AGENT-AWARE OUTPUT DIRECTORY RESOLUTION
# ============================================================================

def _load_kg_adapter(agent_name: str):
    """Best-effort adapter lookup for ``agent_name``. Returns None if unknown."""
    if agent_name == "kqapro":
        from ama_kbqa.framework.adapters.kqapro_adapter import KQAProAdapter
        return KQAProAdapter()
    if agent_name == "sciqa":
        from ama_kbqa.framework.adapters.sciqa_adapter import SciQAAdapter
        return SciQAAdapter()
    return None


def resolve_fewshot_dir(agent_name: Optional[str] = None) -> Path:
    """Resolve the fewshot-examples output directory for ``agent_name``.

    Precedence:
    1. If ``agent_name`` is None, return the legacy KQAPro default — keeps
       existing callers that don't pass an agent name unaffected.
    2. If the agent's KG adapter declares ``domain_settings["fewshot_examples_dir"]``
       (KQAPro does today, see ``KQAProAdapter._create_config``), use that —
       the adapter is the runtime source of truth for KG-specific paths.
    3. Otherwise derive ``db/datasets/<agent_name>/fewshot-examples`` — this
       is the path SciQA uses today since its adapter declares no such field.
    """
    if agent_name is None:
        return _DEFAULT_FEWSHOT_DIR

    try:
        adapter = _load_kg_adapter(agent_name)
    except Exception:
        adapter = None

    if adapter is not None:
        declared = adapter.config.domain_settings.get("fewshot_examples_dir")
        if declared:
            declared_path = Path(declared)
            return declared_path if declared_path.is_absolute() else PROJECT_ROOT / declared_path

    return PROJECT_ROOT / "db" / "datasets" / agent_name / "fewshot-examples"


# ============================================================================
# TOOL ARGUMENT / RESULT ABBREVIATION
# ============================================================================

def abbreviate_tool_args(tool_name: str, args: Dict[str, Any]) -> str:
    """Extract key parameter(s) from tool arguments as a short string."""
    if tool_name == "FindNode":
        return args.get("search_query", str(args)[:60])
    elif tool_name == "FindByAttribute":
        parts = []
        if args.get("attribute_name"):
            parts.append(args["attribute_name"])
        if args.get("value"):
            parts.append(str(args["value"]))
        return ", ".join(parts) if parts else str(args)[:60]
    elif tool_name == "GetAttributeDetails":
        parts = []
        if args.get("base_node_id"):
            parts.append(args["base_node_id"])
        if args.get("attribute_name"):
            parts.append(args["attribute_name"])
        return ", ".join(parts) if parts else str(args)[:60]
    elif tool_name == "GetRelationDetails":
        parts = []
        if args.get("base_node_id"):
            parts.append(args["base_node_id"])
        if args.get("relation_name"):
            parts.append(args["relation_name"])
        return ", ".join(parts) if parts else str(args)[:60]
    elif tool_name == "GetNodeSummary":
        return args.get("node_id", str(args)[:60])
    elif tool_name == "RunSPARQL":
        query = args.get("query", args.get("sparql_query", ""))
        if not query:
            return str(args)[:60]
        query_upper = query.upper().strip()
        if "COUNT" in query_upper:
            return "COUNT " + query[:50].replace("\n", " ")
        elif query_upper.startswith("ASK"):
            return "ASK " + query[:50].replace("\n", " ")
        else:
            return query[:60].replace("\n", " ")
    elif tool_name == "CompareEntities":
        ids = args.get("entity_ids", [])
        attr = args.get("attribute_name", "")
        return f"{','.join(ids[:3])}, {attr}" if ids else str(args)[:60]
    elif tool_name == "VerifyNumericCondition":
        v1 = args.get("value1", "")
        op = args.get("operator", "")
        v2 = args.get("value2", "")
        return f"{v1} {op} {v2}"
    elif tool_name == "ManageJournal":
        action = args.get("action", "")
        content = args.get("content", "")
        return f"{action}: {content[:40]}" if content else action
    else:
        return str(args)[:60]


def abbreviate_tool_result(tool_name: str, result_str: str) -> str:
    """Extract key finding from a tool result string."""
    if not result_str:
        return "no result"

    try:
        data = json.loads(result_str)
    except (json.JSONDecodeError, TypeError):
        return result_str[:80].replace("\n", " ")

    if tool_name == "FindNode":
        if isinstance(data, dict):
            matches = data.get("matches", data.get("results", []))
            if isinstance(matches, list) and matches:
                first = matches[0]
                nid = first.get("id", first.get("node_id", "?"))
                label = first.get("label", first.get("name", ""))
                return f"found {nid} ({label})"
            if data.get("id") or data.get("node_id"):
                nid = data.get("id", data.get("node_id", "?"))
                label = data.get("label", data.get("name", ""))
                return f"found {nid} ({label})"
        return "no matches"
    elif tool_name == "GetAttributeDetails":
        if isinstance(data, dict):
            value = data.get("value", data.get("attribute_value", ""))
            unit = data.get("unit", "")
            if value:
                return f"{value} {unit}".strip()
        return "not found"
    elif tool_name == "GetRelationDetails":
        if isinstance(data, dict):
            targets = data.get("targets", data.get("results", data.get("relations", [])))
            if isinstance(targets, list):
                n = len(targets)
                if n > 0:
                    first_label = ""
                    if isinstance(targets[0], dict):
                        first_label = targets[0].get("label", targets[0].get("name", ""))
                    return f"{n} targets, first: {first_label}" if n > 1 else first_label
                return "0 targets"
        return result_str[:80].replace("\n", " ")
    elif tool_name == "RunSPARQL":
        if isinstance(data, dict):
            results_data = data.get("results", data.get("bindings", []))
            if isinstance(results_data, list):
                n = len(results_data)
                if n > 0:
                    first = results_data[0]
                    if isinstance(first, dict):
                        first_val = next(iter(first.values()), "")
                        if isinstance(first_val, dict):
                            first_val = first_val.get("value", str(first_val)[:30])
                        return f"{n} rows, first: {first_val}"
                return "0 results"
            count = data.get("count", data.get("result", ""))
            if count != "":
                return str(count)
        return result_str[:80].replace("\n", " ")
    elif tool_name == "VerifyNumericCondition":
        if isinstance(data, dict):
            return "TRUE" if data.get("result", False) else "FALSE"
        return str(data)[:40]
    elif tool_name == "CompareEntities":
        if isinstance(data, dict):
            highest = data.get("highest", data.get("result", ""))
            if highest:
                return f"highest: {highest}"
        return result_str[:80].replace("\n", " ")
    elif tool_name == "ManageJournal":
        return "updated"
    else:
        return result_str[:80].replace("\n", " ")


# ============================================================================
# TOOL TRACE EXTRACTION
# ============================================================================

def extract_tool_trace(messages: List[Any], max_steps: int = 8) -> List[Dict]:
    """
    Extract abbreviated tool trace from agent message history.

    Args:
        messages: The agent's message history (list of dicts or objects)
        max_steps: Maximum number of trace steps to keep

    Returns:
        List of trace step dicts with 'tool', 'args', and 'result' keys
    """
    trace = []
    skip_tools = {"GetJournalSummary"}

    for msg in messages:
        if isinstance(msg, dict):
            role = msg.get("role")
            tool_calls = msg.get("tool_calls", [])
            tool_call_id = msg.get("tool_call_id")
            content = msg.get("content", "")
        else:
            role = getattr(msg, "role", None)
            tool_calls = getattr(msg, "tool_calls", [])
            tool_call_id = getattr(msg, "tool_call_id", None)
            content = getattr(msg, "content", "")

        if role == "assistant" and tool_calls:
            for tc in tool_calls:
                if isinstance(tc, dict):
                    tool_name = tc.get("function", {}).get("name", "")
                    args_str = tc.get("function", {}).get("arguments", "{}")
                    call_id = tc.get("id", "")
                else:
                    tool_name = tc.function.name
                    args_str = tc.function.arguments
                    call_id = tc.id

                if tool_name in skip_tools:
                    continue

                try:
                    args = json.loads(args_str) if isinstance(args_str, str) else args_str
                except json.JSONDecodeError:
                    args = {}

                if tool_name == "ManageJournal" and args.get("action") == "read":
                    continue

                trace.append({
                    "tool": tool_name,
                    "args": abbreviate_tool_args(tool_name, args),
                    "call_id": call_id,
                    "result": None
                })

        elif role == "tool" and (tool_call_id if isinstance(msg, dict) else getattr(msg, "tool_call_id", None)):
            tid = tool_call_id if isinstance(msg, dict) else getattr(msg, "tool_call_id", None)
            for step in reversed(trace):
                if step.get("call_id") == tid:
                    step["result"] = abbreviate_tool_result(step["tool"], content)
                    break

    cleaned = []
    for step in trace:
        step.pop("call_id", None)
        if not step["result"]:
            step["result"] = "no result"
        cleaned.append(step)

    return cleaned[:max_steps]


# ============================================================================
# LESSON GENERATION & FEW-SHOT EXPORT
# ============================================================================

def generate_lesson(qtype: str, trace: List[Dict]) -> str:
    """Generate a one-sentence lesson from the tool trace pattern."""
    tools_used = [s["tool"] for s in trace]
    if "RunSPARQL" in tools_used:
        return f"Used SPARQL for {qtype} question after entity discovery."
    primary_tools = [t for t in tools_used if t not in ("ManageJournal",)]
    seen = {}
    unique_tools = []
    for t in primary_tools:
        if t not in seen:
            seen[t] = True
            unique_tools.append(t)
    return f"Solved via: {' -> '.join(unique_tools)}"


def export_fewshot_examples_from_traces(
    results: List[Dict[str, Any]],
    output_dir: Path = None,
    max_tool_count: int = 15,
    max_per_type: int = 5,
    agent_name: Optional[str] = None,
) -> Dict[str, int]:
    """
    Export correct answers with tool traces as few-shot examples.

    Filters for correct, efficient solutions and saves them in the
    tool-trace format grouped by question type.

    Args:
        results: List of batch processing results with tool_trace field
        output_dir: Directory to save examples. If None, resolved from
            ``agent_name`` via :func:`resolve_fewshot_dir` (defaults to the
            legacy KQAPro path if ``agent_name`` is also None).
        max_tool_count: Maximum tool calls to accept (reject verbose runs)
        max_per_type: Maximum examples to keep per question type
        agent_name: Agent that produced ``results`` (e.g. "kqapro", "sciqa").
            Only used to resolve ``output_dir`` when it isn't given
            explicitly — pass this so each agent's traces land in its own
            directory instead of all sharing the KQAPro default.

    Returns:
        Dictionary mapping question types to number of examples exported
    """
    if output_dir is None:
        output_dir = resolve_fewshot_dir(agent_name)

    output_dir.mkdir(exist_ok=True, parents=True)

    examples_by_type: Dict[str, List[Dict]] = {}

    for result in results:
        if not result.get("accuracy", False):
            continue

        qtype = result.get("qtype", "Unknown")
        if qtype == "Unknown":
            continue

        trace = result.get("tool_trace", [])
        if not trace:
            continue

        tool_summary = result.get("tool_call_summary", {})
        tool_count = tool_summary.get("total_calls", len(trace))

        if tool_count > max_tool_count:
            continue

        example = {
            "question": result["question"],
            "answer": str(result.get("answer", result.get("gold_answer", ""))),
            "qtype": qtype,
            "trace": trace,
            "lesson": generate_lesson(qtype, trace),
            "tool_count": tool_count,
            "collected_at": datetime.now().isoformat()
        }

        if qtype not in examples_by_type:
            examples_by_type[qtype] = []
        examples_by_type[qtype].append(example)

    export_counts = {}

    for qtype, examples in examples_by_type.items():
        example_file = output_dir / f"{sanitize_qtype_for_filename(qtype)}.json"

        existing_examples = []
        if example_file.exists():
            try:
                with open(example_file, "r", encoding="utf-8") as f:
                    existing_examples = json.load(f)
            except json.JSONDecodeError:
                print(f"[WARNING] Could not load existing examples from {example_file}, overwriting")

        existing_questions = {ex.get("question", "") for ex in existing_examples}
        new_examples = [ex for ex in examples if ex["question"] not in existing_questions]

        if new_examples:
            all_examples = existing_examples + new_examples
            all_examples.sort(key=lambda x: x.get("tool_count", 999))
            all_examples = all_examples[:max_per_type]

            with open(example_file, "w", encoding="utf-8") as f:
                json.dump(all_examples, f, indent=2, ensure_ascii=False)

            export_counts[qtype] = len(new_examples)
            print(f"[OK] Exported {len(new_examples)} new trace examples for {qtype} (total: {len(all_examples)})")
        else:
            export_counts[qtype] = 0

    return export_counts
