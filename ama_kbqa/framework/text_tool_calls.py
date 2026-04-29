"""
Client-side parser for models that don't emit OpenAI-style structured tool_calls.

Some open-source models (e.g. minimax-m2.7 on the KIT endpoint) reply with bare
text saying things like "I need to call GetAttributeDetails" and never populate
the response's `tool_calls` field. The framework then treats the response as a
final answer and breaks out of the loop, so 0 tool calls = 0 accuracy.

The fix is two-sided:
1. Tell the model exactly how to format tool calls in plain text — a strong
   instruction in the system prompt with a small example.
2. After each LLM response, if `tool_calls` is empty but the content contains
   parseable `<tool_call>{...}</tool_call>` blocks, build synthetic tool_call
   objects so the rest of the agent loop runs unchanged.

This mirrors how Llama 3.1 / Hermes 3 / many local-server adapters handle
tool calling for models without native function-call support. The payload
shape `{"name": "...", "arguments": {...}}` matches the OpenAI tool-call
function field, so the synthetic objects can be consumed by the existing
`_execute_tool_calls` path without modification.
"""
from __future__ import annotations

import json
import re
import uuid
from types import SimpleNamespace
from typing import Any, List, Optional


# Model name substrings that need this treatment. Match by `in` on the model id
# so both "minimax-m2.7-kit" and "kit.minimax-m2.7-229b" trigger.
_TEXT_TOOL_CALL_MODEL_PATTERNS = (
    "minimax-m2.7",
    "kit.minimax-m2.7",
)


def needs_text_tool_calls(model_name: str) -> bool:
    """Return True if this model can't produce OpenAI-style tool_calls and
    needs the text-format parsing path."""
    if not model_name:
        return False
    name = model_name.lower()
    return any(p in name for p in _TEXT_TOOL_CALL_MODEL_PATTERNS)


TEXT_TOOL_CALL_INSTRUCTION = """
TOOL CALL FORMAT (CRITICAL — your endpoint does not support native tool calls):
When you want to invoke a tool, output EXACTLY one or more lines in this format,
with NO other prose on the same line:

<tool_call>{"name": "ToolName", "arguments": {"arg1": "value1"}}</tool_call>

Rules:
- The JSON must be valid and on a single line per call.
- Use the exact tool names from the available tools list.
- You may emit multiple <tool_call>...</tool_call> blocks in one reply to call
  several tools in parallel.
- After the tool results are returned, you may emit MORE tool_call blocks in
  later turns until you have enough data.
- When you have enough data and want to give the FINAL answer, omit the
  tool_call blocks entirely and reply with the answer in plain text.
- Do NOT describe in English what tool you would call — actually emit the
  <tool_call> block. "I need to call X" is wrong; "<tool_call>{...}</tool_call>"
  is right.

Example — looking up an entity:
<tool_call>{"name": "FindNode", "arguments": {"semantic_node_name": "Inception"}}</tool_call>

Example — two parallel calls:
<tool_call>{"name": "FindNode", "arguments": {"semantic_node_name": "Alpha"}}</tool_call>
<tool_call>{"name": "FindNode", "arguments": {"semantic_node_name": "Beta"}}</tool_call>
""".strip()


# Greedy on `.*?` (non-greedy with DOTALL) so multiple blocks are extracted independently.
_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def parse_text_tool_calls(content: Optional[str]) -> List[SimpleNamespace]:
    """Extract tool calls from a plain-text content string.

    Returns a list of duck-typed objects matching the shape that
    `base_agent._execute_tool_calls` consumes (tc.id, tc.type, tc.function.name,
    tc.function.arguments).

    Returns an empty list if no parseable blocks are found.
    """
    if not content:
        return []

    out: List[SimpleNamespace] = []
    for match in _TOOL_CALL_RE.finditer(content):
        raw = match.group(1)
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            # Tolerate trailing commas / minor sloppiness by trying a permissive parse
            cleaned = re.sub(r",\s*([}\]])", r"\1", raw)
            try:
                payload = json.loads(cleaned)
            except json.JSONDecodeError:
                continue

        if not isinstance(payload, dict):
            continue
        name = payload.get("name") or payload.get("tool")
        args = payload.get("arguments")
        if args is None:
            args = payload.get("args", {})
        if not name:
            continue

        # OpenAI's tool_call.function.arguments is a JSON STRING, not a dict.
        # Mirror that so the existing dispatch path can json.loads(...) it as usual.
        if isinstance(args, dict):
            args_json = json.dumps(args)
        elif isinstance(args, str):
            args_json = args
        else:
            args_json = json.dumps({"value": args})

        out.append(SimpleNamespace(
            id=f"text_tc_{uuid.uuid4().hex[:8]}",
            type="function",
            function=SimpleNamespace(name=str(name), arguments=args_json),
        ))
    return out


def strip_tool_call_blocks(content: Optional[str]) -> str:
    """Remove `<tool_call>{...}</tool_call>` blocks from content so the
    remaining prose can be stored as the assistant's "thought" without leaking
    raw JSON into downstream summaries."""
    if not content:
        return ""
    return _TOOL_CALL_RE.sub("", content).strip()
