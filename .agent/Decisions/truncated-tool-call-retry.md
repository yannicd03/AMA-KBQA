# ADR: Truncated Text-Mode Tool-Call Retry

**Date:** 2026-05-05
**Status:** Accepted
**Code:**
- `ama_kbqa/framework/text_tool_calls.py` — `has_truncated_tool_call(content) -> bool` helper added
- `ama_kbqa/framework/base_agent.py` — agent loop "no tool calls" branch extended with truncation detection + corrective user message

## Related Docs

- [Decisions/text-mode-tool-calls.md](./text-mode-tool-calls.md) — original text-tool-call parser; context for why minimax-m2.7 uses the `<tool_call>` XML shim
- [Decisions/qualifier-and-format-prompt-hardening.md](./qualifier-and-format-prompt-hardening.md) — commit `d9ae8de` ships the zero-tool-call retry (RULE 0 enforcement); this ADR is structurally analogous but triggered by a different failure mode

---

## Context

The n=100 audit run (`benchmark_results/minimax-getqualval-2026-05-05-1100`, minimax-m2.7-kit, unstratified, seed=42) showed **6 of 20 failures (30% of all failures, the single largest mode)** caused by the model voluntarily stopping generation mid-`<tool_call>` block.

Example truncated turn:

```
<tool_call>{"name": "FindNode
```

The model terminated output there — no closing brace, no `</tool_call>`. The text-mode parser (`parse_text_tool_calls`) returned zero tool calls (no valid block found). The agent loop then fell through to its "no tool calls returned → break to synthesis" path, and the truncated fragment became the question's final answer.

### Why this is not a token-budget issue

Per-question token tracking (via `token_usage` dict stored in each result file) for the 6 truncation failures:

| Question | Total completion tokens |
|---|---|
| Q range | 504 – 1,195 |
| Median | ~750 |

The configured `chat_max_tokens` is 16,000. These questions were nowhere near the cap. This is a **model-side generation quirk** of minimax-m2.7 — occasionally it stops emitting at a mid-block boundary for reasons unrelated to budget. Raising `chat_max_tokens` would not help.

### Why the existing zero-tool-call retry doesn't cover this

The zero-tool-call retry (commit `d9ae8de`, `agent.zero_tool_call_retry_max`) fires when the model emits a response with zero tool calls **and no `<tool_call>` opener**. That retry's corrective message says "you didn't use any tools — call FindNode now." Sending that message in response to a truncated block would be confusing and incorrect: the model *started* a tool call; it just didn't finish it. A different corrective prompt is needed.

---

## Decision

Add a `has_truncated_tool_call` classifier to `text_tool_calls.py` and a targeted retry in the agent loop's "no tool calls" branch.

### Classifier logic (`text_tool_calls.py`)

```python
def has_truncated_tool_call(content: str) -> bool:
    """
    Returns True iff parse_text_tool_calls yielded zero calls AND
    content contains a <tool_call> opener — i.e., the model started
    a tool call block and did not finish it cleanly.
    """
    calls = parse_text_tool_calls(content)
    if calls:
        return False  # well-formed; not truncated
    return "<tool_call>" in content
```

Five boundary cases covered (verified locally):

| Input | Returns |
|---|---|
| Well-formed `<tool_call>...</tool_call>` block | False |
| `<tool_call>{"name": "FindNode` (no closing) | True |
| `<tool_call>{"name": "X", "arguments": {INVALID JSON}}` | True |
| Plain prose (no `<tool_call>` at all) | False |
| Empty string | False |

### Agent loop change (`base_agent.py`)

In the "no tool calls returned" branch, before the existing zero-tool-call retry:

```python
if self._text_tool_call_mode and has_truncated_tool_call(message.content):
    # Persist the broken assistant turn so the model sees its own output
    messages.append({"role": "assistant", "content": message.content})
    messages.append({
        "role": "user",
        "content": (
            "Your previous response contained an unclosed or unparseable "
            "<tool_call> block. Re-emit the tool call you intended as ONE "
            "complete block on a single line, with valid JSON:\n"
            "<tool_call>{\"name\": \"...\", \"arguments\": {...}}</tool_call>"
        )
    })
    continue  # re-enter the loop without consuming a zero_tool_call_retry
```

**Budget:** Reuses `zero_tool_call_retry_max` (config knob `agent.zero_tool_call_retry_max`) so the retry is bounded. The truncation retry and the zero-tool-call retry share the same counter — combined, they cannot exceed `zero_tool_call_retry_max` iterations beyond a stuck point.

**Persistence of the broken turn:** The broken assistant message is appended to `messages` before the corrective user message. This lets the model see exactly what it emitted, making the correction prompt unambiguous — "the thing you just output had an unclosed block."

### Why only for text-tool-call mode

Native function-call models return structured `tool_calls` objects — truncation at the token level produces an API error (malformed JSON from the endpoint), not a silent zero-tool-call result. This failure mode is exclusive to text-mode models that self-emit `<tool_call>` XML.

### Structural parallel to zero-tool-call retry

| Retry type | Trigger | Corrective message | Commit |
|---|---|---|---|
| Zero-tool-call (RULE 0) | 0 tool calls, no `<tool_call>` opener | "Call FindNode / FindByAttribute first" | `d9ae8de` |
| Truncated-tool-call | 0 parsed calls, `<tool_call>` opener present | "Re-emit the block as one complete line" | this change |

---

## Validation

Sanity-tested locally with all 5 boundary cases — classifier returns correctly for each. Full A/B benchmark comparison (stratified, seed=42, `minimax-kqapro-2026-05-05-fewshot`) in-flight at time of writing.

**Expected impact:** ~6 pp on minimax-m2.7-kit (Mode A failures, 30% of wrongs). If the retry causes the model to re-emit the same truncated block on every retry, add a consecutive-truncation guard before the next iteration.

---

## Invariants for Future Text-Mode Models

- When adding a new text-mode model, check whether it exhibits mid-block truncation: inspect traces for `<tool_call>` openers with no closing tag in the 20 failures.
- The corrective message explicitly shows the expected format — this is intentional; prose-only instructions to "try again" are insufficient for format compliance.
- Do not increase `chat_max_tokens` as a response to this failure mode — it is not a budget issue.
