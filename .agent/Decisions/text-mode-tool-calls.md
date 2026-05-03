# ADR: Text-Mode Tool Calls for Non-Native-Function-Call Models

**Date:** 2026-04-29  
**Commits:** `d0978c9` — "Inject text-mode tool catalog; suppress tools= for text-mode models"; `08cadeb` — "Add minimax text-tool-call parser; tighten new-tool adoption"  
**Status:** Accepted

## Related Docs

- [System/agent_system.md](../System/agent_system.md) — BaseKBQAAgent lifecycle, framework file table
- [Decisions/kqapro-tool-surface-expansion.md](./kqapro-tool-surface-expansion.md) — tool surface this mechanism must expose to text-mode models

---

## Context: The Failure Mode

Smoke-testing `minimax-m2.7-kit` on a KQAPro question produced **0 tool calls and 0 accuracy**. The model emitted English prose ("I need to call GetAttributeDetails to find the answer") instead of OpenAI-structured tool_calls. The KIT wrapper for this model cannot translate prose intent into native function calls, so the agent loop never advanced.

Two routes were considered:

| Option | Assessment |
|--------|-----------|
| (a) Drop the model | Simple, but wastes a capable model |
| (b) Client-side parser | More code, but preserves benchmark coverage |

**Chose (b) per user request.**

### Why the naive parser alone wasn't enough

A first attempt (regex over `message.content` for `<tool_call>` blocks) still produced 0 tool calls on subsequent runs. Root cause: the API call still passed `tools=[...]`, which KIT's wrapper expected to translate into native output — so the model kept generating prose instead of `<tool_call>` XML because it was confused by the dual-signal (native tool schema + no structured output). The fix required **also** suppressing `tools=` at the API layer and inlining a plain-text tool catalog into the system prompt.

### Smoke-test post-fix

`minimax-m2.7` answered a KQAPro question with a clean tool sequence:

```
FindByAttribute → GetAttributeDetails → VerifyNumericCondition
```

LLM-judge verdict: **CORRECT**.

---

## Decision

Implement a per-model **text-tool-call mode** in the framework. When active:

1. `tools=None` is passed to the LLM API (no native function-call schema).
2. Two extra system-prompt messages are prepended:
   - `TEXT_TOOL_CALL_INSTRUCTION` — mandates the `<tool_call>{"name":..., "arguments":...}</tool_call>` format.
   - A plain-text tool catalog built by `build_text_mode_tool_catalog`.
3. After each LLM response: if `message.tool_calls` is empty but content contains parseable `<tool_call>` blocks, `parse_text_tool_calls` constructs synthetic OpenAI-shaped tool_calls and the agent loop continues unchanged.
4. Tool results are appended as `role=user` prose wrapped in `<tool_result name="X">...</tool_result>` (not the `role=tool` schema, which minimax wasn't trained on).

**Activation:** `text_tool_calls.needs_text_tool_calls(model_name)` — currently matches `minimax-m2.7` (substring match). Add future models to that function.

**Key files:**

| File | Role |
|------|------|
| `ama_kbqa/framework/text_tool_calls.py` | New module: `needs_text_tool_calls`, `build_text_mode_tool_catalog`, `parse_text_tool_calls`, `TEXT_TOOL_CALL_INSTRUCTION` |
| `ama_kbqa/framework/base_agent.py` | Init hook (detect mode, inject system prompt); agent loop hook (call parser when `tool_calls` empty); `_execute_tool_calls` extension (wrap results as `role=user`) |

---

## Invariants for Future Text-Mode Models

- Only add a model to `needs_text_tool_calls` after verifying its API endpoint does not support native function calls *and* does support plain-text streaming.
- The `<tool_call>` / `<tool_result>` XML tags are the stable surface — do not change tag names without updating both `TEXT_TOOL_CALL_INSTRUCTION` and `parse_text_tool_calls`.
- Text-mode results must **not** use `role=tool`; the `role=user` wrapping is intentional for model compatibility.
