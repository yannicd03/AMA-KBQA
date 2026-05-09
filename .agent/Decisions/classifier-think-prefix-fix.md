# ADR: Classifier Think-Prefix Fix and `_extract_json_object` Helper

**Status:** Shipped (commit `f490135`, 2026-05-09)

## Related Docs
- [Project Architecture](../System/project_architecture.md) — overall system overview
- [Agent System](../System/agent_system.md) — agent lifecycle, classification hook
- [text-mode-tool-calls](./text-mode-tool-calls.md) — minimax-m2.7 text-mode handling context

---

## Problem

`minimax-m2.7-kit` emits `<think>...</think>` reasoning prefixes in responses to
classification requests **even when `response_format={"type": "json_object"}` is
set**. This caused `json.loads()` to raise on every classification response,
silently falling back to `"Query"` for all questions. Verified empirically: a
stratified n=100 KQAPro benchmark with the broken classifier produced the same
"Query" label for 100/100 questions, defeating all qtype-specific fewshot files
and strategies.

The root cause: `max_tokens=300` was also insufficient for the model to emit the
full `<think>` block plus the JSON object. With a capped token budget the think
block was silently truncated, leaving malformed JSON. Both failures were confirmed
via a direct API probe (`/tmp/classifier_test.py`).

## Decision

1. **`BaseKBQAAgent._extract_json_object` static helper** (`ama_kbqa/framework/base_agent.py`):
   - Strip `<think>...</think>` blocks with `re.sub(r"<think>.*?</think>", "", ..., flags=re.DOTALL)`.
   - Strip markdown code fences.
   - Try `json.loads()` on the cleaned string.
   - Brace-balanced fallback parser: scan for the first `{`, count depth, extract the balanced block.
   - Returns `Optional[Dict[str, Any]]`; `None` on failure.

2. **Classifier `max_tokens` bumped from 300 → 1500** in both `BaseKBQAAgent` (`_classify_question`) and `SciQAAgent` (`agent.py`). This gives the model room to finish the think block and emit complete JSON.

Both agents now call `_extract_json_object(json_content)` instead of `json.loads()` directly.

## Alternatives Considered

- **Strip only the think block in the caller**: fragile; any future model adding reasoning tokens in a different format would break again. Centralising in a helper is more durable.
- **Disable `response_format`**: would require downstream parsing everywhere. Not taken.
- **Model-specific workaround in minimax path only**: `_extract_json_object` is model-agnostic. Keeping it in `BaseKBQAAgent` means all subclasses benefit automatically.

## Empirical Results

Seed=42 stratified n=100 KQAPro comparison:
- Broken classifier (silent "Query" for all): 0.81 accuracy
- Fixed classifier (correct qtype routing): 0.77 accuracy

The fix is architecturally correct and necessary — qtype routing now functions as
designed. The accuracy delta is **not** evidence against the fix. Post-hoc
analysis: Count/Verify/Select improved; Query/QRQ/QAQ regressed. Hypothesis: the
existing QRQ/QAQ fewshot files are small and targeted in ways that can mislead the
agent when qtype is correctly identified. That is a fewshot quality problem, not a
classification problem.

**Do not interpret the 0.77 number as current performance.** It reflects one run
on a specific seed with the existing fewshot corpus. The architectural decision
(accurate classification) is correct; fewshot quality is a separate concern.

## Reusability

`_extract_json_object` is a general-purpose helper on `BaseKBQAAgent`. Any
subclass that needs JSON from an LLM that may emit think blocks or markdown fences
should use it instead of `json.loads()` directly.
