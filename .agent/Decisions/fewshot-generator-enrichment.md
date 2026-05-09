# ADR: Fewshot Generator Enrichment — Config-Driven, Tool-Catalog-Aware

**Status:** Shipped (commits `f6bfcdb` + `25409c5`, 2026-05-09)

## Related Docs
- [Project Architecture](../System/project_architecture.md) — fewshot generator section (§6)
- [Agent System](../System/agent_system.md) — generator workflow and agent integration
- [SOP: Running Batch Processing](../SOP/running_batch_processing.md) — how to invoke the generator

---

## Problem

The fewshot generator had three structural weaknesses:

1. **Hardcoded model/params**: `model_name = "deepseek/deepseek-v3.2-speciale"`, `temperature=0.3`, `max_tokens=4000` were compile-time constants. Changing providers or tuning required code edits.
2. **No tool-catalog context**: The generator analyzed traces without knowing what tools the agent had available. It could not judge whether the agent picked a suboptimal tool when a better one existed.
3. **No way to replay without re-running the agent**: The only way to regenerate fewshot examples from a past benchmark was to re-run the full agent evaluation — expensive, non-deterministic, and risky if prompt changes have since occurred.

## Decisions

### 1. Config-Driven Generator Settings (`[fewshot_generator]` in `config.toml`)

New section with these keys (all optional; defaults shown):

```toml
[fewshot_generator]
provider = "openrouter"
model = "deepseek/deepseek-v4-pro"
temperature = 1.0
max_tokens = 16000
include_tool_descriptions = true
include_full_conversation = true
max_messages = 20         # fallback: max messages when full=false
max_result_chars = 500    # fallback: max tool-result chars when full=false
```

`load_generator_config()` (`fewshot_generator.py`) reads this section at runtime
with the above defaults. Old hardcoded constants removed entirely.

### 2. Tool Catalog Injection (`load_tool_catalog`)

`load_tool_catalog(agent_name, sample_messages?)` (`fewshot_generator.py`) builds
the same tool catalog the agent sees at runtime:

1. **MCP server spawn** (preferred): spawns the agent's MCP server via stdio,
   calls `list_tools`, formats via `build_text_mode_tool_catalog`. Cached per agent
   process. Cost: one subprocess per generator run.
2. **Trace extraction** (fallback): scans `sample_messages` for a system message
   that starts with `"AVAILABLE TOOLS"` (the catalog is injected as a system
   message in text-tool-call mode).
3. **Empty string** (final fallback): generator proceeds without catalog.

The catalog is rendered above the trace in the user prompt with the header
`## Available Tools (the agent had access to all of these)`.

`GENERATOR_SYSTEM_PROMPT` now instructs: "Use the AVAILABLE TOOLS catalog to
judge whether a more direct tool existed than the one the agent picked. If it did,
that belongs in the `pitfall` field or as a `tool_tip`."

### 3. Full-Conversation Mode (`include_full_conversation = true`)

When `true`, `_truncate_messages` preserves every message and every tool result
in full (no length caps). This gives the generator complete fidelity at the cost
of larger prompts. Default `max_tokens = 16000` accommodates this.

When `false`, the prior truncation behavior applies: last `max_messages` messages,
tool results capped to `max_result_chars`.

### 4. Post-Hoc Runner (`ama_kbqa/run_fewshot_generator.py`)

Standalone script that replays the generator over a saved benchmark result
directory **without re-running the agent**:

```bash
uv run python -m ama_kbqa.run_fewshot_generator \
    --result-dir benchmark_results/<run-name> \
    --agent kqapro \
    --output-dir db/datasets/kqapro/fewshot-examples.generated-YYYY-MM-DD
```

Loads `results.json` + `tool_traces/question_NNN.json`, constructs `ReplayResult`
shims (`dataclass` with `.question` and `.full_messages`), and calls
`generate_and_save_fewshot_examples`.

**Hard safety guard**: if `--output-dir` resolves to the live `FEWSHOT_DIR`
(`db/datasets/kqapro/fewshot-examples`), the script aborts immediately with a
clear error message. This is intentional — see Motivation below.

Monkey-patches `fewshot_generator.FEWSHOT_DIR` to the shadow path before the call
and restores it in a `finally` block. The save helpers resolve the module attribute
at call time, so patching is sufficient.

## Motivation for Shadow Output Default

The minimax classifier-fix benchmark (commit `f490135`) demonstrated that a
working fix can still produce a worse aggregate score on the specific seed tested
(0.77 vs 0.81 broken). Fewshot examples from a lower-scoring run could
degrade the corpus if accepted blindly. Therefore:

- All generator output lands in a shadow directory by default.
- Human review is required before promoting to the live `fewshot-examples/`.
- The post-hoc runner enforces this by refusing to write to the live dir.

## Alternatives Considered

- **Auto-promote if argumentation_score >= 4**: rejected; the quality bar on a
  lower-scoring run may still be locally high while being globally suboptimal.
- **Separate config file for generator**: unnecessary; `[fewshot_generator]` in
  `config.toml` is consistent with the existing configuration pattern.
- **Always truncate messages**: discards nuance the generator needs to judge
  path optimality. Full-conversation mode is the better default now that
  `max_tokens = 16000` is the budget.
