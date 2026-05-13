# Root-Cause Tool Generalization After Seed-43 Trace Audit

## Related Docs
- [../System/agent_system.md](../System/agent_system.md)
- [../System/project_architecture.md](../System/project_architecture.md)
- [sciqa-cross-resource-aggregation.md](sciqa-cross-resource-aggregation.md)
- [get-qualifier-value-tool.md](get-qualifier-value-tool.md)

## Context

The 2026-05-13 seed-43 trace audit showed that the remaining failures were not mainly caused by missing prompt reminders. They clustered around reusable capability gaps:

- KQAPro `Verify` answers sometimes had the evidence but violated the `yes` / `no` output contract.
- KQAPro `Count` questions needed heterogeneous OR branches where each branch carries its own relation constraint.
- KQAPro qualifier projection needed synonym tolerance for subscriber/follower wording.
- SciQA row questions with multiple column constraints fell back to raw `RunORKGSPARQL`.
- SciQA aggregation missed metrics split across sibling predicates.
- SciQA paper-metadata frequency questions needed values read from Paper resources, not Contribution rows.

## Decision

Implement general tool-surface fixes rather than row-specific fewshot examples or deterministic question patches:

- Add final answer cleanup in `BaseKBQAAgent` for leaked `<think>` blocks and KQAPro `Verify` answer-shape normalization.
- Extend `CountUnion` with branch-local relation filters: `relation_name`, `relation_target_id(s)`, and `relation_direction`.
- Extend `GetQualifierValue` qualifier aliases for subscriber/follower phrasing.
- Add SciQA `QueryComparisonRows` for multi-filter comparison row projection.
- Extend `AggregateComparisonValues` with `value_predicates` to union sibling metric predicates.
- Extend `FindFrequentValues` with `value_source="subject"` for paper/comparison metadata aggregation.

## Consequences

These changes keep the pipeline probabilistic where the model still has to choose scope, predicates, filters, and tools. The deterministic parts are limited to output formatting and reusable query primitives. That matches the user's stated preference to avoid overfitting exact benchmark questions while still removing root causes exposed by multiple traces.

## Validation

Initial local validation:

- `uv run python -m compileall ama_kbqa/framework/base_agent.py ama_kbqa/server/kqapro_server.py ama_kbqa/server/sciqa_server.py ama_kbqa/agents/sciqa_agent/prompts.py tests/framework/test_exact_constraints.py`
- `uv run pytest tests/framework/test_exact_constraints.py -q` = 8 passed

Full framework validation and Hetzner questionnaire runs should follow after the active benchmark session finishes, so the previous run remains uncontaminated.
