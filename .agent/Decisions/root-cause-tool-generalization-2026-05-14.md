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

Post-deploy validation on 2026-05-14:

- Local and Hetzner `uv run pytest tests/framework -q` = 130 passed.
- Hetzner default questionnaire run: 62/80 = 77.5%.
- Full KIT n=100 seed=43 run:
  - KQAPro MiniMax: 84/100, errors 0.
  - KQAPro Gemma: 79/100, errors 1.
  - SciQA MiniMax: 63/100, errors 2.
  - SciQA Gemma: 67/100, errors 2.
  - Aggregate: 293/400 = 73.25%.

The fix set produced a modest aggregate gain over the prior full seed-43 run
(286/400 = 71.5%). The new traces show that the remaining gap is not mainly
caused by missing row-specific fewshots.

## 2026-05-15 Follow-Up Trace Findings

Root causes still visible in the full n=100 run:

- KQAPro MiniMax rows reported as zero-tool failures are mostly unsafe fast-path
  returns. The fast path calls tools directly but does not append those calls to
  the normal trace/tool summary, then asks the model to synthesize from a small
  context. Several failures are unsupported "not available" answers or wrong
  entity answers. This is both a trace observability bug and an answer-quality
  risk.
- KQAPro `CountEntities` has two reusable issues: incomplete primary
  conditions are not ignored when the real condition is in `not_conditions`
  (`attribute_name` without `attribute_value` polluted the query), and concept
  labels such as "TV series" still need more robust exact concept resolution /
  subclass handling. Gemma sometimes found the correct `CountEntities` result
  and then overrode it with a later raw-SPARQL count.
- KQAPro qualifier questions still fail when the model does not know to call
  `GetQualifierValue`. `GetAttributeDetails("Twitter username")` confirms the
  username but does not expose attached qualifier metadata, so the model cannot
  discover that "subscribers/followers" is a qualifier one call away.
- SciQA exact resource lookup is still brittle. `FindResource(...,
  node_type_filter="Comparison")` can return no results while unfiltered search
  returns a `node_type="comparison"` candidate. The RDF-class filter is too
  strict for resources whose payload class and RDF type disagree or are absent.
- SciQA nested comparison aggregation remains the biggest gap. The model often
  finds a semantically adjacent Comparison, then aggregates the right predicate
  over the wrong scope. Energy-scenario questions repeatedly used adjacent
  comparisons and averaged nested energy-source values that did not match gold.
- SciQA row questions with multiple column constraints are still under-supported
  in practice. `QueryComparisonRows` exists but was rarely adopted; traces show
  the model manually probing resources and raw SPARQL instead of using a
  structured row/filter/projection tool path.
- Context-window failures remain an infra-quality issue: 5/400 questions failed
  with KIT/litellm context-window errors. Current trimming mostly targets old
  tool results, while long assistant thoughts, large tool catalogs, and verbose
  summaries can still push the request beyond provider limits.

Concrete fixes to implement next:

- Replace the KQAPro fast path with an evidence-preserving version that records
  tool calls through the normal executor and only returns when the requested
  relation/attribute value is present in the retrieved data; otherwise fall back
  to the full loop. For benchmarks, disable synthesis-only fast-path answers.
- Harden `CountEntities` to ignore incomplete primary attribute filters, add an
  exact concept-label resolution helper before building the type clause, and
  surface a "trusted answer-producing count" marker so final synthesis prefers
  high-level count tools over later exploratory raw SPARQL.
- Enrich `GetAttributeDetails` responses with compact qualifier metadata for
  each attribute value when qualifiers exist, so qualifier projection is
  discoverable through normal attribute lookup instead of relying on prompt-only
  tool selection.
- Make `FindResource(node_type_filter=...)` use payload `node_type` as a
  fallback when RDF type filtering returns zero, and add lexical label search
  fallback for quoted titles before semantic search gives up.
- Add a SciQA comparison-schema helper or extend `GetComparisonContributions`
  so it exposes nested value predicates and candidate comparison titles compactly;
  this should guide the model to the right comparison and predicate without
  hard-coding benchmark question titles.
- Add an early context-compaction pass that trims old assistant thoughts and
  verbose injected summaries, not only tool results. Lower SciQA's effective
  context threshold for KIT models and preserve compact journal facts for final
  synthesis.
- Extend `analyze_benchmark_run.py` to flag hidden fast-path returns separately
  from true zero-tool failures, using trace messages plus tool summary
  inconsistencies.

## 2026-05-17 Implementation Follow-Up

Implemented the first low-overfit recovery pass:

- `BaseKBQAAgent` fast path is now evidence-only. It executes tools through the
  normal tool-call recorder, mirrors synthetic tool-call/result messages into
  the conversation trace, and returns only directly extracted attribute or
  relation values. It no longer asks the LLM to synthesize from a broad node
  summary inside the fast path.
- The main tool loop now initializes its zero-tool accounting from tool calls
  already made before the loop, so a fallback answer grounded in fast-path
  evidence is not misclassified as a true zero-tool answer.
- `FindResource(node_type_filter=...)` still prefers RDF type matches, but now
  falls back to the Qdrant payload `node_type` when RDF typing yields no
  candidates. If both semantic/type-filter paths fail, it tries a lexical label
  fallback for exact or quoted titles.
- `used_results/` is ignored locally so copied benchmark artifacts remain
  available for analysis without entering git history.

Validation:

- `uv run python -m compileall ama_kbqa/framework/base_agent.py ama_kbqa/server/sciqa_server.py tests/framework/test_exact_constraints.py`
- `uv run pytest tests/framework/test_exact_constraints.py -q` = 10 passed.
- `uv run pytest tests/framework -q` = 132 passed.

These are still root-cause fixes rather than benchmark-row patches: they change
trace integrity, conservative answer gating, and generic resource discovery.

## 2026-05-17 Hetzner Focused Validation

Deployed commit `f17af03` to Hetzner, merged into the deployment branch, rebuilt
the frontend container, and re-ran the framework tests on the server:

- Hetzner `uv run pytest tests/framework -q` = 132 passed.
- Focused KQAPro fast-path panel
  (`benchmark_results/focused-kqapro-fastpath-2026-05-17`):
  - MiniMax: 3/4 = 75%.
  - Gemma: 3/4 = 75%.
  - The qualifier-backed fast-path failures now use recorded tools and pass.
  - Remaining miss in both models is the TBS/Eureka row: the agent finds
    `Eureka Seven` among candidates but refuses to commit to it as the answer.
- Focused SciQA lookup/aggregation panel
  (`benchmark_results/focused-sciqa-lookup-2026-05-17`):
  - MiniMax: 0/7 = 0%.
  - Gemma: 0/7 = 0%, with one context/runtime error.
  - `AggregateComparisonValues` is being adopted, but all focused answers are
    wrong because the agent selects the wrong predicate, comparison scope, or
    nested row path after lookup succeeds.

Decision: do not start another full n=100 benchmark from this state. The KQAPro
trace-integrity fix is validated for the targeted fast-path class, but the SciQA
focused panel shows that the remaining root cause is schema-guided comparison
aggregation, not resource-type lookup alone.

Next implementation direction:

- Add a SciQA comparison-schema/row-path helper that lists candidate predicates,
  nested predicates, labels, units, and contribution counts before aggregation.
- Make `AggregateComparisonValues` require or strongly prefer schema-discovered
  predicates for nested rows, with compact diagnostics when the selected
  predicate is only a sibling/domain label rather than the requested metric.
- Add a row-filter plus companion-return path for questions that ask for values
  attached to the max/min or most-frequent row.
- Keep KQAPro work focused on the remaining commitment failure class, not on
  deterministic row patches.

## 2026-05-17 Schema-Guided SciQA Aggregation Pass

Implemented the next low-overfit SciQA fix:

- Added `InspectComparisonSchema`, a compact Comparison schema tool that reports
  direct contribution predicates and two-hop nested paths with labels, row/value
  counts, sample values, numeric/HAS_VALUE evidence, unit samples when present,
  and ready-to-use `AggregateComparisonValues` hints.
- Updated SciQA prompts so comparison factoid/list/comparison/superlative/
  aggregation questions inspect schema before choosing `value_predicate` or
  `intermediate_predicate`.
- Updated system docs to show the SciQA domain-specific tool tier as 11 tools.

Rationale: the focused SciQA failures were no longer mainly lookup failures.
The agent reached Comparison resources but guessed predicates or nested row
paths. This helper still keeps the model responsible for choosing scope and
metric semantics, while removing the brittle predicate/path discovery step that
was forcing it toward raw SPARQL or wrong `AggregateComparisonValues` calls.

Validation before deployment:

- `uv run python -m compileall ama_kbqa/server/sciqa_server.py ama_kbqa/agents/sciqa_agent/prompts.py tests/framework/test_exact_constraints.py`
- `uv run pytest tests/framework/test_exact_constraints.py -q` = 11 passed.
- `uv run pytest tests/framework -q` = 133 passed.

Follow-up from the first focused schema run:

- Focused SciQA schema validation improved from 0/7 on both KIT models to
  MiniMax 1/7 and Gemma 3/7, but traces showed a remaining generic gap:
  schema inspection exposed nested paths, while `AggregateComparisonValues`
  could not restrict those paths to a named intermediate row such as
  `Atmosphere`.
- Added `intermediate_filter_value` / `intermediate_filter_match` to
  `AggregateComparisonValues` so nested row aggregation can express patterns
  like `Contribution -> Earth System Model -> Atmosphere -> prognostic
  variables` without raw SPARQL.
- Updated `InspectComparisonSchema` hints and SciQA prompts so named nested
  components/categories are passed as `intermediate_filter_value`.
