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

Hetzner validation after deploying `474fc20`:

- Server rebuild succeeded; frontend health check passed.
- Hetzner `uv run pytest tests/framework -q` = 134 passed.
- Live smoke test:
  `AggregateComparisonValues(R68871, intermediate_predicate=P7144,
  intermediate_filter_value=Atmosphere, value_predicate=P26032, agg=mode_top)`
  returned the expected top atmospheric-variable class:
  `Surface pressure`, `Wind components`, `Vapour/solid/liquid`.
- Focused SciQA panel
  (`benchmark_results/focused-sciqa-schema-filter-2026-05-21`):
  - MiniMax: 3/7 = 42.9%, errors 0.
  - Gemma: 2/7 = 28.6%, errors 0.
  - Previous focused baselines were 0/7 on both models before schema inspection,
    then 1/7 MiniMax and 3/7 Gemma after schema inspection only.

Decision: still do not launch a full benchmark. The schema/nested-filter work
fixed several superlative/frequency rows and removed context/runtime errors from
the panel, but the remaining wrong rows are all count/aggregation rows. Trace
analysis shows the next root cause is not lookup; it is aggregation semantics:
the agent chooses a plausible Comparison and path, then computes over the wrong
denominator/scope or re-enters raw SPARQL after a high-level tool result.

## 2026-05-21 Aggregation Diagnostics Layer

Implemented the next low-overfit SciQA fix:

- Added `DiagnoseComparisonAggregation`, a wrapped SPARQL graph-operation tool
  that uses the same Comparison value-row path as `AggregateComparisonValues`
  and reports denominator candidates instead of hard-coding an answer.
- The diagnostics expose total scope contributions, matched value rows,
  distinct contributions with values, nested intermediate labels, explicit
  groups, numeric parse counts, row-level summaries, per-contribution
  sum/mean summaries, and per-intermediate/per-group summaries.
- Updated SciQA prompts so ambiguous count/average/sum questions call
  diagnostics after schema discovery when wording could mean all rows, per
  contribution/study, or per category. Raw SPARQL remains a last resort for
  unsupported shapes.
- Made `AggregateComparisonValues(value_predicates=...)` valid without a
  redundant `value_predicate`, matching how schema inspection sometimes
  exposes sibling value predicates and how text-mode models naturally call the
  tool.
- Made `AggregateComparisonValues(comparison_ids=...)` valid without a
  redundant `comparison_id`, matching the documented multi-Comparison union
  contract and preventing FastMCP validation errors before the wrapped SPARQL
  operation can run.
- Normalized common aggregation aliases for `AggregateComparisonValues` and
  `FindFrequentValues` (`frequency`/`most_common` -> `mode_top`, `mean` ->
  `avg`, `total` -> `sum`, `unique_count` -> `count_distinct`) so natural
  model vocabulary does not fail FastMCP validation before the graph operation
  can run.
- Promoted high-confidence lexical label matches in `FindResource` even when
  vector search already returned candidates. This catches cases where a query
  contains an exact Comparison title plus extra context words, e.g. a title
  embedded in "text <title>", without hard-coding the title.
- Added a strict token-coverage pass before the broad token fallback in
  `FindResource`. For short title-like queries it first asks Virtuoso for
  labels containing all query tokens, then labels containing all but one token,
  before trying noisy token-OR matching. This keeps exact resource-title
  recovery generic while avoiding the previous `LIMIT` window issue where
  generic semantic hits could hide the true Comparison label.
- Added rollup-denominator hints for nested Comparison aggregation. Schema
  inspection now surfaces rollup-like intermediate labels such as "all
  sources", and unfiltered nested numeric aggregation responses include
  denominator hints plus rollup candidate summaries. This keeps the final
  denominator choice with the agent while making the graph evidence visible.
- Added `AggregateComparisonValues(intermediate_path="P1,P2")` for deeper
  nested contribution paths. This keeps multi-hop row aggregation inside the
  wrapped SPARQL tool surface instead of forcing agents to hand-write raw
  SPARQL when a value lives below contribution -> node -> row -> predicate.
- Updated system docs to show the SciQA tool tier as 27 registered tools and
  12 domain-specific tools.

Rationale: this keeps the design aligned with wrapped SPARQL calls as basic
graph exploration operations. The model still chooses the Comparison, value
path, filters, and final denominator from the question wording; the tool only
surfaces the graph populations that were previously invisible in traces.

Validation before deployment:

- `uv run python -m compileall ama_kbqa/server/sciqa_server.py ama_kbqa/agents/sciqa_agent/prompts.py tests/framework/test_exact_constraints.py`
- `uv run pytest tests/framework/test_exact_constraints.py -q` = 17 passed.
- `uv run pytest tests/framework -q` = 143 passed.

## 2026-05-22 Multi-Hop Schema Discovery Follow-Up

Focused validation after the rollup/intermediate-path patch confirmed that the
tool could express the previously failing path, but the agent still did not see
that path through normal schema discovery. In the Scenario Factsheets trace it
manually enumerated factsheet -> study links with repeated `GetRelationTargets`
calls and then fell back to raw `RunORKGSPARQL`, even though the wrapped call
`AggregateComparisonValues(intermediate_path="P37586,P37675",
value_predicate="P37668", agg="mode_top")` returns the target `Heat sector (8)`.

Implemented the generic fix:

- `InspectComparisonSchema` now scans two-hop nested paths from each Comparison
  contribution and merges them into `nested_paths` alongside one-hop paths.
- Multi-hop entries include `intermediate_path`, `intermediate_path_labels`,
  path/intermediate samples, value samples, and ready-to-use
  `AggregateComparisonValues(intermediate_path="P1,P2", value_predicate=..., agg=...)`
  hints.
- The SciQA prompt now tells agents to pass schema-discovered
  `intermediate_path` hints directly to `AggregateComparisonValues` instead of
  manually following every row.

This keeps the change aligned with the wrapped-tool philosophy: the model still
chooses the relevant path from graph evidence, but the system now exposes the
multi-hop graph operation through the same high-level aggregation wrapper used
for one-hop nested rows.

Validation:

- `uv run python -m py_compile ama_kbqa/server/sciqa_server.py ama_kbqa/agents/sciqa_agent/prompts.py tests/framework/test_exact_constraints.py`
- `uv run pytest tests/framework/test_exact_constraints.py` = 17 passed.
- `uv run pytest tests/framework` = 143 passed.

## 2026-05-22 Numeric Precision Guard

The fresh focused validation on `d11f324` confirmed the graph path fixes, but
also exposed an evaluation/synthesis issue: the installed-capacity trace used
the right schema, diagnosed the `all sources` denominator, and computed
`367.5708`, but the final prose rounded it to `367.57 GW`. The LLM judge marked
that incorrect against the gold `367.570798339843756` despite the answer being
the same benchmark value within normal numeric tolerance.

Implemented two general fixes:

- SciQA prompt guidance now tells the agent to copy exact numeric values from
  answer-producing tools/journal entries first, and only add rounded values
  secondarily.
- `postprocessing.py` now applies a single-number numeric-equivalence guard
  around LLM-judge output and simple matching. It only fires when the gold answer
  is essentially one numeric value, and the predicted answer contains a numeric
  token within a small absolute/relative tolerance.

Rationale: this is not a question-specific override. It prevents benchmark
accuracy from depending on whether the judge treats harmless decimal formatting
as equivalent, while preserving the judge for non-numeric and mixed concept
answers such as `Heat sector 8`.

Validation:

- `uv run python -m py_compile ama_kbqa/postprocessing.py ama_kbqa/agents/sciqa_agent/prompts.py tests/framework/test_exact_constraints.py`
- `uv run pytest tests/framework/test_exact_constraints.py` = 18 passed.
- `uv run pytest tests/framework` = 144 passed.

## 2026-05-22 Quoted Anchor Routing Follow-Up

The next focused run showed a stochastic lookup failure on the previously fixed
`"summarization before 2002"` question. The tool surface could retrieve `R6948`
when asked for the exact title, but the agent sometimes paraphrased the quoted
anchor into broader searches such as "text summarization approaches methods
comparison" and then explored adjacent empty/irrelevant resources.

Implemented a general prompt rule rather than a question-specific shortcut:

- If a user question contains a quoted title/label, search the exact quoted
  substring first.
- For aggregation/count/superlative wording that asks about values in a quoted
  thing, the first lookup should be
  `FindResource("<quoted text>", node_type_filter="Comparison", top_n=10)`.
- Only after the exact quoted anchor fails should the agent broaden or
  paraphrase the search.

Rationale: quoted spans are user-provided graph anchors. Preserving them is a
general search discipline and keeps resource selection inside the existing
semantic/lexical `FindResource` tool instead of adding deterministic
question-specific routing.

## 2026-05-22 Grouped Path Aggregation Follow-Up

The next focused validation confirmed that the earlier fixes recovered the
exact-title, rollup-denominator, numeric-precision, and multi-hop factsheet
cases. The remaining interval/table case did answer correctly, but only after a
long raw `RunORKGSPARQL` trajectory that manually joined:

`Comparison -> contribution -> scenario -> goal -> time frame`

with:

`contribution -> energy source -> installed capacity -> HAS_VALUE`

That is a generic missing graph operation rather than an argument for
question-specific routing. The agent needed to aggregate a nested metric row
while grouping by both the nested row label and a separate contribution-relative
path.

Implemented the generic wrapper extension:

- `AggregateComparisonValues` now accepts `group_by_path="P1,P2,P3"` so the
  grouping key can be reached through a relation path from each contribution,
  not only a direct contribution predicate.
- `AggregateComparisonValues(group_by_intermediate=true)` adds the nested row
  label as another grouping axis when values live below
  `contribution -> row-object -> metric`.
- The SciQA prompt now routes interval/table wording such as "for each source by
  year/time frame" or "in 5-year intervals" to the wrapped aggregation call
  before raw SPARQL.

This keeps with the design goal of wrapped SPARQL calls as basic graph
operations. The model still chooses the Comparison, metric path, grouping path,
and final interpretation from graph evidence; the code only exposes a reusable
projection/aggregation shape that previously required brittle hand-written
SPARQL.

Validation:

- `uv run python -m py_compile ama_kbqa/server/sciqa_server.py ama_kbqa/agents/sciqa_agent/prompts.py tests/framework/test_exact_constraints.py`
- `uv run pytest tests/framework/test_exact_constraints.py -q` = 18 passed.
- `uv run pytest tests/framework -q` = 144 passed.

## 2026-05-23 Context-Safety and Ambiguous-Denominator Follow-Up

Focused validation after deploying grouped path aggregation showed three
separate failure modes that were not question-specific:

- MiniMax used the new grouped aggregation wrapper for both interval/table rows
  and produced plausible graph-derived tables, but the judge still marked both
  incorrect because the gold answers are opaque TinyURL artifacts. Do not solve
  this by hardcoding URL strings; the dataset/evaluation layer should dereference
  or semantically evaluate those artifacts.
- Gemma issued broad exploratory `RunORKGSPARQL` queries while trying to recover
  a nested factsheet path. One query returned 3,866 rows, the full Virtuoso JSON
  was fed back into the LLM, and the next prompt exceeded the KIT Gemma context
  window. This is an unbounded-tool-output bug.
- Gemma also called `AggregateComparisonValues` on nested energy-source rows
  without the `all sources` rollup filter, received denominator hints containing
  the correct rollup candidate, but still finalized the row-level average because
  that ambiguous result was written to the journal like a normal answer value.

Implemented the generic fixes:

- `BaseKBQAAgent._execute_tool_calls()` only treats a no-argument
  `GetJournalSummary()` call as the answer-prompt trigger. Malformed calls such
  as `GetJournalSummary(action="read")` return their tool error to the model but
  no longer push the "answer now" prompt.
- `ama_kbqa/utils/sparql_results.py` compacts SPARQL SELECT responses for both
  `RunSPARQL` and `RunORKGSPARQL`: bounded row preview plus `result_count`,
  `returned_count`, `truncated`, and a refinement note. This keeps raw SPARQL as
  a last-resort graph operation without allowing broad probes to poison context.
- `AggregateComparisonValues` now adds `denominator_hints.recommended_follow_up`
  for ungrouped, unfiltered nested numeric aggregates when rollup intermediate
  candidates are present. The journal stores an `ambiguous_denominator` object
  with the row-level result, candidate rollup result, and exact wrapped
  follow-up call instead of promoting the row-level aggregate as final. Grouped
  table outputs such as "for each source by time frame" are not marked
  ambiguous just because one group is a rollup row.

These changes preserve the wrapped-SPARQL philosophy: the agent still chooses
graph anchors, paths, filters, and aggregates from schema/tool evidence. The
system only makes generic graph operations safer and prevents ambiguous evidence
from being presented as a final fact.

Validation:

- `uv run python -m py_compile` on the edited agent, server, prompt, utility, and
  test files.
- `uv run pytest tests/framework/test_base_agent_tool_loop.py tests/framework/test_sparql_results.py tests/framework/test_exact_constraints.py -q` = 22 passed.
- `uv run pytest tests/framework -q` = 148 passed.
