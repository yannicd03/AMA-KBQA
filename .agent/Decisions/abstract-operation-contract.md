# ADR: Abstract Operation Contract (Operations Layer)

**Date:** 2026-06-06
**Commits:** 42187d0, 203fd3e (branch `yannic-dev`)
**Status:** Accepted

## Related Docs
- [Project Architecture](../System/project_architecture.md) — overall system overview; framework and server sections
- [Agent System](../System/agent_system.md) — agent lifecycle and tool loop context
- [RelationPathStep Schema Hardening](./relation-path-step-schema-hardening.md) — 2026-07-25 parameter-level companion: named-key validation for the `follow_path` op's `relation_path` argument in both servers, closing a raw-`KeyError` gap this ADR's Decision 4 didn't cover

---

## Context

The SEMANTiCS 2026 paper claims that onboarding a new knowledge graph "reduces to implementing thin SPARQL wrappers against the abstract tool interface". Before this change, no such interface existed in code. A drift audit found:

- KQAPro exposed 29 tools, SciQA/ORKG exposed 27 tools.
- Only 4 tool names overlapped across the two servers.
- Roughly 45% of tools on each side were KG-idiosyncratic with no counterpart.

Four linked decisions were made together in response to this gap, all on 2026-06-06.

---

## Decision 1: Property addressing stays KG-native

**Chosen:** Operations accept KG-native property designators. KQAPro uses human-readable label-based names (e.g. `"date_of_birth"`) because its URIs are label-based. ORKG uses opaque P-ids (e.g. `"P29"`) because that is what its graph stores.

**Rejected:**
- *Canonical ID scheme* (all KGs map to a shared ID space): cross-scheme resolution in ORKG is vector-search based and inherently fuzzy, which would inject nondeterminism into deterministic operations like `verify_numeric` and `count`.
- *Canonical name scheme* (all KGs map to a shared vocabulary): ORKG P-ids do not carry stable labels; bridging them requires the same fuzzy lookup.

**Consequence:** The abstract operation contract specifies `property` as a KG-native designator. The adapter documents the form. Each KG resolves properties inside its own MCP server, keeping the comparison step exact.

---

## Decision 2: Response envelopes stay per-server; unification deferred

**Chosen:** Each server keeps its own response format. KQAPro tools return typed Pydantic models. SciQA tools return JSON strings. The previously-present `framework/types.py` (a shared response contract) was deleted because no production code imported it.

**Rejected:**
- *Immediate unification of envelopes*: the response envelope is LLM-visible text, embedded in benchmark traces and prompts. Changing it would invalidate the benchmark numbers in the submitted SEMANTiCS 2026 paper.

**Consequence:** Unification is deferred until after the paper submission. The dead `types.py` file is removed to avoid false signals that a shared contract already exists.

---

## Decision 3: Two-level count contract (required minimal + optional aggregate tier)

**Chosen:** The abstract operation contract is split into two levels:
- **Required (11 ops):** `find_entity`, `get_label`, `get_labels`, `get_summary`, `get_relation_targets`, `reverse_lookup`, `follow_path`, `compare`, `count`, `verify_numeric`, `run_sparql`. Every KG adapter must bind all 11.
- **Optional (3 ops):** `aggregate`, `frequent_values`, `select_extreme`. A KG binds these only when its data model supports them.

Counting is in the required tier (`count`), but the two KGs bind it differently:
- KQAPro binds `count` to `CountEntities` (a condition-algebra COUNT tool).
- SciQA binds `count` to `AggregateComparisonValues` with `agg="count"` (ORKG's table-shaped aggregator).
- SciQA also binds the optional `aggregate` operation to `AggregateComparisonValues` and `frequent_values` to `FindFrequentValues`.

**Rejected:**
- *Force-unifying one counting signature*: KQAPro counting operates via KoPL condition algebra over reified attributes; ORKG counting is aggregation over comparison contribution tables. These mirror different data models. A forced unified signature would either break KQAPro's condition branching or ORKG's group-by semantics.

**Consequence:** `CoverageReport.ok` returns True for a KG that binds all 11 required operations, even if it binds none of the optional 3. The optional tier is additive.

---

## Decision 4: KG-flavored LLM-facing tool names are preserved; binding layer provides comparability

**Chosen:** MCP server tools keep their KG-flavored names: `FindNode` / `FindResource`, `GetNodeSummary` / `GetResourceSummary`, `RunSPARQL` / `RunORKGSPARQL`, etc. The binding layer in `BaseKGAdapter.get_operation_bindings()` maps these to abstract operation names, making them comparable across KGs without renaming them.

**Rejected:**
- *Renaming MCP tools to canonical names*: tool names appear in system prompts, fewshot examples, `QTYPE_TOOL_MAP`, and all recorded benchmark traces. Renaming would break prompts, fewshots, and benchmark reproducibility simultaneously.

**Consequence:** The abstract operation contract is verifiable without any LLM-visible surface changes. `validate_bindings()` and `validate_operation_coverage()` check completeness mechanically.

---

## Implementation

### New files
- `ama_kbqa/framework/operations.py` — `AtomicOperation`, `ATOMIC_OPERATIONS` dict (14 entries: 11 required + 3 optional), `CoverageReport`, `validate_bindings()`.
- `ama_kbqa/framework/deterministic.py` — `parse_numeric`, `NumericComparison`, `compare_numeric`. Shared math core extracted from the previously-duplicated `VerifyNumericCondition` in both servers; output strings are byte-identical to originals.

### Modified files
- `ama_kbqa/framework/adapters/base_adapter.py` — added abstract method `get_operation_bindings()` and concrete `validate_operation_coverage()` alongside existing `_create_config()`.
- `ama_kbqa/framework/adapters/kqapro_adapter.py` — implements `get_operation_bindings()` (11 required + `select_extreme`).
- `ama_kbqa/framework/adapters/sciqa_adapter.py` — implements `get_operation_bindings()` (11 required, `count` bound to `AggregateComparisonValues`; optional `aggregate` and `frequent_values` also bound); added `owl:` prefix to `sparql_prefixes` (the `RunORKGSPARQL` docstring advertises `owl:` to the LLM, so it must be injectable from the adapter).
- `ama_kbqa/server/kqapro_server.py` — `NS_*` constants and `SPARQL_PREFIXES` now sourced from `_ADAPTER = KQAProAdapter()` at module level instead of hardcoded strings.
- `ama_kbqa/server/sciqa_server.py` — same refactoring to `_ADAPTER = SciQAAdapter()`.

### Deleted files
- `ama_kbqa/framework/types.py` — dead shared response contract; no production code imported it.
- `create_default_kqapro_config()` and `create_default_sciqa_config()` factory functions in `config.py` — third copy of KG configs; removed.
- `tests/framework/test_types.py` — tests for the deleted module.

### New tests
- `tests/framework/test_operations.py` (14 tests) — registry shape, two-level semantics, adapter coverage for both KGs, plus a source-level regex check that every bound tool name appears as a `def` in its server file (guards binding rot without importing server runtime deps).
- `tests/framework/test_deterministic.py` (14 tests) — `parse_numeric` variants, `compare_numeric` correctness and error paths; pins byte-identical output strings.

### Test count
206 → 219 (net +13 from 28 added − 2 deleted test files).
