"""
Abstract atomic graph operations: the KG-agnostic operation contract.

This module declares the canonical set of operations that a knowledge-graph
backend must expose for the agent framework to function:

- ``atomic`` operations: single-purpose graph primitives (entity discovery,
  label resolution, relation traversal, reverse lookup, path following).
- ``deterministic`` operations: prebuilt tools whose result is computed
  without any LLM involvement (comparison, counting, numeric verification).
- ``escape_hatch``: raw SPARQL access for patterns the prebuilt tools do
  not cover.

The contract is two-level: ``required=True`` operations must be bound by
every KG backend; ``required=False`` operations are an optional tier that a
KG binds only when its data model supports them (e.g. superlative selection
over reified attributes, or group-by aggregation over comparison tables).

Property addressing: parameters named ``property`` take a *KG-native
property designator*. Each KG keeps its own scheme (KQAPro uses
human-readable names because its URIs are label-based; ORKG uses opaque
P-ids); the adapter documents and validates the designator form. The
contract deliberately does not impose a universal scheme, because resolving
names to IDs is fuzzy (vector search) in some KGs and would inject
nondeterminism into deterministic operations.

Concrete MCP servers keep their KG-flavored tool names (those names are
part of prompts, fewshots, and recorded benchmark traces, so they must not
change). Instead, each KG adapter *binds* its concrete tool names to these
abstract operations via ``BaseKGAdapter.get_operation_bindings()``.
``validate_bindings()`` checks a binding map for completeness, making the
claim "supporting a new KG means implementing this operation set"
mechanically verifiable.
"""

from dataclasses import dataclass, field
from typing import Dict, Literal, Mapping, Tuple

OperationKind = Literal["atomic", "deterministic", "escape_hatch"]


@dataclass(frozen=True)
class AtomicOperation:
    """A single abstract operation in the KG-agnostic contract."""

    name: str
    description: str
    canonical_params: Tuple[str, ...]
    kind: OperationKind
    required: bool = True


ATOMIC_OPERATIONS: Dict[str, AtomicOperation] = {
    op.name: op
    for op in [
        # --- Atomic graph primitives -------------------------------------
        AtomicOperation(
            name="find_entity",
            description="Semantic (vector) search for entities/resources by name or description.",
            canonical_params=("query", "top_n", "type_filter"),
            kind="atomic",
        ),
        AtomicOperation(
            name="get_label",
            description="Resolve a single entity/resource ID to its human-readable label.",
            canonical_params=("entity_id",),
            kind="atomic",
        ),
        AtomicOperation(
            name="get_labels",
            description="Batch-resolve multiple entity/resource IDs to labels in one call.",
            canonical_params=("entity_ids",),
            kind="atomic",
        ),
        AtomicOperation(
            name="get_summary",
            description="Retrieve all attributes and relations of one entity/resource in one call.",
            canonical_params=("entity_id",),
            kind="atomic",
        ),
        AtomicOperation(
            name="get_relation_targets",
            description=(
                "Follow one relation/predicate (KG-native property designator) "
                "from an entity and return the targets."
            ),
            canonical_params=("entity_id", "property"),
            kind="atomic",
        ),
        AtomicOperation(
            name="reverse_lookup",
            description=(
                "Find entities by an attribute/predicate VALUE (codes, IDs, literals); "
                "the property is a KG-native designator."
            ),
            canonical_params=("property", "value"),
            kind="atomic",
        ),
        AtomicOperation(
            name="follow_path",
            description="Navigate a multi-hop relation path from a start entity in one call.",
            canonical_params=("start_id", "relation_path"),
            kind="atomic",
        ),
        # --- Deterministic prebuilt tools --------------------------------
        AtomicOperation(
            name="compare",
            description=(
                "Fetch one property (KG-native designator) across multiple entities "
                "and return sorted results."
            ),
            canonical_params=("entity_ids", "property"),
            kind="deterministic",
        ),
        AtomicOperation(
            name="count",
            description=(
                "Count entities matching a KG-native filter via SPARQL aggregation. "
                "The minimal contract is count(filter) -> int; richer aggregation "
                "lives in the optional tier (aggregate, frequent_values)."
            ),
            canonical_params=("filter",),
            kind="deterministic",
        ),
        AtomicOperation(
            name="verify_numeric",
            description="Deterministic numeric/date comparison with a TRUE/FALSE verdict.",
            canonical_params=("value1", "operator", "value2", "unit"),
            kind="deterministic",
        ),
        # --- Optional tier: bound only by KGs whose data model supports them
        AtomicOperation(
            name="aggregate",
            description=(
                "Group-by aggregation (avg/sum/min/max/count) of a property over a "
                "scoped set of entities (e.g. ORKG comparison tables)."
            ),
            canonical_params=("scope", "property", "agg", "group_by"),
            kind="deterministic",
            required=False,
        ),
        AtomicOperation(
            name="frequent_values",
            description=(
                "Find the most frequent values of a property across a global or "
                "field-scoped entity set."
            ),
            canonical_params=("property", "scope"),
            kind="deterministic",
            required=False,
        ),
        AtomicOperation(
            name="select_extreme",
            description=(
                "Superlative selection: pick the entity with the max/min value of a "
                "property within a scope."
            ),
            canonical_params=("scope", "property", "direction"),
            kind="deterministic",
            required=False,
        ),
        # --- Escape hatch -------------------------------------------------
        AtomicOperation(
            name="run_sparql",
            description="Execute raw SPARQL against the KG endpoint (prefixes auto-injected).",
            canonical_params=("query",),
            kind="escape_hatch",
        ),
    ]
}


@dataclass(frozen=True)
class CoverageReport:
    """Result of checking a KG's operation bindings against the contract."""

    bound: Dict[str, str] = field(default_factory=dict)
    missing: Tuple[str, ...] = ()
    unknown: Tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        """True if every required operation is bound and no binding is unknown."""
        return not self.missing and not self.unknown


def validate_bindings(bindings: Mapping[str, str]) -> CoverageReport:
    """
    Check a {abstract_operation: concrete_tool_name} map for completeness.

    Returns a CoverageReport listing required operations that have no
    binding (``missing``) and bindings that reference operations not in
    the contract (``unknown``).
    """
    missing = tuple(
        name
        for name, op in ATOMIC_OPERATIONS.items()
        if op.required and name not in bindings
    )
    unknown = tuple(name for name in bindings if name not in ATOMIC_OPERATIONS)
    return CoverageReport(bound=dict(bindings), missing=missing, unknown=unknown)
