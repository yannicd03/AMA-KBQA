# Generic Framework for RDF Knowledge Graph Tools

This document defines a unified framework for creating MCP (Model Context Protocol) tools that work with any RDF-based knowledge graph. The framework abstracts common operations that differ in implementation details but share the same semantic purpose across different knowledge graphs.

## Table of Contents

1. [Framework Overview](#framework-overview)
2. [Core Tool Categories](#core-tool-categories)
3. [Abstract Tool Definitions](#abstract-tool-definitions)
4. [Configuration Layer](#configuration-layer)
5. [State Management Pattern](#state-management-pattern)
6. [Implementation Guidelines](#implementation-guidelines)
7. [Comparison: KQAPro vs SciQA](#comparison-kqapro-vs-sciqa)

---

## Framework Overview

### Problem Statement

When integrating a new RDF knowledge graph into the system, developers currently need to:
1. Write custom tools from scratch
2. Learn the specific ontology and namespace conventions
3. Implement domain-specific query patterns
4. Handle state management and journaling

This leads to code duplication and inconsistent tool behaviors across different knowledge graphs.

### Solution: Layered Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Agent Layer                                 │
│           (Uses tools through standard interface)                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                 Abstract Tool Interface                          │
│    (Defines standard inputs/outputs for each tool category)      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              Knowledge Graph Adapter Layer                       │
│    (Implements namespace mapping, URI formatting, prefixes)      │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ KQAPro       │  │ SciQA/ORKG   │  │ Future KG    │          │
│  │ Adapter      │  │ Adapter      │  │ Adapter      │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                 Shared Infrastructure                            │
│         (Qdrant, Virtuoso, Embedding Models)                     │
└─────────────────────────────────────────────────────────────────┘
```

---

## Core Tool Categories

Based on analysis of existing implementations (KQAPro: 21 tools, SciQA: 13 tools), tools fall into these functional categories:

### Category 1: Entity Discovery

**Purpose:** Find the smallest semantic units (nodes) in the knowledge graph.

| Function | Description | KQAPro Tool | SciQA Tool |
|----------|-------------|-------------|------------|
| Semantic Node Search | Find nodes by natural language description | `FindNode` | `FindResource` |
| Semantic Relation Search | Find predicates/relations by description | (via FindNode) | `FindPredicate` |
| Reverse Lookup | Find nodes by attribute value | `FindByAttribute` | - |

**Generic Interface:**
```python
def discover_entity(
    query: str,
    search_type: Literal["semantic", "exact", "attribute_value"],
    attribute_name: Optional[str] = None,  # For attribute_value search
    top_n: int = 5
) -> List[EntityMatch]
```

---

### Category 2: Identity Resolution

**Purpose:** Convert internal identifiers to human-readable labels.

| Function | Description | KQAPro Tool | SciQA Tool |
|----------|-------------|-------------|------------|
| Single Label Lookup | Get label for one ID | `GetNodeLabel` | `GetResourceLabel` |
| Batch Label Lookup | Get labels for multiple IDs | `BatchGetNodeLabels` | `BatchGetResourceLabels` |

**Generic Interface:**
```python
def resolve_labels(
    node_ids: Union[str, List[str]]
) -> Dict[str, str]  # {id: label}
```

---

### Category 3: Node Exploration

**Purpose:** Retrieve comprehensive information about a specific node.

| Function | Description | KQAPro Tool | SciQA Tool |
|----------|-------------|-------------|------------|
| Full Node Summary | Get all attributes and relations | `GetNodeSummary` | `GetResourceDetails` |
| Specific Attribute | Get values for one attribute | `GetAttributeDetails` | (via GetResourceDetails) |
| Attribute with Context | Get attribute with qualifiers | `GetAttributeWithQualifiers` | - |
| Temporal Query | Get attribute at specific date | `TemporalAttributeQuery` | - |

**Generic Interface:**
```python
def explore_node(
    node_id: str,
    detail_level: Literal["summary", "full", "attribute_only"],
    attribute_name: Optional[str] = None,
    include_qualifiers: bool = False,
    target_date: Optional[str] = None
) -> NodeDetails
```

---

### Category 4: Relation Navigation

**Purpose:** Traverse connections between nodes.

| Function | Description | KQAPro Tool | SciQA Tool |
|----------|-------------|-------------|------------|
| Direct Relations | Get targets of specific relation | `GetRelationDetails` | `GetRelationTargets` |
| Multi-hop Navigation | Follow relation chains | `FindEntitiesByRelationPath` | - |
| Bidirectional Check | Check both directions | (built into GetRelationDetails) | - |

**Generic Interface:**
```python
def navigate_relations(
    start_node_id: str,
    relation_path: List[RelationStep],  # [{relation, direction}]
    include_intermediate: bool = False
) -> NavigationResult
```

---

### Category 5: Schema Introspection

**Purpose:** Validate and discover schema elements (attributes, relations, types).

| Function | Description | KQAPro Tool | SciQA Tool |
|----------|-------------|-------------|------------|
| Attribute Validation | Check if attribute exists | `GetSchemaForAttribute` | - |
| Fuzzy Matching | Find similar attribute names | `GetSchemaForAttribute` | - |
| Type Discovery | Get available types | (via FindNode results) | (via FindResource) |

**Generic Interface:**
```python
def validate_schema_element(
    element_name: str,
    element_type: Literal["attribute", "relation", "type"],
    fuzzy_match: bool = True
) -> SchemaValidationResult
```

---

### Category 6: Statement Qualifiers (RDF Reification)

**Purpose:** Access metadata attached to statements (when, where, how, etc.).

| Function | Description | KQAPro Tool | SciQA Tool |
|----------|-------------|-------------|------------|
| Attribute Qualifiers | Get qualifiers for attribute statement | `GetEdgeQualifiers` | - |
| Relation Qualifiers | Get qualifiers for relation statement | `GetQualifiersByPredicate` | - |

**Generic Interface:**
```python
def get_statement_qualifiers(
    subject_id: str,
    predicate_name: str,
    object_value: Optional[str] = None,
    statement_type: Literal["attribute", "relation"] = "attribute"
) -> QualifierResult
```

**Note:** Not all knowledge graphs use RDF reification. This category is optional.

---

### Category 7: Comparison & Verification

**Purpose:** Compare values across entities and perform mathematical verification.

| Function | Description | KQAPro Tool | SciQA Tool |
|----------|-------------|-------------|------------|
| Multi-Entity Comparison | Compare attribute across entities | `CompareEntities` | - |
| Numeric Verification | Perform mathematical comparison | `VerifyNumericCondition` | - |

**Generic Interface:**
```python
def compare_entities(
    entity_ids: List[str],
    attribute_name: str,
    sort_order: Literal["asc", "desc"] = "desc"
) -> ComparisonResult

def verify_condition(
    value1: str,
    operator: Literal["<", ">", "<=", ">=", "==", "!="],
    value2: str,
    value_type: Literal["numeric", "date", "string"] = "numeric"
) -> VerificationResult
```

---

### Category 8: Raw Query Execution

**Purpose:** Execute arbitrary SPARQL queries for complex operations.

| Function | Description | KQAPro Tool | SciQA Tool |
|----------|-------------|-------------|------------|
| SPARQL Execution | Execute SELECT/ASK queries | `RunSPARQL` | `RunORKGSPARQL` |

**Generic Interface:**
```python
def execute_sparql(
    query: str,
    inject_prefixes: bool = True,
    inject_graph: bool = True
) -> SPARQLResult
```

---

### Category 9: State Management

**Purpose:** Track agent progress, prevent loops, enable synthesis.

| Function | Description | KQAPro Tool | SciQA Tool |
|----------|-------------|-------------|------------|
| Journal Management | Update scratchpad state | `ManageJournal` | `ManageJournal` |
| Progress Summary | Get formatted discovery summary | `GetJournalSummary` | `GetJournalSummary` |

**Generic Interface:**
```python
def manage_state(
    action: Literal["read", "update", "clear"],
    update_type: Optional[str] = None,
    content: Optional[str] = None
) -> JournalState

def get_discovery_summary() -> str
```

---

### Category 10: Domain-Specific Tools

**Purpose:** Provide shortcuts for common domain-specific patterns.

| Domain | Function | SciQA Tool |
|--------|----------|------------|
| Academic Papers | Get paper contributions | `GetPaperContributions` |
| Academic Papers | Get paper authors | `GetPaperAuthors` |
| Academic Papers | Get contribution methods | `GetContributionMethods` |
| Research Fields | Get papers in field | `GetResearchFieldPapers` |

**Note:** These are syntactic sugar over generic relation navigation but significantly improve agent efficiency for common patterns in the domain.

---

## Abstract Tool Definitions

### EntityMatch (Base Response Type)

```python
@dataclass
class EntityMatch:
    """Standardized entity match result."""
    id: str                          # Internal identifier (e.g., "Q54089", "R12345")
    label: str                       # Human-readable name
    node_type: str                   # Type classification (entity, concept, paper, etc.)
    relevance_score: float           # Vector similarity or 1.0 for exact match
    available_attributes: List[str]  # Discoverable attributes
    available_relations: List[str]   # Discoverable relations
    metadata: Dict[str, Any]         # Additional KG-specific metadata
```

### NodeDetails (Exploration Response)

```python
@dataclass
class NodeDetails:
    """Complete node information."""
    id: str
    label: str
    node_type: str
    attributes: Dict[str, List[AttributeValue]]  # {attr_name: [values]}
    relations: Dict[str, List[str]]              # {relation_name: [target_ids]}
    stats: Dict[str, int]                        # {attribute_count, relation_count}
    status: str
```

### AttributeValue

```python
@dataclass
class AttributeValue:
    """Attribute value with optional metadata."""
    value: Any                       # The actual value
    unit: Optional[str]              # Unit of measurement if applicable
    qualifiers: Dict[str, Any]       # Contextual metadata (dates, methods, etc.)
    datatype: str                    # XSD datatype or "entity" for references
```

### JournalState (State Tracking)

```python
@dataclass
class JournalState:
    """Agent scratchpad state - universal across all KGs."""
    question_text: str = ""
    question_type: str = ""
    target_entities: List[str] = field(default_factory=list)

    # Exploration tracking
    visited_nodes: Dict[str, str] = field(default_factory=dict)  # {id: label}
    found_values: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    verified_facts: List[Dict] = field(default_factory=list)
    failed_attempts: List[str] = field(default_factory=list)

    # Progress tracking
    completed_steps: List[str] = field(default_factory=list)
    current_plan: List[str] = field(default_factory=list)
    partial_answer: str = ""
```

---

## Configuration Layer

Each knowledge graph adapter requires a configuration that defines:

### Namespace Configuration

```python
@dataclass
class NamespaceConfig:
    """Namespace definitions for a knowledge graph."""
    entity_prefix: str         # e.g., "http://kqapro.org/entity/"
    property_prefix: str       # e.g., "http://kqapro.org/property/"
    attribute_prefix: str      # e.g., "http://kqapro.org/attribute/"
    qualifier_prefix: str      # e.g., "http://kqapro.org/qualifier/"
    unit_prefix: str          # e.g., "http://kqapro.org/unit/"

    # SPARQL prefix declarations
    sparql_prefixes: str      # Full PREFIX block for SPARQL queries
```

**Example: KQAPro**
```python
KQAPRO_NAMESPACES = NamespaceConfig(
    entity_prefix="http://kqapro.org/entity/",
    property_prefix="http://kqapro.org/property/",
    attribute_prefix="http://kqapro.org/attribute/",
    qualifier_prefix="http://kqapro.org/qualifier/",
    unit_prefix="http://kqapro.org/unit/",
    sparql_prefixes="""
PREFIX ex:   <http://kqapro.org/entity/>
PREFIX prop: <http://kqapro.org/property/>
PREFIX attr: <http://kqapro.org/attribute/>
PREFIX qual: <http://kqapro.org/qualifier/>
PREFIX unit: <http://kqapro.org/unit/>
"""
)
```

**Example: SciQA/ORKG**
```python
SCIQA_NAMESPACES = NamespaceConfig(
    entity_prefix="http://orkg.org/orkg/resource/",
    property_prefix="http://orkg.org/orkg/predicate/",
    attribute_prefix="http://orkg.org/orkg/predicate/",  # Same as property
    qualifier_prefix=None,  # ORKG doesn't use RDF reification
    unit_prefix=None,
    sparql_prefixes="""
PREFIX orkgr: <http://orkg.org/orkg/resource/>
PREFIX orkgp: <http://orkg.org/orkg/predicate/>
PREFIX orkgc: <http://orkg.org/orkg/class/>
"""
)
```

### Vector Collection Configuration

```python
@dataclass
class VectorConfig:
    """Qdrant collection configuration."""
    entity_collection: str     # Collection name for entities
    relation_collection: str   # Collection name for relations
    entity_threshold: float    # Minimum similarity score
    relation_threshold: float
```

### Graph Configuration

```python
@dataclass
class GraphConfig:
    """Virtuoso graph configuration."""
    graph_uri: str             # Named graph URI (e.g., "http://sciqa.org/kg")
    supports_reification: bool # Whether KG uses RDF statement reification
    has_temporal_data: bool    # Whether KG has temporal qualifiers
```

---

## State Management Pattern

### Auto-Update Pattern

All tools should auto-update the journal state to:
1. Track visited nodes
2. Store discovered values
3. Log verified facts
4. Track failed attempts
5. Record completed steps

```python
# Pattern for auto-updating journal
def some_tool(node_id: str, ...) -> Result:
    result = execute_query(...)

    if result.success:
        # Update visited nodes
        session_journal.visited_nodes[node_id] = result.label

        # Store discovered values
        if node_id not in session_journal.found_values:
            session_journal.found_values[node_id] = {}
        session_journal.found_values[node_id][attr_name] = result.values

        # Add verified fact
        session_journal.verified_facts.append({
            "subject": node_id,
            "attribute": attr_name,
            "value": result.values[0],
            "source": "ToolName"
        })

        # Log completion
        session_journal.completed_steps.append(
            f"Retrieved {attr_name} for {result.label}"
        )
    else:
        # Log failure
        session_journal.failed_attempts.append(
            f"ToolName({node_id}, {attr_name}): {result.error}"
        )

    return result
```

---

## Implementation Guidelines

### Adding a New Knowledge Graph

1. **Create Namespace Configuration**
   - Define all namespace prefixes
   - Create SPARQL prefix block
   - Determine if reification is used

2. **Populate Vector Database**
   - Extract entities with labels and types
   - Extract relations/predicates
   - Generate embeddings and store in Qdrant

3. **Implement Core Tools**
   - Start with Category 1-4 (discovery, resolution, exploration, navigation)
   - Add Category 8 (raw SPARQL) as fallback
   - Add Category 9 (state management)

4. **Add Optional Tools**
   - Category 5 (schema introspection) if needed
   - Category 6 (qualifiers) if KG uses reification
   - Category 7 (comparison) if needed

5. **Add Domain-Specific Tools**
   - Identify common query patterns in your domain
   - Create convenience tools that wrap relation navigation

### Tool Naming Conventions

| Generic Function | KQAPro Pattern | SciQA Pattern | Recommended |
|-----------------|----------------|---------------|-------------|
| Find entity | `FindNode` | `FindResource` | `Find{DomainEntity}` |
| Get label | `GetNodeLabel` | `GetResourceLabel` | `Get{DomainEntity}Label` |
| Get details | `GetNodeSummary` | `GetResourceDetails` | `Get{DomainEntity}Details` |
| Get relations | `GetRelationDetails` | `GetRelationTargets` | `GetRelationTargets` |
| Run SPARQL | `RunSPARQL` | `RunORKGSPARQL` | `Run{KGName}SPARQL` |

---

## Comparison: KQAPro vs SciQA

### Feature Comparison

| Feature | KQAPro | SciQA |
|---------|--------|-------|
| **RDF Reification** | Yes (qualifiers) | No |
| **Temporal Data** | Yes (point in time) | No |
| **Attribute/Property Split** | Yes (attr: vs prop:) | No (same prefix) |
| **Unit Handling** | Yes (unit: prefix) | No |
| **Named Graph** | No | Yes (http://sciqa.org/kg) |
| **Domain-Specific Tools** | No | Yes (papers, authors) |

### Tool Count Comparison

| Category | KQAPro | SciQA |
|----------|--------|-------|
| Entity Discovery | 2 | 2 |
| Identity Resolution | 2 | 2 |
| Node Exploration | 4 | 1 |
| Relation Navigation | 3 | 1 |
| Schema Introspection | 1 | 0 |
| Statement Qualifiers | 2 | 0 |
| Comparison/Verification | 2 | 0 |
| Raw Query | 1 | 1 |
| State Management | 2 | 2 |
| Domain-Specific | 0 | 4 |
| **Total** | **19** | **13** |

### Observations

1. **KQAPro is more complex** due to RDF reification (qualifiers), temporal data, and separate attribute/property namespaces.

2. **SciQA is simpler** with a flat predicate structure, but adds domain-specific tools for academic paper navigation.

3. **Core tools are universal**: Entity discovery, identity resolution, node exploration, relation navigation, SPARQL execution, and state management appear in both.

4. **Qualifier tools are optional**: Only needed for KGs that use RDF statement reification.

5. **Domain-specific tools improve efficiency**: SciQA's paper/author/contribution tools are shortcuts that would otherwise require multiple generic tool calls.

---

## Recommended Minimum Tool Set

For any new RDF knowledge graph, implement at minimum:

### Tier 1: Essential (Must Have)
1. `FindEntity` - Semantic search for nodes
2. `GetEntityLabel` / `BatchGetEntityLabels` - ID to label resolution
3. `GetEntityDetails` - Full node summary
4. `GetRelationTargets` - Follow relations
5. `RunSPARQL` - Fallback for complex queries
6. `ManageJournal` / `GetJournalSummary` - State management

### Tier 2: Recommended
7. `FindByAttribute` - Reverse lookup by value
8. `ValidateSchemaElement` - Attribute/relation name validation
9. `NavigateRelationPath` - Multi-hop navigation
10. `CompareEntities` - Multi-entity comparison

### Tier 3: Optional (If Applicable)
11. `GetStatementQualifiers` - If KG uses RDF reification
12. `TemporalQuery` - If KG has temporal data
13. `VerifyCondition` - For verification questions
14. Domain-specific tools - Based on common query patterns

---

## Future Work

1. **Generic Base Classes**: Create abstract base classes in Python that define the standard interface for each tool category.

2. **Configuration-Driven Tools**: Implement tools that read namespace configuration and generate appropriate SPARQL queries dynamically.

3. **Auto-Discovery**: Tools that automatically discover available attributes and relations from the KG schema.

4. **Cross-KG Queries**: Support for queries that span multiple knowledge graphs.

5. **Tool Selection Agent**: An agent that automatically selects the right tools based on question analysis and KG capabilities.
