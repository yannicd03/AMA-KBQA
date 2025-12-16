# AMA KBQA Tools Reference

Complete reference for all MCP tools available to the KQAPro Agent.

**Total Tools:** 21
**Server Location:** `ama_kbqa/server/kqapro_server.py`

---

## Table of Contents

- [Discovery & Search Tools](#discovery--search-tools)
- [Node Exploration Tools](#node-exploration-tools)
- [Label Resolution Tools](#label-resolution-tools)
- [Schema Validation Tools](#schema-validation-tools)
- [Attribute & Temporal Tools](#attribute--temporal-tools)
- [Relation Navigation Tools](#relation-navigation-tools)
- [Qualifier Tools](#qualifier-tools)
- [Comparison & Verification Tools](#comparison--verification-tools)
- [Query & Classification Tools](#query--classification-tools)
- [State Management Tools](#state-management-tools)

---

## Discovery & Search Tools

### 1. FindNode
**Purpose:** Find entities or concepts in the knowledge graph using semantic search

**Location:** kqapro_server.py:1669

**Parameters:**
- `semantic_node_name` (str) - Entity name, concept, or technical ID to search for

**What it does:**
- Performs HYBRID search combining exact filtering and semantic vector search
- Works for both technical IDs (e.g., "GGZX52") and semantic concepts (e.g., "Boston")
- Returns node metadata including available attributes and predicates
- Two-phase search: First tries exact match, then falls back to vector similarity

**When to use:**
- Looking for a named entity (person, place, organization)
- Searching for a general concept (e.g., "Director", "Film")
- When you don't have a specific attribute value to search by

**Returns:** `SearchResponse` - List of matching nodes with relevance scores, available attributes, and available predicates

**Journal Updates:** Auto-updates `visited_nodes` with top 5 matches

---

### 2. FindByAttribute
**Purpose:** Reverse lookup - find entities by their attribute VALUE

**Location:** kqapro_server.py:3245

**Parameters:**
- `value` (str) - The attribute value to search for (e.g., "UKE11", "94332", "http://...")
- `attribute_name` (str) - The attribute name (e.g., "NUTS code", "GameID", "official website")

**What it does:**
- Searches for entities that have a specific attribute value
- Designed for technical IDs, codes, URLs, and other non-semantic values
- More precise than semantic search for exact value matching
- Uses SPARQL query with FILTER to find exact matches

**When to use:**
- Question contains a UNIQUE identifier (GameID, ISBN, NUTS code, URL)
- Searching for entities with specific attribute values
- When semantic search won't work (technical codes, URLs)

**Returns:** `SearchResponse` - List of entities that have the specified attribute value

**Journal Updates:** Auto-updates `visited_nodes` with top 5 matches

---

## Node Exploration Tools

### 3. GetNodeSummary
**Purpose:** Get ALL data about a node in ONE call

**Location:** kqapro_server.py:1774

**Parameters:**
- `node_id` (str) - The entity ID (e.g., "Q54089")

**What it does:**
- Fetches comprehensive node information including ALL attributes and relations
- Reduces multiple tool calls to a single request
- Returns structured summary with metadata, attributes, relations, and statistics
- Uses a single SPARQL query to fetch all predicates and objects

**When to use:**
- Need multiple attributes from the same node
- Want to see everything available for a node at once
- Exploring what data exists for an entity
- Reduces iterations compared to calling GetAttributeDetails multiple times

**Returns:** `Dict[str, Any]` - Complete node summary with:
- `node_id`, `name`, `node_type`
- `attributes`: Dict of attribute_name → list of values
- `relations`: Dict of relation_name → list of related_node_ids
- `summary_stats`: Counts of attributes and relations
- `status`: Success/error message

**Journal Updates:**
- Auto-updates `found_values` with all discovered attributes
- Adds completion step

---

### 4. GetAttributeDetails
**Purpose:** Get full details of a SPECIFIC attribute for a node

**Location:** kqapro_server.py:1974

**Parameters:**
- `base_node_id` (str) - The entity ID (e.g., "Q100")
- `attribute_name` (str) - The attribute to query (e.g., "population", "area")

**What it does:**
- Retrieves all values for a specific attribute
- Automatically resolves RDF blank nodes for quantities with units
- Returns numeric values and units separately
- Single SPARQL query with OPTIONAL clauses for blank node resolution

**When to use:**
- Need a specific attribute value (when you know the attribute name)
- Following up on available_attributes from FindNode results
- Single-attribute queries

**Returns:** `AttributeDetailsResponse` - List of attribute values with:
- `node_id`, `attribute_name`
- `values`: List of dicts with `value`, `unit` (if applicable), `type`
- `status`: Success/error message

**Journal Updates:**
- Auto-updates `found_values[node_id][attribute_name]`
- Adds verified facts (first 3 values)
- Adds completion step

---

### 5. GetRelationDetails
**Purpose:** Get all entities connected via a SPECIFIC relation

**Location:** kqapro_server.py:2564

**Parameters:**
- `base_node_id` (str) - The entity ID (e.g., "Q100")
- `relation_name` (str) - The relation predicate (e.g., "country", "capital of")

**What it does:**
- Retrieves all nodes connected via specified relation
- Automatically checks BOTH directions (subject→object and object←subject)
- Returns full details of related entities with direction information

**When to use:**
- Following relationships between entities
- Finding connected entities (e.g., "films directed by X")
- When you know the exact relation name from available_predicates

**Returns:** `RelationDetailsResponse` - List of related nodes with:
- `node_id`, `relation_name`
- `triples`: List of dicts with `related_id`, `related_uri`, `direction`
- `status`: Success/error message

**Journal Updates:**
- Adds verified facts (first 5 relations)
- Adds completion step

---

## Label Resolution Tools

### 6. GetNodeLabel
**Purpose:** Resolve entity ID to human-readable label

**Location:** kqapro_server.py:770

**Parameters:**
- `node_id` (str) - Entity ID to resolve (e.g., "Q1860", "Q217008")

**What it does:**
- Queries SPARQL for rdfs:label of entity
- Returns label and node type (entity/concept)
- CRITICAL for qualifier handling - always use when encountering entity IDs in qualifiers

**When to use:**
- ALWAYS use when you encounter entity IDs in qualifiers
- Common pattern: "original_language: Q1860" → call GetNodeLabel("Q1860") → "English"
- Converting entity IDs to human-readable names for final answers

**Returns:** JSON string with `node_id`, `label`, `node_type`, `status`

**Journal Updates:** Auto-updates `visited_nodes[node_id]` with label

---

### 7. BatchGetNodeLabels
**Purpose:** Efficiently resolve MULTIPLE entity IDs to labels in a single call

**Location:** kqapro_server.py:857

**Parameters:**
- `node_ids` (list[str]) - List of entity IDs to resolve (e.g., ["Q1860", "Q217008", "Q699224"])

**What it does:**
- Single SPARQL query with VALUES clause resolves all IDs at once
- Much more efficient than calling GetNodeLabel multiple times
- Returns resolved labels and list of IDs that couldn't be resolved

**When to use:**
- Use instead of calling GetNodeLabel multiple times
- Processing qualifier results with many entity IDs
- Batch label resolution for performance

**Returns:** JSON string with:
- `resolved`: Dict of {entity_id: label}
- `not_found`: List of IDs without labels
- `status`: Success message with counts

**Journal Updates:**
- Auto-updates `visited_nodes` for all resolved entities
- Logs failed attempts for not_found entities

---

## Schema Validation Tools

### 8. GetSchemaForAttribute
**Purpose:** Validate and fuzzy-match attribute names before using FindByAttribute

**Location:** kqapro_server.py:941

**Parameters:**
- `attribute_name` (str) - Attribute to search for (e.g., "GameID", "NUTS code")
- `fuzzy` (bool, default=True) - Enable fuzzy matching

**What it does:**
- Queries all distinct attributes from Virtuoso
- Checks for exact match first
- If no exact match, uses embeddings for fuzzy matching
- Returns top 3 fuzzy matches with similarity scores and usage counts

**When to use:**
- CRITICAL: ALWAYS use BEFORE FindByAttribute with technical IDs
- Prevents failures from wrong attribute names (e.g., "GameID" vs "Nintendo GameID")
- When unsure of exact attribute name

**Returns:** JSON string with:
- `query`: Your search term
- `exact_match`: Exact attribute name if found
- `fuzzy_matches`: List of similar attributes with similarity scores
- `recommendation`: Best match to use
- `status`: Success/error message

---

## Attribute & Temporal Tools

### 9. GetAttributeWithQualifiers
**Purpose:** Get attribute values INCLUDING all their qualifiers (context metadata)

**Location:** kqapro_server.py:2131

**Parameters:**
- `base_node_id` (str) - The entity ID (e.g., "Q54089")
- `attribute_name` (str) - The attribute name (e.g., "population", "director")

**What it does:**
- Retrieves attribute values with ALL their qualifiers (dates, locations, roles, etc.)
- One-stop solution for qualified facts in a single SPARQL query
- Includes temporal, spatial, and contextual metadata
- Automatically resolves blank nodes and extracts qualifiers

**When to use:**
- Questions with temporal constraints: "as of 2015", "in 2020", "on January 1st"
- Questions about qualified facts: "population at a specific time"
- Any attribute that might have context (date, location, role, determination method, etc.)
- When you need to filter by qualifier value (e.g., find population value for 2015)

**Returns:** `Dict[str, Any]` with:
- `node_id`, `attribute_name`
- `values`: List of dicts with:
  - `value`: The attribute value
  - `unit`: Unit of measurement (if applicable)
  - `qualifiers`: Dict of qualifier_name → qualifier_value (e.g., {"point in time": "2015-01-01", "determination method": "United States Census"})
- `status`: Success/error message

**Journal Updates:**
- Auto-updates `found_values[node_id][attribute_name]`
- Adds verified facts with qualifiers (first 3)
- Adds completion step

**Example:**
```python
result = GetAttributeWithQualifiers("Q54089", "population")
# Returns population values with their dates, determination methods, etc.
# You can then filter for specific qualifier values
```

---

### 10. TemporalAttributeQuery
**Purpose:** Get attribute value as of a SPECIFIC DATE (easiest temporal tool)

**Location:** kqapro_server.py:2373

**Parameters:**
- `base_node_id` (str) - The entity ID (e.g., "Q54089")
- `attribute_name` (str) - The attribute name (e.g., "population")
- `target_date` (str) - Target date in ISO format (YYYY-MM-DD) or just year (YYYY)
- `tolerance_days` (int, default=365) - Max days difference to accept

**What it does:**
- Finds attribute value closest to the specified date
- Automatically calls GetAttributeWithQualifiers and filters by date
- Handles date parsing and returns closest match within tolerance
- EASIEST way to answer "What was X's Y in [year]?" questions

**When to use:**
- Question has explicit date reference ("in 2015", "on 1998-04-09")
- Temporal queries with specific dates
- Prefer this over GetAttributeWithQualifiers for date-specific queries

**Returns:** `Dict[str, Any]` with:
- `node_id`, `attribute_name`, `target_date`
- `found_date`: Actual date of value found
- `value`: The attribute value
- `unit`: Unit of measurement
- `date_difference_days`: Days between target and found date
- `all_qualifiers`: Other qualifiers for this value
- `status`: Success/error message

**Journal Updates:** Adds verified fact to journal

**Example:**
```python
result = TemporalAttributeQuery("Q54089", "population", "2015-01-01")
# Returns: population value from 2015 (or closest within tolerance)
```

---

## Relation Navigation Tools

### 11. ExploreNeighborhood
**⚠️ DEPRECATED - Will be removed in future version**

**Purpose:** Semantically find relations you don't know the exact name for

**Location:** kqapro_server.py:2690

**Deprecation Notice:**
This tool is redundant. Use instead:
- `GetNodeSummary(node_id)` - Gets ALL relations at once
- `GetRelationDetails(node_id, relation_name)` - For specific relations
- `FindNode` already returns `available_predicates` for semantic matching

**Parameters:**
- `base_node_id` (str) - The entity ID (e.g., "Q64")
- `semantic_relation_name` (str) - Semantic description of relation (e.g., "population", "born in")

**What it does:**
- Uses vector search on relations collection to find candidate relations
- Verifies candidate relations exist via SPARQL
- Returns verified matches with actual data

**When to use:**
- ❌ Don't use - functionality is redundant
- ✅ Use GetNodeSummary or GetRelationDetails instead

**Returns:** `NeighborhoodResponse` with:
- `base_node`: The queried node ID
- `verified_match`: Best matching relation with objects and confidence (if found)
- `candidates_checked`: List of predicates checked
- `status`: Success/error message

**Journal Updates:**
- Adds verified facts when match found
- Adds completion step
- Logs failed attempts if no match

---

### 12. FindEntitiesByRelationPath
**Purpose:** Navigate multi-hop relation paths in ONE query

**Location:** kqapro_server.py:2901

**Parameters:**
- `start_node_id` (str) - Starting entity ID (e.g., "Q678410")
- `relation_path` (list[Dict[str, str]]) - Path steps, each with:
  - `relation` (str): Relation name (e.g., "location of formation")
  - `direction` (str): "forward" (A→B) or "backward" (B→A)

**What it does:**
- Follows multi-hop relation chains (A→B→C→D) in ONE SPARQL query
- Executes entire path in one request instead of sequential calls
- Returns all entities found at the end of the path
- Includes intermediate nodes for debugging

**When to use:**
- Multi-hop questions (e.g., "Films directed by people born in Boston")
- Complex relationship chains (>2 hops)
- When you need to navigate through multiple relations
- STRONGLY preferred over sequential GetRelationDetails calls

**Returns:** `Dict[str, Any]` with:
- `start_node_id`, `relation_path`, `path_length`
- `entities_found`: List of entity IDs at end of path
- `intermediate_nodes`: Dict showing nodes at each hop
- `status`: Success/error message

**Journal Updates:**
- Adds verified fact with path and results
- Adds completion step

**Example:**
```python
# Find counties that border counties that border Cecil County
result = FindEntitiesByRelationPath("Q385365", [
    {"relation": "shares border with", "direction": "forward"},
    {"relation": "shares border with", "direction": "forward"}
])
```

---

## Qualifier Tools

### 13. GetEdgeQualifiers
**Purpose:** Extract ALL qualifiers from a specific ATTRIBUTE statement

**Location:** kqapro_server.py:1091

**Parameters:**
- `base_node_id` (str) - Entity ID (e.g., "Q217008")
- `attribute_name` (str) - Attribute name (e.g., "official website")
- `attribute_value` (str, optional) - Specific value to filter by

**What it does:**
- Retrieves all qualifiers for an attribute statement
- Uses attr: namespace and RDF statement reification
- Automatically resolves entity IDs in qualifiers to labels
- Tries alternative query patterns if first pattern fails

**When to use:**
- CRITICAL for QueryAttrQualifier questions
- Questions ask about metadata/context of an attribute
- Examples: "What language is associated with website X?", "Where was X published on date Y?"

**Returns:** JSON string with:
- `base_node`, `attribute`, `value`
- `qualifiers`: Dict of qualifier names → values (with entity labels resolved)
- `status`: Success/error message

**Journal Updates:**
- Adds verified facts with qualifier structure
- Logs failed attempts if no qualifiers found

---

### 14. GetQualifiersByPredicate
**Purpose:** Find ALL qualifiers for a specific RELATION statement

**Location:** kqapro_server.py:1285

**Parameters:**
- `base_node_id` (str) - Source entity ID (e.g., "Q100")
- `relation_name` (str) - Relation/predicate name (e.g., "award received")
- `target_node_id` (str, optional) - Target entity to filter by

**What it does:**
- Retrieves all qualifiers for a relation statement
- Uses prop: namespace and RDF statement reification
- Automatically resolves entity IDs in qualifiers to labels
- Handles both target node label and qualifiers in one query

**When to use:**
- CRITICAL for complex award/relation questions
- Questions ask about context of a relationship
- Example: "What film won award X with winner Y?"

**Returns:** JSON string with:
- `base_node`, `relation`, `target`, `target_label`
- `qualifiers`: Dict of qualifier names → values (with labels)
- `status`: Success/error message

**Journal Updates:**
- Adds verified facts with qualifier structure
- Logs failed attempts if no qualifiers found

---

## Comparison & Verification Tools

### 15. CompareEntities
**Purpose:** Compare an attribute across multiple entities and get sorted results

**Location:** kqapro_server.py:3486

**Parameters:**
- `entity_ids` (list[str]) - Entity IDs to compare (e.g., ["Q100", "Q64"])
- `attribute_name` (str) - Attribute to compare (e.g., "population", "duration")

**What it does:**
- Fetches attribute for ALL entities in ONE SPARQL query using VALUES clause
- Automatically sorts results by attribute value
- Returns leaderboard-style comparison
- Handles numeric values and units

**When to use:**
- "Which is taller/longer/larger" questions (SelectBetween, SelectAmong)
- Comparing attributes across multiple entities
- Sorting/ranking entities by a property
- MUCH faster than calling GetAttributeDetails for each entity

**Returns:** `CompareEntitiesResponse` with:
- `attribute_name`, `sorted_by`
- `results`: List of `ComparisonResult` objects (sorted) with:
  - `entity_id`, `entity_name`
  - `value`: Attribute value (with unit if applicable)
  - `normalized_value`: Float value for sorting
- `status`: Success/error message

**Journal Updates:**
- Auto-updates `found_values` for all entities
- Adds completion step

---

### 16. VerifyNumericCondition
**Purpose:** Perform DETERMINISTIC mathematical comparison (LLMs are bad at math!)

**Location:** kqapro_server.py:3357

**Parameters:**
- `value1` (str) - First value (can include units like "150 million")
- `operator` (str) - Comparison operator: `<`, `>`, `<=`, `>=`, `==`, `!=`
- `value2` (str) - Second value (can include units)
- `unit` (str, optional) - Unit description for context

**What it does:**
- Performs precise mathematical comparison
- Returns definitive TRUE/FALSE verdict
- Handles numbers, dates, and values with units (e.g., "150 million", "1.5k")
- Provides explanation of comparison

**When to use:**
- ANY numeric comparison ("Is X > Y?", "Is X taller than Y?")
- Verify questions requiring TRUE/FALSE answers
- Date comparisons
- ALWAYS use for math instead of relying on LLM reasoning

**Returns:** `NumericComparisonResponse` with:
- `verdict`: "TRUE", "FALSE", or "ERROR"
- `explanation`: Human-readable explanation
- `value1`, `value2`: Normalized values
- `operator`: The operator used

**Journal Updates:**
- Adds verified fact with explanation
- Adds completion step
- Logs failed attempts on error

---

## Query & Classification Tools

### 17. RunSPARQL
**Purpose:** Execute raw SPARQL queries for complex operations

**Location:** kqapro_server.py:3100

**Parameters:**
- `query` (str) - Valid SPARQL SELECT query

**What it does:**
- Executes arbitrary SPARQL queries against knowledge graph
- Auto-injects standard prefixes (ex:, prop:, attr:, qual:, unit:, rdfs:, rdf:, xsd:)
- Supports complex queries with aggregations, filters, joins
- Fallback for operations other tools cannot handle

**When to use:**
- Complex queries requiring COUNT, GROUP BY, aggregations
- Multi-constraint queries (multiple WHERE clauses)
- When specialized tools fail or are inefficient
- Large-scale queries (e.g., "How many cities in China?")
- Custom SPARQL patterns not covered by other tools

**Important Notes:**
- DO NOT define prefixes in your query - they are auto-injected
- Use proper namespace prefixes:
  - `ex:Q12345` for entities
  - `prop:P1082` for relations
  - `attr:population` for attributes
  - `qual:P585` for qualifiers
  - `unit:square_kilometre` for units
- Try simpler tools first (GetAttributeDetails, GetRelationDetails, etc.)
- If syntax error, use specialized tools instead

**Returns:** `SPARQLResponse` with:
- `vars`: List of variable names in SELECT clause
- `bindings`: List of result rows (dicts mapping var → value)
- `raw_json`: Full raw JSON response from Virtuoso

**Journal Updates:**
- Adds verified facts with query results (first 10)
- Adds completion step
- Logs failed attempts on error

---

## State Management Tools

### 18. ManageJournal
**Purpose:** Track progress and prevent loops (scratchpad management)

**Location:** kqapro_server.py:1489

**Parameters:**
- `action` (str) - Type of update:
  - `add_visited` - (DEPRECATED - auto-updated by FindNode)
  - `add_fact` - (DEPRECATED - auto-updated by tools)
  - `update_plan` - Update reasoning plan (manual)
  - `set_qtype` - Set question type
  - `set_target` - Add target entity or attribute
  - `set_partial_answer` - Store intermediate answer reasoning
  - `read` - Read current journal state
- `content` (str) - Text content to add/update

**What it does:**
- Maintains scratchpad/journal of agent progress
- Tracks visited nodes, verified facts, current plan, completed steps
- Most updates happen AUTOMATICALLY via other tools
- Mainly for manual planning and intermediate answers

**When to use:**
- RARE - Most updates are automatic
- Manually update reasoning plan (update_plan)
- Set partial answer before synthesis (set_partial_answer)
- Explicitly mark progress in complex multi-step reasoning

**Returns:** String - Full current journal content (formatted)

**Auto-Updates:**
- `visited_nodes` - Auto-updated by FindNode, BatchGetNodeLabels
- `found_values` - Auto-updated by GetAttributeDetails, GetAttributeWithQualifiers, CompareEntities
- `verified_facts` - Auto-updated by various tools
- `failed_attempts` - Auto-logged when tools fail

---

### 19. GetJournalSummary
**Purpose:** Get formatted summary of ALL discoveries (CRITICAL before final answer!)

**Location:** kqapro_server.py:1546

**Parameters:** None

**What it does:**
- Returns formatted summary of everything discovered
- Shows visited nodes, discovered values, verified facts, completed steps
- Provides organized view of all gathered information
- Highlights which nodes have data vs which don't

**When to use:**
- **MANDATORY before formulating final answer**
- Periodic reviews during long investigations (auto-injected every 5 iterations)
- When you need to review what you've learned so far

**Returns:** String - Formatted summary showing:
- Question type and target entities
- **DISCOVERED VALUES** (MOST IMPORTANT - use these for your answer!)
- Explored nodes (with indicators of which have data)
- Completed steps and current plan
- Failed attempts
- Statistics (nodes explored, entities with data, steps completed)

---

## Tool Selection Strategy

### Quick Decision Tree

**1. Looking for an entity?**
- Has unique ID/code/URL → `FindByAttribute` (after validating with `GetSchemaForAttribute`)
- Named entity/concept → `FindNode`

**2. Need attribute value?**
- Specific date required → `TemporalAttributeQuery`
- Need qualifiers/context → `GetAttributeWithQualifiers`
- Just the value → `GetAttributeDetails`
- Multiple attributes → `GetNodeSummary`

**3. Need relationships?**
- Know exact relation → `GetRelationDetails`
- Semantic search for relation → `GetNodeSummary` (deprecated: ~~ExploreNeighborhood~~)
- Multi-hop path → `FindEntitiesByRelationPath`

**4. Need qualifiers?**
- For attributes → `GetEdgeQualifiers` or `GetAttributeWithQualifiers`
- For relations → `GetQualifiersByPredicate`

**5. Comparison/Verification?**
- Numeric comparison → `VerifyNumericCondition`
- Compare multiple entities → `CompareEntities`

**6. Complex query?**
- Multi-constraint, aggregations → `RunSPARQL`

**7. Resolve entity IDs?**
- Single ID → `GetNodeLabel`
- Multiple IDs → `BatchGetNodeLabels`

**8. Before final answer?**
- **ALWAYS call** `GetJournalSummary`

---

## Common Pitfalls

### DON'T:
- Call FindNode repeatedly with same search term (indicates loop)
- Use RunSPARQL before trying specialized tools
- Fetch attributes one-by-one when GetNodeSummary would work
- Compare numbers manually - use VerifyNumericCondition
- Forget to call GetJournalSummary before answering
- Use FindByAttribute without first validating attribute name with GetSchemaForAttribute
- Call GetNodeLabel multiple times - use BatchGetNodeLabels instead

### DO:
- Use FindByAttribute for unique identifiers (after schema validation)
- Use TemporalAttributeQuery for date-specific queries
- Use CompareEntities for sorting/ranking
- Use GetNodeSummary to reduce iterations
- Review available_attributes/predicates from FindNode results
- Check GetJournalSummary regularly
- Use BatchGetNodeLabels for resolving multiple entity IDs

---

## Performance Tips

**Reduce Iterations:**
- Prefer `GetNodeSummary` over multiple `GetAttributeDetails` calls
- Prefer `FindEntitiesByRelationPath` over sequential `GetRelationDetails` calls
- Use `CompareEntities` for batch comparisons
- Use `BatchGetNodeLabels` instead of multiple `GetNodeLabel` calls
- Use `GetAttributeWithQualifiers` instead of separate attribute + qualifier queries

**Prevent Loops:**
- Journal automatically tracks visited nodes
- Loop detection fires after 3 identical calls or oscillating patterns
- Review GetJournalSummary to avoid revisiting explored paths

**Handle Failures:**
- If FindNode fails twice, try FindByAttribute or search for connected entity
- If GetAttributeDetails fails, review available_attributes from FindNode
- If relation not found, try GetNodeSummary to see all available relations
- If FindByAttribute fails, validate attribute name with GetSchemaForAttribute first

---

## Auto-Injected Context

The following information is automatically injected by the pre-agent hook:
- **Question Type** - Classified from the question
- **Extracted Entities** - Identified entities/concepts from the question
- **Extracted Relations** - Identified relations from the question
- **Few-Shot Examples** - Curated examples for the question type
- **Strategy Guidance** - Type-specific reasoning approach

This context enrichment happens automatically before the agent starts processing.

---

## Auto-Updated Journal Tracking

The following journal fields are automatically updated:
- `visited_nodes` - When FindNode, GetNodeLabel, BatchGetNodeLabels are called
- `found_values` - When GetAttributeDetails, GetAttributeWithQualifiers, GetNodeSummary, CompareEntities are called
- `verified_facts` - When tools verify information
- `failed_attempts` - When tools return errors
- `completed_steps` - When tools complete successfully

You should focus on using tools; the journal tracks progress automatically.
