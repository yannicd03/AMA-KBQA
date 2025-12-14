# AMA KBQA Tools Reference

Complete reference for all MCP tools available to the KQAPro Agent.

**Total Tools:** 17
**Server Location:** `ama_kbqa/server/kqapro_server.py`

---

## Table of Contents

- [Discovery & Search Tools](#discovery--search-tools)
- [Node Exploration Tools](#node-exploration-tools)
- [Attribute & Temporal Tools](#attribute--temporal-tools)
- [Relation Navigation Tools](#relation-navigation-tools)
- [Comparison & Verification Tools](#comparison--verification-tools)
- [Query & Classification Tools](#query--classification-tools)
- [State Management Tools](#state-management-tools)

---

## Discovery & Search Tools

### 1. FindNode
**Purpose:** Find entities or concepts in the knowledge graph using semantic search

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

**Returns:** List of matching nodes with relevance scores, available attributes, and available predicates

---

### 2. EntityExtraction
**Purpose:** Extract entities and relations from natural language questions

**Parameters:**
- `query` (str) - Natural language question to analyze

**What it does:**
- Uses LLM in JSON mode to identify entities/concepts and relations
- Automatically called by pre-agent hook for all questions
- Provides structured extraction for agent reasoning

**When to use:**
- Automatically called by pre-agent hook
- Manually use when you need to re-analyze a complex question mid-execution

**Returns:** Lists of extracted entities/concepts and relations

---

### 3. FindByAttribute
**Purpose:** Reverse lookup - find entities by their attribute VALUE

**Parameters:**
- `value` (str) - The attribute value to search for (e.g., "UKE11", "94332", "http://...")
- `attribute_name` (str) - The attribute name (e.g., "NUTS code", "GameID", "official website")

**What it does:**
- Searches for entities that have a specific attribute value
- Designed for technical IDs, codes, URLs, and other non-semantic values
- More precise than semantic search for exact value matching

**When to use:**
- Question contains a UNIQUE identifier (GameID, ISBN, NUTS code, URL)
- Searching for entities with specific attribute values
- When semantic search won't work (technical codes, URLs)

**Returns:** List of entities that have the specified attribute value

---

## Node Exploration Tools

### 4. GetNodeSummary
**Purpose:** Get ALL data about a node in ONE call

**Parameters:**
- `node_id` (str) - The entity ID (e.g., "Q54089")

**What it does:**
- Fetches comprehensive node information including ALL attributes and relations
- Reduces multiple tool calls to a single request
- Returns structured summary with metadata, attributes, relations, and statistics

**When to use:**
- Need multiple attributes from the same node
- Want to see everything available for a node at once
- Exploring what data exists for an entity
- Reduces iterations compared to calling GetAttributeDetails multiple times

**Returns:** Complete node summary with all attributes, relations, and metadata

---

### 5. GetAttributeDetails
**Purpose:** Get full details of a SPECIFIC attribute for a node

**Parameters:**
- `base_node_id` (str) - The entity ID (e.g., "Q100")
- `attribute_name` (str) - The attribute to query (e.g., "population", "area")

**What it does:**
- Retrieves all values for a specific attribute
- Automatically resolves RDF blank nodes for quantities with units
- Returns numeric values and units separately

**When to use:**
- Need a specific attribute value (when you know the attribute name)
- Following up on available_attributes from FindNode results
- Single-attribute queries

**Returns:** List of attribute values with unit resolution

---

### 6. GetRelationDetails
**Purpose:** Get all entities connected via a SPECIFIC relation

**Parameters:**
- `base_node_id` (str) - The entity ID (e.g., "Q100")
- `relation_name` (str) - The relation predicate (e.g., "country", "capital of")

**What it does:**
- Retrieves all nodes connected via specified relation
- Automatically checks BOTH directions (subject→object and object←subject)
- Returns full details of related entities

**When to use:**
- Following relationships between entities
- Finding connected entities (e.g., "films directed by X")
- When you know the exact relation name from available_predicates

**Returns:** List of related nodes with full triple details

---

## Attribute & Temporal Tools

### 7. GetAttributeWithQualifiers
**Purpose:** Get attribute values INCLUDING their qualifiers (context metadata)

**Parameters:**
- `base_node_id` (str) - The entity ID (e.g., "Q54089")
- `attribute_name` (str) - The attribute name (e.g., "population", "director")

**What it does:**
- Retrieves attribute values with ALL their qualifiers (dates, locations, roles, etc.)
- One-stop solution for qualified facts
- Includes temporal, spatial, and contextual metadata

**When to use:**
- Question asks about temporal context ("as of date", "in 2015")
- Need to know WHEN, WHERE, or HOW for a fact
- Attribute values change over time
- Question specifies conditions for the fact

**Returns:** Attribute values with complete qualifier metadata (dates, locations, roles, determination methods)

---

### 8. TemporalAttributeQuery
**Purpose:** Get attribute value as of a SPECIFIC DATE (easiest temporal tool)

**Parameters:**
- `base_node_id` (str) - The entity ID (e.g., "Q54089")
- `attribute_name` (str) - The attribute name (e.g., "population")
- `target_date` (str) - Target date in ISO format (YYYY-MM-DD) or just year (YYYY)
- `tolerance_days` (int, default=365) - Max days difference to accept

**What it does:**
- Finds attribute value closest to the specified date
- Automatically filters by date qualifiers
- Handles date parsing and returns closest match within tolerance
- EASIEST way to answer "What was X's Y in [year]?" questions

**When to use:**
- Question has explicit date reference ("in 2015", "on 1998-04-09")
- Temporal queries with specific dates
- Prefer this over GetAttributeWithQualifiers for date-specific queries

**Returns:** Value closest to target date with date difference and unit information

---

## Relation Navigation Tools

### 9. ExploreNeighborhood
**Purpose:** Semantically find relations you don't know the exact name for

**Parameters:**
- `base_node_id` (str) - The entity ID (e.g., "Q64")
- `semantic_relation_name` (str) - Semantic description of relation (e.g., "population", "born in")

**What it does:**
- Uses vector search to find candidate relations matching semantic description
- Verifies candidate relations exist via SPARQL
- Returns verified matches with actual data

**When to use:**
- Don't know the exact relation/predicate name
- Semantic search for relations
- Exploring what relations exist for a node
- When GetRelationDetails fails because you guessed the wrong relation name

**Returns:** Verified relation matches with objects and confidence scores

---

### 10. FindEntitiesByRelationPath
**Purpose:** Navigate multi-hop relation paths in ONE query

**Parameters:**
- `start_node_id` (str) - Starting entity ID (e.g., "Q678410")
- `relation_path` (list of dicts) - Path steps with "relation" and "direction" ("forward"/"backward")

**What it does:**
- Follows multi-hop relation chains (A→B→C→D)
- Executes entire path in ONE SPARQL query instead of sequential calls
- Returns all entities found at the end of the path
- Includes intermediate nodes for debugging

**When to use:**
- Multi-hop questions (e.g., "Films directed by people born in Boston")
- Complex relationship chains (>2 hops)
- When you need to navigate through multiple relations
- STRONGLY preferred over sequential GetRelationDetails calls

**Returns:** Entities found at end of path, with intermediate nodes and path metadata

---

### 11. GetEdgeQualifiers
**Purpose:** Get qualifiers (metadata) for a SPECIFIC relationship instance

**Parameters:**
- `subject_id` (str) - The entity ID (e.g., "Q100")
- `predicate_name` (str) - The relation name (e.g., "official website")
- `target_id` (str) - The specific target value or entity ID

**What it does:**
- Retrieves 'facts about a fact' (qualifiers on edges)
- Gets context like language, start time, location, or role for a specific relation
- Provides fine-grained metadata about relationships

**When to use:**
- Question asks about relationship details ("role in", "capacity as")
- Need context for a SPECIFIC relationship instance
- Temporal/spatial context for relationships
- Example: "What language was the website (published on date X) in?"

**Returns:** Qualifiers for the specified relationship (time, location, role, etc.)

---

## Comparison & Verification Tools

### 12. CompareEntities
**Purpose:** Compare an attribute across multiple entities and get sorted results

**Parameters:**
- `entity_ids` (list of str) - Entity IDs to compare (e.g., ["Q100", "Q64"])
- `attribute_name` (str) - Attribute to compare (e.g., "population", "duration")

**What it does:**
- Fetches attribute for ALL entities in ONE call
- Automatically sorts results by attribute value
- Returns leaderboard-style comparison
- Handles numeric values and units

**When to use:**
- "Which is taller/longer/larger" questions (SelectBetween, SelectAmong)
- Comparing attributes across multiple entities
- Sorting/ranking entities by a property
- MUCH faster than calling GetAttributeDetails for each entity

**Returns:** Sorted list of entities with attribute values and rankings

---

### 13. VerifyNumericCondition
**Purpose:** Perform DETERMINISTIC mathematical comparison (LLMs are bad at math!)

**Parameters:**
- `value1` (str) - First value (can include units like "150 million")
- `operator` (str) - Comparison operator (`<`, `>`, `<=`, `>=`, `==`, `!=`)
- `value2` (str) - Second value (can include units)
- `unit` (str, optional) - Unit description for context

**What it does:**
- Performs precise mathematical comparison
- Returns definitive TRUE/FALSE verdict
- Handles numbers, dates, and values with units
- Provides explanation of comparison

**When to use:**
- ANY numeric comparison ("Is X > Y?", "Is X taller than Y?")
- Verify questions requiring TRUE/FALSE answers
- Date comparisons
- ALWAYS use for math instead of relying on LLM reasoning

**Returns:** Verdict (TRUE/FALSE/ERROR) with explanation and normalized values

---

## Query & Classification Tools

### 14. RunSPARQL
**Purpose:** Execute raw SPARQL queries for complex operations

**Parameters:**
- `query` (str) - Valid SPARQL SELECT query using system prefixes

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

**Returns:** Raw SPARQL results with variables and bindings

**Important Notes:**
- DO NOT define prefixes in your query - they are auto-injected
- Use proper namespace prefixes (ex: for entities, prop: for relations, attr: for attributes)
- Validate SPARQL syntax carefully to avoid errors

---

### 15. QtypePrediction
**Purpose:** Classify question type and retrieve relevant few-shot examples

**Parameters:**
- `question` (str) - Question to classify

**What it does:**
- Classifies question into KQAPro taxonomy (Count, Verify, QueryAttr, etc.)
- Loads curated few-shot examples specific to that question type
- Provides strategic guidance for answering that type of question
- Automatically called by pre-agent hook for all questions

**When to use:**
- Automatically called by pre-agent hook - rarely need to call manually
- Can be re-called if question complexity changes mid-execution

**Returns:** Question type classification and curated few-shot examples

**Question Types:**
- **Count:** Aggregation questions ("How many...")
- **Verify:** Boolean questions ("Is...", "Does...", "Did...")
- **QueryAttr:** Direct attribute lookup
- **QueryAttrQualifier:** Attribute with context (time/place)
- **QueryRelation:** Identify relationship between entities
- **QueryRelationQualifier:** Relationship with context
- **QueryName:** Reverse lookup / identification
- **SelectAmong:** Superlative/sorting ("tallest", "longest")
- **SelectBetween:** Binary comparison
- **Query:** General factual questions

---

## State Management Tools

### 16. ManageJournal
**Purpose:** Track progress and prevent loops (scratchpad management)

**Parameters:**
- `action` (str) - Type of update: `add_visited`, `add_fact`, `update_plan`, `set_qtype`, `set_target`, `set_partial_answer`, `read`
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

**Returns:** Full current journal content

**Auto-Updates:**
- `visited_nodes` - Auto-updated when calling FindNode
- `found_values` - Auto-updated when calling GetAttributeDetails
- `verified_facts` - Auto-updated by various tools
- `failed_attempts` - Auto-logged when tools fail

---

### 17. GetJournalSummary
**Purpose:** Get formatted summary of ALL discoveries (CRITICAL before final answer!)

**Parameters:** None

**What it does:**
- Returns formatted summary of everything discovered
- Shows visited nodes, discovered values, verified facts, completed steps
- Provides organized view of all gathered information

**When to use:**
- **MANDATORY before formulating final answer**
- Periodic reviews during long investigations (auto-injected every 5 iterations)
- When you need to review what you've learned so far

**Returns:** Formatted summary string showing:
- Question type and target entities
- Explored nodes
- Discovered values (MOST IMPORTANT)
- Completed steps and current plan
- Failed attempts
- Statistics

---

## Tool Selection Strategy

### Quick Decision Tree

**1. Looking for an entity?**
- Has unique ID/code/URL → `FindByAttribute`
- Named entity/concept → `FindNode`

**2. Need attribute value?**
- Specific date required → `TemporalAttributeQuery`
- Need qualifiers/context → `GetAttributeWithQualifiers`
- Just the value → `GetAttributeDetails`
- Multiple attributes → `GetNodeSummary`

**3. Need relationships?**
- Know exact relation → `GetRelationDetails`
- Semantic search for relation → `ExploreNeighborhood`
- Multi-hop path → `FindEntitiesByRelationPath`

**4. Comparison/Verification?**
- Numeric comparison → `VerifyNumericCondition`
- Compare multiple entities → `CompareEntities`

**5. Complex query?**
- Multi-constraint, aggregations → `RunSPARQL`

**6. Before final answer?**
- **ALWAYS call** `GetJournalSummary`

---

## Common Pitfalls

### DON'T:
- Call FindNode repeatedly with same search term (indicates loop)
- Use RunSPARQL before trying specialized tools
- Fetch attributes one-by-one when GetNodeSummary would work
- Compare numbers manually - use VerifyNumericCondition
- Forget to call GetJournalSummary before answering

### DO:
- Use FindByAttribute for unique identifiers
- Use TemporalAttributeQuery for date-specific queries
- Use CompareEntities for sorting/ranking
- Use GetNodeSummary to reduce iterations
- Review available_attributes/predicates from FindNode results
- Check GetJournalSummary regularly

---

## Performance Tips

**Reduce Iterations:**
- Prefer `GetNodeSummary` over multiple `GetAttributeDetails` calls
- Prefer `FindEntitiesByRelationPath` over sequential `GetRelationDetails` calls
- Use `CompareEntities` for batch comparisons

**Prevent Loops:**
- Journal automatically tracks visited nodes
- Loop detection fires after 3 identical calls or oscillating patterns
- Review GetJournalSummary to avoid revisiting explored paths

**Handle Failures:**
- If FindNode fails twice, try FindByAttribute or search for connected entity
- If GetAttributeDetails fails, review available_attributes from FindNode
- If relation not found, try ExploreNeighborhood for semantic matching

---

## Auto-Injected Context

The following information is automatically injected by the pre-agent hook:
- **Question Type** - From QtypePrediction
- **Extracted Entities** - From EntityExtraction
- **Extracted Relations** - From EntityExtraction
- **Few-Shot Examples** - Curated examples for the question type
- **Strategy Guidance** - Type-specific reasoning approach

You do NOT need to call these manually unless re-analyzing mid-execution.

---

## Auto-Updated Journal Tracking

The following journal fields are automatically updated:
- `visited_nodes` - When FindNode is called
- `found_values` - When GetAttributeDetails is called
- `verified_facts` - When tools verify information
- `failed_attempts` - When tools return errors

You should focus on using tools; the journal tracks progress automatically.
