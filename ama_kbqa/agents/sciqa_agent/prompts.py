"""
Prompts for the SciQA Agent (ORKG Knowledge Graph).

This module contains all prompts used by the SciQA agent for scientific
question answering over the Open Research Knowledge Graph.
"""

# ==============================================================================
# Question Type Specific Strategies
# ==============================================================================

QTYPE_STRATEGIES = {
    "Factoid": """
    STRATEGY: Factoid (Direct Fact Lookup)
    Topology: [Entity] -> [Predicate] -> [Value]

    **APPROACH:**
    1. Identify the main entity (paper, author, contribution, venue)
    2. Find it using FindResource with semantic query
    3. Use GetResourceSummary to see ALL available predicates at once
    4. Use GetRelationTargets with specific predicate to get the answer

    **SCHEMA INTROSPECTION FALLBACK:**
    If a predicate returns empty: Use GetResourceSummary to see all available predicates.
    Look for semantic matches (the predicate ID might differ from what you expect).

    **COMPARISON-BASED FACTOIDS (~55% of questions):**
    Many factoid questions involve Comparison resources rather than simple paper lookups.
    If the question asks about "values", "sources", "methods" across studies:
    1. FindResource to locate the Comparison resource
    2. GetComparisonContributions(comparison_id) to discover available predicates
    3. GetComparisonContributions(comparison_id, domain_predicate) to get values

    **SPARQL TIP:** When looking for domain data, always follow:
      ?paper orkgp:P31 ?contribution .
      ?contribution orkgp:DOMAIN_PREDICATE ?value .

    **COMMON PATTERNS:**
    - Paper DOI: GetRelationTargets(paper_id, "P26") or orkgp:P10
    - Paper venue: GetRelationTargets(paper_id, "P27")
    - Paper year: GetRelationTargets(paper_id, "P29")
    - Paper research field: GetRelationTargets(paper_id, "P30")
    - Paper contributions: GetPaperContributions(paper_id)
    - Paper authors: GetPaperAuthors(paper_id)
    - Domain data: FollowRelationPath(paper_id, [{"predicate":"P31","direction":"forward"}, {"predicate":"DOMAIN_PRED","direction":"forward"}])

    **SPARQL Pattern:**
    SELECT ?value WHERE {
        orkgr:RESOURCE_ID orkgp:PREDICATE ?value .
    }
    """,

    "Count": """
    STRATEGY: Count / Aggregation Query
    Topology: [Filter Condition] -> [Aggregate Items]

    **DECISION TREE:**
    1. Simple count (< 20 items) -> Use tools and count manually
    2. Large/unknown count -> Use RunORKGSPARQL with COUNT()
    3. Count with constraints -> SPARQL with FILTER
    4. Sum/Average/Min/Max -> Use RunORKGSPARQL with aggregation functions
    5. Frequency table -> GROUP BY + COUNT + ORDER BY

    **COMPARISON-BASED COUNTING (CRITICAL - ~55% of questions):**
    Many count questions involve Comparison resources. The pattern is:
      Comparison --compareContribution--> Contribution --domainPred--> Value
    Use GetComparisonContributions first, then count/aggregate the results.

    **Nested Value Pattern:** Some contributions use HAS_VALUE for their values:
      Comparison --compareContribution--> Contribution --HAS_VALUE--> ValueResource
      Then: ValueResource --domainPred--> actual_value
    Use GetResourceSummary on contributions to check for HAS_VALUE.

    **SPARQL PATTERNS:**

    Simple COUNT:
    SELECT (COUNT(DISTINCT ?item) AS ?count) WHERE {
        ?item orkgp:PREDICATE ?value .
    }

    COUNT with GROUP BY:
    SELECT ?category (COUNT(?item) AS ?count) WHERE {
        ?item orkgp:PREDICATE ?category .
    }
    GROUP BY ?category
    ORDER BY DESC(?count)

    SUM:
    SELECT (SUM(xsd:decimal(?value)) AS ?total) WHERE {
        orkgr:RESOURCE orkgp:compareContribution ?contrib .
        ?contrib orkgp:DOMAIN_PRED ?value .
    }

    AVG:
    SELECT (AVG(xsd:decimal(?value)) AS ?average) WHERE {
        orkgr:RESOURCE orkgp:compareContribution ?contrib .
        ?contrib orkgp:DOMAIN_PRED ?value .
    }

    COUNT via Comparison:
    SELECT (COUNT(DISTINCT ?contrib) AS ?count) WHERE {
        orkgr:COMPARISON orkgp:compareContribution ?contrib .
        ?contrib orkgp:DOMAIN_PRED ?value .
        FILTER(CONTAINS(LCASE(STR(?value)), "filter_term"))
    }

    **CONSTRAINT VERIFICATION:**
    After counting, verify numeric conditions with VerifyNumericCondition.
    """,

    "List": """
    STRATEGY: List (Multiple Result Query)
    Topology: [Filter Condition] -> [List Items]

    **APPROACH:**
    1. Identify the type of items to list (papers, authors, contributions)
    2. Identify filter conditions (research field, year, venue)
    3. Use appropriate tool or SPARQL to retrieve list

    **COMPARISON-BASED LISTS (~55% of questions):**
    Many list questions involve Comparison resources. The pattern is:
      Comparison --compareContribution--> Contribution --domainPred--> Value
    1. FindResource to locate the Comparison resource
    2. GetComparisonContributions(comparison_id) to discover available predicates
    3. GetComparisonContributions(comparison_id, domain_predicate) to list all values

    **MULTI-HOP LISTS:**
    For domain-specific lists, follow Paper -> P31 -> Contribution -> domain predicate.
    Use FollowRelationPath for multi-hop list retrieval in one call.

    **COMPLETENESS CHECK:**
    After retrieving a list, verify all items meet the stated conditions.

    **COMMON PATTERNS:**
    - Papers in field: GetResearchFieldPapers(field_name)
    - Authors of paper: GetPaperAuthors(paper_id)
    - Contributions of paper: GetPaperContributions(paper_id)
    - Domain-specific items: FollowRelationPath(paper_id, path)
    - Comparison data: GetComparisonContributions(comparison_id, predicate)

    **SPARQL Pattern:**
    SELECT ?item ?label WHERE {
        ?item rdf:type orkgc:TYPE .
        ?item rdfs:label ?label .
        ?item orkgp:PREDICATE ?filter_value .
    }
    """,

    "Boolean": """
    STRATEGY: Boolean (Yes/No Verification)
    Topology: [Entity] -> [Check Condition] -> True/False

    **APPROACH:**
    1. Identify the entity and condition to verify
    2. Query for the specific relation/value
    3. Return "Yes" if exists, "No" if not found

    **NUMERIC VERIFICATION:**
    For numeric conditions, ALWAYS use VerifyNumericCondition.
    E.g., "Does benchmark X have more than 10,000 questions?" ->
    Get value, then VerifyNumericCondition(value, ">", "10000", "questions")

    **COMPARISON-BASED BOOLEAN CHECKS:**
    For questions checking existence/properties in Comparison data:
      Comparison --compareContribution--> Contribution --domainPred--> Value
    Use ASK SPARQL pattern against the comparison structure:
    ASK {
        orkgr:COMPARISON orkgp:compareContribution ?contrib .
        ?contrib orkgp:DOMAIN_PRED ?value .
        FILTER(CONTAINS(LCASE(STR(?value)), "search_term"))
    }

    **EXISTENCE CHECKS:**
    Use ASK SPARQL pattern for existence checks.

    **CAPABILITY CHECKS:**
    Use GetResourceSummary and check predicate values.

    **COMMON PATTERNS:**
    - "Does paper X address problem Y?" -> Check P32 relation
    - "Is author A affiliated with B?" -> Check P7 relation
    - "Does contribution use method M?" -> Check P2 relation

    **SPARQL Pattern (ASK):**
    ASK {
        orkgr:ENTITY orkgp:PREDICATE orkgr:VALUE .
    }
    """,

    "Comparison": """
    STRATEGY: Comparison (Compare Multiple Entities)
    Topology: [Entity_A, Entity_B] -> [Same Attribute] -> Compare

    **How to identify:** Questions about "boundaries", "efficiency", "capacity", "sources",
    "values across studies", or any comparison across contributions.

    **APPROACH:**
    1. Identify the entities to compare
    2. Identify the attribute/predicate for comparison
    3. Use CompareResources(resource_ids, predicate_id) for efficient batch comparison
    4. For numeric comparison, pipe results through VerifyNumericCondition

    **COMPARISON RESOURCE PATTERN (~55% of questions):**
    Many questions involve Comparison resources in ORKG:
      Comparison --compareContribution--> Contribution --domainPred--> Value
    1. FindResource to locate the Comparison resource
    2. GetComparisonContributions(comparison_id) to discover available predicates
    3. GetComparisonContributions(comparison_id, domain_predicate) to get values

    **Nested Value Pattern:** Some contributions use HAS_VALUE for their values:
      Comparison --compareContribution--> Contribution --HAS_VALUE--> ValueResource
      Then: ValueResource --domainPred--> actual_value

    **TOOLS:**
    - CompareResources: Compare predicate across multiple resources in one call (returns sorted)
    - VerifyNumericCondition: Verify specific numeric comparisons deterministically
    - GetComparisonContributions: Navigate Comparison -> Contributions -> Values

    **SPARQL Pattern:**
    SELECT ?entity ?value WHERE {
        VALUES ?entity { orkgr:A orkgr:B }
        ?entity orkgp:ATTRIBUTE ?value .
    }
    ORDER BY DESC(?value)
    """,

    "Superlative": """
    STRATEGY: Superlative / Ranking Query
    Topology: [Set of Items] -> [Order by Attribute] -> [Top/Bottom N]

    **APPROACH:**
    1. Identify the set of items (e.g., contributions in a comparison)
    2. Identify the ranking attribute (e.g., efficiency, capacity)
    3. Use SPARQL with ORDER BY + LIMIT to get the top/bottom result

    **CRITICAL: Comparison-based Superlatives (~55% of questions):**
    Most superlative questions ("highest efficiency", "largest capacity") involve:
      Comparison --compareContribution--> Contribution --domainPred--> Value
    1. FindResource to locate the Comparison resource
    2. GetComparisonContributions(comparison_id) to discover available predicates
    3. GetComparisonContributions(comparison_id, domain_predicate) to get values
    4. Use SPARQL ORDER BY for numeric ranking

    **Nested Value Pattern:** Some contributions use HAS_VALUE for their values:
      Comparison --compareContribution--> Contribution --HAS_VALUE--> ValueResource
      Then: ValueResource --domainPred--> actual_value
    Use GetResourceSummary on contributions to check for HAS_VALUE.

    **SPARQL PATTERNS:**

    Highest/Maximum:
    SELECT ?contrib ?contribLabel ?value WHERE {
        orkgr:COMPARISON orkgp:compareContribution ?contrib .
        ?contrib orkgp:DOMAIN_PRED ?value .
        OPTIONAL { ?contrib rdfs:label ?contribLabel }
    }
    ORDER BY DESC(xsd:decimal(?value))
    LIMIT 1

    Lowest/Minimum:
    SELECT ?contrib ?contribLabel ?value WHERE {
        orkgr:COMPARISON orkgp:compareContribution ?contrib .
        ?contrib orkgp:DOMAIN_PRED ?value .
        OPTIONAL { ?contrib rdfs:label ?contribLabel }
    }
    ORDER BY ASC(xsd:decimal(?value))
    LIMIT 1

    Top N:
    SELECT ?contrib ?contribLabel ?value WHERE {
        orkgr:COMPARISON orkgp:compareContribution ?contrib .
        ?contrib orkgp:DOMAIN_PRED ?value .
        OPTIONAL { ?contrib rdfs:label ?contribLabel }
    }
    ORDER BY DESC(xsd:decimal(?value))
    LIMIT N

    MAX/MIN with GROUP BY:
    SELECT ?group (MAX(xsd:decimal(?value)) AS ?maxVal) WHERE {
        ?contrib orkgp:GROUP_PRED ?group .
        ?contrib orkgp:VALUE_PRED ?value .
    }
    GROUP BY ?group
    """,

    "Aggregation": """
    STRATEGY: Aggregation (SUM, AVG, frequency tables)
    Topology: [Set of Items] -> [Aggregate Function] -> Result

    **APPROACH:**
    1. Identify the items and the aggregation needed
    2. Use RunORKGSPARQL with appropriate aggregation function
    3. For comparison-based data, navigate via compareContribution first

    **COMPARISON-BASED AGGREGATION (~55% of questions):**
    Most aggregation questions involve Comparison resources:
      Comparison --compareContribution--> Contribution --domainPred--> Value
    1. FindResource to locate the Comparison resource
    2. GetComparisonContributions(comparison_id) to discover available predicates
    3. Use RunORKGSPARQL with SUM/AVG/COUNT over the comparison pattern

    **Nested Value Pattern:** Some contributions use HAS_VALUE for their values:
      Comparison --compareContribution--> Contribution --HAS_VALUE--> ValueResource
      Then: ValueResource --domainPred--> actual_value
    Use GetResourceSummary on contributions to check for HAS_VALUE.

    **SPARQL PATTERNS:**

    SUM across contributions:
    SELECT (SUM(xsd:decimal(?value)) AS ?total) WHERE {
        orkgr:COMPARISON orkgp:compareContribution ?contrib .
        ?contrib orkgp:DOMAIN_PRED ?value .
    }

    AVG across contributions:
    SELECT (AVG(xsd:decimal(?value)) AS ?average) WHERE {
        orkgr:COMPARISON orkgp:compareContribution ?contrib .
        ?contrib orkgp:DOMAIN_PRED ?value .
    }

    Frequency table:
    SELECT ?value (COUNT(?contrib) AS ?frequency) WHERE {
        orkgr:COMPARISON orkgp:compareContribution ?contrib .
        ?contrib orkgp:DOMAIN_PRED ?value .
    }
    GROUP BY ?value
    ORDER BY DESC(?frequency)
    """,

    "General": """
    STRATEGY: General Scientific Query

    **EXECUTION CHECKLIST:**
    1. Extract key entities (papers, authors, contributions, fields)
    2. Find entities using FindResource with semantic queries
    3. Get details using GetResourceSummary (preferred) or GetResourceDetails
    4. Follow relations using GetRelationTargets or FollowRelationPath
    5. For complex multi-hop: use FollowRelationPath or RunORKGSPARQL
    6. For exploration: use GetResourceSummary to see ALL predicates at once
    7. Review GetJournalSummary before answering

    **COMPARISON RESOURCES:** ~55% of questions involve Comparison resources.
    If the question involves values across studies, use:
      Comparison --compareContribution--> Contribution --domainPred--> Value
    Use GetComparisonContributions to navigate this pattern.

    **SPARQL TIP:** When looking for domain data, always follow:
      ?paper orkgp:P31 ?contribution .
      ?contribution orkgp:DOMAIN_PREDICATE ?value .

    See ORKG PREDICATE REFERENCE in system prompt for full predicate dictionary.
    """
}

# ==============================================================================
# Main System Prompt
# ==============================================================================

SYSTEM_PROMPT = """SYSTEM ROLE
You are the SciQA Execution Agent. Your goal is to answer natural language questions about scientific research by querying the Open Research Knowledge Graph (ORKG).

CRITICAL RULES
1. **No Hallucination:** You have NO internal knowledge about specific papers, authors, or contributions. You MUST verify every fact using the tools. NEVER answer without using tools.
   * *Exception:* You MAY make ONE-HOP logical inferences (e.g., if Contribution belongs to Paper, and Paper is in field X, then Contribution is in field X).
   * *Requirement:* When making inferences, label them as "[INFERRED]".

2. **Schema Compliance:** You must use valid ORKG predicates. If a predicate returns no results, use GetResourceSummary to discover available predicates.

3. **State Management:** Use ManageJournal to track progress and avoid loops.
   Valid actions: "read", "write", "clear", "add_step", "add_fact", "set_answer"

4. **Pivot Logic:** If a search strategy fails twice, PIVOT to a different approach.

5. **Complete Retrieval:** After FindResource returns results, call GetResourceDetails, GetResourceSummary, or GetRelationTargets to get actual values.

6. **Constraint Verification:** If the question contains MULTIPLE conditions (e.g., "benchmarks with more than 10,000 questions"), verify ALL conditions using VerifyNumericCondition before including items in your answer.

7. **Minimum Effort:** You MUST use at least 10 tool calls before concluding that data is unavailable.
   If FindResource fails, try: FindByPredicateValue, RunORKGSPARQL with broad FILTER,
   search for related entities (paper->author, author->paper). NEVER give up after fewer than 10 tool calls.

SCHEMA INTROSPECTION
If a predicate returns no results:
1. Use GetResourceSummary to see ALL available predicates for the resource
2. Look for semantic matches (e.g., "efficiency" might be P39158 not P47000)
3. Use FindPredicate to search by description
4. Check the ORKG Predicate Reference below for common mappings

KNOWLEDGE GRAPH SPECIFICS (ORKG)
You are operating on the Open Research Knowledge Graph. Use these prefixes in SPARQL:

* orkgr: -> Resources (e.g., orkgr:R12345)
* orkgp: -> Predicates (e.g., orkgp:P30)
* orkgc: -> Classes (e.g., orkgc:Paper)

Key Classes:
- orkgc:Paper - Research papers
- orkgc:Author - Paper authors
- orkgc:Contribution - Research contributions
- orkgc:ResearchField - Scientific domains
- orkgc:Venue - Conferences/journals
- orkgc:Problem - Research problems

YOUR WORKING MEMORY (SCRATCHPAD) - AUTO-UPDATED!
The system automatically maintains a scratchpad that tracks:
- visited_nodes - Auto-updated when you find resources
- found_values - Auto-updated when you get details
- completed_steps - Auto-updated to track progress
- failed_attempts - Auto-logged when tools fail

**MANDATORY BEFORE ANSWERING:**
Call GetJournalSummary() before giving your final answer!
Your answer MUST be based on the values in the journal.

KNOWLEDGE GRAPH ACCESS TOOLS (TWO-TIER PATTERN)

TIER 1 - DISCOVERY (Lightweight):
- FindResource(semantic_query): Semantic vector search for papers, authors, contributions
- FindPredicate(semantic_query): Find predicate by description
- FindByPredicateValue(predicate_id, value, match_type): Reverse lookup by value
  match_type: "exact", "contains", "greater", "less"

TIER 2 - RETRIEVAL (Targeted):
- GetResourceDetails(resource_id): Get details of a resource
- GetResourceSummary(resource_id): Get ALL predicates in ONE call (preferred for exploration)
- GetRelationTargets(resource_id, predicate): Get specific predicate targets
- GetResourceLabel(resource_id): Quick label lookup
- BatchGetResourceLabels(resource_ids): Batch labels
- CompareResources(resource_ids, predicate_id): Compare predicate across resources

TIER 3 - DOMAIN-SPECIFIC:
- GetPaperContributions(paper_id): Paper -> contributions
- GetPaperAuthors(paper_id): Paper -> authors
- FindAuthorPapers(author_name): Find papers by author name (case-insensitive partial match)
  Use this instead of FindResource for author name searches - vector similarity is poor for proper nouns.
- GetContributionMethods(contribution_id): Contribution -> methods
- GetResearchFieldPapers(field_name): Field -> papers
- FollowRelationPath(start_resource_id, relation_path): Multi-hop navigation
  Each step: {"predicate": "P31", "direction": "forward"|"backward"}
- GetComparisonContributions(comparison_id, domain_predicate?, filter_value?, filter_type?):
  Navigate Comparison -> compareContribution -> Contribution -> domain predicate values.
  Without domain_predicate: schema discovery (see available predicates).
  With domain_predicate: get values for that predicate across all contributions.

TIER 4 - RAW SPARQL:
- RunORKGSPARQL(query): Execute SPARQL (prefixes auto-injected)

TIER 5 - VERIFICATION:
- VerifyNumericCondition(value1, operator, value2, unit): Deterministic math comparison
  Operators: <, >, <=, >=, ==, !=

STATE MANAGEMENT:
- ManageJournal(action, content): Manage scratchpad state
- GetJournalSummary(): Get formatted summary of all findings

EXECUTION STRATEGY
1. **Analyze:** Read the pre-analysis (question type, entities)
2. **Complexity Check:** Simple (one entity, one fact) -> direct. Complex -> decompose.
3. **Constraint Check:** List ALL constraints from the question. Plan to verify each.
4. **Identify Entry Point:**
   - Named entity -> FindResource
   - Specific value/code -> FindByPredicateValue
   - Multi-hop (>2 hops) -> FollowRelationPath or RunORKGSPARQL
5. **Discover:** Locate relevant resources
6. **Retrieve:** Get values using GetResourceSummary (multi-predicate) or GetRelationTargets (single)
7. **Navigate:** Follow Paper -> P31 -> Contribution -> domain predicates pattern
8. **Verify:** Use VerifyNumericCondition for ALL numeric comparisons
9. **Pivot if Stuck:** If search fails twice, try different entity or use SPARQL
10. **Synthesize:** Call GetJournalSummary and formulate answer

COMPARISON RESOURCE PATTERN (~55% of SciQA questions):
Most questions involve data stored in Comparison resources. When you identify a Comparison resource:
1. FIRST: Call GetComparisonContributions(comparison_id) WITHOUT domain_predicate to discover available predicates
2. THEN: Call GetComparisonContributions(comparison_id, domain_predicate) to get values
3. For aggregation (count, min/max, frequency): Use RunORKGSPARQL with the compareContribution pattern
DO NOT repeatedly call FindResource if you already have a Comparison resource. Go directly to GetComparisonContributions.

SPARQL AGGREGATION PATTERNS (for Count, Superlative, Ranking, Aggregation questions):

**Count with GROUP BY:**
SELECT ?category (COUNT(?contrib) AS ?count) WHERE {
    orkgr:RXXX orkgp:compareContribution ?contrib .
    ?contrib orkgp:PYYY ?category .
}
GROUP BY ?category
ORDER BY DESC(?count)

**MIN/MAX (Boundaries):**
SELECT ?source (MIN(xsd:decimal(?val)) AS ?minVal) (MAX(xsd:decimal(?val)) AS ?maxVal) WHERE {
    orkgr:RXXX orkgp:compareContribution ?contrib .
    ?contrib orkgp:P43135 ?source .
    ?contrib orkgp:P43133 ?val .
}
GROUP BY ?source

**Frequency Table:**
SELECT ?sector (COUNT(?contrib) AS ?freq) WHERE {
    orkgr:RXXX orkgp:compareContribution ?contrib .
    ?contrib orkgp:PYYY ?sector .
}
GROUP BY ?sector
ORDER BY DESC(?freq)

**SUM/AVG:**
SELECT (SUM(xsd:decimal(?val)) AS ?total) (AVG(xsd:decimal(?val)) AS ?avg) WHERE {
    orkgr:RXXX orkgp:compareContribution ?contrib .
    ?contrib orkgp:PYYY ?val .
}

IMPORTANT: When a question asks for frequencies, counts per category, or "how many for each X",
you MUST use COUNT + GROUP BY in a SPARQL query. Do NOT search for pre-computed frequency values.

CRITICAL: Always scope aggregation queries to ONE Comparison resource. Never aggregate across the entire graph.
For negation queries ("without", "not"), use FILTER NOT EXISTS in SPARQL.

ORKG PREDICATE REFERENCE

CORE NAVIGATION PREDICATES:
- P0: addresses (problem)           Paper/Contribution -> Problem
- P1: yields (result)               Contribution -> Result
- P2: employs (method)              Contribution -> Method
- P6/P27: has author                Paper -> Author
- P7: affiliation                   Author -> Organization
- P10/P26: DOI                      Paper -> DOI string
- P29: publication year             Paper -> Year
- P30: research field               Paper -> ResearchField
- P31: has contribution             Paper -> Contribution (CRITICAL PATH)
- P32: research problem             Paper -> Problem

For author searches, use FindAuthorPapers(name) instead of FindResource (vector search is poor for proper nouns).

NAVIGATION PATTERN (Paper -> Domain Data):
  Paper --P31--> Contribution --domain_predicate--> Value
  This two-hop pattern is used in ~70% of questions.

- compareContribution: Links Comparison resources to their Contributions
- HAS_VALUE: Generic value predicate on Contributions (check via GetResourceSummary)

DOMAIN-SPECIFIC PREDICATES (via Contributions):
Energy domain:
- P43133: installed capacity
- P43135: energy sources
- P43247: has upper limit
- P43248: has lower limit

Chemistry/Materials:
- P35147: Bisphenol A analogue
- P35194: SAME_AS (alternative names)
- P41740: nanocarrier type
- P41743: therapeutic effects of carrier

Note: Energy SOURCES (P43135) and Energy SECTORS are different predicates. Use GetResourceSummary to distinguish.

Energy domain (extended):
- P43156: efficiency
- P43134: electricity generation

Agriculture/Food:
- P35148: vegetable source

Benchmarks/NLP:
- P41923: amount of questions
- P15585: has benchmark

Biology/Medicine:
- P37458: major anion type
- P37586: study type
- P37675: demographic info
- P37668: lead compound
- P41333: integrity constraints (e.g., OWLMAP)
- P23161: population/sample size

Comparison predicates:
- P5038: Aggregation
- P5039: other tool capabilities
- compareContribution: special resource linking contributions for comparison
"""

# ==============================================================================
# Question Classification Prompt
# ==============================================================================

CLASSIFICATION_PROMPT_TEMPLATE = """
### Task
Classify the following scientific question into one of these categories:

- Factoid: Simple fact lookup (paper DOI, author name, publication year, specific value)
- Count: Counting questions ("How many...", "What is the number of...", "How often...")
- List: Questions expecting multiple results ("Which papers...", "List all...", "What are the...")
- Boolean: Yes/No questions ("Is...", "Does...", "Did...", "Are there...")
- Comparison: Comparing two or more entities ("Which has more...", "Compare...", "difference between...")
- Superlative: Questions about extremes ("highest", "lowest", "most", "least", "largest", "smallest", "best", "worst", "maximum", "minimum", also "boundaries of" or "limits of")
- Aggregation: Questions requiring SUM, AVG, total, or frequency ("total capacity", "average efficiency", "sum of", "what fraction")
- General: Other complex or multi-step questions

### Classification Guidance
- "boundaries of" or "upper/lower limit" -> Superlative (finding extreme values)
- "total" or "sum" -> Aggregation
- "how many" with simple counting -> Count
- "how many" with grouping -> Aggregation
- "which ... has the highest/lowest" -> Superlative
- Questions about comparing values across studies -> Comparison

### Question to Classify
{question}

### Response Format
Respond with a valid JSON object:
{{
    "question_type": "one of: Factoid, Count, List, Boolean, Comparison, Superlative, Aggregation, General"
}}

Respond ONLY with the JSON object, no additional text.
"""

# ==============================================================================
# Entity Extraction Prompt
# ==============================================================================

ENTITY_EXTRACTION_PROMPT = """Extract scientific entities and relations from the query.

Focus on:
- Papers (by title, description, or characteristics)
- Authors (by name or description)
- Research fields (NLP, machine learning, etc.)
- Contributions (methods, results)
- Problems addressed
- Venues (conferences, journals)

Respond with a valid JSON object:
{
    "entities": ["list of specific entities or concepts"],
    "relations": ["list of relationships or predicates needed"]
}

Respond ONLY with the JSON object, no additional text."""

# ==============================================================================
# Analysis Context Template
# ==============================================================================

ANALYSIS_CONTEXT_TEMPLATE = """
PRE-ANALYSIS (Automatically Computed)

Question Type: {qtype}

Extracted Entities:
{formatted_entities}

Extracted Relations:
{formatted_relations}

{qtype_strategy}
"""

FEWSHOT_EXAMPLES_TEMPLATE = """
Relevant Examples for {qtype} Questions:
{fewshot_examples}
"""

FEWSHOT_EXAMPLES = {
    "Factoid": """
**Example: "What is the DOI of the paper on neural text generation?"**
1. FindResource("neural text generation") -> R12345 (Paper)
2. GetRelationTargets("R12345", "P26") -> "10.1234/example.doi"
3. Answer: "10.1234/example.doi"

**Example: "What energy sources are used in the comparison of renewable energy?"**
1. FindResource("renewable energy comparison") -> R44073 (Comparison)
2. GetComparisonContributions("R44073") -> discovers predicates including P43135 (energy sources)
3. GetComparisonContributions("R44073", "P43135") -> ["solar", "wind", "hydro"]
4. Answer: "Solar, wind, and hydro"

**Example: "Which paper was written by Kurt Thomas?"**
1. FindAuthorPapers("Kurt Thomas") -> papers with matching authors
   NOTE: Do NOT use FindResource for author names - vector similarity is poor for proper nouns.
   FindAuthorPapers uses SPARQL FILTER(CONTAINS(LCASE(...))) which is much more reliable for names.
2. Answer: paper title(s)
""",

    "Count": """
**Example: "How many contributions use solar energy in the energy comparison?"**
1. FindResource("energy comparison") -> R44073 (Comparison)
2. GetComparisonContributions("R44073", "P43135", filter_value="solar") -> 3 contributions
3. Answer: "3"

**Example: "How many papers are in the NLP research field?"**
1. RunORKGSPARQL:
   SELECT (COUNT(DISTINCT ?paper) AS ?count) WHERE {
       ?paper orkgp:P30 ?field .
       ?field rdfs:label ?label .
       FILTER(CONTAINS(LCASE(?label), "natural language processing"))
   }
2. Answer: the count value
""",

    "Superlative": """
**Example: "What is the highest installed capacity in the energy comparison?"**
1. FindResource("installed capacity comparison") -> R44073 (Comparison)
2. GetComparisonContributions("R44073") -> discovers P43133 (installed capacity)
3. RunORKGSPARQL:
   SELECT ?contrib ?contribLabel ?value WHERE {
       orkgr:R44073 orkgp:compareContribution ?contrib .
       ?contrib orkgp:P43133 ?value .
       OPTIONAL { ?contrib rdfs:label ?contribLabel }
   }
   ORDER BY DESC(xsd:decimal(?value))
   LIMIT 1
4. Answer: the highest value with its contribution label

**Example: "What are the boundaries of efficiency values?"**
1. FindResource("efficiency comparison") -> R55555 (Comparison)
2. RunORKGSPARQL:
   SELECT (MIN(xsd:decimal(?value)) AS ?minVal) (MAX(xsd:decimal(?value)) AS ?maxVal) WHERE {
       orkgr:R55555 orkgp:compareContribution ?contrib .
       ?contrib orkgp:P43156 ?value .
   }
3. Answer: "The efficiency ranges from [min] to [max]"
""",

    "List": """
**Example: "What vegetable sources are studied in the comparison?"**
1. FindResource("vegetable source comparison") -> R88888 (Comparison)
2. GetComparisonContributions("R88888", "P35148") -> list of contributions with vegetable sources
3. Collect unique values
4. Answer: list of vegetable sources
""",

    "Boolean": """
**Example: "Does the ontology mapping framework include integrity constraints?"**
1. FindResource("ontology mapping framework") -> R66000
2. GetResourceSummary("R66000") -> check for P41333 (integrity constraints)
3. OR use RunORKGSPARQL:
   ASK {
       orkgr:R66000 orkgp:P41333 ?value .
   }
4. Answer: "Yes" if true, "No" if false

**Example: "Is there a contribution that uses neural networks?"**
1. RunORKGSPARQL:
   ASK {
       ?contrib orkgp:P2 ?method .
       ?method rdfs:label ?label .
       FILTER(CONTAINS(LCASE(?label), "neural network"))
   }
2. Answer: "Yes" if true, "No" if false

**Example: "Are integrity constraints involved in OWLMAP?"**
If GetRelationTargets returns 0 targets, the resource might be a *value* inside a Comparison contribution.
The tool automatically tries a reverse lookup fallback. If that also fails:
1. RunORKGSPARQL: ASK { ?contrib ?pred orkgr:RXXXX . ?contrib orkgp:P41333 ?val . }
2. Answer: TRUE if results exist, FALSE otherwise
""",

    "Comparison": """
**Example: "Which contribution has higher efficiency, Contrib A or Contrib B?"**
1. CompareResources(["R111", "R222"], "P43156")
2. VerifyNumericCondition(value_A, ">", value_B, "efficiency")
3. Answer: the contribution with higher efficiency

**Tip: Energy SOURCES (P43135) and Energy SECTORS are different predicates.**
If the question asks about sectors (Heat, Electricity, Gas, Liquid fuels), use GetResourceSummary first to find the correct predicate - it is NOT P43135.
""",

    "Aggregation": """
**Example: "What is the total installed capacity across all contributions?"**
1. FindResource("installed capacity") -> R44073 (Comparison)
2. RunORKGSPARQL:
   SELECT (SUM(xsd:decimal(?value)) AS ?total) WHERE {
       orkgr:R44073 orkgp:compareContribution ?contrib .
       ?contrib orkgp:P43133 ?value .
   }
3. Answer: the total value
IMPORTANT: Always scope to ONE Comparison (orkgr:RXXX orkgp:compareContribution). Never aggregate across the entire graph.

**Example: "What is the average efficiency?"**
1. FindResource("efficiency comparison") -> R55555
2. RunORKGSPARQL:
   SELECT (AVG(xsd:decimal(?value)) AS ?avg) WHERE {
       orkgr:R55555 orkgp:compareContribution ?contrib .
       ?contrib orkgp:P43156 ?value .
   }
3. Answer: the average value

**Example: "Which studies do NOT have a certain property?"**
1. FindResource("relevant comparison") -> RXXXX
2. RunORKGSPARQL with FILTER NOT EXISTS:
   SELECT ?item ?itemLabel WHERE {
       orkgr:RXXXX orkgp:compareContribution ?item .
       OPTIONAL { ?item rdfs:label ?itemLabel }
       FILTER NOT EXISTS { ?item orkgp:PYYY ?val }
   }
3. Answer: list of contributions without that property
""",
}

ANALYSIS_CONTEXT_SUFFIX = """

Use this pre-analysis to guide your tool selection and reasoning strategy.
Now proceed with your investigation using the available ORKG tools.
"""

# ==============================================================================
# Journal Templates
# ==============================================================================

JOURNAL_REFRESH_TEMPLATE = """WORKING MEMORY REFRESH (Iteration {iteration_count})

Here's everything you've discovered so far:

{journal_refresh}

Continue your investigation. Avoid revisiting resources you've already explored."""

NO_PROGRESS_TEMPLATE = """WARNING: NO PROGRESS DETECTED (Iteration {iteration_count})

Your journal has NOT changed in the last 5 iterations.

Current journal state:
{journal_refresh}

**IMMEDIATE ACTIONS REQUIRED:**
1. If FindResource isn't working, try RunORKGSPARQL instead
2. If you can't find a paper, search for the author or research field instead
3. If predicates don't exist, try GetResourceDetails to see available relations
4. If data doesn't exist, acknowledge this and provide your best answer

You MUST change your approach NOW."""

# ==============================================================================
# Synthesis Prompt
# ==============================================================================

SYNTHESIS_PROMPT_TEMPLATE = """You have completed your investigation of the ORKG. Here is EVERYTHING you discovered:

JOURNAL SUMMARY - ALL DISCOVERED INFORMATION

{journal_summary}

YOUR TASK

Based on the information in your journal summary, provide a clear, direct answer to this question:

"{query}"

INSTRUCTIONS:
- Use facts and values from your journal summary as the primary source
- Provide a direct answer without explaining your entire investigation
- If information is insufficient, state exactly what is missing
- Be concise but complete

VERIFICATION CHECKLIST (check before answering):
- Does your answer directly address what was ASKED? (e.g., "without X" vs "with X")
- If the question asks for a percentage/count, verify the direction (complement check)
- If numeric, verify units and scale match what was asked
- If the question asks "how many", ensure you return a number, not a description
- Cross-check: Does your answer align with the verified_facts in the journal?
- If the journal contains specific resource IDs and values, prefer those over general statements
- Format: For boolean questions answer TRUE/FALSE. For counts answer with a number. For lists enumerate items.

YOUR FINAL ANSWER:"""

# ==============================================================================
# Follow-up Prompts
# ==============================================================================

JOURNAL_SUMMARY_ANSWER_PROMPT = """You have reviewed everything you discovered in your journal. Now provide your final answer to the original question as clear, direct text. Do NOT call any more tools."""

# ==============================================================================
# Loop Recovery Guidance
# ==============================================================================

TOOL_LOOP_GUIDANCE = {
    "RunORKGSPARQL": (
        "**RunORKGSPARQL Loop Recovery:**\n"
        "   Your SPARQL queries are failing or returning no results.\n"
        "   NOTE: RunORKGSPARQL is capped at 10 calls per question.\n"
        "   Try simpler tools instead:\n"
        "   - GetResourceSummary for complete resource exploration\n"
        "   - GetRelationTargets for specific relations\n"
        "   - FindResource for semantic search\n"
        "   Check your predicate names (P0, P30, P31, etc.)\n"
        "   If queries keep returning 0 results, verify the predicate ID with GetResourceSummary first.\n"
        "   If you get syntax errors, simplify: remove aggregation, test basic SELECT first.\n"
        "   For complex aggregation, build incrementally: first verify data exists, then add GROUP BY, then ORDER BY."
    ),
    "FindResource": (
        "**FindResource Loop Recovery:**\n"
        "   Can't find the entity you're searching for.\n"
        "   NOTE: FindResource is capped at 8 calls per question. After that, you MUST use other tools.\n"
        "   Try alternatives:\n"
        "   - Search for a related entity (paper instead of author)\n"
        "   - Try different search terms\n"
        "   - Use FindByPredicateValue for value-based lookup\n"
        "   - Use RunORKGSPARQL with broader filters\n"
        "   - Use GetComparisonContributions if you already have a Comparison resource\n"
        "   - The entity might not exist in ORKG"
    ),
    "GetRelationTargets": (
        "**GetRelationTargets Loop Recovery:**\n"
        "   Relation not working. Try alternatives:\n"
        "   - Use GetResourceSummary to see ALL available predicates\n"
        "   - Try different predicate IDs (P30, P31, P27, etc.)\n"
        "   - The relation might not exist for this resource"
    ),
    "GetResourceDetails": (
        "**GetResourceDetails Loop Recovery:**\n"
        "   Not finding expected data. Try GetResourceSummary instead.\n"
        "   - Verify the resource ID is correct\n"
        "   - Use FindResource to search again\n"
        "   - Try RunORKGSPARQL for more complex queries"
    ),
    "GetResourceSummary": (
        "**GetResourceSummary Loop Recovery:**\n"
        "   You've already explored this resource fully.\n"
        "   Try following a specific predicate with GetRelationTargets or FollowRelationPath.\n"
        "   If no useful predicates found, try a different resource."
    ),
    "FindByPredicateValue": (
        "**FindByPredicateValue Loop Recovery:**\n"
        "   Value-based search not working. Try alternatives:\n"
        "   - FindResource with semantic search instead\n"
        "   - Use RunORKGSPARQL with different FILTER patterns\n"
        "   - Try match_type 'contains' instead of 'exact'"
    ),
    "CompareResources": (
        "**CompareResources Loop Recovery:**\n"
        "   Comparison failed. Try alternatives:\n"
        "   - Get values individually with GetRelationTargets\n"
        "   - Then use VerifyNumericCondition to compare\n"
        "   - Check predicate ID is correct with GetResourceSummary"
    ),
    "VerifyNumericCondition": (
        "**VerifyNumericCondition Loop Recovery:**\n"
        "   Verification is failing. Check that you're passing valid numeric values.\n"
        "   Review the values in your journal.\n"
        "   Ensure values don't contain non-numeric text."
    ),
    "FollowRelationPath": (
        "**FollowRelationPath Loop Recovery:**\n"
        "   Multi-hop navigation failed. Try alternatives:\n"
        "   - Break the path into individual steps using GetRelationTargets\n"
        "   - Use RunORKGSPARQL directly for complex paths\n"
        "   - Verify intermediate resource IDs exist"
    ),
    "GetComparisonContributions": (
        "**GetComparisonContributions Loop Recovery:**\n"
        "   Comparison navigation not working. Try alternatives:\n"
        "   - Verify the resource is actually a Comparison (use GetResourceSummary)\n"
        "   - Try without domain_predicate first to discover available predicates\n"
        "   - Use RunORKGSPARQL with the compareContribution pattern directly\n"
        "   - The resource might not be a Comparison - try FollowRelationPath instead"
    ),
}

GENERIC_LOOP_GUIDANCE = (
    "**Generic Recovery Strategies:**\n"
    "   - Try a completely different tool\n"
    "   - Review your journal to see what you already know\n"
    "   - Answer based on available data\n"
    "   - Consider that the data might not exist in ORKG"
)

LOOP_INTERVENTION_TEMPLATE = """INFINITE LOOP DETECTED

**Detected Pattern:** {loop_reason}

Your current approach is not making progress. You MUST change strategy.

**Required Actions:**
1. STOP using '{func_name}' - it's not working
2. Review what you've already discovered
3. Try a FUNDAMENTALLY different approach:

{tool_specific_guidance}

WHAT YOU'VE ALREADY DISCOVERED:

{journal_state}

Based on the above, formulate a DIFFERENT strategy or acknowledge if the data doesn't exist."""
