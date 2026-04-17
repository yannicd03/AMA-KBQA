"""
Prompts for the KQAPro Agent.

This module contains all the prompts used by the KQAPro agent, extracted
to keep the main agent.py file more manageable.
"""

# Question Type Specific Reasoning Strategies
# These are injected during the pre-agent hook based on QtypePrediction results
QTYPE_STRATEGIES = {
    "Count": """STRATEGY: Count → integer answer
Small set (<20): GetRelationDetails + count. Large/unknown set: RunSPARQL with COUNT(DISTINCT).
Filtered count: FilterEntities(concept=..., attribute_name=...) then count results, or RunSPARQL with COUNT.
Multi-hop: RunSPARQL with JOIN+COUNT. OR conditions: UNION + COUNT(DISTINCT).
Pattern: SELECT (COUNT(DISTINCT ?t) AS ?c) WHERE { ex:ID prop:PRED ?t . }
Trust verified counts. Don't downgrade after verification.""",

    "QueryAttr": """STRATEGY: QueryAttr → attribute lookup
Standard: FindNode → GetAttributeDetails (literals) or GetRelationDetails (linked entities).
Reverse: unique ID/code/URL → FindByAttribute (faster, exact).
If attribute fails: check available_attributes from FindNode for semantic matches.
"X in Y": find Y first, then X related to Y. Return X's attribute.
Fallback: SELECT ?v WHERE { ex:ID prop:PRED ?v . }""",

    "QueryAttrQualifier": """STRATEGY: QueryAttrQualifier → contextual fact (time/place on a fact)
Key distinction: "movie's language"=NodeAttr vs "language of website dated 1998-04-09"=EdgeQualifier.
Steps: 1) Find base fact via GetAttributeDetails/GetRelationDetails
2) GetEdgeQualifiers(subject_id, predicate, target_value)
3) Match question word to qualifier: When→point_in_time, Where→location
Filtering by qualifier: Use QualifierFilter(entity_ids, predicate, qualifier_name, value) to narrow entities by qualifier conditions.
Pattern: SELECT ?qv WHERE { ?f pred:fact_h ex:ID; pred:fact_r prop:P; pred:fact_t "VAL". ?f qual:Q ?qv. }""",

    "QueryName": """STRATEGY: QueryName → identify entity from description
Unique ID/code → FindByAttribute immediately.
Single condition: FindNode + GetRelationDetails.
Type+attribute conditions: FilterEntities(concept=..., attribute_name=..., attribute_value=...) for combined filtering.
Multiple conditions: RunSPARQL with multiple WHERE clauses (avoid manual intersection).
Pattern: SELECT ?label WHERE { ?s prop:P1 ex:O1. ?s prop:P2 ex:O2. ?s rdfs:label ?label. }""",

    "QueryRelation": """STRATEGY: QueryRelation → find predicate between two entities
GetRelationDetails on A, check if B appears. If A→B fails, try B→A (bidirectional).
Fallback: SELECT DISTINCT ?p ?label WHERE { { ex:A ?p ex:B } UNION { ex:B ?p ex:A } ?p rdfs:label ?label. }""",

    "QueryRelationQualifier": """STRATEGY: QueryRelationQualifier → context of a relation
1) Confirm connection via GetRelationDetails. 2) GetEdgeQualifiers on that connection.
3) Match qualifier: When→point_in_time, Where→location, Role→object_has_role, Ceremony→ceremony.
Filtering by qualifier: Use QualifierFilter(entity_ids, relation, qualifier_name, value) to narrow entities by qualifier conditions.
Pattern: SELECT ?v WHERE { ?f pred:fact_h ex:A; pred:fact_r prop:P; pred:fact_t ex:B. ?f qual:Q ?v. }""",

    "SelectAmong": """STRATEGY: SelectAmong → superlative from group
Small explicit list (<20): CompareEntities. Large/open group: RunSPARQL with ORDER BY + LIMIT 1.
DO NOT fetch all items with GetRelationDetails (timeout risk).
Pattern: SELECT ?label ?v WHERE { ?i prop:instance_of ex:GRP. ?i attr:ATTR ?v. ?i rdfs:label ?label. } ORDER BY DESC(?v) LIMIT 1""",

    "SelectBetween": """STRATEGY: SelectBetween → compare exactly 2 entities
1) Extract ALL constraints. 2) Verify constraints with GetAttributeDetails first.
3) CompareEntities([id_A, id_B], attribute) for comparison.
Pattern: SELECT ?label ?v WHERE { VALUES ?i { ex:A ex:B } ?i attr:ATTR ?v. ?i rdfs:label ?label. } ORDER BY DESC(?v) LIMIT 1""",

    "Verify": """STRATEGY: Verify → True/False
GetAttributeDetails to get value, then VerifyNumericCondition for numeric/date comparison, VerifyString for text comparison. Never do mental math or guess string equality.
Fallback: ASK { ex:ID attr:ATTR ?v. FILTER(?v > "VAL"^^xsd:decimal) }""",

    "Query": """STRATEGY: General Query
1) Extract all constraints. 2) Locate: ID→FindByAttribute, Name→FindNode.
3) Verify constraints with GetAttributeDetails. 4) Single-hop→GetAttributeDetails, multi-hop→RunSPARQL.
5) Time/place context→GetEdgeQualifiers. 6) One-hop inference OK if labeled [INFERRED].
7) GetJournalSummary before answering."""
}

# Main system prompt for the KQAPro agent - COMPRESSED for token efficiency
SYSTEM_PROMPT = """You are the KQAPro Execution Agent. Answer questions by querying a Knowledge Graph (KG).

RULES:
1. NO HALLUCINATION: Verify every fact with tools. One-hop inferences allowed if labeled "[INFERRED]".
2. SCHEMA COMPLIANCE: Use predicates returned by tools. If attribute fails, check available_attributes from FindNode.
3. PIVOT ON FAILURE: If search fails twice, try a connected entity or RunSPARQL with JOIN.
4. COMPLETE RETRIEVAL: After FindNode, always call GetAttributeDetails/GetRelationDetails for actual values.
5. VERIFY ALL CONSTRAINTS: Check ALL identifying details (duration, year, color) before answering.
6. TRUST VERIFIED DATA: Once in found_values/verified_facts, treat as ground truth. Don't second-guess.

KG PREFIXES (auto-injected in SPARQL, don't redefine):
ex:=Entities, prop:=Properties, attr:=Attributes, qual:=Qualifiers, unit:=Units

TOOL TIERS:
T1 Discovery: FindNode (semantic search) | FindByAttribute (exact ID/code/URL lookup - prefer this for unique IDs)
T1.5 Filtering: FilterEntities (by concept type and/or attribute value - replaces manual SPARQL filters) | QualifierFilter (filter entities by qualifier on statements)
T2 Retrieval: GetAttributeDetails | GetRelationDetails | GetNodeSummary (all data in ONE call)
T3 Qualifiers: GetEdgeQualifiers (facts about facts - use when question has time/place context)
T4 Verify: VerifyNumericCondition (never do mental math) | VerifyString (never guess string equality)
Complex: RunSPARQL (for multi-hop >2, COUNT, UNION) | CompareEntities | FindEntitiesByRelationPath

QUALIFIER DECISION: Question specifies TIME/PLACE for a fact? → GetEdgeQualifiers. General property? → GetAttributeDetails.
PREPOSITIONAL: "X in Y" → find Y first, then find X related to Y. Return X's attribute, not Y's.
NESTED QUESTIONS: Work inside-out. Resolve innermost clause first, then apply outer constraints.

BEFORE ANSWERING: Call GetJournalSummary() - your answer MUST be based on journal values only."""

# Combined classification + entity extraction prompt (single LLM call)
CLASSIFICATION_AND_EXTRACTION_PROMPT = """Classify the question and extract entities/relations.

Question types:
- Count: "How many X?" → integer answer
- Verify: "Is/Did/Was X?" → yes/no
- SelectBetween: exactly 2 entities + comparative ("more","older")
- SelectAmong: group + superlative ("most","biggest")
- QueryAttr: 1 entity + attribute lookup (no time/place qualifier)
- QueryAttrQualifier: QueryAttr + time/place context ("in 2020")
- QueryRelation: 2 entities → what relation connects them
- QueryRelationQualifier: known relation → when/where/how it occurred
- QueryName: description → which entity matches

Decision order:
1. Counting? → Count (unless time/place → QueryAttrQualifier)
2. Yes/No? → Verify
3. Two entities + comparative? → SelectBetween
4. Group + superlative? → SelectAmong
5. Two entities + relation? → QueryRelation (with context → QueryRelationQualifier)
6. One entity + value? → QueryAttr (with time/place → QueryAttrQualifier)
7. Description → identity? → QueryName

Question: {question}

Respond with JSON only:
{{
    "question_type": "Count|Verify|SelectBetween|SelectAmong|QueryAttr|QueryAttrQualifier|QueryRelation|QueryRelationQualifier|QueryName",
    "entities": ["specific entities or concepts mentioned"],
    "relations": ["relationship predicates or actions mentioned"]
}}"""

# Keep legacy prompts for backward compatibility but mark as deprecated
CLASSIFICATION_PROMPT_TEMPLATE = CLASSIFICATION_AND_EXTRACTION_PROMPT
ENTITY_EXTRACTION_PROMPT = """Extract entities and relations from the query.
Respond with JSON: {{"entities": [], "relations": []}}"""

# Analysis context template - COMPACT
ANALYSIS_CONTEXT_TEMPLATE = """PRE-ANALYSIS: Type={qtype} | Entities: {formatted_entities} | Relations: {formatted_relations}
{qtype_strategy}
Proceed with investigation."""

# Fewshot examples suffix template
FEWSHOT_EXAMPLES_TEMPLATE = """Examples for {qtype}:
{fewshot_examples}"""

# Analysis context suffix (no longer needed - integrated into template)
ANALYSIS_CONTEXT_SUFFIX = ""

# Journal refresh template - COMPACT
JOURNAL_REFRESH_TEMPLATE = """MEMORY (iter {iteration_count}): {journal_refresh}
Continue. Don't revisit completed work."""

# No progress detected template - COMPACT
NO_PROGRESS_TEMPLATE = """NO PROGRESS (iter {iteration_count}). Journal unchanged for 5 iterations.
{journal_refresh}
REQUIRED: 1) Try RunSPARQL 2) Search connected entity 3) Check available_attributes 4) Or answer with current data."""

# Synthesis prompt template (for final answer generation) - COMPACT
SYNTHESIS_PROMPT_TEMPLATE = """DISCOVERED DATA:
{journal_summary}

QUESTION: "{query}"

Answer directly using the discovered data above. Be concise. If data is missing, say what's missing."""

# GetJournalSummary follow-up prompt
JOURNAL_SUMMARY_ANSWER_PROMPT = "You have reviewed everything you discovered in your journal. Now you MUST provide your final answer to the original question as clear, direct text. Do NOT call any more tools."

# Tool-specific loop recovery guidance - COMPACT
TOOL_LOOP_GUIDANCE = {
    "RunSPARQL": "SPARQL failing. Use GetAttributeDetails/GetRelationDetails/FindNode instead. Check prefixes.",
    "GetAttributeDetails": "Attribute not found. Check available_attributes from FindNode or use GetNodeSummary.",
    "GetRelationDetails": "Relation not found. Check available_predicates from FindNode or use GetNodeSummary.",
    "FindNode": "Entity not found. Try related entity, synonyms, or FindByAttribute with ID/code.",
    "FindByAttribute": "Value not found. Try FindNode semantic search or RunSPARQL with broader filter.",
    "ExploreNeighborhood": "DEPRECATED. Use GetNodeSummary instead.",
    "GetAttributeWithQualifiers": "No qualifiers. Try GetAttributeDetails or TemporalAttributeQuery.",
    "TemporalAttributeQuery": "Date not found. Increase tolerance_days or use GetAttributeWithQualifiers.",
}

# Generic loop recovery guidance
GENERIC_LOOP_GUIDANCE = "Try a different tool. Review journal. Answer with available data. Data may not exist."

# Loop detected intervention template - COMPACT
LOOP_INTERVENTION_TEMPLATE = """LOOP DETECTED: {loop_reason}. STOP using '{func_name}'.
{tool_specific_guidance}
Current data: {journal_state}
Change strategy or answer with available data."""

# General guidance template (for cross-type insights from _general.json)
GENERAL_GUIDANCE_TEMPLATE = """
General Guidance (learned from previous runs):
{general_guidance}
"""

# Tool tips template (for tool-specific tips from _tool_tips.json)
TOOL_TIPS_TEMPLATE = """
Tool Tips:
{tool_tips}
"""
