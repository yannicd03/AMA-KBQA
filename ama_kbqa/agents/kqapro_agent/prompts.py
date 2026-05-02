"""
Prompts for the KQAPro Agent.

This module contains all the prompts used by the KQAPro agent, extracted
to keep the main agent.py file more manageable.
"""

# Question Type Specific Reasoning Strategies
# These are injected during the pre-agent hook based on QtypePrediction results
QTYPE_STRATEGIES = {
    "Count": """STRATEGY: Count → integer answer
USE CountEntities. It is the only tool you should reach for. It returns the EXACT count via SPARQL COUNT(DISTINCT) with NO truncation.

  - "How many cities in Germany?" → first FindNode("Germany") → Q-id, then CountEntities(concept="city", entity_ids=...) — or use a relation pattern via FilterEntities + entity_ids.
  - "How many counties pop > 7800 OR < 40M?" → CountEntities(concept="county of Pennsylvania", attribute_name="population", attribute_value="7800", operator=">", or_conditions=[{"attribute_name":"population","attribute_value":"40000000","operator":"<"}])
  - "How many mammals?" → CountEntities(concept="mammal", transitive_concept=True)
  - "How many films directed by Nolan?" → GetRelationDetails(Q25191, director, inverse) → entity_ids; then CountEntities(entity_ids=...).

FORBIDDEN: FilterEntities + len(matches) [caps at limit=50, returns wrong number].
FORBIDDEN as default: hand-written RunSPARQL COUNT [error-prone with KQAPro's namespaces]. Reach for it ONLY for genuinely 3+ hop joins where CountEntities can't express the join.

Trust the integer CountEntities returns. Don't downgrade after verification.""",

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
THE ANSWER IS A QUALIFIER VALUE, NOT A NEW ENTITY. The relation itself is already known;
the question asks *when/where/at-what-event/in-what-role* it held. Do not collapse to the
person, film, or award — return the qualifier (a ceremony, date, place, or role label).

1) Confirm connection via GetRelationDetails.
2) GetEdgeQualifiers on that connection.
3) Match qualifier:
     When → point_in_time (date) OR ceremony/edition (event) — pick event if the
        question frames it as an occasion ("at which ceremony", "during which").
     Where → location
     Role → object_has_role
     Ceremony → ceremony
4) If multiple qualifiers, pick the one whose type matches the question's wh-word.

COMMON TRAP: "Who was the winning individual WHEN [film] was the recipient for [award]?"
    → Even though it says "who", the wh-scope is the ceremony edition, not the editor.
    Answer with the ceremony qualifier (e.g. "19th Academy Awards"), not the person.
    Tell-tale: the subject+relation+object triple is fully specified in the question,
    so the only unknown left is a qualifier.

Filtering by qualifier: Use QualifierFilter(entity_ids, relation, qualifier_name, value) to narrow entities by qualifier conditions.
Pattern: SELECT ?v WHERE { ?f pred:fact_h ex:A; pred:fact_r prop:P; pred:fact_t ex:B. ?f qual:Q ?v. }""",

    "SelectAmong": """STRATEGY: SelectAmong → superlative from group
USE SelectExtreme. It runs SPARQL ORDER BY on the server and returns the actual winner — do NOT make the LLM compare values from CompareEntities by hand.

  - "Smallest former French region with pop != 97000?" → SelectExtreme(concept="former French region", attribute_name="population", mode="min", filter_attribute_name="population", filter_attribute_value="97000", filter_operator="!=")
  - "Top 3 most-populous cities?" → SelectExtreme(concept="city", attribute_name="population", mode="max", k=3)
  - "Longest film?" → SelectExtreme(concept="film", attribute_name="duration", mode="max")

FORBIDDEN as default: GetRelationDetails-fetch-everything-then-compare [timeout risk on large concepts].
FORBIDDEN as default: RunSPARQL with hand-written ORDER BY [you'll get the namespaces or the bnode-unwrap wrong].

EMPTY RESULT FALLBACK (do NOT give up):
 1. Verify attribute name with FindNode on one example instance → inspect available_attributes.
 2. Try transitive_concept=True (the concept may need its subclasses, e.g. feature_film under film).
 3. Drop or relax the filter_attribute_* and re-run.
 4. Last resort: return best candidate from partial data with [INFERRED]. NEVER "could not be identified".""",

    "SelectBetween": """STRATEGY: SelectBetween → compare exactly 2 entities
USE SelectExtreme(entity_ids=[id_A, id_B], attribute_name=..., mode="max"|"min"). It returns the winner directly via SPARQL ORDER BY.

For dates: "Who is older?" → mode="min" on date_of_birth (earlier date = older).
For dates: "More recent?" → mode="max".

1) FindNode each entity to get its Q-id. 2) Verify each entity's constraints with GetAttributeDetails (films may share titles). 3) SelectExtreme over [id_A, id_B].

CompareEntities is fine for diagnostic display ("show me both values") but SelectExtreme is the answer-producing call -- the LLM should not pick the winner by reading numbers from a table.""",

    "Verify": """STRATEGY: Verify → yes/no answer

🔴 OUTPUT FORMAT (HARD RULE): For Verify questions, the FINAL ANSWER must be exactly the
single word "yes" or "no" — even if the question is phrased as "Which..." or "Was...".
Do NOT name the entity, do NOT explain. Examples of phrasings that look like Wh-questions
but are actually Verify (gold answer is yes/no):
  - "Which US city in Washoe County occupies over 100 km²?"  → answer "yes" / "no"
  - "Was Frank Marshall (ISNI ...) not born in 1922?"        → answer "yes" / "no"
  - "Is the title of the derivative work equal to '...' ?"   → answer "yes" / "no"

For fact-existence ("Is 129586 the visa number of X?", "Did Nolan direct Inception?", "Is X an instance of Y?"): USE VerifyFact(subject_id, predicate, target). It runs a single SPARQL ASK and returns TRUE/FALSE. predicate_type='auto' tries attribute first, then relation. VerifyFact now auto-resolves a label target ("Netherlands") to its Q-id when the predicate is a relation, but you should still prefer passing Q-ids when you have them.

For numeric/date COMPARISON ("Is the population > 1M?", "Was X released after 2000?"): GetAttributeDetails to fetch the value, then VerifyNumericCondition for the inequality. VerifyFact does NOT handle inequalities.

For text equality ("Is the capital named Y?"): VerifyString.

Never do mental math or guess string equality. Never reason "GetAttributeDetails returned nothing therefore the answer is no" — use VerifyFact for a definitive answer instead.""",

    "Query": """STRATEGY: General Query
1) Extract all constraints. 2) Locate: ID→FindByAttribute, Name→FindNode.
3) Verify constraints with GetAttributeDetails. 4) Single-hop→GetAttributeDetails, multi-hop→RunSPARQL.
5) Time/place context→GetEdgeQualifiers. 6) One-hop inference OK if labeled [INFERRED].
7) GetJournalSummary before answering."""
}

# Main system prompt for the KQAPro agent - COMPRESSED for token efficiency
SYSTEM_PROMPT = """You are the KQAPro Execution Agent. Answer questions by querying a Knowledge Graph (KG).

RULES:
0. MANDATORY TOOL USE: You MUST call at least one tool (FindNode, FindByAttribute, RunSPARQL, …)
   before producing any final answer. NEVER answer from your training-data memory. If you find
   yourself about to write a final answer with zero tool calls in the conversation, STOP and
   call FindNode or FindByAttribute on an entity from the question first. Saying "Missing data"
   without having queried the KG is a hard error. The KG often has the answer; the agent that
   gives up early loses points the agent that probes one more time wins.
1. NO HALLUCINATION: Verify every fact with tools. One-hop inferences allowed if labeled "[INFERRED]".
2. SCHEMA COMPLIANCE: Use predicates returned by tools. If attribute fails, check available_attributes from FindNode.
3. PIVOT ON FAILURE: If search fails twice, try a connected entity or RunSPARQL with JOIN.
   Specifically: an empty GetRelationDetails on a band/group/award means you should try the
   INVERSE direction (members link to the band, not the band to members). An empty attribute
   lookup on an entity may mean the value lives on a QUALIFIER of a related statement
   (street_address on place_of_birth, number_of_matches on member_of) — reach for
   GetEdgeQualifiers / GetAttributeWithQualifiers before declaring "not in KG".
4. COMPLETE RETRIEVAL: After FindNode, always call GetAttributeDetails/GetRelationDetails for actual values.
5. VERIFY ALL CONSTRAINTS: Check ALL identifying details (duration, year, color) before answering.
6. TRUST VERIFIED DATA: Once in found_values/verified_facts, treat as ground truth. Don't second-guess.

KG PREFIXES (auto-injected in SPARQL, don't redefine):
ex:=Entities, prop:=Properties, attr:=Attributes, qual:=Qualifiers, unit:=Units

TOOL TIERS:
T1 Discovery: FindNode (semantic search) | FindByAttribute (exact ID/code/URL lookup - prefer this for unique IDs)
T1.5 Filtering: FilterEntities (by concept type and/or attribute value, supports or_conditions and transitive_concept) | QualifierFilter (by qualifier on statements)
T2 Retrieval: GetAttributeDetails | GetRelationDetails | GetNodeSummary (all data in ONE call)
T3 Qualifiers: GetEdgeQualifiers (facts about facts - use when question has time/place context)
T4 Verify: VerifyFact (deterministic ASK for "does this fact exist") | VerifyNumericCondition (never do mental math) | VerifyString (never guess string equality)
T5 Aggregate: CountEntities (exact, no truncation, supports OR/transitive) | SelectExtreme (argmax/argmin via SPARQL ORDER BY)
Complex: RunSPARQL (multi-hop >2, unusual joins) | CompareEntities | FindEntitiesByRelationPath

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

# Synthesis prompt template (for final answer generation) - COMPACT (benchmark)
SYNTHESIS_PROMPT_TEMPLATE = """DISCOVERED DATA:
{journal_summary}

QUESTION: "{query}"

Answer directly using the discovered data above. Be concise. If data is missing, say what's missing."""

# Synthesis prompt template - CONVERSATIONAL (user-facing)
SYNTHESIS_PROMPT_TEMPLATE_CONVERSATIONAL = """DISCOVERED DATA:
{journal_summary}

QUESTION: "{query}"

Write a clear, friendly, human-readable answer for the user.

Guidelines:
- Lead with the direct answer in a natural sentence (not just a bare value).
- Add 1–3 sentences of helpful supporting context drawn from the discovered data
  (e.g., related entities, dates, categories) when it aids understanding.
- You may use short lists or paragraphs; keep it tight — no filler.
- Do NOT invent facts beyond the discovered data. If something is missing or
  uncertain, say so plainly.
- Do not describe your tool-calling process; speak to the user about the answer."""

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
    "CountEntities": "Count was 0 / unexpected. If 0 AND your concept has subclasses (e.g. 'woodwind instrument' has saxophone, trumpet etc.), retry with transitive_concept=True. Otherwise check the concept label via FindNode (exact-string mismatch is common), or drop the attribute filter to count by concept alone first. Do NOT use transitive_concept=True on flat concepts like 'country' or 'province' — it can over-count by pulling in unrelated subclasses.",
    "SelectExtreme": "No winner returned. Verify the attribute_name via FindNode, drop the filter_attribute pre-filter, or fall back to CompareEntities on a smaller candidate set. If your concept is a genuine hierarchy (e.g. 'instrument' covering all subspecies), retry with transitive_concept=True; do NOT enable it for flat concepts.",
    "VerifyFact": "Returned FALSE / not found. (Note: VerifyFact now auto-resolves a label target like 'Netherlands' to its Q-id for relations, so label/Q-id mismatch is no longer a silent FALSE source.) Try predicate_type='auto' if you specified one, check the predicate spelling via FindNode's available_predicates, or use GetAttributeDetails to inspect the actual stored value.",
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
