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
  - "How many X have attribute A OR are used by Y?" → use CountUnion with one branch
    for the global concept/attribute filter and one branch for the explicit IDs from
    Y's relation. Do NOT put those entity_ids on CountEntities globally; that
    intersects the branches instead of unioning them.

FORBIDDEN: FilterEntities + len(matches) [caps at limit=50, returns wrong number].
FORBIDDEN as default: hand-written RunSPARQL COUNT [error-prone with KQAPro's namespaces]. Reach for it ONLY for genuinely 3+ hop joins where CountEntities can't express the join.

Trust the integer CountEntities returns. Don't downgrade after verification.""",

    "QueryAttr": """STRATEGY: QueryAttr → attribute lookup
Standard: FindNode → GetAttributeDetails (literals) or GetRelationDetails (linked entities).
Reverse: unique ID/code/URL → FindByAttribute (faster, exact).

AWARD/NOMINATION QUALIFIER TRAP: If the question asks "What film/work was PERSON
nominated for AWARD?" or "for which work did PERSON receive AWARD?", it is a
relation qualifier question even if classification says QueryAttr. Verify the
PERSON -> award relation, then call GetQualifierValue(person_id, "nominated for"
or "award received", award_id, "for_work", predicate_type="relation").

EMPTY-RESULT FALLBACK ORDER (do NOT skip steps):
 1. Re-check available_attributes from the FindNode result for a near-match name
    (e.g., question says "subscribers" → attribute may be "number of subscribers").
 2. Call GetNodeSummary(node_id) — returns EVERY attribute and predicate on the node in
    one call. This is the canonical "what does this node have?" probe. Subscriber counts,
    follower counts, ISBNs, ISNI numbers, durations, populations, codes are stored as
    ATTRIBUTES, not relations. ExploreNeighborhood searches RELATION embeddings and will
    NEVER surface them — do not reach for it before GetNodeSummary has confirmed no
    matching attribute exists.
 3. Only if GetNodeSummary shows no candidate attribute, fall back to ExploreNeighborhood
    (in case the question's noun is actually a related entity, not an attribute).
 4. Last resort: RunSPARQL with `SELECT ?p ?v WHERE { ex:ID ?p ?v . }` to dump everything.

"X in Y": find Y first, then X related to Y. Return X's attribute.""",

    "QueryAttrQualifier": """STRATEGY: QueryAttrQualifier → contextual fact (time/place on a fact)
Key distinction: "movie's language"=NodeAttr vs "language of website dated 1998-04-09"=EdgeQualifier.
Steps: 1) Find base fact via GetAttributeDetails/GetRelationDetails
2) GetQualifierValue(subject_id, predicate, target_value, qualifier_name) when the
   qualifier is known; use GetEdgeQualifiers only to discover unknown qualifier keys.
3) Match question word to qualifier: When→point_in_time, Where→location

MANDATORY GetQualifierValue WHEN YOU KNOW THE QUALIFIER NAME: Once you've identified the
target statement (subject, predicate, target) AND you know which qualifier you want
(e.g., "start time", "point in time", "location"), call
GetQualifierValue(subject_id, predicate, target, qualifier_name). It projects ONLY that
qualifier — no full-dict scan, fewer distractors, smaller payload, auto-handles backward
direction. Use GetEdgeQualifiers / GetQualifiersByPredicate ONLY when you need to discover
which qualifiers exist on the statement.

QUALIFIER ALIASES:
  - matches played / appearances for sports-team membership → number_of_matches
  - mapped relation type / maps to / relation type for an external identifier → relation_type
  - applies to which part / grammatical form / demonym form → applies_to_part
  - known under / recorded in database / mentioned as work contributor → inspect identifier/name
    qualifiers before returning the plain title/name attribute.

DIRECTION RULE (critical for backward edges): When GetRelationDetails returns
`direction: "backward"` for a triple, the canonical statement is
`<related_id> prop:<relation> <base_id>` — the RELATED entity is the subject, the BASE
entity is the object. To pull qualifiers off that statement, call GetEdgeQualifiers /
GetQualifiersByPredicate with `subject = related_id` and `target = base_id`, NOT the
other way around. Calling with the directions swapped silently returns 0 qualifiers and
makes you think the data is missing.

FILTERING-BY-QUALIFIER (PREFER THIS over per-entity loops): When the question is "find
the X among {entities} where qualifier_K = V" (e.g., "the member of Cardiff City whose
start_time is 1991"), DO NOT iterate GetEdgeQualifiers per entity — call
QualifierFilter(entity_ids=[Q1,Q2,...], relation="member of sports team",
qualifier_name="start_time", value="1991") in ONE call. It runs the SPARQL filter
server-side and returns just the matching entities. Per-entity loops burn iterations
and frequently trip the loop detector.

Pattern: SELECT ?qv WHERE { ?f pred:fact_h ex:ID; pred:fact_r prop:P; pred:fact_t "VAL". ?f qual:Q ?qv. }""",

    "QueryName": """STRATEGY: QueryName → identify entity from description
Unique ID/code → FindByAttribute immediately.
Single condition: FindNode + GetRelationDetails.
Type+attribute conditions: FilterEntities(concept=..., attribute_name=..., attribute_value=...) for combined filtering.
Multiple conditions: RunSPARQL with multiple WHERE clauses (avoid manual intersection).

AWARD/NOMINATION QUALIFIER TRAP: "What film/work was PERSON nominated for AWARD?"
means the requested entity is the `for_work` qualifier on PERSON --nominated_for-->
AWARD. Use GetQualifierValue(person_id, "nominated for", award_id, "for_work",
predicate_type="relation") before raw SPARQL or generic QueryName search.

Pattern: SELECT ?label WHERE { ?s prop:P1 ex:O1. ?s prop:P2 ex:O2. ?s rdfs:label ?label. }""",

    "QueryRelation": """STRATEGY: QueryRelation → find predicate between two entities

🔴 OUTPUT FORMAT (HARD RULE): The FINAL ANSWER must be the BARE predicate label exactly
as it appears in available_predicates — e.g., "occupation", "cast member", "director",
"spouse". NOT a sentence, NOT "X has the relation Y to Z", NOT "X is the spouse of Y".
Just the label. The judge measures the bare label; narrative answers fail even when
they semantically contain the right relation.

Preferred: FindNode A and B, then call GetRelationBetween(A_id, B_id). It returns
subject_to_object and object_to_subject separately; for "How is A related to B?" answer
with the predicate in subject_to_object. Do NOT answer with the inverse predicate from B
to A unless no direct A→B predicate exists and the question wording supports inverse-only.

AWARD/NOMINATION QUALIFIER TRAP: If the question asks "What film/work was PERSON
nominated for AWARD?" or "for which work did PERSON receive AWARD?", this is NOT
asking for the bare relation `nominated for` / `award received`. First verify the
PERSON -> award relation, then call:
GetQualifierValue(person_id, "nominated for" or "award received", award_id,
                  "for_work", predicate_type="relation")
Answer the returned film/work label.

Fallback: GetRelationDetails on A, check if B appears. If A→B fails, try B→A (bidirectional).
Fallback: SELECT DISTINCT ?p ?label WHERE { { ex:A ?p ex:B } UNION { ex:B ?p ex:A } ?p rdfs:label ?label. }

PASSIVE-VOICE GRAMMAR TRAP (symmetric temporal relations: followed_by, preceded_by,
replaced_by, succeeded_by, follows): Passive voice INVERTS direction.
  - "X was followed by Y"  → X is BEFORE Y, edge is `X --followed_by--> Y` (forward from X)
  - "X follows Y"          → X is AFTER Y,  edge is `Y --followed_by--> X` (forward from Y)
  - "What were followed by Y?"  → answer is the PREDECESSOR(s) of Y, i.e., entities X
                                  with `X --followed_by--> Y` (Y is the forward target).
  - "What followed Y?"          → answer is the SUCCESSOR(s) of Y, i.e., entities X
                                  with `Y --followed_by--> X` (X is the forward target).
GetRelationDetails returns BOTH directions when the relation is bidirectional. Parse the
grammar of the wh-clause to pick the right one. Worked example:
  Q: "What Olympic Games were followed by the 1980 Olympics?"
  → grammar: "X were followed by 1980" → X is before 1980 → answer is the predecessor
  → look for the result on Q8450 (1980) where direction is "backward" (i.e., the OTHER
     entity is the source of `followed_by` pointing at 1980) → 1976 Olympics, NOT 1984.
This rule applies to any temporally-symmetric Wikidata predicate, not just Olympics.""",

    "QueryRelationQualifier": """STRATEGY: QueryRelationQualifier → context of a relation
THE ANSWER IS A QUALIFIER VALUE, NOT A NEW ENTITY. The relation itself is already known;
the question asks *when/where/at-what-event/in-what-role* it held. Do not collapse to the
person, film, or award — return the qualifier (a ceremony, date, place, or role label).

1) Confirm connection via GetRelationDetails.
2) MANDATORY when qualifier name is known: GetQualifierValue(subject_id, predicate, target, qualifier_name) — direct
   projection of one qualifier (e.g., "for work", "point in time", "ceremony"). Use this
   when you know which qualifier the wh-word targets. Fallback to GetEdgeQualifiers only
   to discover which qualifiers exist.
3) Match qualifier:
     When → point_in_time (date) OR ceremony/edition (event) — pick event if the
        question frames it as an occasion ("at which ceremony", "during which").
     Where → location
     Role → object_has_role
     Ceremony → ceremony
     Matches played / appearances → number_of_matches
     Relation type / maps to → relation_type
     Applies to which part / demonym form → applies_to_part
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
FORBIDDEN: comparing only a sampled subset of relation-derived candidates. If the candidate
set comes from a relation (e.g., all release regions of a film), enumerate ALL relation
targets, filter the set, then call SelectExtreme(entity_ids=all_filtered_candidates, ...).

EVIDENCE CHECK before final answer: your trace must show (a) where the candidate set
came from, (b) which filter removed/kept candidates, and (c) the SelectExtreme result
over the complete filtered set. If any of those are missing, do not finalize yet.

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
1) Extract all constraints. 2) Locate: exact attribute/value→FindByAttribute, Name→FindNode.
3) Verify exact constraints with GetAttributeDetails/GetNodeSummary before trusting a same-name entity.
4) Single-hop→GetAttributeDetails/GetRelationDetails, multi-hop→RunSPARQL.
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
0a. NO PREMATURE "DATA NOT IN KG" (HARD RULE): Before finalizing any answer that contains
    phrases like "not available", "not in the knowledge graph", "could not be determined",
    "no record found", or "data is missing", you MUST have called GetNodeSummary on the
    target entity at least once in this conversation. GetNodeSummary returns EVERY attribute
    and EVERY predicate on the node in one call — if the data exists, it will appear in that
    output. A single failed GetAttributeDetails / GetRelationDetails / ExploreNeighborhood
    is NOT enough evidence that the data is absent: the wrong attribute name, the wrong
    relation direction, or a relation-vs-attribute mismatch can each produce empty results
    while the data sits one call away. Concluding "missing" without GetNodeSummary is a
    hard error that costs guaranteed points.
1. NO HALLUCINATION: Verify every fact with tools. One-hop inferences allowed if labeled "[INFERRED]".
1a. EXACT CONSTRAINTS: If the question states an exact attribute/value condition
    (official name, ID/code, date of birth, URL, ISNI, UMLS CUI, ICD, ISWC, etc.),
    use FindByAttribute or explicitly verify that condition. Do not semantic-search
    the whole phrase as if it were an entity name, and do not answer from a same-name
    entity that does not satisfy the exact constraints.
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
T1 Discovery: FindNode (semantic search) | FindByAttribute (exact ID/code/URL lookup - prefer this for unique IDs) | LookupEntityByName (deterministic, non-vector name lookup - use when FindNode's results look plausible but wrong, or when you hold an over-specified mention like "Texas metropolitan area" for a KB entity actually named "Texas")
T1.5 Filtering: FilterEntities (by concept type and/or attribute value, supports or_conditions and transitive_concept) | QualifierFilter (by qualifier on statements)
T2 Retrieval: GetAttributeDetails | GetRelationDetails | GetNodeSummary (all data in ONE call)
T3 Qualifiers: GetQualifierValue (preferred single qualifier projection) | GetEdgeQualifiers/GetQualifiersByPredicate (qualifier discovery)
T4 Verify: VerifyFact (deterministic ASK for "does this fact exist") | VerifyNumericCondition (never do mental math) | VerifyString (never guess string equality)
T5 Aggregate: CountEntities (exact, no truncation, supports OR/transitive) | CountUnion (heterogeneous OR branches) | SelectExtreme (argmax/argmin via SPARQL ORDER BY)
Complex: GetRelationBetween (exact direct/inverse predicate labels) | RunSPARQL (multi-hop >2, unusual joins; SELECT output is capped, so refine broad queries with COUNT/GROUP BY/LIMIT) | CompareEntities | FindEntitiesByRelationPath

BATCH INDEPENDENT CALLS: When your next steps need several lookups that do not depend on
each other's results (e.g. FindNode on two different entities, GetNodeSummary on several
candidates), emit them as MULTIPLE tool calls in ONE turn. They run concurrently; one
batched turn is far cheaper than one round-trip per call. Only sequence calls whose
arguments need a previous result.

QUALIFIER DECISION: Question specifies a known qualifier value target? → GetQualifierValue. Need to discover qualifier keys first? → GetEdgeQualifiers/GetQualifiersByPredicate. General property? → GetAttributeDetails.
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
REQUIRED: change the failure mode, not just the wording:
1) If exact attributes/IDs/codes/names/dates are stated, use FindByAttribute or verify them.
2) If a name is ambiguous, disambiguate by explicit constraints before answering.
3) If doing Count/Select, prove the candidate set is complete before aggregating/selecting.
4) Use RunSPARQL only when high-level tools cannot express the graph shape."""

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
    "FindNode": "Entity not found, or the results you got back don't actually match what you're looking for (confident-looking but wrong). Try LookupEntityByName (deterministic, no vectors) — especially if your search term was an over-specified mention (e.g. 'Texas metropolitan area' when the KB entity is just 'Texas'). Otherwise try a related entity, synonyms, or FindByAttribute with ID/code.",
    "FindByAttribute": "Value not found. Try FindNode semantic search, LookupEntityByName for a literal name, or RunSPARQL with broader filter.",
    "LookupEntityByName": "No lexical match either. Try the other `mode` ('exact'/'prefix'/'contains'), fall back to FindNode's semantic search, or the entity may not exist in this KB.",
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
