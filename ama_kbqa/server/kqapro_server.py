import os
import sys
import time
import json
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import AsyncIterator
from pathlib import Path
from functools import wraps

from fastmcp import FastMCP, Context
from qdrant_client import QdrantClient
from openai import OpenAI
from loguru import logger
from pydantic import BaseModel, Field, ConfigDict
from typing import Literal, Any, Optional
from SPARQLWrapper import SPARQLWrapper, JSON
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

REPO_ROOT = Path(__file__).resolve().parents[2]
FEWSHOT_EXAMPLES_DIR = REPO_ROOT / "db" / "datasets" / "kqapro" / "fewshot-examples"

# Import configuration utilities
sys.path.insert(0, str(REPO_ROOT))
from ama_kbqa.config import (
    get_chat_client,
    get_embedding_client,
    get_chat_model_name,
    get_embedding_model_name,
    get_chat_temperature,
    get_chat_max_tokens,
    get_qdrant_host,
    get_qdrant_port,
    get_virtuoso_endpoint,
    get_collection_entities,
    get_collection_relations,
    get_top_n,
    get_score_threshold,
)

# Configure logger
log_dir = REPO_ROOT / "logs"
log_dir.mkdir(exist_ok=True)
logger.add(
    log_dir / "kqapro_server.log",
    rotation="10 MB",
    retention="7 days",
    level="DEBUG",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"
)

# --- Configuration from config.toml ---
QDRANT_HOST = get_qdrant_host()
QDRANT_PORT = get_qdrant_port()
COLLECTION_ENTITIES = get_collection_entities()
COLLECTION_RELATIONS = get_collection_relations()
VIRTUOSO_ENDPOINT = get_virtuoso_endpoint()
EMBEDDING_MODEL = get_embedding_model_name()
CHAT_MODEL = get_chat_model_name()
CHAT_TEMPERATURE = get_chat_temperature()
CHAT_MAX_TOKENS = get_chat_max_tokens()
TOP_N = get_top_n()
SCORE_THRESHHOLD = get_score_threshold()

# --- 1. Define a Context Class for Type Safety ---

NS_ENTITY = "http://kqapro.org/entity/"
NS_PROPERTY = "http://kqapro.org/property/"

# We inject these prefixes into every SPARQL query for convenience/safety
SPARQL_PREFIXES = """
PREFIX ex:   <http://kqapro.org/entity/>
PREFIX prop: <http://kqapro.org/property/>
PREFIX attr: <http://kqapro.org/attribute/>
PREFIX qual: <http://kqapro.org/qualifier/>
PREFIX unit: <http://kqapro.org/unit/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX xsd:  <http://www.w3.org/2001/XMLSchema#>
"""


class AppContext(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    qdrant: QdrantClient
    chat_client: OpenAI  # Client for chat/reasoning tasks
    embedding_client: OpenAI  # Client for embedding tasks
    sparql: Any  # SPARQLWrapper is not easily Pydantic-serializable, usually fine as Any


class NodeMatch(BaseModel):
    """Represents a single node found in the Knowledge Graph."""
    original_id: str = Field(
        ...,
        description="The unique identifier in the KB (e.g., 'Q937')."
    )
    name: str = Field(
        ...,
        description="The human-readable label of the entity or concept."
    )
    node_type: Literal["entity", "concept", "unknown"] = Field(
        ...,
        description="Categorization of the node."
    )
    relevance_score: float = Field(
        ...,
        description="Vector similarity score (0-1), higher is better."
    )
    # --- NEW FIELD FOR SHAPE INFORMED PROMPTING ---
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="The full node payload containing attributes, relations, and schema info."
    )


class ExtractionResponse(BaseModel):
    """Structured response for entity and relation extraction."""
    entities_concepts: list[str] = Field(
        ...,
        alias="entities/concepts",
        description="List of specific entities or general concepts identified in the query."
    )
    relations: list[str] = Field(
        ...,
        description="List of relationship predicates or actions identified in the query."
    )


class SearchResponse(BaseModel):
    """The top-level response object for the search tool."""
    matches: list[NodeMatch] = Field(
        default_factory=list,
        description="List of matching nodes sorted by relevance."
    )
    result_count: int = Field(
        ...,
        description="Total number of matches returned."
    )


class RelationMatch(BaseModel):
    """Represents a verified relationship found in the Graph DB."""
    predicate_used: str = Field(..., description="The exact predicate URI/ID verified in Virtuoso.")
    semantic_label: str = Field(...,
                                description="The readable label (e.g. 'has_population') matched via vector search.")
    objects: list[dict] = Field(
        ..., description="List of objects (targets) found for this relation. Each dict is a simplified triple object.")
    confidence: float = Field(..., description="Vector similarity score of the predicate.")


class NeighborhoodResponse(BaseModel):
    """Result of exploring a node's neighborhood."""
    base_node: str = Field(..., description="The ID of the node being explored.")
    verified_match: Optional[RelationMatch] = Field(
        None, description="The best matching relation that actually exists.")
    candidates_checked: list[str] = Field(default_factory=list, description="List of predicates checked but rejected.")
    status: str = Field(..., description="Status message.")


class SPARQLResponse(BaseModel):
    """Raw results from a SPARQL query."""
    vars: list[str] = Field(..., description="List of variable names in the SELECT clause.")
    bindings: list[dict[str, Any]
                   ] = Field(..., description="List of rows. Each row is a dict mapping variable name to value.")
    raw_json: dict[str, Any] = Field(..., description="The full raw JSON response from Virtuoso.")


class QtypePredictionResponse(BaseModel):
    """Predicted question type classification."""
    question_type: Literal[
        "Count",
        "Verify",
        "SelectBetween",
        "SelectAmong",
        "QueryAttr",
        "QueryAttrQualifier",
        "QueryRelation",
        "QueryRelationQualifier",
        "QueryName"
    ] = Field(..., description="The classified question type from KQAPro taxonomy.")


class JournalState(BaseModel):
    """The scratchpad state for the current reasoning session."""
    visited_nodes: list[str] = Field(default_factory=list, description="IDs of nodes already explored.")
    verified_facts: list[str] = Field(default_factory=list, description="Triples that have been verified via SPARQL.")
    current_plan: list[str] = Field(default_factory=list, description="Step-by-step plan for the remaining steps.")

    def to_str(self) -> str:
        return (
            f"--- CURRENT JOURNAL ---\n"
            f"VISITED NODES: {', '.join(self.visited_nodes)}\n"
            f"VERIFIED FACTS: {'; '.join(self.verified_facts)}\n"
            f"NEXT STEPS: {'; '.join(self.current_plan)}\n"
            f"-----------------------"
        )


# Global state container (resets when the agent process restarts the server)
# Since your agent.py restarts the server for every 'ask', this resets automatically per question.
session_journal = JournalState()

# --- 2. Define the Lifespan Manager ---


def format_entity_uri(node_id: str) -> str:
    """Formats a raw ID (e.g. 'Q64') into a KQAPRO Entity URI."""
    if node_id.startswith("<") and node_id.endswith(">"):
        return node_id  # Already a URI
    if "http" in node_id:
        return f"<{node_id}>"  # Raw URL string

    # Default: Append to Entity Namespace
    return f"<{NS_ENTITY}{node_id}>"


def format_property_uri(predicate_id: str) -> str:
    """Formats a raw ID (e.g. 'P1082' or 'has_name') into a KQAPRO Property URI."""
    if predicate_id.startswith("<") and predicate_id.endswith(">"):
        return predicate_id
    if "http" in predicate_id:
        return f"<{predicate_id}>"

    # Default: Append to Property Namespace
    return f"<{NS_PROPERTY}{predicate_id}>"


@asynccontextmanager
async def server_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    """
    Manages the lifecycle of the server.
    Code before 'yield' runs on startup.
    Code after 'yield' runs on shutdown.
    """
    logger.info("Starting up: Connecting to Qdrant & LLM providers...")

    try:
        # Initialize Clients using config.toml
        qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)  # TODO Use Async Client instead?

        # Quick connectivity check
        qdrant.get_collections()

        # Get LLM clients from config
        chat_client = get_chat_client()
        embedding_client = get_embedding_client()

        sparql = SPARQLWrapper(VIRTUOSO_ENDPOINT)
        sparql.setReturnFormat(JSON)
        # Yield the context so tools can access it
        yield AppContext(
            qdrant=qdrant,
            chat_client=chat_client,
            embedding_client=embedding_client,
            sparql=sparql
        )

    except Exception as e:
        logger.error(f"Something went wrong: {e}")

    finally:
        # Cleanup code (runs on shutdown)
        logger.error("🔌 Shutting down: Closing connections...")
        # Qdrant client handles its own closing usually, but you can add explicit closes here if needed
        qdrant.close()
        sys.exit(1)

# --- 3. Initialize FastMCP with Lifespan ---
mcp = FastMCP("KG-Search-Server", lifespan=server_lifespan)

# --- 4. Helper Function (now needs the client passed in) ---


def get_embedding(client: OpenAI, text: str) -> list[float]:
    text = text.replace("\n", " ")
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=[text],
        encoding_format="float"
    )
    return response.data[0].embedding


def log_tool_duration(func):
    """Decorator to log the duration of tool execution."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        tool_name = func.__name__
        start_time = time.time()
        logger.info(f"[{tool_name}] Starting execution...")

        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            logger.info(f"[{tool_name}] Completed in {duration:.2f}s")
            return result
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"[{tool_name}] Failed after {duration:.2f}s - Error: {e}")
            raise

    return wrapper


def load_fewshot_examples(max_per_type: int = 10) -> str:
    """
    Load few-shot examples from the fewshot-examples directory.

    Loads up to max_per_type examples for each question type from JSON files.
    Returns a formatted string to be injected into the classification prompt.

    Args:
        max_per_type: Maximum number of examples to load per question type

    Returns:
        Formatted string containing few-shot examples, or empty string if none available
    """
    if not FEWSHOT_EXAMPLES_DIR.exists():
        logger.warning(f"Few-shot examples directory not found: {FEWSHOT_EXAMPLES_DIR}")
        return ""

    qtypes = [
        "Count", "Verify", "SelectBetween", "SelectAmong",
        "QueryAttr", "QueryAttrQualifier", "QueryRelation",
        "QueryRelationQualifier", "QueryName"
    ]

    all_examples = []

    for qtype in qtypes:
        example_file = FEWSHOT_EXAMPLES_DIR / f"{qtype}.json"

        if not example_file.exists():
            logger.debug(f"Few-shot file not found: {example_file}")
            continue

        try:
            with open(example_file, "r", encoding="utf-8") as f:
                examples = json.load(f)

            if not examples:
                continue

            # Limit to max_per_type examples
            examples = examples[:max_per_type]

            for example in examples:
                all_examples.append({
                    "qtype": qtype,
                    "question": example.get("question", ""),
                    "reasoning": example.get("reasoning", ""),
                    "lesson_learned": example.get("lesson_learned", "")
                })

            logger.info(f"Loaded {len(examples)} few-shot examples for {qtype}")

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in {example_file}: {e}")
            continue
        except Exception as e:
            logger.error(f"Error loading {example_file}: {e}")
            continue

    if not all_examples:
        logger.info("No few-shot examples available")
        return ""

    # Format examples for the prompt
    formatted_examples = "\n\n### Few-Shot Examples (Curated from Past Classifications)\n\n"
    formatted_examples += "Here are examples of correctly classified questions with reasoning:\n\n"

    for i, example in enumerate(all_examples, 1):
        formatted_examples += f"**Example {i}:**\n"
        formatted_examples += f"Question: {example['question']}\n"
        formatted_examples += f"Correct Type: {example['qtype']}\n"

        if example['reasoning']:
            formatted_examples += f"Reasoning: {example['reasoning']}\n"

        if example['lesson_learned']:
            formatted_examples += f"Lesson Learned: {example['lesson_learned']}\n"

        formatted_examples += "\n"

    logger.info(f"Loaded total of {len(all_examples)} few-shot examples across all question types")
    return formatted_examples


# --- 5. Refactored Tool using Context ---


@mcp.tool
@log_tool_duration
def QtypePrediction(question_to_classify: str, context: Context) -> QtypePredictionResponse:
    """
    Classifies a single question using the LLM with curated few-shot examples.

    Loads up to 10 examples per question type from db/datasets/kqapro/fewshot-examples
    to improve classification accuracy through few-shot learning.

    Args:
        question_to_classify (str): The question to classify.

    Returns:
        QtypePredictionResponse: The predicted question type classification.
    """
    app_context: AppContext = context.request_context.lifespan_context

    # Load few-shot examples from curated files
    fewshot_examples = load_fewshot_examples(max_per_type=10)

    # The prompt now uses the pre-formatted examples directly
    prompt = f"""
        ### Task
        You are a question classification assistant.
        
        Your goal is to classify the following question into **exactly one** of the 9 KQA-Pro categories:
        
        - Count
        - Verify
        - SelectBetween
        - SelectAmong
        - QueryAttr
        - QueryAttrQualifier
        - QueryRelation
        - QueryRelationQualifier
        - QueryName
        
        You must reason through a structured decision process before answering.  
        At each step, evaluate whether a specific type fits based on the question's content.  
        If a later step reveals a better fit, you are allowed to go back and revise the earlier decision.
        
        Your final answer must be a **single line**:
        Qtype: <TYPE>
        
        
        ────────────────────────────────────────
        ### Explanation of Each Question Type
        
        1. **Count** Use this type if the question asks directly for a **number or quantity** of things.  
        Typical phrases include “How many…?”, “What is the number of…?”, or “Count the…”.  
        The expected answer is a non-negative integer.  
        Note: even if entities are mentioned, the focus must be on **counting** them, not on what they are or when something happened.
        
        2. **Verify** Choose this type if the question can be answered with a clear “yes” or “no”.  
        It will usually be phrased as a **factual check**, e.g., “Is…?”, “Did…?”, “Was…?”, and refers to a full statement.  
        Only use Verify if the statement is **complete enough** to verify independently — no missing subjects or vague phrases.
        
        3. **SelectBetween** This type applies when the question explicitly names **exactly two distinct entities** and compares them on a **single measurable attribute**.  
        Comparative words such as “more”, “older”, “faster”, or “better” must appear.  
        Avoid choosing SelectBetween if more than two entities are listed or if no comparison is being made.
        
        4. **SelectAmong** Use this type when a group or class of entities is involved and the question asks which one has an **extreme property** (e.g., the biggest, fastest, most successful).  
        A superlative is usually present — “most”, “least”, “biggest”, “oldest”, etc.  
        If a list is given or a general class (e.g., “Which planet…”), and only one is being selected as “best” or “most”, this is SelectAmong.
        
        5. **QueryAttr** Select this type if the question names a specific entity (like a person, company, city) and asks for a **literal attribute** (date, population, height, etc.).  
        Examples include “What is the population of Tokyo?” or “When was Google founded?”  
        Do not choose QueryAttr if the question also includes a time or place constraint — in that case, prefer QueryAttrQualifier.
        
        6. **QueryAttrQualifier** This type is a refinement of QueryAttr: it still asks for a property of a single entity, but now with a **qualifying context** like “in 2020”, “at night”, or “during WWII”.  
        The key difference is that QueryAttrQualifier adds a **constraint or filter** to the value being requested.
        
        7. **QueryRelation** Use this type if the question involves two entities and asks **what connects them**.  
        Typical patterns include: “Who directed Inception?”, “How is X related to Y?”, “Who founded Tesla?”  
        The expected answer is the **name of the relation** or **the entity that serves as a link**.
        
        8. **QueryRelationQualifier** This type builds on QueryRelation. Use it when the relation is already assumed or known, and the question now asks about **its context** — such as when it occurred, in what role, or under what conditions.  
        For example: “When did X direct Y?” or “In what role did X work at Y?”
        
        9. **QueryName** This applies when the question gives a description (using attributes, relations, or actions) and asks **who or what entity** matches it.  
        Examples: “Who discovered penicillin?”, “Which scientist developed relativity?”  
        Here, the subject or object is **unknown**, and the question seeks the **name of the entity**.
        
        ────────────────────────────────────────
        ### Structural Comparison Table (Yes/No Logic)
        
        | Type                  | Asks count | Yes/No | Attribute | Needs qualifier | Two entities | Superlative | Comparison | Needs name |
        |-----------------------|------------|--------|-----------|------------------|---------------|-------------|------------|-------------|
        | Count                 | Yes        | No     | No        | No               | No            | No          | No         | No          |
        | Verify                | No         | Yes    | No        | No               | No            | No          | No         | No          |
        | SelectBetween         | No         | No     | No        | No               | Yes           | No          | Yes        | No          |
        | SelectAmong           | No         | No     | No        | No               | Often         | Yes         | No         | No          |
        | QueryAttr             | No         | No     | Yes       | No               | No            | No          | No         | No          |
        | QueryAttrQualifier    | No         | No     | Yes       | Yes              | No            | No          | No         | No          |
        | QueryRelation         | No         | No     | No        | No               | Yes           | No          | No         | No          |
        | QueryRelationQualif.  | No         | No     | No        | Yes              | Yes           | No          | No         | No          |
        | QueryName             | No         | No     | No        | No               | No            | No          | No         | Yes         |

        ────────────────────────────────────────
        {fewshot_examples}
        ────────────────────────────────────────
        ### Classification Logic: Step-by-Step Reasoning
        
        You must now classify the input question by walking through this chain of thought:
        
        **Step 1** Is the question primarily asking for a **number** of things?  
        → If yes, the correct type is likely **Count**.  
        → However, if it adds time/place context (e.g. “in 2020”), consider revisiting this as **QueryAttrQualifier**.
        
        **Step 2** Is the question a **yes/no statement** that can be verified as true or false?  
        → If yes, this points to **Verify**.  
        → But if it instead expects a specific name or value, this is incorrect.
        
        **Step 3** Does the question mention **exactly two entities**, and compare them on a property?  
        → If yes, and words like “more”, “less”, “faster” appear → choose **SelectBetween**.  
        → If only one item is selected from a group → go to Step 4 instead.
        
        **Step 4** Does the question include a **superlative** like “most”, “least”, “biggest”, or refer to a group/list of candidates?  
        → If yes → this is likely **SelectAmong**.
        
        **Step 5** Does the question involve **two named entities** and ask what **relation** connects them?  
        → If yes → choose **QueryRelation**.  
        → If the question asks **when/where/how** that relation took place → choose **QueryRelationQualifier**.
        
        **Step 6** Does the question mention **one entity** and ask for a **specific value** (e.g., date, amount, status)?  
        → If yes, and no qualifier is present → this is **QueryAttr**.  
        → If there is a time/place condition → switch to **QueryAttrQualifier**.
        
        **Step 7** Does the question ask **who or what** matches a description, where the entity is **not explicitly named**?  
        → If yes → this is **QueryName**.
        
        You may revisit previous steps if you realize a better fit based on qualifiers, phrasing, or intent.
        
        ────────────────────────────────────────
        ### Scratchpad (internal reasoning — DO NOT SHOW TO USER)
        <scratch>
        
        ────────────────────────────────────────
        ### Final
        Output exactly one line:
        Qtype: <TYPE>
        
        ────────────────────────────────────────
        ### Question
        {question_to_classify}
    """

    messages = [{"role": "system", "content": prompt}]
    try:
        # Use the chat client from the app_context
        response = app_context.chat_client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            temperature=CHAT_TEMPERATURE,
        )

        # Parse the standard OpenAI object response
        content = response.choices[0].message.content.strip()

        # Extract the type after "Qtype: "
        if "Qtype: " in content:
            choice = content.split("Qtype: ")[-1].strip()
        else:
            # Fallback if the model outputted just the type directly
            choice = content

        logger.info(f"Question: {question_to_classify}")
        logger.info(f"Predicted qtype: {choice}")

        return QtypePredictionResponse(question_type=choice)

    except Exception as e:
        logger.error(f"Error in QtypePrediction: {e}")
        raise


@mcp.tool
@log_tool_duration
def ManageJournal(
    action: Literal["add_visited", "add_fact", "update_plan", "read"],
    content: str,
    context: Context
) -> str:
    """
    Use this tool to keep track of your progress. 
    ALWAYS use this after verifying a fact or exploring a node to prevent loops.

    Args:
        action: The type of update to perform.
        content: The text content to add (e.g., node ID, fact, or plan step).

    Returns:
        The FULL current content of the journal to refresh your memory.
    """
    global session_journal

    if action == "add_visited":
        if content not in session_journal.visited_nodes:
            session_journal.visited_nodes.append(content)

    elif action == "add_fact":
        if content not in session_journal.verified_facts:
            session_journal.verified_facts.append(content)

    elif action == "update_plan":
        # We overwrite the plan as it changes dynamically
        session_journal.current_plan = [content]

    # 'read' action just falls through to return the state

    return session_journal.to_str()


@mcp.tool
@log_tool_duration
def EntityExtraction(query: str, context: Context) -> ExtractionResponse:
    """
    Extracts entities/concepts and relations from a natural language query
    using structured output.
    """
    app_context: AppContext = context.request_context.lifespan_context

    model = CHAT_MODEL

    try:
        logger.info(f"EntityExtraction called with query: {query[:100]}...")

        completion = app_context.chat_client.beta.chat.completions.parse(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "Extract the semantic entities/concepts and relations from the user query."
                },
                {"role": "user", "content": query}
            ],
            response_format=ExtractionResponse,
            timeout=30.0  # Add 30-second timeout
        )

        logger.info(f"EntityExtraction completed successfully")
        return completion.choices[0].message.parsed

    except Exception as e:
        logger.error(f"Entity Extraction failed: {e}")
        # Return empty lists on failure to maintain type safety
        return ExtractionResponse(**{"entities/concepts": [], "relations": []})


@mcp.tool
@log_tool_duration
def FindNode(semantic_node_name: str, context: Context) -> SearchResponse:
    """
    Performs a semantic vector search to identify relevant nodes (Entities or Concepts) within the Knowledge Graph.

    Use this tool to resolve natural language descriptions into concrete Knowledge Graph nodes. 
    It retrieves the top matches based on vector similarity and returns their full context.

    Key Features:
    - **Semantic Resolution:** Can find nodes even without exact name matches (e.g., inputting "The capital of France" will find "Paris").
    - **Shape Retrieval:** Returns the full `metadata` payload (attributes and relations) for every match. This payload provides the "Shape" required for generating accurate, schema-aware SPARQL queries or answers.

    Args:
        semantic_node_name (str): The search query. This can be a specific entity name (e.g., "Berlin") or a descriptive phrase (e.g., "German cities with a population over 3 million").
        context (Context): The FastMCP request context containing the active database connections.

    Returns:
        SearchResponse: A structured object containing a list of `NodeMatch` items, each with its original ID, relevance score, and full metadata payload.
    """
    # 1. Get Context
    app_context: AppContext = context.request_context.lifespan_context

    try:
        # 2. Generate Embedding
        vector = get_embedding(app_context.embedding_client, semantic_node_name)

        # 3. Search Qdrant
        # We request the payload explicitly (though it is True by default)
        search_results = app_context.qdrant.search(
            collection_name=COLLECTION_ENTITIES,
            query_vector=vector,
            limit=TOP_N,
            with_payload=True,
            score_threshold=SCORE_THRESHHOLD
        )

        # 4. Map to Pydantic Models
        matches = []
        for point in search_results:
            payload = point.payload or {}

            match = NodeMatch(
                original_id=payload.get("original_id", "N/A"),
                name=payload.get("name", "Unknown"),
                node_type=payload.get("node_type", "unknown"),
                relevance_score=point.score,
                # Pass the entire dictionary as metadata
                metadata=payload
            )
            matches.append(match)

        # 5. Return Structured Response
        return SearchResponse(
            matches=matches,
            result_count=len(matches)
        )

    except Exception as e:
        logger.error(f"Search failed: {e}")
        return SearchResponse(matches=[], result_count=0)


@mcp.tool
@log_tool_duration
def ExploreNeighborhood(base_node_id: str, semantic_relation_name: str, context: Context) -> NeighborhoodResponse:
    """
    Finds specific facts about a node by semantically matching relations and verifying them in the Graph DB.

    Args:
        base_node_id (str): The unique ID of the node (e.g., "Q64") found via FindNode.
        semantic_relation_name (str): The relation to find (e.g., "population", "born in").

    Returns:
        NeighborhoodResponse: Verified triples found in the Virtuoso database.
    """
    ctx: AppContext = context.request_context.lifespan_context
    sparql: SPARQLWrapper = ctx.sparql

    # 1. Embed the relation query
    vector = get_embedding(ctx.embedding_client, semantic_relation_name)

    # 2. Find candidates in Qdrant
    candidates = ctx.qdrant.search(
        collection_name=COLLECTION_RELATIONS,
        query_vector=vector,
        limit=TOP_N,
        with_payload=True
    )

    checked_log = []

    # Apply Entity formatting to the Subject
    base_uri = format_entity_uri(base_node_id)

    logger.info(f"Exploring {base_uri} for relation '{semantic_relation_name}'")

    # 3. Iterate and Verify via SPARQL
    for candidate in candidates:
        predicate_raw = candidate.payload.get('predicate')

        # Apply Property formatting to the Predicate
        pred_uri = format_property_uri(predicate_raw)

        checked_log.append(f"{predicate_raw} ({candidate.score:.2f})")

        # Construct SPARQL query with prefixes
        query = f"""
        {SPARQL_PREFIXES}
        
        SELECT ?o WHERE {{
            {base_uri} {pred_uri} ?o .
        }} LIMIT 10
        """

        try:
            sparql.setQuery(query)
            results = sparql.query().convert()
            bindings = results["results"]["bindings"]

            if bindings:
                objects_found = []
                for b in bindings:
                    obj_val = b['o']['value']
                    obj_type = b['o']['type']
                    objects_found.append({"value": obj_val, "type": obj_type})

                logger.info(f"Verified match: {pred_uri} -> {len(objects_found)} objects")

                return NeighborhoodResponse(
                    base_node=base_node_id,
                    verified_match=RelationMatch(
                        predicate_used=predicate_raw,
                        semantic_label=predicate_raw,
                        objects=objects_found,
                        confidence=candidate.score
                    ),
                    candidates_checked=checked_log,
                    status="Match Found"
                )

        except Exception as e:
            logger.warning(f"SPARQL Error checking {pred_uri}: {e}")
            continue

    return NeighborhoodResponse(
        base_node=base_node_id,
        verified_match=None,
        candidates_checked=checked_log,
        status="No existing relation found among top candidates."
    )


@mcp.tool
@log_tool_duration
def RunSPARQL(query: str, context: Context) -> SPARQLResponse:
    """
    Executes an arbitrary SPARQL query against the Knowledge Graph.

    Use this tool when you need complex logic (aggregations, multi-hop, filters) 
    that cannot be satisfied by simple neighborhood exploration.

    Args:
        query (str): A valid SPARQL query string.

    Returns:
        SPARQLResponse: The structured results containing variables and bindings.
    """
    ctx: AppContext = context.request_context.lifespan_context
    sparql: SPARQLWrapper = ctx.sparql

    # Auto-inject prefixes if they aren't present
    full_query = query
    if "PREFIX" not in query:
        full_query = f"{SPARQL_PREFIXES}\n{query}"

    logger.info(f"Executing arbitrary SPARQL query:\n{full_query}")

    try:
        sparql.setQuery(full_query)
        # Convert result to Python dict
        raw_results = sparql.query().convert()

        # Parse standard SPARQL JSON format
        head_vars = raw_results.get("head", {}).get("vars", [])
        bindings = raw_results.get("results", {}).get("bindings", [])

        # Simplify bindings for the LLM (extract just the values)
        simplified_rows = []
        for row in bindings:
            simple_row = {}
            for var in head_vars:
                if var in row:
                    simple_row[var] = row[var]["value"]
            simplified_rows.append(simple_row)

        return SPARQLResponse(
            vars=head_vars,
            bindings=simplified_rows,
            raw_json=raw_results
        )

    except Exception as e:
        logger.error(f"SPARQL Execution Error: {e}")
        # Return empty/error structure so the agent knows it failed
        return SPARQLResponse(
            vars=[],
            bindings=[],
            raw_json={"error": str(e)}
        )


if __name__ == "__main__":
    mcp.run(transport="stdio")
