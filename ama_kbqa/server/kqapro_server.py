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
from qdrant_client.http import models
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
    # --- REDUCED METADATA FOR CONTEXT EFFICIENCY ---
    available_attributes: list[str] = Field(
        default_factory=list,
        description="List of unique attribute keys available for this node (e.g., 'population', 'area', 'inception')."
    )
    available_predicates: list[str] = Field(
        default_factory=list,
        description="List of unique relation predicates available for this node (e.g., 'country', 'located in time zone')."
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


class AttributeDetailsResponse(BaseModel):
    """Response containing full attribute details from the knowledge graph."""
    node_id: str = Field(..., description="The node ID queried.")
    attribute_name: str = Field(..., description="The attribute name queried.")
    values: list[dict[str, Any]] = Field(
        default_factory=list,
        description="List of attribute value bindings from SPARQL query."
    )
    status: str = Field(..., description="Status message.")


class RelationDetailsResponse(BaseModel):
    """Response containing full relation details from the knowledge graph."""
    node_id: str = Field(..., description="The node ID queried.")
    relation_name: str = Field(..., description="The relation/predicate name queried.")
    triples: list[dict[str, Any]] = Field(
        default_factory=list,
        description="List of related nodes and their details from SPARQL query."
    )
    status: str = Field(..., description="Status message.")


class SPARQLResponse(BaseModel):
    """Raw results from a SPARQL query."""
    vars: list[str] = Field(..., description="List of variable names in the SELECT clause.")
    bindings: list[dict[str, Any]
                   ] = Field(..., description="List of rows. Each row is a dict mapping variable name to value.")
    raw_json: dict[str, Any] = Field(..., description="The full raw JSON response from Virtuoso.")


class QtypePredictionResponse(BaseModel):
    """Predicted question type classification with curated examples."""
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
    fewshot_examples: str = Field(
        default="",
        description="Curated few-shot examples for this question type to guide the agent in answering effectively."
    )


class QualifierResponse(BaseModel):
    """Response containing qualifiers for a specific statement/fact."""
    base_node_id: str = Field(..., description="The Subject ID.")
    predicate: str = Field(..., description="The relation name.")
    target_node: str = Field(..., description="The Object/Value ID or string.")
    qualifiers: list[dict[str, Any]] = Field(
        default_factory=list,
        description="List of qualifiers found (e.g. time, location, role)."
    )
    status: str = Field(..., description="Status message.")


class NumericComparisonResponse(BaseModel):
    """Response from numeric/date comparison verification."""
    verdict: Literal["TRUE", "FALSE", "ERROR"] = Field(..., description="The comparison result.")
    explanation: str = Field(..., description="Human-readable explanation of the comparison.")
    value1: str = Field(..., description="First value (normalized).")
    value2: str = Field(..., description="Second value (normalized).")
    operator: str = Field(..., description="Comparison operator used.")


class ComparisonResult(BaseModel):
    """Single entity's attribute value in a comparison."""
    entity_id: str = Field(..., description="Entity ID.")
    entity_name: str = Field(..., description="Human-readable entity name.")
    value: Any = Field(..., description="The attribute value (numeric, string, or dict with value/unit).")
    normalized_value: Optional[float] = Field(None, description="Numeric value for sorting (if applicable).")


class CompareEntitiesResponse(BaseModel):
    """Response from comparing an attribute across multiple entities."""
    attribute_name: str = Field(..., description="The attribute being compared.")
    results: list[ComparisonResult] = Field(default_factory=list, description="Sorted list of entities with values.")
    sorted_by: str = Field(..., description="How the results are sorted.")
    status: str = Field(..., description="Status message.")


class JournalState(BaseModel):
    """The scratchpad state for the current reasoning session."""

    # Question Understanding (Phase 2)
    question_text: str = Field(default="", description="The original question being answered")
    question_type: str = Field(default="", description="Question type: Count, Verify, SelectBetween, etc.")
    target_entities: list[str] = Field(default_factory=list, description="Entity names we're looking for")
    target_attributes: list[str] = Field(default_factory=list, description="Attributes we need to find")

    # Exploration Tracking (Phase 1 + 2)
    visited_nodes: dict[str, str] = Field(
        default_factory=dict, description="Map of {node_id: node_name} already explored")
    verified_facts: list[dict] = Field(default_factory=list, description="Verified facts with structure")
    failed_attempts: list[str] = Field(default_factory=list, description="Track what didn't work to avoid repeating")

    # Intermediate Results (Phase 1 - CRITICAL!)
    found_values: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description="Map of {entity_id: {attribute: value}} storing all discovered values"
    )

    # Reasoning Chain (Phase 2)
    current_plan: list[str] = Field(default_factory=list, description="Step-by-step plan for remaining steps")
    completed_steps: list[str] = Field(default_factory=list, description="Steps that have been completed")

    # Answer Building (Phase 2)
    partial_answer: str = Field(default="", description="Intermediate answer being constructed")

    def to_str(self) -> str:
        """Enhanced visualization with better structure and readability."""
        lines = ["=" * 70]
        lines.append("SCRATCHPAD STATE")
        lines.append("=" * 70)

        # Question context
        if self.question_type:
            lines.append(f"Question Type: {self.question_type}")
        if self.target_entities:
            lines.append(f"Target Entities: {', '.join(self.target_entities)}")

        # Explored nodes
        if self.visited_nodes:
            lines.append(f"\nEXPLORED NODES ({len(self.visited_nodes)}):")
            for node_id, node_name in list(self.visited_nodes.items())[:5]:
                lines.append(f"  • {node_name} ({node_id})")
            if len(self.visited_nodes) > 5:
                lines.append(f"  ... and {len(self.visited_nodes) - 5} more")

        # Discovered values (MOST IMPORTANT!)
        if self.found_values:
            lines.append(f"\nDISCOVERED VALUES:")
            for entity_id, attrs in self.found_values.items():
                entity_name = self.visited_nodes.get(entity_id, entity_id)
                lines.append(f"  {entity_name}:")
                for attr_name, attr_data in attrs.items():
                    if isinstance(attr_data, list) and attr_data:
                        # Handle list of values (from GetAttributeDetails)
                        for val_item in attr_data[:3]:  # Show first 3
                            if isinstance(val_item, dict):
                                val_str = val_item.get("value", "?")
                                unit_str = val_item.get("unit", "")
                                lines.append(f"    - {attr_name}: {val_str} {unit_str}".strip())
                            else:
                                lines.append(f"    - {attr_name}: {val_item}")
                    else:
                        lines.append(f"    - {attr_name}: {attr_data}")

        # Progress tracking
        if self.completed_steps:
            lines.append(f"\nCOMPLETED STEPS ({len(self.completed_steps)}):")
            for step in self.completed_steps[-3:]:  # Last 3
                lines.append(f"  ✓ {step}")

        # Current plan
        if self.current_plan:
            lines.append(f"\nNEXT STEPS:")
            for i, step in enumerate(self.current_plan[:3], 1):  # Next 3
                lines.append(f"  {i}. {step}")

        # Failed attempts (for debugging)
        if self.failed_attempts:
            lines.append(f"\nFAILED ATTEMPTS ({len(self.failed_attempts)}):")
            for attempt in self.failed_attempts[-2:]:  # Last 2
                lines.append(f"  ✗ {attempt}")

        # Partial answer
        if self.partial_answer:
            lines.append(f"\nPARTIAL ANSWER: {self.partial_answer}")

        # Statistics
        lines.append(
            f"\nSTATS: {len(self.visited_nodes)} nodes, {len(self.found_values)} entities with data, {len(self.completed_steps)} steps done")

        lines.append("=" * 70)
        return "\n".join(lines)


# Global state container (resets when the agent process restarts the server)
# Since your agent.py restarts the server for every 'ask', this resets automatically per question.
session_journal = JournalState()

# --- 2. Define the Lifespan Manager ---


def format_entity_uri(node_id: str) -> str:
    """Formats a raw ID (e.g. 'Q64') into a KQAPRO Entity URI."""

    node_id = node_id.replace(" ", "_")

    if node_id.startswith("<") and node_id.endswith(">"):
        return node_id  # Already a URI
    if "http" in node_id:
        return f"<{node_id}>"  # Raw URL string

    # Default: Append to Entity Namespace
    return f"<{NS_ENTITY}{node_id}>"


def format_property_uri(predicate_id: str) -> str:
    """Formats a raw ID (e.g. 'P1082' or 'has_name') into a KQAPRO Property URI."""

    predicate_id = predicate_id.replace(" ", "_")

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

    qdrant = None
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
        logger.error(f"Startup error: {e}")
        sys.exit(1)

    finally:
        # Cleanup code (runs on shutdown)
        logger.info("🔌 Shutting down: Closing connections...")
        if qdrant is not None:
            qdrant.close()

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

        # Extract and log key parameters (first 2 non-context args)
        param_info = ""
        if args:
            # Skip 'self' if it's a method
            arg_start = 1 if hasattr(args[0], '__class__') else 0
            # Get up to 2 args, avoiding Context objects
            display_args = []
            for arg in args[arg_start:arg_start+3]:
                if not isinstance(arg, Context):
                    arg_repr = str(arg)[:50]  # Truncate long args
                    display_args.append(arg_repr)
            if display_args:
                param_info = f" with params: {', '.join(display_args)}"

        logger.info(f"[{tool_name}] ▶️  Starting execution{param_info}")

        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time

            # Log result size/type
            result_info = ""
            if isinstance(result, str):
                result_info = f" (returned {len(result)} chars)"
            elif isinstance(result, dict):
                result_info = f" (returned dict with {len(result)} keys)"
            elif hasattr(result, '__dict__'):
                result_info = f" (returned {type(result).__name__})"

            logger.info(f"[{tool_name}] ✅ Completed in {duration:.2f}s{result_info}")
            return result
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"[{tool_name}] ❌ Failed after {duration:.2f}s - Error: {e}", exc_info=True)
            raise

    return wrapper


def load_fewshot_examples(max_per_type: int = 10, specific_qtype: Optional[str] = None) -> str:
    """
    Load few-shot examples from the fewshot-examples directory.

    Loads up to max_per_type examples for each question type from JSON files.
    Returns a formatted string to be injected into the classification prompt.

    Args:
        max_per_type: Maximum number of examples to load per question type
        specific_qtype: If provided, only load examples for this specific question type

    Returns:
        Formatted string containing few-shot examples, or empty string if none available
    """
    if not FEWSHOT_EXAMPLES_DIR.exists():
        logger.warning(f"Few-shot examples directory not found: {FEWSHOT_EXAMPLES_DIR}")
        return ""

    # If specific_qtype is provided, only load that type's examples
    if specific_qtype:
        qtypes = [specific_qtype]
    else:
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

class _QtypeClassification(BaseModel):
    """Internal model for structured qtype classification output."""
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


def _build_classification_prompt(question: str, fewshot_examples: str = "") -> str:
    """
    Build the classification prompt with optional few-shot examples.

    Args:
        question: The question to classify
        fewshot_examples: Pre-formatted few-shot examples section (optional)

    Returns:
        The complete prompt string
    """
    return f"""
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


        ────────────────────────────────────────
        ### Explanation of Each Question Type

        1. **Count** Use this type if the question asks directly for a **number or quantity** of things.
        Typical phrases include "How many…?", "What is the number of…?", or "Count the…".
        The expected answer is a non-negative integer.
        Note: even if entities are mentioned, the focus must be on **counting** them, not on what they are or when something happened.

        2. **Verify** Choose this type if the question can be answered with a clear "yes" or "no".
        It will usually be phrased as a **factual check**, e.g., "Is…?", "Did…?", "Was…?", and refers to a full statement.
        Only use Verify if the statement is **complete enough** to verify independently — no missing subjects or vague phrases.

        3. **SelectBetween** This type applies when the question explicitly names **exactly two distinct entities** and compares them on a **single measurable attribute**.
        Comparative words such as "more", "older", "faster", or "better" must appear.
        Avoid choosing SelectBetween if more than two entities are listed or if no comparison is being made.

        4. **SelectAmong** Use this type when a group or class of entities is involved and the question asks which one has an **extreme property** (e.g., the biggest, fastest, most successful).
        A superlative is usually present — "most", "least", "biggest", "oldest", etc.
        If a list is given or a general class (e.g., "Which planet…"), and only one is being selected as "best" or "most", this is SelectAmong.

        5. **QueryAttr** Select this type if the question names a specific entity (like a person, company, city) and asks for a **literal attribute** (date, population, height, etc.).
        Examples include "What is the population of Tokyo?" or "When was Google founded?"
        Do not choose QueryAttr if the question also includes a time or place constraint — in that case, prefer QueryAttrQualifier.

        6. **QueryAttrQualifier** This type is a refinement of QueryAttr: it still asks for a property of a single entity, but now with a **qualifying context** like "in 2020", "at night", or "during WWII".
        The key difference is that QueryAttrQualifier adds a **constraint or filter** to the value being requested.

        7. **QueryRelation** Use this type if the question involves two entities and asks **what connects them**.
        Typical patterns include: "Who directed Inception?", "How is X related to Y?", "Who founded Tesla?"
        The expected answer is the **name of the relation** or **the entity that serves as a link**.

        8. **QueryRelationQualifier** This type builds on QueryRelation. Use it when the relation is already assumed or known, and the question now asks about **its context** — such as when it occurred, in what role, or under what conditions.
        For example: "When did X direct Y?" or "In what role did X work at Y?"

        9. **QueryName** This applies when the question gives a description (using attributes, relations, or actions) and asks **who or what entity** matches it.
        Examples: "Who discovered penicillin?", "Which scientist developed relativity?"
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
        → However, if it adds time/place context (e.g. "in 2020"), consider revisiting this as **QueryAttrQualifier**.

        **Step 2** Is the question a **yes/no statement** that can be verified as true or false?
        → If yes, this points to **Verify**.
        → But if it instead expects a specific name or value, this is incorrect.

        **Step 3** Does the question mention **exactly two entities**, and compare them on a property?
        → If yes, and words like "more", "less", "faster" appear → choose **SelectBetween**.
        → If only one item is selected from a group → go to Step 4 instead.

        **Step 4** Does the question include a **superlative** like "most", "least", "biggest", or refer to a group/list of candidates?
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
        ### Question to Classify
        {question}
    """


def _classify_question(app_context: AppContext, question: str, fewshot_examples: str) -> str:
    """
    Helper function to classify a question using the LLM with JSON mode.

    Args:
        app_context: The application context with LLM clients
        question: The question to classify
        fewshot_examples: Pre-formatted few-shot examples (can be empty)

    Returns:
        The predicted question type as a string
    """
    prompt = _build_classification_prompt(question, fewshot_examples)

    # Add JSON schema instruction to the prompt
    json_instruction = """

You MUST respond with a valid JSON object matching this exact schema:
{
    "question_type": "one of: Count, Verify, SelectBetween, SelectAmong, QueryAttr, QueryAttrQualifier, QueryRelation, QueryRelationQualifier, QueryName"
}

Respond ONLY with the JSON object, no additional text."""

    messages = [{"role": "system", "content": prompt + json_instruction}]

    try:
        # Use JSON mode instead of structured output
        completion = app_context.chat_client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            temperature=CHAT_TEMPERATURE,
            response_format={"type": "json_object"},
        )

        # Parse JSON response
        json_content = completion.choices[0].message.content
        response_dict = json.loads(json_content)

        # Validate against Pydantic model
        parsed_response = _QtypeClassification(**response_dict)
        return parsed_response.question_type

    except Exception as e:
        logger.error(f"Error during classification: {e}")
        raise


@mcp.tool
@log_tool_duration
def QtypePrediction(question_to_classify: str, context: Context) -> QtypePredictionResponse:
    """
    Classifies a question and returns curated few-shot examples for that question type.

    This tool first classifies the question WITHOUT few-shot examples to identify its type.
    Then it loads curated examples specific to that question type to provide guidance
    for the agent in answering the question effectively.

    The few-shot examples are NOT used to improve classification accuracy, but rather
    to provide the agent with relevant context about how to approach questions of this type.

    Args:
        question_to_classify (str): The question to classify.

    Returns:
        QtypePredictionResponse: Contains the predicted question type and curated examples
                                 specific to that type to guide the agent.
    """
    app_context: AppContext = context.request_context.lifespan_context

    # Classify the question WITHOUT few-shot examples
    logger.info(f"Classifying question: {question_to_classify}")
    predicted_qtype = _classify_question(app_context, question_to_classify, fewshot_examples="")
    logger.info(f"Predicted qtype: {predicted_qtype}")

    # Load few-shot examples specific to this question type
    logger.info(f"Loading few-shot examples for {predicted_qtype}")
    specific_examples = load_fewshot_examples(max_per_type=10, specific_qtype=predicted_qtype)

    if specific_examples:
        logger.info(f"Loaded examples for {predicted_qtype}")
    else:
        logger.info(f"No few-shot examples available for {predicted_qtype}")

    return QtypePredictionResponse(
        question_type=predicted_qtype,
        fewshot_examples=specific_examples
    )


@mcp.tool
@log_tool_duration
def ManageJournal(
    action: Literal["add_visited", "add_fact", "update_plan", "set_qtype", "set_target", "set_partial_answer", "read"],
    content: str,
    context: Context
) -> str:
    """
    Use this tool to keep track of your progress and prevent loops.

    NOTE: Most updates happen automatically via tools. Use this mainly for planning and partial answers.

    Args:
        action: The type of update to perform:
            - "add_visited": (DEPRECATED - auto-updated by FindNode) Log a visited node
            - "add_fact": (DEPRECATED - auto-updated by tools) Save a verified fact
            - "update_plan": Update your reasoning plan (manual)
            - "set_qtype": Set the question type (e.g., "Count", "SelectBetween")
            - "set_target": Add a target entity or attribute you're looking for
            - "set_partial_answer": Store your intermediate answer reasoning
            - "read": Read the current journal state
        content: The text content to add. Can be empty string for "read".

    Returns:
        The FULL current content of the journal to refresh your memory.
    """

    if action == "add_visited":
        # Backward compatibility - but tools now auto-update this
        node_id = content.strip()
        if node_id and node_id not in session_journal.visited_nodes:
            session_journal.visited_nodes[node_id] = node_id  # Store as {id: name}, name will be updated by FindNode

    elif action == "add_fact":
        # Backward compatibility - but tools now auto-update this
        if content and content not in [str(f) for f in session_journal.verified_facts]:
            session_journal.verified_facts.append({"fact": content, "source": "manual"})

    elif action == "update_plan":
        # We overwrite the plan as it changes dynamically
        session_journal.current_plan = [content] if content else []

    elif action == "set_qtype":
        session_journal.question_type = content

    elif action == "set_target":
        if content and content not in session_journal.target_entities:
            session_journal.target_entities.append(content)

    elif action == "set_partial_answer":
        session_journal.partial_answer = content

    # 'read' action just falls through to return the state

    return session_journal.to_str()


@mcp.tool
@log_tool_duration
def GetJournalSummary(context: Context) -> str:
    """
    Get a formatted summary of everything discovered so far in the scratchpad.

    **CRITICAL: Use this tool before formulating your final answer!**

    This tool shows you all the values you've discovered, which nodes you've visited,
    and what facts have been verified. Your answer MUST be based on what's in the journal.

    Returns:
        Formatted summary of all discoveries, ready to use for answering the question.
    """

    summary_lines = ["=" * 70]
    summary_lines.append("JOURNAL SUMMARY - EVERYTHING DISCOVERED SO FAR")
    summary_lines.append("=" * 70)

    # Question context
    if session_journal.question_type:
        summary_lines.append(f"\nQuestion Type: {session_journal.question_type}")
    if session_journal.target_entities:
        summary_lines.append(f"Looking for: {', '.join(session_journal.target_entities)}")

    # Most important: DISCOVERED VALUES
    if session_journal.found_values:
        summary_lines.append(f"\n📊 DISCOVERED VALUES (USE THESE FOR YOUR ANSWER!):")
        for entity_id, attrs in session_journal.found_values.items():
            entity_name = session_journal.visited_nodes.get(entity_id, entity_id)
            summary_lines.append(f"\n  {entity_name} ({entity_id}):")
            for attr_name, attr_data in attrs.items():
                if isinstance(attr_data, list) and attr_data:
                    for val_item in attr_data[:3]:
                        if isinstance(val_item, dict):
                            val_str = val_item.get("value", "?")
                            unit_str = val_item.get("unit", "")
                            summary_lines.append(f"    ✓ {attr_name}: {val_str} {unit_str}".strip())
                        else:
                            summary_lines.append(f"    ✓ {attr_name}: {val_item}")
                else:
                    summary_lines.append(f"    ✓ {attr_name}: {attr_data}")
    else:
        summary_lines.append(f"\n⚠️  NO VALUES DISCOVERED YET - You need to call GetAttributeDetails!")

    # Progress
    summary_lines.append(f"\n📈 PROGRESS:")
    summary_lines.append(f"  • Nodes explored: {len(session_journal.visited_nodes)}")
    summary_lines.append(f"  • Facts verified: {len(session_journal.verified_facts)}")
    summary_lines.append(f"  • Steps completed: {len(session_journal.completed_steps)}")

    # What we found
    if session_journal.visited_nodes:
        summary_lines.append(f"\n🔍 EXPLORED NODES:")
        for node_id, node_name in list(session_journal.visited_nodes.items())[:5]:
            has_data = "✓ HAS DATA" if node_id in session_journal.found_values else "○ no data yet"
            summary_lines.append(f"  • {node_name} ({node_id}) - {has_data}")

    # Partial answer
    if session_journal.partial_answer:
        summary_lines.append(f"\n💭 PARTIAL ANSWER: {session_journal.partial_answer}")

    # Next steps
    if session_journal.current_plan:
        summary_lines.append(f"\n📋 NEXT STEPS:")
        for i, step in enumerate(session_journal.current_plan[:3], 1):
            summary_lines.append(f"  {i}. {step}")

    summary_lines.append("=" * 70)
    summary_lines.append("✅ Use the DISCOVERED VALUES above to formulate your final answer.")
    summary_lines.append("=" * 70)

    summary_text = "\n".join(summary_lines)
    logger.info(f"GetJournalSummary called - {len(session_journal.found_values)} entities with data")

    return summary_text


@mcp.tool
@log_tool_duration
def EntityExtraction(query: str, context: Context) -> ExtractionResponse:
    """
    Extracts entities/concepts and relations from a natural language query
    using JSON mode.
    """
    app_context: AppContext = context.request_context.lifespan_context

    model = CHAT_MODEL

    try:
        logger.info(f"EntityExtraction called with query: {query[:100]}...")

        system_prompt = """Extract the semantic entities/concepts and relations from the user query.

You MUST respond with a valid JSON object matching this exact schema:
{
    "entities/concepts": ["list of specific entities or general concepts"],
    "relations": ["list of relationship predicates or actions"]
}

Respond ONLY with the JSON object, no additional text."""

        completion = app_context.chat_client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {"role": "user", "content": query}
            ],
            response_format={"type": "json_object"},
            timeout=30.0  # Add 30-second timeout
        )

        # Parse JSON response
        json_content = completion.choices[0].message.content
        response_dict = json.loads(json_content)

        # Validate against Pydantic model
        parsed_response = ExtractionResponse(**response_dict)

        logger.info(f"EntityExtraction completed successfully")
        return parsed_response

    except Exception as e:
        logger.error(f"Entity Extraction failed: {e}")
        # Return empty lists on failure to maintain type safety
        return ExtractionResponse(**{"entities/concepts": [], "relations": []})


@mcp.tool
@log_tool_duration
def FindNode(semantic_node_name: str, context: Context) -> SearchResponse:
    """
    Performs a HYBRID search (Qdrant Filters + Semantic Vectors) to find nodes.

    This tool is ROBUST: It uses exact filtering to find Technical IDs (e.g. "GGZX52", "UKE11")
    and Vector Search to find semantic concepts (e.g. "Boston").
    """
    app_context: AppContext = context.request_context.lifespan_context

    search_term_clean = semantic_node_name.strip()
    logger.info(f"FindNode: Searching for '{search_term_clean}'")

    exact_matches = []

    try:
        # PHASE 1: Qdrant Exact Filter (Robust Match)
        # We look in 'original_id', 'name', AND deep inside 'attributes' values

        # Note: 'attributes.value.value' path works if Qdrant indexed the JSON payload structure
        should_conditions = [
            models.FieldCondition(key="original_id", match=models.MatchValue(value=search_term_clean)),
            models.FieldCondition(key="name", match=models.MatchValue(value=search_term_clean)),
            # Searching inside nested attributes for IDs (e.g. searching a Visa Number or GameID)
            models.FieldCondition(key="attributes.value.value", match=models.MatchValue(value=search_term_clean))
        ]

        filter_query = models.Filter(should=should_conditions)

        # Use scroll to get exact matches ignoring vector score
        scroll_results, _ = app_context.qdrant.scroll(
            collection_name=COLLECTION_ENTITIES,
            scroll_filter=filter_query,
            limit=5,
            with_payload=True
        )

        for point in scroll_results:
            payload = point.payload or {}

            # Extract schema info (same as before)
            attributes = payload.get("attributes", [])
            unique_attrs = sorted(list(set(a.get("key") for a in attributes if a.get("key"))))
            relations = payload.get("relations", [])
            unique_preds = sorted(list(set(r.get("predicate") for r in relations if r.get("predicate"))))

            exact_matches.append(NodeMatch(
                original_id=payload.get("original_id", "N/A"),
                name=payload.get("name", "Unknown"),
                node_type=payload.get("node_type", "entity"),
                relevance_score=1.0,
                available_attributes=unique_attrs,
                available_predicates=unique_preds
            ))

        logger.info(f"FindNode: Phase 1 (Qdrant Filter) found {len(exact_matches)} matches")

    except Exception as e:
        logger.warning(f"FindNode: Phase 1 (Filter) failed: {e}")

    # PHASE 2: Semantic Search (Vector)
    vector = get_embedding(app_context.embedding_client, semantic_node_name)
    search_results = app_context.qdrant.search(
        collection_name=COLLECTION_ENTITIES,
        query_vector=vector,
        limit=TOP_N,
        with_payload=True,
        score_threshold=SCORE_THRESHHOLD
    )

    semantic_matches = []
    for point in search_results:
        payload = point.payload or {}
        attributes = payload.get("attributes", [])
        unique_attrs = sorted(list(set(a.get("key") for a in attributes if a.get("key"))))
        relations = payload.get("relations", [])
        unique_preds = sorted(list(set(r.get("predicate") for r in relations if r.get("predicate"))))

        semantic_matches.append(NodeMatch(
            original_id=payload.get("original_id", "N/A"),
            name=payload.get("name", "Unknown"),
            node_type=payload.get("node_type", "unknown"),
            relevance_score=point.score,
            available_attributes=unique_attrs,
            available_predicates=unique_preds
        ))

    # PHASE 3: Merge (Exact matches first)
    combined = {m.original_id: m for m in exact_matches}
    for m in semantic_matches:
        if m.original_id not in combined:
            combined[m.original_id] = m

    matches = sorted(combined.values(), key=lambda x: x.relevance_score, reverse=True)[:TOP_N]

    # Auto-update Journal (no `global` needed - only modifying attributes)
    if matches:
        for m in matches[:5]:
            session_journal.visited_nodes[m.original_id] = m.name
        session_journal.completed_steps.append(f"Found {len(matches)} nodes for '{semantic_node_name}'")

    return SearchResponse(matches=matches, result_count=len(matches))


@mcp.tool
@log_tool_duration
def GetAttributeDetails(base_node_id: str, attribute_name: str, context: Context) -> AttributeDetailsResponse:
    """
    Retrieves the full details of a specific attribute for a given node from the knowledge graph.

    Use this tool when you need the actual value(s) of an attribute that was discovered via FindNode.
    This queries Virtuoso directly to get all values, types, units, and qualifiers for the specified attribute.

    **Automatic Blank Node Resolution**: If the attribute value is stored as an RDF blank node
    (common for quantities with units like "150 million dollars" or "146 minutes"), this tool
    automatically resolves the blank node and returns the numeric value and unit separately.

    Args:
        base_node_id (str): The unique ID of the node (e.g., "Q100").
        attribute_name (str): The attribute key to query (e.g., "population", "area", "cost", "duration").
        context (Context): The FastMCP request context.

    Returns:
        AttributeDetailsResponse: Contains all values found for this attribute. For quantities,
                                 returns dict with "value" (numeric) and "unit" (string) keys.
    """
    ctx: AppContext = context.request_context.lifespan_context
    sparql: SPARQLWrapper = ctx.sparql

    # Format the entity URI
    base_uri = format_entity_uri(base_node_id)

    # Construct attribute URI - attributes use the attr: prefix
    # Replace spaces with underscores to ensure valid URI
    sanitized_attr_name = attribute_name.replace(" ", "_")
    attr_uri = f"<http://kqapro.org/attribute/{sanitized_attr_name}>"

    logger.info(f"Querying attribute details: {base_uri} -> {attribute_name}")

    # Query for attribute values with blank node resolution in one query
    # This handles both direct values and blank nodes with rdf:value and unit
    query = f"""
    {SPARQL_PREFIXES}

    SELECT ?value ?type ?numericValue ?unit WHERE {{
        {base_uri} {attr_uri} ?value .
        OPTIONAL {{ ?value rdf:type ?type }}
        OPTIONAL {{
            ?value rdf:value ?numericValue .
            OPTIONAL {{ ?value unit:unit ?unit }}
        }}
    }}
    """

    try:
        sparql.setQuery(query)
        results = sparql.query().convert()
        bindings = results.get("results", {}).get("bindings", [])

        if not bindings:
            logger.info(f"No attribute values found for {attribute_name} on {base_node_id}")
            return AttributeDetailsResponse(
                node_id=base_node_id,
                attribute_name=attribute_name,
                values=[],
                status=f"No values found for attribute '{attribute_name}' on node {base_node_id}"
            )

        # Simplify bindings - blank nodes are now automatically resolved in the query
        simplified_values = []
        for binding in bindings:
            simple_val = {}
            value_data = binding.get("value", {})
            value_str = value_data.get("value", "")
            value_type = value_data.get("type", "")

            # Check if we have a resolved numeric value (from blank node)
            if "numericValue" in binding and binding["numericValue"].get("value"):
                # This was a blank node, and we successfully resolved it in the query
                numeric_value = binding["numericValue"]["value"]
                simple_val["value"] = numeric_value

                # Add unit if present
                if "unit" in binding and binding["unit"].get("value"):
                    unit_uri = binding["unit"]["value"]
                    # Extract unit name from URI (e.g., http://kqapro.org/unit/minute → minute)
                    if "/" in unit_uri:
                        simple_val["unit"] = unit_uri.split("/")[-1]
                    else:
                        simple_val["unit"] = unit_uri

                simple_val["resolved_from_bnode"] = value_str
                logger.info(f"Resolved blank node {value_str}: {numeric_value} {simple_val.get('unit', '')}")

            elif value_type == "bnode" or value_str.startswith("nodeID://"):
                # This is a blank node but we couldn't resolve it (no rdf:value found)
                simple_val["value"] = value_str
                simple_val["type"] = "bnode (unresolved)"
                logger.warning(f"Could not resolve blank node {value_str}")

            else:
                # Regular value, not a blank node
                simple_val["value"] = value_str
                if "type" in binding:
                    simple_val["type"] = binding["type"]["value"]

            simplified_values.append(simple_val)

        logger.info(f"Found {len(simplified_values)} value(s) for {attribute_name}")

        # AUTO-UPDATE JOURNAL (Phase 1 - CRITICAL!) (no `global` needed - only modifying attributes)
        if simplified_values:
            # Initialize entity in found_values if not present
            if base_node_id not in session_journal.found_values:
                session_journal.found_values[base_node_id] = {}

            # Store the attribute values
            session_journal.found_values[base_node_id][attribute_name] = simplified_values

            # Log as verified fact
            for val in simplified_values[:3]:  # First 3 values
                fact_entry = {
                    "subject": base_node_id,
                    "attribute": attribute_name,
                    "value": val.get("value") if isinstance(val, dict) else val,
                    "source": "GetAttributeDetails"
                }
                if isinstance(val, dict) and "unit" in val:
                    fact_entry["unit"] = val["unit"]

                session_journal.verified_facts.append(fact_entry)

            # Log completion
            node_name = session_journal.visited_nodes.get(base_node_id, base_node_id)
            session_journal.completed_steps.append(
                f"Retrieved {attribute_name} for {node_name}"
            )

            logger.info(f"Journal auto-updated: Stored {attribute_name} values for {base_node_id}")

        return AttributeDetailsResponse(
            node_id=base_node_id,
            attribute_name=attribute_name,
            values=simplified_values,
            status=f"Found {len(simplified_values)} value(s)"
        )

    except Exception as e:
        logger.error(f"Error querying attribute details: {e}")
        # Log failure
        session_journal.failed_attempts.append(
            f"GetAttributeDetails({base_node_id}, {attribute_name}): {str(e)[:100]}"
        )
        return AttributeDetailsResponse(
            node_id=base_node_id,
            attribute_name=attribute_name,
            values=[],
            status=f"Error: {str(e)}"
        )


@mcp.tool
@log_tool_duration
def GetRelationDetails(base_node_id: str, relation_name: str, context: Context) -> RelationDetailsResponse:
    """
    Retrieves the full details of a specific relation for a given node from the knowledge graph.

    Use this tool when you need to find what nodes are connected via a specific relation that was
    discovered via FindNode. This queries Virtuoso directly to get all connected nodes.

    The tool automatically checks both forward (subject -> object) and backward (object <- subject)
    directions to find all connections.

    Args:
        base_node_id (str): The unique ID of the node (e.g., "Q100").
        relation_name (str): The relation predicate to query (e.g., "country", "capital of").
        context (Context): The FastMCP request context.

    Returns:
        RelationDetailsResponse: Contains all nodes connected via this relation.
    """
    ctx: AppContext = context.request_context.lifespan_context
    sparql: SPARQLWrapper = ctx.sparql

    # Format the entity URI
    base_uri = format_entity_uri(base_node_id)

    # Construct property URI - relations use the prop: prefix
    # Replace spaces with underscores to ensure valid URI
    sanitized_relation_name = relation_name.replace(" ", "_")
    prop_uri = f"<http://kqapro.org/property/{sanitized_relation_name}>"

    logger.info(f"Querying relation details: {base_uri} -> {relation_name}")

    # Query for both forward and backward relations
    query = f"""
    {SPARQL_PREFIXES}

    SELECT ?related ?direction WHERE {{
        {{
            {base_uri} {prop_uri} ?related .
            BIND("forward" AS ?direction)
        }}
        UNION
        {{
            ?related {prop_uri} {base_uri} .
            BIND("backward" AS ?direction)
        }}
    }}
    """

    try:
        sparql.setQuery(query)
        results = sparql.query().convert()
        bindings = results.get("results", {}).get("bindings", [])

        if not bindings:
            logger.info(f"No relations found for {relation_name} on {base_node_id}")
            return RelationDetailsResponse(
                node_id=base_node_id,
                relation_name=relation_name,
                triples=[],
                status=f"No relations found for '{relation_name}' on node {base_node_id}"
            )

        # Simplify bindings and extract entity IDs
        simplified_triples = []
        for binding in bindings:
            triple = {}
            if "related" in binding:
                related_uri = binding["related"]["value"]
                # Extract entity ID from URI (e.g., http://kqapro.org/entity/Q30 -> Q30)
                if "/entity/" in related_uri:
                    triple["related_id"] = related_uri.split("/entity/")[-1]
                else:
                    triple["related_id"] = related_uri
                triple["related_uri"] = related_uri

            if "direction" in binding:
                triple["direction"] = binding["direction"]["value"]

            simplified_triples.append(triple)

        logger.info(f"Found {len(simplified_triples)} relation(s) for {relation_name}")

        # Auto-update Journal (no `global` needed - only modifying attributes)
        if simplified_triples:
            for triple in simplified_triples[:5]:  # Log first 5 relations
                fact_entry = {
                    "subject": base_node_id,
                    "relation": relation_name,
                    "related_id": triple.get("related_id"),
                    "direction": triple.get("direction"),
                    "source": "GetRelationDetails"
                }
                session_journal.verified_facts.append(fact_entry)

            node_name = session_journal.visited_nodes.get(base_node_id, base_node_id)
            session_journal.completed_steps.append(
                f"Found {len(simplified_triples)} relations for {node_name} -> {relation_name}"
            )

            logger.info(f"Journal auto-updated: Stored {len(simplified_triples)} relations for {base_node_id}")

        return RelationDetailsResponse(
            node_id=base_node_id,
            relation_name=relation_name,
            triples=simplified_triples,
            status=f"Found {len(simplified_triples)} relation(s)"
        )

    except Exception as e:
        logger.error(f"Error querying relation details: {e}")
        # Log failure
        session_journal.failed_attempts.append(
            f"GetRelationDetails({base_node_id}, {relation_name}): {str(e)[:100]}"
        )
        return RelationDetailsResponse(
            node_id=base_node_id,
            relation_name=relation_name,
            triples=[],
            status=f"Error: {str(e)}"
        )


@mcp.tool
@log_tool_duration
def GetEdgeQualifiers(subject_id: str, predicate_name: str, target_id: str, context: Context) -> QualifierResponse:
    """
    Retrieves 'facts about a fact' (Qualifiers) for a specific relationship.

    Use this when you have found a relation (e.g., Movie -> has_website -> URL) but need
    extra context like 'language', 'start time', 'location', or 'role'.

    Args:
        subject_id: The Entity ID (e.g. "Q100").
        predicate_name: The relation name (e.g. "official website").
        target_id: The specific value or Entity ID of the target (e.g. "http://..." or "Q30").
    """
    ctx: AppContext = context.request_context.lifespan_context
    sparql: SPARQLWrapper = ctx.sparql

    subject_uri = format_entity_uri(subject_id)
    sanitized_pred = predicate_name.replace(" ", "_")

    # Heuristic for target: if it looks like a URI/ID, wrap it. Else treat as string.
    if target_id.startswith("http") or target_id.startswith("Q"):
        # For URIs, we don't quote, but we need to ensure correct format
        target_filter = f'?target = <{target_id}> || ?target = <{NS_ENTITY}{target_id}> || STR(?target) = "{target_id}"'
    else:
        # For literals
        target_filter = f'STR(?target) = "{target_id}"'

    query = f"""
    {SPARQL_PREFIXES}
    SELECT DISTINCT ?qualifier_pred ?qualifier_val WHERE {{
        # Find the Fact Node (Reified Statement)
        ?fact_node prop:fact_h {subject_uri} .
        ?fact_node prop:fact_r prop:{sanitized_pred} .
        ?fact_node prop:fact_t ?target .

        FILTER({target_filter})

        # Get qualifiers
        ?fact_node ?qualifier_pred ?qualifier_val .

        # Exclude system predicates
        FILTER(?qualifier_pred != prop:fact_h && ?qualifier_pred != prop:fact_r && ?qualifier_pred != prop:fact_t)
    }}
    """

    try:
        sparql.setQuery(query)
        results = sparql.query().convert()
        bindings = results.get("results", {}).get("bindings", [])

        qualifiers = []
        for b in bindings:
            # Clean up predicate name for display
            pred_raw = b['qualifier_pred']['value']
            pred_name = pred_raw.split('/')[-1]
            val = b['qualifier_val']['value']
            qualifiers.append({"qualifier": pred_name, "value": val})

        # Auto-update journal (no `global` needed - only modifying attributes)
        if qualifiers:
            fact_str = f"Qualifiers for {subject_id} -> {predicate_name} -> {target_id}: {qualifiers}"
            session_journal.verified_facts.append({"fact": fact_str, "source": "GetEdgeQualifiers"})

        return QualifierResponse(
            base_node_id=subject_id,
            predicate=predicate_name,
            target_node=target_id,
            qualifiers=qualifiers,
            status=f"Found {len(qualifiers)} qualifiers"
        )
    except Exception as e:
        logger.error(f"GetEdgeQualifiers failed: {e}")
        return QualifierResponse(
            base_node_id=subject_id,
            predicate=predicate_name,
            target_node=target_id,
            status=f"Error: {str(e)}"
        )


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

                # Auto-update Journal (no `global` needed - only modifying attributes)
                fact_entry = {
                    "subject": base_node_id,
                    "predicate": predicate_raw,
                    "objects": objects_found[:5],  # Store first 5 objects
                    "confidence": candidate.score,
                    "source": "ExploreNeighborhood"
                }
                session_journal.verified_facts.append(fact_entry)

                node_name = session_journal.visited_nodes.get(base_node_id, base_node_id)
                session_journal.completed_steps.append(
                    f"Explored {node_name} -> {predicate_raw}: found {len(objects_found)} objects"
                )

                logger.info(f"Journal auto-updated: Stored ExploreNeighborhood results for {base_node_id}")

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

    # Log failed exploration attempt
    session_journal.failed_attempts.append(
        f"ExploreNeighborhood({base_node_id}, {semantic_relation_name}): No match found"
    )

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

    logger.info(f"RunSPARQL: Executing query (length: {len(full_query)} chars)")
    logger.debug(f"RunSPARQL: Full query:\n{full_query}")

    try:
        sparql.setQuery(full_query)
        logger.debug(f"RunSPARQL: Query set, executing...")
        # Convert result to Python dict
        raw_results = sparql.query().convert()
        logger.debug(f"RunSPARQL: Query executed successfully")

        # Parse standard SPARQL JSON format
        head_vars = raw_results.get("head", {}).get("vars", [])
        bindings = raw_results.get("results", {}).get("bindings", [])
        logger.info(f"RunSPARQL: Query returned {len(bindings)} result(s) with variables: {head_vars}")

        # Simplify bindings for the LLM (extract just the values)
        simplified_rows = []
        for row in bindings:
            simple_row = {}
            for var in head_vars:
                if var in row:
                    simple_row[var] = row[var]["value"]
            simplified_rows.append(simple_row)

        if not bindings:
            logger.warning(f"RunSPARQL: Query returned NO results (0 bindings)")
        else:
            # Auto-update Journal with SPARQL results (no `global` needed - only modifying attributes)
            fact_entry = {
                "query_type": "SPARQL",
                "variables": head_vars,
                "result_count": len(simplified_rows),
                "results": simplified_rows[:10],  # Store first 10 results to avoid bloat
                "source": "RunSPARQL"
            }
            session_journal.verified_facts.append(fact_entry)

            # Add completion step
            session_journal.completed_steps.append(
                f"Executed SPARQL query: {len(simplified_rows)} results with variables {head_vars}"
            )

            logger.info(f"Journal auto-updated: Stored {len(simplified_rows)} SPARQL results")

        return SPARQLResponse(
            vars=head_vars,
            bindings=simplified_rows,
            raw_json=raw_results
        )

    except Exception as e:
        logger.error(f"RunSPARQL: SPARQL Execution Error: {e}", exc_info=True)
        logger.error(f"RunSPARQL: Failed query was:\n{full_query}")

        # Log failure
        session_journal.failed_attempts.append(
            f"RunSPARQL failed: {str(e)[:100]}"
        )

        # Return empty/error structure so the agent knows it failed
        return SPARQLResponse(
            vars=[],
            bindings=[],
            raw_json={"error": str(e)}
        )


@mcp.tool
@log_tool_duration
def FindByAttribute(value: str, attribute_name: str, context: Context) -> SearchResponse:
    """
    Performs a reverse lookup: finds entities by their attribute VALUE instead of name.

    This tool is designed for searching technical IDs, codes, URLs, and other non-semantic values
    where vector search fails (e.g., NUTS codes like "UKE11", ISBN numbers, Game IDs, URLs).

    **When to use:**
    - User provides a specific code, ID, or technical identifier
    - Searching for entities by URL (e.g., official website)
    - Searching for entities with specific attribute values (e.g., "population of 1000000")

    **Examples:**
    - "What city has NUTS code UKE11?" → FindByAttribute(value="UKE11", attribute_name="NUTS code")
    - "Which entity has official website http://...?" → FindByAttribute(value="http://...", attribute_name="official website")

    Args:
        value (str): The attribute value to search for (e.g., "UKE11", "94332", "http://...")
        attribute_name (str): The attribute name (e.g., "NUTS code", "population", "official website")

    Returns:
        SearchResponse: List of entities that have this attribute value.
    """
    ctx: AppContext = context.request_context.lifespan_context
    sparql: SPARQLWrapper = ctx.sparql

    logger.info(f"FindByAttribute: Searching for entities with {attribute_name}={value}")

    # Sanitize attribute name for URI
    sanitized_attr_name = attribute_name.replace(" ", "_")
    attr_uri = f"<http://kqapro.org/attribute/{sanitized_attr_name}>"

    # Build SPARQL query to find entities with this attribute value
    # We need to handle both direct values and blank nodes
    query = f"""
    {SPARQL_PREFIXES}

    SELECT DISTINCT ?entity ?entityName WHERE {{
        ?entity {attr_uri} ?attrValue .
        OPTIONAL {{ ?entity rdfs:label ?entityName }}

        # Match direct string/URI values
        FILTER(
            STR(?attrValue) = "{value}" ||
            ?attrValue = <{value}> ||
            # Also check if it's a blank node with rdf:value
            EXISTS {{
                ?attrValue rdf:value ?numVal .
                FILTER(STR(?numVal) = "{value}")
            }}
        )
    }}
    LIMIT 50
    """

    try:
        sparql.setQuery(query)
        results = sparql.query().convert()
        bindings = results.get("results", {}).get("bindings", [])

        if not bindings:
            logger.info(f"No entities found with {attribute_name}={value}")
            return SearchResponse(matches=[], result_count=0)

        # Convert SPARQL results to NodeMatch objects
        matches = []
        for binding in bindings:
            entity_uri = binding.get("entity", {}).get("value", "")
            entity_name = binding.get("entityName", {}).get("value", "Unknown")

            # Extract entity ID from URI
            if "/entity/" in entity_uri:
                entity_id = entity_uri.split("/entity/")[-1]
            else:
                entity_id = entity_uri

            # We don't have full metadata here, so we'll fetch it from Qdrant if needed
            # For now, create a basic NodeMatch
            matches.append(NodeMatch(
                original_id=entity_id,
                name=entity_name,
                node_type="entity",
                relevance_score=1.0,  # Exact match
                available_attributes=[],  # Could be populated if needed
                available_predicates=[]
            ))

        # Auto-update Journal (no `global` needed - only modifying attributes)
        for m in matches[:5]:
            session_journal.visited_nodes[m.original_id] = m.name
        session_journal.completed_steps.append(
            f"Found {len(matches)} entities with {attribute_name}={value}"
        )

        logger.info(f"FindByAttribute: Found {len(matches)} entities")
        return SearchResponse(matches=matches, result_count=len(matches))

    except Exception as e:
        logger.error(f"FindByAttribute failed: {e}")
        session_journal.failed_attempts.append(
            f"FindByAttribute({attribute_name}={value}): {str(e)[:100]}"
        )
        return SearchResponse(matches=[], result_count=0)


@mcp.tool
@log_tool_duration
def VerifyNumericCondition(
    value1: str,
    operator: Literal["<", ">", "<=", ">=", "==", "!="],
    value2: str,
    unit: str = "",
    context: Context = None
) -> NumericComparisonResponse:
    """
    Performs deterministic mathematical comparison between two values.

    This tool is the "Math Judge" - it returns a definitive TRUE/FALSE verdict for numeric
    comparisons, solving the problem where LLMs are bad at precise math or forget to state
    the final answer.

    **When to use:**
    - Any "Verify" question involving numbers, dates, or counts
    - Questions asking "Is X greater/less than Y?"
    - Questions comparing durations, populations, areas, costs, etc.

    **Supported comparisons:**
    - Numbers (including decimals)
    - Dates (ISO format: YYYY-MM-DD)
    - Values with units (e.g., "150 million", "146 minutes")

    **Examples:**
    - "Is the duration less than 238.9 minutes?" → VerifyNumericCondition("146", "<", "238.9", "minutes")
    - "Is the population greater than 1 million?" → VerifyNumericCondition("1500000", ">", "1000000", "people")
    - "Was it released before 2020?" → VerifyNumericCondition("2019-05-15", "<", "2020-01-01")

    Args:
        value1 (str): First value (can include units like "150 million")
        operator (str): Comparison operator: <, >, <=, >=, ==, !=
        value2 (str): Second value (can include units)
        unit (str): Optional unit description for context (e.g., "dollars", "minutes", "people")

    Returns:
        NumericComparisonResponse: Verdict (TRUE/FALSE/ERROR) with explanation.
    """
    logger.info(f"VerifyNumericCondition: {value1} {operator} {value2} ({unit})")

    def parse_numeric(val: str) -> float:
        """Parse numeric value, handling common formats like '150 million', '1.5k', dates."""
        val = val.strip().lower()

        # Handle dates (convert to timestamp for comparison)
        if "-" in val and len(val) >= 10:
            try:
                from datetime import datetime
                dt = datetime.fromisoformat(val.split("T")[0])
                return dt.timestamp()
            except:
                pass

        # Handle multipliers
        multipliers = {
            "trillion": 1e12, "billion": 1e9, "million": 1e6,
            "thousand": 1e3, "hundred": 1e2,
            "k": 1e3, "m": 1e6, "b": 1e9, "t": 1e12
        }

        # Extract number and multiplier
        import re
        match = re.match(r"([+-]?[\d.,]+)\s*([a-z]+)?", val)
        if match:
            num_str = match.group(1).replace(",", "")
            mult_str = match.group(2) or ""

            num = float(num_str)
            mult = multipliers.get(mult_str, 1.0)
            return num * mult

        # Fallback: try direct conversion
        return float(val.replace(",", ""))

    try:
        # Parse both values
        num1 = parse_numeric(value1)
        num2 = parse_numeric(value2)

        # Perform comparison
        comparisons = {
            "<": num1 < num2,
            ">": num1 > num2,
            "<=": num1 <= num2,
            ">=": num1 >= num2,
            "==": abs(num1 - num2) < 1e-9,  # Float equality tolerance
            "!=": abs(num1 - num2) >= 1e-9
        }

        result = comparisons[operator]
        verdict = "TRUE" if result else "FALSE"

        # Build explanation
        unit_str = f" {unit}" if unit else ""
        explanation = f"{value1}{unit_str} {operator} {value2}{unit_str} → {num1} {operator} {num2} = {verdict}"

        logger.info(f"VerifyNumericCondition result: {verdict}")

        # Auto-update Journal (no `global` needed - only modifying attributes)
        session_journal.verified_facts.append({
            "fact": explanation,
            "source": "VerifyNumericCondition"
        })
        session_journal.completed_steps.append(f"Verified: {explanation}")

        return NumericComparisonResponse(
            verdict=verdict,
            explanation=explanation,
            value1=f"{num1}{unit_str}",
            value2=f"{num2}{unit_str}",
            operator=operator
        )

    except Exception as e:
        logger.error(f"VerifyNumericCondition failed: {e}")
        session_journal.failed_attempts.append(
            f"VerifyNumericCondition({value1} {operator} {value2}): {str(e)[:100]}"
        )
        return NumericComparisonResponse(
            verdict="ERROR",
            explanation=f"Could not compare values: {str(e)}",
            value1=value1,
            value2=value2,
            operator=operator
        )


@mcp.tool
@log_tool_duration
def CompareEntities(
    entity_ids: list[str],
    attribute_name: str,
    context: Context
) -> CompareEntitiesResponse:
    """
    Simultaneously compares an attribute across multiple entities and returns sorted results.

    This is the "Leaderboard" tool - instead of fetching attributes one-by-one (which causes
    context drift), it batches all requests and returns a sorted comparison.

    **When to use:**
    - "SelectBetween" questions (comparing two items)
    - "SelectAmong" questions (finding the best in a group)
    - Any question asking "which is bigger/longer/more expensive/etc."

    **Examples:**
    - "Which movie is longer, X or Y?" → CompareEntities(ids=["Q1", "Q2"], attribute_name="duration")
    - "Which of these cities has the highest population?" → CompareEntities(ids=["Q64", "Q100", "Q90"], attribute_name="population")
    - "Which film cost more?" → CompareEntities(ids=["Q123", "Q456"], attribute_name="cost")

    Args:
        entity_ids (list[str]): List of entity IDs to compare (e.g., ["Q100", "Q64"])
        attribute_name (str): The attribute to compare (e.g., "population", "duration", "cost")

    Returns:
        CompareEntitiesResponse: Sorted list of entities with their attribute values.
    """
    ctx: AppContext = context.request_context.lifespan_context
    sparql: SPARQLWrapper = ctx.sparql

    logger.info(f"CompareEntities: Comparing {attribute_name} for {len(entity_ids)} entities")

    # Sanitize attribute name
    sanitized_attr_name = attribute_name.replace(" ", "_")
    attr_uri = f"<http://kqapro.org/attribute/{sanitized_attr_name}>"

    # Build SPARQL query with VALUES clause for multiple entities
    entity_uris = " ".join([format_entity_uri(eid) for eid in entity_ids])

    query = f"""
    {SPARQL_PREFIXES}

    SELECT ?entity ?entityName ?value ?numericValue ?unit WHERE {{
        VALUES ?entity {{ {entity_uris} }}

        ?entity {attr_uri} ?value .
        OPTIONAL {{ ?entity rdfs:label ?entityName }}

        # Try to resolve blank nodes
        OPTIONAL {{
            ?value rdf:value ?numericValue .
            OPTIONAL {{ ?value unit:unit ?unit }}
        }}
    }}
    """

    try:
        sparql.setQuery(query)
        results = sparql.query().convert()
        bindings = results.get("results", {}).get("bindings", [])

        if not bindings:
            logger.warning(f"No attribute values found for {attribute_name}")
            return CompareEntitiesResponse(
                attribute_name=attribute_name,
                results=[],
                sorted_by="none",
                status=f"No values found for attribute '{attribute_name}'"
            )

        # Process results
        entity_values = {}
        for binding in bindings:
            entity_uri = binding.get("entity", {}).get("value", "")
            entity_name = binding.get("entityName", {}).get("value", "Unknown")

            # Extract entity ID
            if "/entity/" in entity_uri:
                entity_id = entity_uri.split("/entity/")[-1]
            else:
                entity_id = entity_uri

            # Get value (prefer numeric resolution from blank nodes)
            if "numericValue" in binding and binding["numericValue"].get("value"):
                numeric_val = binding["numericValue"]["value"]
                unit_val = binding.get("unit", {}).get("value", "")

                # Extract unit name from URI
                if "/" in unit_val:
                    unit_val = unit_val.split("/")[-1]

                value_obj = {
                    "value": numeric_val,
                    "unit": unit_val
                }

                # Try to parse as float for sorting
                try:
                    normalized_val = float(numeric_val)
                except:
                    normalized_val = None
            else:
                # Direct value (string or number)
                value_str = binding.get("value", {}).get("value", "")
                value_obj = value_str

                # Try to parse as float for sorting
                try:
                    normalized_val = float(value_str)
                except:
                    normalized_val = None

            # Store (use first value if multiple)
            if entity_id not in entity_values:
                entity_values[entity_id] = {
                    "name": entity_name,
                    "value": value_obj,
                    "normalized": normalized_val
                }

        # Create comparison results
        comparison_results = []
        for eid in entity_ids:
            if eid in entity_values:
                data = entity_values[eid]
                comparison_results.append(ComparisonResult(
                    entity_id=eid,
                    entity_name=data["name"],
                    value=data["value"],
                    normalized_value=data["normalized"]
                ))
            else:
                # Entity not found or no value
                comparison_results.append(ComparisonResult(
                    entity_id=eid,
                    entity_name=session_journal.visited_nodes.get(eid, eid),
                    value="N/A",
                    normalized_value=None
                ))

        # Sort by normalized value (descending)
        sorted_results = sorted(
            comparison_results,
            key=lambda x: x.normalized_value if x.normalized_value is not None else float('-inf'),
            reverse=True
        )

        # Auto-update Journal (no `global` needed - only modifying attributes)
        for result in sorted_results:
            if result.entity_id not in session_journal.found_values:
                session_journal.found_values[result.entity_id] = {}
            session_journal.found_values[result.entity_id][attribute_name] = result.value

        session_journal.completed_steps.append(
            f"Compared {attribute_name} for {len(entity_ids)} entities"
        )

        # Build status message
        sorted_by = "numeric value (descending)" if any(
            r.normalized_value for r in sorted_results) else "order provided"

        logger.info(f"CompareEntities: Successfully compared {len(comparison_results)} entities")

        return CompareEntitiesResponse(
            attribute_name=attribute_name,
            results=sorted_results,
            sorted_by=sorted_by,
            status=f"Successfully compared {len(comparison_results)} entities"
        )

    except Exception as e:
        logger.error(f"CompareEntities failed: {e}")
        session_journal.failed_attempts.append(
            f"CompareEntities({attribute_name}): {str(e)[:100]}"
        )
        return CompareEntitiesResponse(
            attribute_name=attribute_name,
            results=[],
            sorted_by="error",
            status=f"Error: {str(e)}"
        )


if __name__ == "__main__":
    mcp.run(transport="stdio")
