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
import qdrant_client

from fastmcp import FastMCP, Context
from qdrant_client import QdrantClient
from qdrant_client.http import models
from openai import OpenAI
from loguru import logger
from pydantic import BaseModel, Field, ConfigDict
from typing import Literal, Any, Optional, Dict
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


@mcp.tool()
async def GetNodeLabel(
    app_context: Context,
    node_id: str
) -> str:
    """
    Resolve entity ID (e.g., Q1860) to human-readable label.
    
    **CRITICAL for Qualifier Handling:**
    - ALWAYS use this when you encounter entity IDs in qualifiers
    - Common pattern: "original_language: Q1860" → call GetNodeLabel("Q1860") → "English"
    
    Args:
        node_id: Entity ID to resolve (e.g., "Q1860", "Q217008")
    
    Returns:
        JSON with node_id, label, and status
    """
    app = app_context.request_context.lifespan_context
    
    try:
        session_journal.visited_nodes[node_id] = f"(resolving label)"
        
        query = f"""
        SELECT ?label WHERE {{
            ex:{node_id} rdfs:label ?label .
        }}
        """
        
        full_query = SPARQL_PREFIXES + query
        
        # FIX: sparql direkt vom app context nutzen, NICHT app.qdrant.sparql
        app.sparql.setQuery(full_query)
        app.sparql.setReturnFormat(JSON)
        
        results = app.sparql.query().convert()
        bindings = results.get("results", {}).get("bindings", [])
        
        if not bindings:
            response = {
                "node_id": node_id,
                "label": None,
                "node_type": "unknown",
                "status": f"No label found for {node_id}"
            }
            logger.warning(f"GetNodeLabel: No label found for {node_id}")
            return json.dumps(response, indent=2)
        
        label = bindings[0].get("label", {}).get("value", "")
        session_journal.visited_nodes[node_id] = label
        
        # Type check
        type_query = f"""
        SELECT ?type WHERE {{
            ex:{node_id} rdf:type ?type .
        }} LIMIT 1
        """
        type_full_query = SPARQL_PREFIXES + type_query
        app.sparql.setQuery(type_full_query)
        type_results = app.sparql.query().convert()
        type_bindings = type_results.get("results", {}).get("bindings", [])
        
        node_type = "entity"
        if type_bindings:
            type_uri = type_bindings[0].get("type", {}).get("value", "")
            if "concept" in type_uri.lower():
                node_type = "concept"
        
        response = {
            "node_id": node_id,
            "label": label,
            "node_type": node_type,
            "status": f"Found label for {node_id}"
        }
        
        return json.dumps(response, indent=2)
        
    except Exception as e:
        error_msg = f"Error resolving label for {node_id}: {str(e)}"
        logger.error(error_msg)
        return json.dumps({"error": error_msg, "node_id": node_id}, indent=2)


# ============================================================================
# TOOL 2: BatchGetNodeLabels
# ============================================================================

@mcp.tool()
async def BatchGetNodeLabels(
    app_context: Context,
    node_ids: list[str]
) -> str:
    """
    Efficiently resolve MULTIPLE entity IDs to labels in a single SPARQL call.
    
    **Performance Optimization:**
    - Use this instead of calling GetNodeLabel multiple times
    - Single SPARQL query resolves all IDs at once
    
    Args:
        node_ids: List of entity IDs to resolve (e.g., ["Q1860", "Q217008", "Q699224"])
    
    Returns:
        JSON with resolved labels and not_found list
    """
    app = app_context.request_context.lifespan_context

    try:
        if not node_ids:
            return json.dumps({"status": "No IDs"}, indent=2)
        
        unique_ids = list(set(node_ids))
        values_clause = " ".join([f"ex:{nid}" for nid in unique_ids])
        
        query = f"""
        SELECT ?entity ?label WHERE {{
            VALUES ?entity {{ {values_clause} }}
            OPTIONAL {{ ?entity rdfs:label ?label . }}
        }}
        """
        
        full_query = SPARQL_PREFIXES + query
        
        # FIX: sparql vom app context
        app.sparql.setQuery(full_query)
        app.sparql.setReturnFormat(JSON)
        
        results = app.sparql.query().convert()
        bindings = results.get("results", {}).get("bindings", [])
        
        resolved = {}
        found_ids = set()
        
        for binding in bindings:
            entity_uri = binding.get("entity", {}).get("value", "")
            label = binding.get("label", {}).get("value", None)
            
            # Extract entity ID from URI
            entity_id = entity_uri.split("/")[-1]
            
            if label:
                resolved[entity_id] = label
                found_ids.add(entity_id)
                # Update journal
                session_journal.visited_nodes[entity_id] = label
        
        not_found = [nid for nid in unique_ids if nid not in found_ids]
        
        # Log failed resolutions
        for nid in not_found:
            session_journal.failed_attempts.append(f"BatchGetNodeLabels: {nid} not found")
        
        response = {
            "resolved": resolved,
            "not_found": not_found,
            "status": f"Resolved {len(resolved)}/{len(unique_ids)} entities"
        }
        
        logger.info(f"BatchGetNodeLabels: Resolved {len(resolved)}/{len(unique_ids)} IDs")
        return json.dumps(response, indent=2)
        
    except Exception as e:
        error_msg = f"Error in batch label resolution: {str(e)}"
        logger.error(error_msg)
        return json.dumps({"error": error_msg}, indent=2)


# ============================================================================
# TOOL 3: GetSchemaForAttribute
# ============================================================================

@mcp.tool()
async def GetSchemaForAttribute(
    app_context: Context,
    attribute_name: str,
    fuzzy: bool = True
) -> str:
    """
    Validate and fuzzy-match attribute names before using FindByAttribute.
    
    **CRITICAL for Attribute Searches:**
    - ALWAYS use this BEFORE FindByAttribute with technical IDs
    - Prevents failures from wrong attribute names (e.g., "GameID" vs "Nintendo GameID")
    
    Args:
        attribute_name: Attribute to search for (e.g., "GameID", "NUTS code")
        fuzzy: Enable fuzzy matching (default: True)
    
    Returns:
        JSON with exact_match, fuzzy_matches, and recommendation
    """
    try:
        # Query all distinct attributes from Virtuoso
        query = """
        SELECT DISTINCT ?attr WHERE {
            ?s ?attr ?o .
            FILTER(STRSTARTS(STR(?attr), "http://kqapro.org/attribute/"))
        }
        """
        
        full_query = SPARQL_PREFIXES + query
        app_context = app_context.request_context.lifespan_context
        app_context.sparql.setQuery(full_query)         # ✅ CORRECT!
        app_context.sparql.setReturnFormat(JSON)        # ✅ CORRECT!
    
        results = app_context.sparql.query().convert()  # ✅ CORRECT!
        bindings = results.get("results", {}).get("bindings", [])
        
        # Extract attribute names
        all_attributes = []
        for binding in bindings:
            attr_uri = binding.get("attr", {}).get("value", "")
            attr_name = attr_uri.split("/")[-1].replace("_", " ")
            all_attributes.append(attr_name)
        
        # Normalize query
        query_normalized = attribute_name.lower().strip().replace("_", " ")
        
        # Check for exact match
        exact_match = None
        for attr in all_attributes:
            if attr.lower() == query_normalized:
                exact_match = attr
                break
        
        if exact_match:
            response = {
                "query": attribute_name,
                "exact_match": exact_match,
                "fuzzy_matches": [],
                "recommendation": exact_match,
                "status": f"Found exact match: {exact_match}"
            }
            logger.info(f"GetSchemaForAttribute: Exact match for '{attribute_name}' → {exact_match}")
            return json.dumps(response, indent=2)
        
        if not fuzzy:
            response = {
                "query": attribute_name,
                "exact_match": None,
                "fuzzy_matches": [],
                "recommendation": None,
                "status": f"No exact match found for '{attribute_name}' (fuzzy disabled)"
            }
            return json.dumps(response, indent=2)
        
        # Fuzzy matching using embeddings
        # Generate embedding for query
        emb_response = app_context.embedding_client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=[query_normalized]
        )
        query = emb_response.data[0].embedding
        
        # Generate embeddings for all attributes
        attr_embeddings = app_context.embedding_client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=all_attributes[:100]  # Limit to avoid API limits
        )
        
        # Compute cosine similarities
        from numpy import dot
        from numpy.linalg import norm
        
        similarities = []
        for i, attr in enumerate(all_attributes[:100]):
            attr_vec = attr_embeddings.data[i].embedding
            similarity = dot(query, attr_vec) / (norm(query) * norm(attr_vec))
            similarities.append((attr, float(similarity)))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Get top 3 fuzzy matches
        fuzzy_matches = []
        for attr, sim in similarities[:3]:
            if sim > 0.6:  # Threshold for relevance
                # Count usage in KB
                count_query = f"""
                SELECT (COUNT(?s) AS ?count) WHERE {{
                    ?s attr:{attr.replace(" ", "_")} ?o .
                }}
                """
                count_full_query = SPARQL_PREFIXES + count_query
                app_context.sparql.setQuery(count_full_query)        # ✅ CORRECT!
                count_results = app_context.sparql.query().convert() # ✅ CORRECT!
                count_bindings = count_results.get("results", {}).get("bindings", [])
                
                usage_count = 0
                if count_bindings:
                    usage_count = int(count_bindings[0].get("count", {}).get("value", 0))
                
                fuzzy_matches.append({
                    "attribute": attr,
                    "similarity": round(sim, 3),
                    "sample_usage": f"Used in {usage_count} entities"
                })
        
        recommendation = fuzzy_matches[0]["attribute"] if fuzzy_matches else None
        
        response = {
            "query": attribute_name,
            "exact_match": None,
            "fuzzy_matches": fuzzy_matches,
            "recommendation": recommendation,
            "status": f"Found {len(fuzzy_matches)} fuzzy matches"
        }
        
        logger.info(f"GetSchemaForAttribute: Fuzzy matches for '{attribute_name}' → {[m['attribute'] for m in fuzzy_matches]}")
        return json.dumps(response, indent=2)
        
    except Exception as e:
        error_msg = f"Error in schema validation: {str(e)}"
        logger.error(error_msg)
        return json.dumps({"error": error_msg}, indent=2)


# ============================================================================
# TOOL 4: GetEdgeQualifiers
# ============================================================================

@mcp.tool()
async def GetEdgeQualifiers(
    app_context: Context,
    base_node_id: str,
    attribute_name: str,
    attribute_value: Optional[str] = None
) -> str:
    """
    Extract ALL qualifiers from a specific attribute statement.
    
    **CRITICAL for QueryAttrQualifier Questions:**
    - Use this when questions ask about metadata/context of an attribute
    - Examples: "What language is associated with website X?", "Where was X published on date Y?"
    - Automatically resolves entity IDs in qualifiers to labels
    
    Args:
        base_node_id: Entity ID (e.g., "Q217008")
        attribute_name: Attribute name (e.g., "official website")
        attribute_value: Optional specific value to filter by
    
    Returns:
        JSON with base_node, attribute, value, qualifiers, and status
    """
    try:
        # Normalize attribute name
        attr_normalized = attribute_name.replace(" ", "_")
        
        # Build SPARQL query
        value_filter = ""
        if attribute_value:
            value_filter = f'FILTER(?value = "{attribute_value}")'
        
        query = f"""
        SELECT ?value ?qkey ?qval ?qlabel WHERE {{
            ex:{base_node_id} attr:{attr_normalized} ?bnode .
            ?bnode rdf:value ?value .
            {value_filter}
            
            # Find the statement node
            ?stmt rdf:subject ex:{base_node_id} ;
                  rdf:predicate attr:{attr_normalized} ;
                  rdf:object ?bnode .
            
            # Get qualifiers
            ?stmt ?qprop ?qval .
            FILTER(STRSTARTS(STR(?qprop), "http://kqapro.org/qualifier/"))
            
            # Extract qualifier key name
            BIND(REPLACE(STR(?qprop), ".*/(.*)", "$1") AS ?qkey)
            
            # Try to resolve entity labels
            OPTIONAL {{
                ?qval rdfs:label ?qlabel .
            }}
        }}
        """
        
        full_query = SPARQL_PREFIXES + query
        app_context = app_context.request_context.lifespan_context
        app_context.sparql.setQuery(full_query)             # ✅ CORRECT!
        app_context.sparql.setReturnFormat(JSON)            # ✅ CORRECT!

        results = app_context.sparql.query().convert()      # ✅ CORRECT!
        bindings = results.get("results", {}).get("bindings", [])
        
        if not bindings:
            # Try alternative query pattern (for relations instead of attributes)
            query_alt = f"""
            SELECT ?value ?qkey ?qval ?qlabel WHERE {{
                ex:{base_node_id} prop:{attr_normalized} ?target .
                
                # Find the statement node
                ?stmt rdf:subject ex:{base_node_id} ;
                      rdf:predicate prop:{attr_normalized} ;
                      rdf:object ?target .
                
                # Get qualifiers
                ?stmt ?qprop ?qval .
                FILTER(STRSTARTS(STR(?qprop), "http://kqapro.org/qualifier/"))
                
                BIND(REPLACE(STR(?qprop), ".*/(.*)", "$1") AS ?qkey)
                
                OPTIONAL {{
                    ?qval rdfs:label ?qlabel .
                }}
                
                # Get target label as value
                OPTIONAL {{
                    ?target rdfs:label ?value .
                }}
            }}
            """
            
            full_query_alt = SPARQL_PREFIXES + query_alt
            app_context.sparql.setQuery(full_query_alt)     # ✅ CORRECT!
            results = app_context.sparql.query().convert()  # ✅ CORRECT!
            bindings = results.get("results", {}).get("bindings", [])
        
        if not bindings:
            response = {
                "base_node": base_node_id,
                "attribute": attribute_name,
                "value": attribute_value,
                "qualifiers": {},
                "status": f"No qualifiers found for {attribute_name} on {base_node_id}"
            }
            logger.warning(f"GetEdgeQualifiers: No qualifiers found")
            session_journal.failed_attempts.append(f"GetEdgeQualifiers({base_node_id}, {attribute_name}): No qualifiers")
            return json.dumps(response, indent=2)
        
        # Parse qualifiers
        attr_value = bindings[0].get("value", {}).get("value", attribute_value)
        qualifiers = {}
        entity_ids_to_resolve = []
        
        for binding in bindings:
            qkey = binding.get("qkey", {}).get("value", "")
            qval_node = binding.get("qval", {})
            qlabel = binding.get("qlabel", {}).get("value", None)
            
            qval_uri = qval_node.get("value", "")
            qval_type = qval_node.get("type", "literal")
            
            if qkey not in qualifiers:
                qualifiers[qkey] = []
            
            # Determine qualifier value type
            if qval_type == "uri" and not qlabel:
                # Entity ID without label - need to resolve
                entity_id = qval_uri.split("/")[-1]
                entity_ids_to_resolve.append(entity_id)
                qualifiers[qkey].append({
                    "entity_id": entity_id,
                    "type": "entity"
                })
            elif qlabel:
                # Entity with label already resolved
                entity_id = qval_uri.split("/")[-1]
                qualifiers[qkey].append({
                    "entity_id": entity_id,
                    "entity_label": qlabel,
                    "type": "entity"
                })
            else:
                # Literal value (string, date, number)
                qualifiers[qkey].append({
                    "value": qval_uri,
                    "type": "literal"
                })
        
        # Batch resolve entity IDs
        if entity_ids_to_resolve:
            batch_result = await BatchGetNodeLabels(app_context, entity_ids_to_resolve)
            batch_data = json.loads(batch_result)
            resolved = batch_data.get("resolved", {})
            
            # Inject labels
            for qkey, qvalues in qualifiers.items():
                for qval in qvalues:
                    if qval.get("type") == "entity" and "entity_label" not in qval:
                        entity_id = qval["entity_id"]
                        if entity_id in resolved:
                            qval["entity_label"] = resolved[entity_id]
        
        # Log to journal
        session_journal.verified_facts.append({
            "type": "edge_qualifiers",
            "node": base_node_id,
            "attribute": attribute_name,
            "qualifiers": list(qualifiers.keys())
        })
        
        response = {
            "base_node": base_node_id,
            "attribute": attribute_name,
            "value": attr_value,
            "qualifiers": qualifiers,
            "status": f"Found {len(qualifiers)} qualifier types"
        }
        
        logger.info(f"GetEdgeQualifiers: Found qualifiers for {base_node_id}.{attribute_name}")
        return json.dumps(response, indent=2)
        
    except Exception as e:
        error_msg = f"Error extracting edge qualifiers: {str(e)}"
        logger.error(error_msg)
        session_journal.failed_attempts.append(f"GetEdgeQualifiers: {str(e)}")
        return json.dumps({"error": error_msg}, indent=2)


# ============================================================================
# TOOL 5: GetQualifiersByPredicate
# ============================================================================

@mcp.tool()
async def GetQualifiersByPredicate(
    app_context: Context,
    base_node_id: str,
    relation_name: str,
    target_node_id: Optional[str] = None
) -> str:
    """
    Find ALL qualifiers for a specific RELATION statement (not attribute).
    
    **CRITICAL for Complex Award/Relation Questions:**
    - Use when questions ask about context of a relationship
    - Example: "What film won award X with winner Y?"
    - Automatically resolves entity IDs to labels
    
    Args:
        base_node_id: Source entity ID (e.g., "Q100")
        relation_name: Relation/predicate name (e.g., "award received")
        target_node_id: Optional target entity to filter by
    
    Returns:
        JSON with base_node, relation, target, qualifiers, and status
    """
    try:
        # Normalize relation name
        rel_normalized = relation_name.replace(" ", "_")
        
        # Build target filter
        target_filter = ""
        if target_node_id:
            target_filter = f"FILTER(?target = ex:{target_node_id})"
        
        query = f"""
        SELECT ?target ?targetLabel ?qkey ?qval ?qlabel WHERE {{
            ex:{base_node_id} prop:{rel_normalized} ?target .
            {target_filter}
            
            # Get target label
            OPTIONAL {{ ?target rdfs:label ?targetLabel . }}
            
            # Find the statement node
            ?stmt rdf:subject ex:{base_node_id} ;
                  rdf:predicate prop:{rel_normalized} ;
                  rdf:object ?target .
            
            # Get qualifiers
            ?stmt ?qprop ?qval .
            FILTER(STRSTARTS(STR(?qprop), "http://kqapro.org/qualifier/"))
            
            BIND(REPLACE(STR(?qprop), ".*/(.*)", "$1") AS ?qkey)
            
            # Try to resolve qualifier entity labels
            OPTIONAL {{
                ?qval rdfs:label ?qlabel .
            }}
        }}
        """
        
        full_query = SPARQL_PREFIXES + query
        app_context = app_context.request_context.lifespan_context
        app_context.sparql.setQuery(full_query)             # ✅ CORRECT!
        app_context.sparql.setReturnFormat(JSON)            # ✅ CORRECT!

        results = app_context.sparql.query().convert()      # ✅ CORRECT!
        bindings = results.get("results", {}).get("bindings", [])
        
        if not bindings:
            response = {
                "base_node": base_node_id,
                "relation": relation_name,
                "target": target_node_id,
                "qualifiers": {},
                "status": f"No qualifiers found for {relation_name} from {base_node_id}"
            }
            logger.warning(f"GetQualifiersByPredicate: No qualifiers found")
            session_journal.failed_attempts.append(f"GetQualifiersByPredicate({base_node_id}, {relation_name}): No qualifiers")
            return json.dumps(response, indent=2)
        
        # Parse results
        target_uri = bindings[0].get("target", {}).get("value", "")
        target_id = target_uri.split("/")[-1] if target_uri else target_node_id
        target_label = bindings[0].get("targetLabel", {}).get("value", target_id)
        
        qualifiers = {}
        entity_ids_to_resolve = []
        
        for binding in bindings:
            qkey = binding.get("qkey", {}).get("value", "")
            qval_node = binding.get("qval", {})
            qlabel = binding.get("qlabel", {}).get("value", None)
            
            qval_uri = qval_node.get("value", "")
            qval_type = qval_node.get("type", "literal")
            
            if qkey not in qualifiers:
                qualifiers[qkey] = []
            
            if qval_type == "uri" and not qlabel:
                entity_id = qval_uri.split("/")[-1]
                entity_ids_to_resolve.append(entity_id)
                qualifiers[qkey].append({
                    "entity_id": entity_id,
                    "type": "entity"
                })
            elif qlabel:
                entity_id = qval_uri.split("/")[-1]
                qualifiers[qkey].append({
                    "entity_id": entity_id,
                    "label": qlabel,
                    "type": "entity"
                })
            else:
                qualifiers[qkey].append({
                    "value": qval_uri,
                    "type": "literal"
                })
        
        # Batch resolve entity IDs
        if entity_ids_to_resolve:
            batch_result = await BatchGetNodeLabels(app_context, entity_ids_to_resolve)
            batch_data = json.loads(batch_result)
            resolved = batch_data.get("resolved", {})
            
            for qkey, qvalues in qualifiers.items():
                for qval in qvalues:
                    if qval.get("type") == "entity" and "label" not in qval:
                        entity_id = qval["entity_id"]
                        if entity_id in resolved:
                            qval["label"] = resolved[entity_id]
        
        # Log to journal
        session_journal.verified_facts.append({
            "type": "relation_qualifiers",
            "node": base_node_id,
            "relation": relation_name,
            "target": target_id,
            "qualifiers": list(qualifiers.keys())
        })
        
        response = {
            "base_node": base_node_id,
            "relation": relation_name,
            "target": target_id,
            "target_label": target_label,
            "qualifiers": qualifiers,
            "status": f"Found qualifiers for {relation_name} statement"
        }
        
        logger.info(f"GetQualifiersByPredicate: Found qualifiers for {base_node_id} -> {relation_name} -> {target_id}")
        return json.dumps(response, indent=2)
        
    except Exception as e:
        error_msg = f"Error extracting relation qualifiers: {str(e)}"
        logger.error(error_msg)
        session_journal.failed_attempts.append(f"GetQualifiersByPredicate: {str(e)}")
        return json.dumps({"error": error_msg}, indent=2)



@mcp.tool
@log_tool_duration
def ManageJournal(
    action: Literal["add_visited", "add_fact", "update_plan", "set_qtype", "set_target", "set_partial_answer", "read", "clear"],
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
            - "clear": Reset the journal to empty state
        content: The text content to add. Can be empty string for "read".

    Returns:
        The FULL current content of the journal to refresh your memory.
    """
    global session_journal

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

    elif action == "clear":
        session_journal = JournalState()

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


def _find_node_impl(semantic_node_name: str, context: Context) -> SearchResponse:
    """
    Internal implementation of FindNode.
    Can be called directly by other tools without going through MCP decorators.
    """
    app_context: AppContext = context.request_context.lifespan_context
    search_term_clean = semantic_node_name.strip()
    logger.info(f"FindNode: Searching for '{search_term_clean}'")

    # Debugging: Version prüfen
    try:
        logger.info(f"DEBUG: Qdrant Client Version: {qdrant_client.__version__}")
    except:
        pass

    exact_matches = []

    # --- PHASE 1: Exact Filter ---
    try:
        should_conditions = [
            models.FieldCondition(key="original_id", match=models.MatchValue(value=search_term_clean)),
            models.FieldCondition(key="name", match=models.MatchValue(value=search_term_clean)),
            models.FieldCondition(key="attributes.value.value", match=models.MatchValue(value=search_term_clean))
        ]
        filter_query = models.Filter(should=should_conditions)
        
        # .scroll() ist stabil
        scroll_results, _ = app_context.qdrant.scroll(
            collection_name=COLLECTION_ENTITIES,
            scroll_filter=filter_query,
            limit=5,
            with_payload=True
        )
        
        for point in scroll_results:
            payload = point.payload or {}
            attributes = payload.get("attributes", [])
            unique_attrs = sorted(list(set(a.get("key") for a in attributes if a.get("key"))))
            relations = payload.get("relations", [])
            unique_preds = sorted(list(set(r.get("predicate") for r in relations if r.get("predicate"))))
            
            exact_matches.append(NodeMatch(
                original_id=payload.get("original_id", "N/A"),
                name=payload.get("name", "Unknown"),
                node_type="entity",
                relevance_score=1.0,
                available_attributes=unique_attrs,
                available_predicates=unique_preds
            ))
    except Exception as e:
        logger.warning(f"FindNode: Phase 1 failed: {e}")

    # --- PHASE 2: Semantic Search ---
    vector = get_embedding(app_context.embedding_client, semantic_node_name)
    search_results = []

    try:
        # Nur Standard .search() nutzen - kein Legacy Fallback mehr!
        search_results = app_context.qdrant.search(
            collection_name=COLLECTION_ENTITIES,
            query_vector=vector,
            limit=TOP_N,
            with_payload=True,
            score_threshold=SCORE_THRESHHOLD
        )
    except Exception as e:
        logger.error(f"FindNode: Semantic search failed: {e}")
        search_results = []

    # Mapping logic
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
            node_type="unknown",
            relevance_score=point.score,
            available_attributes=unique_attrs,
            available_predicates=unique_preds
        ))

    # Merge & Sort
    combined = {m.original_id: m for m in exact_matches}
    for m in semantic_matches:
        if m.original_id not in combined:
            combined[m.original_id] = m

    matches = sorted(combined.values(), key=lambda x: x.relevance_score, reverse=True)[:TOP_N]
    
    if matches:
        for m in matches[:5]:
            session_journal.visited_nodes[m.original_id] = m.name
        session_journal.completed_steps.append(f"Found {len(matches)} nodes for '{semantic_node_name}'")

    return SearchResponse(matches=matches, result_count=len(matches))


@mcp.tool
@log_tool_duration
def FindNode(semantic_node_name: str, context: Context) -> SearchResponse:
    """
    Performs a HYBRID search (Qdrant Filters + Semantic Vectors).
    Optimized for qdrant-client 1.16+.
    """
    return _find_node_impl(semantic_node_name, context)


@mcp.tool
@log_tool_duration
def GetNodeSummary(node_id: str, context: Context) -> Dict[str, Any]:
    """
    Get a comprehensive summary of a node with ALL its data in ONE call.

    🎯 Use this when you need to fully explore a node's attributes and relations.

    Instead of:
      1. FindNode → get node metadata
      2. For each attribute: GetAttributeDetails
      3. For each relation: GetRelationDetails

    This tool fetches EVERYTHING at once, dramatically reducing iterations.

    ✅ WHEN TO USE:
    - After finding a node, when you need to explore what data it has
    - When you need multiple attributes/relations from the same node
    - To get a complete picture before deciding what to query next

    📝 EXAMPLES:

    Example 1: Explore an entity
    Q: "Tell me about Barnstable County"
    → GetNodeSummary("Q54089")
    → Returns: All attributes (population, area, FIPS code, etc.) and relations

    Example 2: Multi-attribute question
    Q: "What is the population and area of Tokyo?"
    → GetNodeSummary("Q1490")
    → Get both population and area values in one call

    Args:
        node_id (str): The entity ID (e.g., "Q54089")

    Returns:
        Dict with:
        - node_id: The queried node
        - name: Human-readable name
        - node_type: "entity" or "concept"
        - attributes: Dict of {attribute_name: [values]}
        - relations: Dict of {relation_name: [related_node_ids]}
        - summary_stats: Counts of attributes and relations
        - status: Success/error message

    Example return:
    {
      "node_id": "Q54089",
      "name": "Barnstable County",
      "node_type": "entity",
      "attributes": {
        "population": ["215918", "214990"],
        "area": ["1000.5 square_kilometre"],
        "FIPS 6-4 (US counties)": ["25001"]
      },
      "relations": {
        "country": ["Q30"],
        "located in time zone": ["Q941"]
      },
      "summary_stats": {
        "attribute_count": 8,
        "relation_count": 5
      },
      "status": "Success"
    }
    """
    app_context: AppContext = context.request_context.lifespan_context
    sparql: SPARQLWrapper = app_context.sparql

    base_uri = format_entity_uri(node_id)
    logger.info(f"GetNodeSummary: Fetching complete data for {node_id}")

    # Query to get ALL attributes and relations in one SPARQL query
    query = f"""
    {SPARQL_PREFIXES}

    SELECT ?pred ?obj ?objValue ?objUnit WHERE {{
        # Get all predicates from this node
        {base_uri} ?pred ?obj .

        # Try to resolve blank nodes for attributes
        OPTIONAL {{
            ?obj rdf:value ?objValue .
            OPTIONAL {{ ?obj unit:unit ?objUnit }}
        }}
    }}
    """

    try:
        sparql.setQuery(query)
        results = sparql.query().convert()
        bindings = results.get("results", {}).get("bindings", [])

        # Also get node name from Qdrant
        node_name = session_journal.visited_nodes.get(node_id, node_id)

        # If not in journal, try to find it
        if node_name == node_id:
            search_result = _find_node_impl(node_id, context)
            if search_result.matches:
                node_name = search_result.matches[0].name
                node_type = search_result.matches[0].node_type
            else:
                node_type = "unknown"
        else:
            node_type = "entity"  # Assume entity if in journal

        # Organize results into attributes and relations
        attributes = {}
        relations = {}

        for binding in bindings:
            pred_uri = binding.get("pred", {}).get("value", "")

            # Skip RDF system predicates
            if pred_uri.startswith("http://www.w3.org/1999/02/22-rdf-syntax-ns#") or \
               pred_uri.startswith("http://www.w3.org/2000/01/rdf-schema#label"):
                continue

            # Extract predicate name
            pred_name = pred_uri.split("/")[-1]

            # Determine if this is an attribute or relation
            is_attribute = "/attribute/" in pred_uri
            is_relation = "/property/" in pred_uri

            # Get the object value
            if "objValue" in binding and binding["objValue"].get("value"):
                # Resolved blank node
                value = binding["objValue"]["value"]
                if "objUnit" in binding and binding["objUnit"].get("value"):
                    unit_uri = binding["objUnit"]["value"]
                    unit = unit_uri.split("/")[-1] if "/" in unit_uri else unit_uri
                    value = f"{value} {unit}"
            else:
                # Direct value or unresolved
                value = binding.get("obj", {}).get("value", "")

            # Categorize
            if is_attribute:
                if pred_name not in attributes:
                    attributes[pred_name] = []
                if value not in attributes[pred_name]:  # Avoid duplicates
                    attributes[pred_name].append(value)

            elif is_relation:
                # Extract entity ID if it's a URI
                if value.startswith("http://kqapro.org/entity/"):
                    value = value.split("/entity/")[-1]

                if pred_name not in relations:
                    relations[pred_name] = []
                if value not in relations[pred_name]:  # Avoid duplicates
                    relations[pred_name].append(value)

        logger.info(f"GetNodeSummary: Found {len(attributes)} attributes, {len(relations)} relations")

        # Auto-update journal with all discovered data
        if attributes:
            if node_id not in session_journal.found_values:
                session_journal.found_values[node_id] = {}

            for attr_name, attr_values in attributes.items():
                session_journal.found_values[node_id][attr_name] = attr_values

        session_journal.completed_steps.append(
            f"Explored {node_name}: {len(attributes)} attributes, {len(relations)} relations"
        )

        return {
            "node_id": node_id,
            "name": node_name,
            "node_type": node_type,
            "attributes": attributes,
            "relations": relations,
            "summary_stats": {
                "attribute_count": len(attributes),
                "relation_count": len(relations),
                "total_attribute_values": sum(len(v) for v in attributes.values()),
                "total_related_nodes": sum(len(v) for v in relations.values())
            },
            "status": "Success"
        }

    except Exception as e:
        logger.error(f"GetNodeSummary failed: {e}")
        session_journal.failed_attempts.append(
            f"GetNodeSummary({node_id}): {str(e)[:100]}"
        )
        return {
            "node_id": node_id,
            "name": node_name if 'node_name' in locals() else node_id,
            "node_type": "unknown",
            "attributes": {},
            "relations": {},
            "summary_stats": {"attribute_count": 0, "relation_count": 0},
            "status": f"Error: {str(e)}"
        }


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
    app_context: AppContext = context.request_context.lifespan_context
    sparql: SPARQLWrapper = app_context.sparql

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
def GetAttributeWithQualifiers(
    base_node_id: str,
    attribute_name: str,
    context: Context
) -> Dict[str, Any]:
    """
    Retrieves attribute values INCLUDING all their qualifiers (dates, locations, roles, etc.).

    🎯 This is the ONE-STOP tool for temporal queries and qualified facts.

    Instead of the old workflow:
      1. GetAttributeDetails → get bnode IDs
      2. For each bnode, call GetEdgeQualifiers

    You get EVERYTHING in ONE call with this tool.

    ✅ WHEN TO USE:
    - Questions with temporal constraints: "as of 2015", "in 2020", "on January 1st"
    - Questions about qualified facts: "population at a specific time"
    - Any attribute that might have context (date, location, role, determination method, etc.)
    - When you need to filter by qualifier value (e.g., find population value for 2015)

    📝 EXAMPLES:

    Example 1: Temporal population query
    Q: "What was the population of Barnstable County in 2015?"
    → GetAttributeWithQualifiers("Q54089", "population")
    → Returns all population values with their "point in time" qualifiers
    → You can then filter for date matching 2015

    Example 2: Director with time context
    Q: "When did Christopher Nolan direct Inception?"
    → GetAttributeWithQualifiers("Q25188", "director")
    → Returns directors with their "start time" qualifiers

    Example 3: Population with determination method
    Q: "What census data exists for Tokyo?"
    → GetAttributeWithQualifiers("Q1490", "population")
    → Returns population values with qualifiers like "determination method": "census"

    Args:
        base_node_id (str): The entity ID (e.g., "Q54089" for Barnstable County)
        attribute_name (str): The attribute name (e.g., "population", "director")

    Returns:
        Dict with:
        - node_id: The entity queried
        - attribute_name: The attribute queried
        - values: List of value objects, each containing:
          - value: The attribute value (numeric or string)
          - unit: (if applicable) The unit of measurement
          - qualifiers: Dict of qualifier names → values
            e.g., {"point in time": "2015-01-01", "determination method": "United States Census"}
        - status: Success/error message

    Example return value:
    {
      "node_id": "Q54089",
      "attribute_name": "population",
      "values": [
        {
          "value": "215918",
          "unit": "1",
          "qualifiers": {
            "point in time": "2015-01-01",
            "determination method": "United States Census"
          }
        },
        {
          "value": "214990",
          "unit": "1",
          "qualifiers": {
            "point in time": "2014-01-01"
          }
        }
      ],
      "status": "Found 2 value(s) with qualifiers"
    }

    💡 TIP: After getting the results, you can filter the values list by qualifier.
    For example, to find the 2015 population, look for the value where
    qualifiers["point in time"] starts with "2015".
    """
    app_context: AppContext = context.request_context.lifespan_context
    sparql: SPARQLWrapper = app_context.sparql

    # Format the entity URI
    base_uri = format_entity_uri(base_node_id)

    # Construct attribute URI
    sanitized_attr_name = attribute_name.replace(" ", "_")
    attr_uri = f"<http://kqapro.org/attribute/{sanitized_attr_name}>"

    logger.info(f"GetAttributeWithQualifiers: {base_uri} -> {attribute_name}")

    # Comprehensive query that gets values AND their qualifiers in one go
    query = f"""
    {SPARQL_PREFIXES}

    SELECT ?valueNode ?value ?numericValue ?unit ?qualPred ?qualVal WHERE {{
        {base_uri} {attr_uri} ?valueNode .

        # Resolve blank nodes (for quantities with units)
        OPTIONAL {{
            ?valueNode rdf:value ?numericValue .
            OPTIONAL {{ ?valueNode unit:unit ?unit }}
        }}

        # Get all qualifiers for this value
        OPTIONAL {{
            ?valueNode ?qualPred ?qualVal .
            # Filter out structural predicates
            FILTER(?qualPred != rdf:type && ?qualPred != rdf:value && ?qualPred != unit:unit)
        }}

        # Determine the actual value to return
        BIND(IF(BOUND(?numericValue), ?numericValue, ?valueNode) AS ?value)
    }}
    """

    try:
        sparql.setQuery(query)
        results = sparql.query().convert()
        bindings = results.get("results", {}).get("bindings", [])

        if not bindings:
            logger.info(f"No values found for {attribute_name} on {base_node_id}")
            return {
                "node_id": base_node_id,
                "attribute_name": attribute_name,
                "values": [],
                "status": f"No values found for attribute '{attribute_name}' on node {base_node_id}"
            }

        # Group results by value node (each value can have multiple qualifiers)
        value_groups = {}
        for binding in bindings:
            value_node = binding.get("valueNode", {}).get("value", "")

            # Initialize this value group if not seen
            if value_node not in value_groups:
                value_obj = {}

                # Get the actual value
                if "numericValue" in binding and binding["numericValue"].get("value"):
                    value_obj["value"] = binding["numericValue"]["value"]

                    # Add unit if present
                    if "unit" in binding and binding["unit"].get("value"):
                        unit_uri = binding["unit"]["value"]
                        value_obj["unit"] = unit_uri.split("/")[-1] if "/" in unit_uri else unit_uri
                    else:
                        value_obj["unit"] = ""

                    value_obj["resolved_from_bnode"] = value_node
                    logger.debug(f"Resolved blank node {value_node}: {value_obj['value']}")
                else:
                    # Direct value (not a blank node)
                    value_obj["value"] = binding.get("value", {}).get("value", "")

                value_obj["qualifiers"] = {}
                value_groups[value_node] = value_obj

            # Add qualifier if present
            if "qualPred" in binding and "qualVal" in binding:
                qual_pred_uri = binding["qualPred"]["value"]
                qual_val = binding["qualVal"]["value"]

                # Extract readable predicate name
                qual_pred_name = qual_pred_uri.split("/")[-1]

                # Handle common qualifiers specially for readability
                qualifier_mappings = {
                    "P585": "point in time",
                    "P459": "determination method",
                    "P580": "start time",
                    "P582": "end time",
                    "P276": "location",
                    "P805": "statement is subject of",
                    "P1932": "object has role",
                }

                display_name = qualifier_mappings.get(qual_pred_name, qual_pred_name)
                value_groups[value_node]["qualifiers"][display_name] = qual_val

        # Convert to list format
        values_list = list(value_groups.values())

        logger.info(f"Found {len(values_list)} value(s) with qualifiers for {attribute_name}")

        # AUTO-UPDATE JOURNAL
        if values_list:
            # Store in found_values
            if base_node_id not in session_journal.found_values:
                session_journal.found_values[base_node_id] = {}

            session_journal.found_values[base_node_id][attribute_name] = values_list

            # Log as verified facts
            for val in values_list[:3]:  # First 3 values
                fact_entry = {
                    "subject": base_node_id,
                    "attribute": attribute_name,
                    "value": val.get("value"),
                    "qualifiers": val.get("qualifiers", {}),
                    "source": "GetAttributeWithQualifiers"
                }
                if "unit" in val:
                    fact_entry["unit"] = val["unit"]

                session_journal.verified_facts.append(fact_entry)

            # Log completion
            node_name = session_journal.visited_nodes.get(base_node_id, base_node_id)
            session_journal.completed_steps.append(
                f"Retrieved {attribute_name} with qualifiers for {node_name}"
            )

            logger.info(f"Journal auto-updated: Stored {attribute_name} with qualifiers for {base_node_id}")

        return {
            "node_id": base_node_id,
            "attribute_name": attribute_name,
            "values": values_list,
            "status": f"Found {len(values_list)} value(s) with qualifiers"
        }

    except Exception as e:
        logger.error(f"GetAttributeWithQualifiers failed: {e}")
        session_journal.failed_attempts.append(
            f"GetAttributeWithQualifiers({base_node_id}, {attribute_name}): {str(e)[:100]}"
        )
        return {
            "node_id": base_node_id,
            "attribute_name": attribute_name,
            "values": [],
            "status": f"Error: {str(e)}"
        }


@mcp.tool
@log_tool_duration
def TemporalAttributeQuery(
    base_node_id: str,
    attribute_name: str,
    target_date: str,
    tolerance_days: int = 365,
    context: Context = None
) -> Dict[str, Any]:
    """
    Get an attribute value as of a specific date (temporal query made easy).

    🎯 This is the EASIEST way to answer "What was X's Y in 2015?" questions.

    Instead of:
      1. GetAttributeWithQualifiers → get all values with dates
      2. Manually filter by date
      3. Handle date parsing and comparison

    This tool does ALL of that for you automatically.

    ✅ WHEN TO USE:
    - Questions with specific dates: "on 2015-01-01", "in 2020", "as of January 1st"
    - Questions asking for values at a point in time
    - When you need the closest value to a specific date

    📝 EXAMPLES:

    Example 1: Population on specific date
    Q: "What was the population of Barnstable County on 2015-01-01?"
    → TemporalAttributeQuery("Q54089", "population", "2015-01-01")
    → Returns: The population value closest to that date

    Example 2: Value in a year (finds closest)
    Q: "What was Tokyo's population in 2020?"
    → TemporalAttributeQuery("Q1490", "population", "2020-01-01", tolerance_days=365)
    → Returns: Population value from 2020 (within 1 year tolerance)

    Example 3: Boolean check with date
    Q: "On 2015-01-01 was the population of X greater than 56000?"
    1. result = TemporalAttributeQuery("Q54089", "population", "2015-01-01")
    2. VerifyNumericCondition(result["value"], ">", "56000")

    Args:
        base_node_id (str): The entity ID (e.g., "Q54089")
        attribute_name (str): The attribute name (e.g., "population")
        target_date (str): Target date in ISO format (YYYY-MM-DD) or just year (YYYY)
        tolerance_days (int): Max days difference to accept (default: 365 days = 1 year)

    Returns:
        Dict with:
        - node_id: The entity queried
        - attribute_name: The attribute queried
        - target_date: The date you asked for
        - found_date: The actual date of the value found (might be slightly different)
        - value: The attribute value
        - unit: (if applicable) The unit
        - date_difference_days: How many days between target and found date
        - all_qualifiers: All other qualifiers for this value
        - status: Success/error message

    Example return:
    {
      "node_id": "Q54089",
      "attribute_name": "population",
      "target_date": "2015-01-01",
      "found_date": "2015-01-01",
      "value": "215918",
      "unit": "1",
      "date_difference_days": 0,
      "all_qualifiers": {"determination method": "United States Census"},
      "status": "Found exact match"
    }
    """
    from datetime import datetime, timedelta

    logger.info(f"TemporalAttributeQuery: {base_node_id} -> {attribute_name} @ {target_date}")

    # First, get all values with qualifiers using our other tool
    qualified_response = GetAttributeWithQualifiers(base_node_id, attribute_name, context)

    if not qualified_response.get("values"):
        return {
            "node_id": base_node_id,
            "attribute_name": attribute_name,
            "target_date": target_date,
            "status": f"No values found for '{attribute_name}' on node {base_node_id}"
        }

    # Parse target date (handle both YYYY and YYYY-MM-DD)
    try:
        if len(target_date) == 4:  # Just year
            target_dt = datetime(int(target_date), 1, 1)
        else:
            target_dt = datetime.fromisoformat(target_date.split("T")[0])
    except Exception as e:
        return {
            "node_id": base_node_id,
            "attribute_name": attribute_name,
            "target_date": target_date,
            "status": f"Invalid date format '{target_date}'. Use YYYY or YYYY-MM-DD"
        }

    # Find values with "point in time" qualifiers
    temporal_values = []
    for val_obj in qualified_response["values"]:
        qualifiers = val_obj.get("qualifiers", {})

        # Look for temporal qualifiers (point in time, start time, end time)
        date_str = qualifiers.get("point in time") or qualifiers.get("start time") or qualifiers.get("end time")

        if date_str:
            try:
                # Parse the qualifier date
                val_dt = datetime.fromisoformat(date_str.split("T")[0])

                # Calculate difference
                diff_days = abs((val_dt - target_dt).days)

                temporal_values.append({
                    "value_obj": val_obj,
                    "date": val_dt,
                    "date_str": date_str,
                    "diff_days": diff_days
                })
            except Exception as e:
                logger.warning(f"Could not parse date qualifier '{date_str}': {e}")
                continue

    if not temporal_values:
        # No temporal qualifiers found - return first value with a warning
        first_val = qualified_response["values"][0]
        return {
            "node_id": base_node_id,
            "attribute_name": attribute_name,
            "target_date": target_date,
            "value": first_val.get("value"),
            "unit": first_val.get("unit", ""),
            "all_qualifiers": first_val.get("qualifiers", {}),
            "status": "No temporal qualifiers found - returning first value (may not match date)"
        }

    # Sort by date difference (closest first)
    temporal_values.sort(key=lambda x: x["diff_days"])

    # Get the closest match
    best_match = temporal_values[0]

    # Check if within tolerance
    if best_match["diff_days"] > tolerance_days:
        return {
            "node_id": base_node_id,
            "attribute_name": attribute_name,
            "target_date": target_date,
            "found_date": best_match["date_str"],
            "value": best_match["value_obj"].get("value"),
            "unit": best_match["value_obj"].get("unit", ""),
            "date_difference_days": best_match["diff_days"],
            "all_qualifiers": best_match["value_obj"].get("qualifiers", {}),
            "status": f"Closest value is {best_match['diff_days']} days away (exceeds tolerance of {tolerance_days} days)"
        }

    # Found a match within tolerance
    val_obj = best_match["value_obj"]
    other_qualifiers = {k: v for k, v in val_obj.get("qualifiers", {}).items()
                       if k not in ["point in time", "start time", "end time"]}

    status_msg = "Found exact match" if best_match["diff_days"] == 0 else f"Found value {best_match['diff_days']} days away"

    result = {
        "node_id": base_node_id,
        "attribute_name": attribute_name,
        "target_date": target_date,
        "found_date": best_match["date_str"],
        "value": val_obj.get("value"),
        "unit": val_obj.get("unit", ""),
        "date_difference_days": best_match["diff_days"],
        "all_qualifiers": other_qualifiers,
        "status": status_msg
    }

    # Auto-update journal
    session_journal.verified_facts.append({
        "fact": f"{attribute_name} of {base_node_id} on {target_date} = {result['value']}",
        "source": "TemporalAttributeQuery"
    })

    logger.info(f"TemporalAttributeQuery: {status_msg}")
    return result


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
    app_context: AppContext = context.request_context.lifespan_context
    sparql: SPARQLWrapper = app_context.sparql

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


from qdrant_client import models
# Stelle sicher, dass du das importiert hast: 
# from qdrant_client import QdrantClient (nur für Typing nötig, nicht zur Laufzeit hier)

@mcp.tool
@log_tool_duration
def ExploreNeighborhood(base_node_id: str, semantic_relation_name: str, context: Context) -> NeighborhoodResponse:
    """
    **DEPRECATED: This tool will be removed in a future version.**

    This functionality is redundant. Use instead:
    - GetNodeSummary(node_id) to get ALL relations at once
    - GetRelationDetails(node_id, relation_name) for specific relations
    - FindNode already returns available_predicates for semantic matching

    Finds specific facts about a node by semantically matching relations and verifying them in the Graph DB.
    """
    logger.warning(
        f"⚠️ DEPRECATED: ExploreNeighborhood is deprecated and will be removed. "
        f"Use GetNodeSummary or GetRelationDetails instead."
    )

    app_context: AppContext = context.request_context.lifespan_context  # ✅ Named 'app_context'
    sparql: SPARQLWrapper = app_context.sparql
    
    # --- DEBUGGING START ---
    # Wir prüfen, was app_context.qdrant wirklich ist.
    client_type = type(app_context.sparql).__name__
    logger.info(f"🔍 Debug: app_context.qdrant is of type '{client_type}'")
    logger.info(f"🔍 Debug: Available methods: {[m for m in dir(app_context.sparql) if not m.startswith('_')][:5]}...")
    # --- DEBUGGING END ---

    # 1. Embed the relation query
    vector = get_embedding(app_context.embedding_client, semantic_relation_name)  # ✅ Using 'app_context'

    # 2. Find candidates in Qdrant
    # HINWEIS: Wenn dies fehlschlägt, ist app_context.qdrant falsch initialisiert (siehe unten).
    try:
        candidates = app_context.qdrant.search(
            collection_name=COLLECTION_RELATIONS,
            query_vector=vector,
            limit=TOP_N,
            with_payload=True
        )
    except AttributeError:
        # Fallback: Falls es ein LangChain-Objekt ist oder der Client 'query_points' nutzt
        logger.error(f"⚠️ 'search' method missing on {client_type}. Trying fallback/checking imports.")
        raise RuntimeError(
            f"Der Qdrant-Client ({client_type}) hat keine 'search'-Methode. "
            "Prüfe 'kqapro_server.py': Importiere 'from qdrant_client import QdrantClient'."
        )

    checked_log = []

    # Apply Entity formatting to the Subject
    base_uri = format_entity_uri(base_node_id)

    logger.info(f"Exploring {base_uri} for relation '{semantic_relation_name}'")

    # 3. Iterate and Verify via SPARQL
    for candidate in candidates:
        # Handle payload access safely (manche Clients geben dicts, manche Objekte zurück)
        payload = candidate.payload if hasattr(candidate, 'payload') else candidate
        if not isinstance(payload, dict): 
             # Falls es ein ScoredPoint Objekt ist
             payload = payload.dict() if hasattr(payload, 'dict') else {}

        predicate_raw = payload.get('predicate')

        if not predicate_raw:
            continue

        # Apply Property formatting to the Predicate
        pred_uri = format_property_uri(predicate_raw)

        score = candidate.score if hasattr(candidate, 'score') else 0.0
        checked_log.append(f"{predicate_raw} ({score:.2f})")

        # Construct SPARQL query
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

                # Auto-update Journal
                fact_entry = {
                    "subject": base_node_id,
                    "predicate": predicate_raw,
                    "objects": objects_found[:5],
                    "confidence": score,
                    "source": "ExploreNeighborhood"
                }
                session_journal.verified_facts.append(fact_entry)

                node_name = session_journal.visited_nodes.get(base_node_id, base_node_id)
                session_journal.completed_steps.append(
                    f"Explored {node_name} -> {predicate_raw}: found {len(objects_found)} objects"
                )

                return NeighborhoodResponse(
                    base_node=base_node_id,
                    verified_match=RelationMatch(
                        predicate_used=predicate_raw,
                        semantic_label=predicate_raw,
                        objects=objects_found,
                        confidence=score
                    ),
                    candidates_checked=checked_log,
                    status="Match Found"
                )

        except Exception as e:
            logger.warning(f"SPARQL Error checking {pred_uri}: {e}")
            continue

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
def FindEntitiesByRelationPath(
    start_node_id: str,
    relation_path: list[Dict[str, str]],
    context: Context
) -> Dict[str, Any]:
    """
    Navigate multi-hop relation paths to find connected entities.

    🎯 Use this for questions that require following relationships across multiple nodes.

    Instead of:
      1. GetRelationDetails(A, "rel1") → get B
      2. GetRelationDetails(B, "rel2") → get C
      3. GetRelationDetails(C, "rel3") → get D

    This tool does multi-hop navigation in ONE SPARQL query.

    ✅ WHEN TO USE:
    - Questions with nested relationships: "X's Y's Z"
    - Multi-hop queries: "county that borders the county that borders X"
    - Finding entities via indirect connections

    📝 EXAMPLES:

    Example 1: Two-hop navigation
    Q: "What is the local dialing code of the county town that is the formation location of Marillion?"
    Step 1: Marillion → location_of_formation → ?location
    Step 2: ?location → local_dialing_code → ?code

    → FindEntitiesByRelationPath("Q678410", [
        {"relation": "location of formation", "direction": "forward"},
      ])
    → Then get attribute from the result

    Example 2: Border relationships
    Q: "Which counties border the county that borders Cecil County?"

    → FindEntitiesByRelationPath("Q385365", [
        {"relation": "shares border with", "direction": "forward"},
        {"relation": "shares border with", "direction": "forward"}
      ])
    → Returns all counties 2 hops away via "shares border with"

    Example 3: Ownership chain
    Q: "What companies are owned by companies owned by X?"

    → FindEntitiesByRelationPath("Q23844", [
        {"relation": "owner of", "direction": "forward"},
        {"relation": "owner of", "direction": "forward"}
      ])

    Args:
        start_node_id (str): Starting entity ID (e.g., "Q678410")
        relation_path (list[Dict]): List of relation steps, each with:
          - relation (str): Relation name (e.g., "location of formation")
          - direction (str): "forward" (A→B) or "backward" (B→A)

    Returns:
        Dict with:
        - start_node_id: The starting entity
        - relation_path: The path you requested
        - entities_found: List of entity IDs reached at the end of the path
        - path_length: Number of hops
        - intermediate_nodes: Dict showing nodes at each hop (for debugging)
        - status: Success/error message

    Example return:
    {
      "start_node_id": "Q678410",
      "relation_path": [{"relation": "location of formation", "direction": "forward"}],
      "entities_found": ["Q213474"],
      "path_length": 1,
      "intermediate_nodes": {
        "hop_0": ["Q678410"],
        "hop_1": ["Q213474"]
      },
      "status": "Found 1 entity/entities"
    }
    """
    app_context: AppContext = context.request_context.lifespan_context
    sparql: SPARQLWrapper = app_context.sparql

    logger.info(f"FindEntitiesByRelationPath: Starting from {start_node_id}, {len(relation_path)} hops")

    # Build SPARQL query dynamically based on path
    start_uri = format_entity_uri(start_node_id)

    # Initialize query
    query_parts = [SPARQL_PREFIXES]

    # Build SELECT clause (select all intermediate variables)
    hop_vars = [f"?hop{i}" for i in range(len(relation_path) + 1)]
    select_clause = "SELECT DISTINCT " + " ".join(hop_vars)
    query_parts.append(select_clause)

    # Build WHERE clause
    where_clauses = []
    where_clauses.append(f"BIND({start_uri} AS ?hop0)")

    for i, step in enumerate(relation_path):
        relation_name = step["relation"]
        direction = step.get("direction", "forward")

        # Sanitize relation name and create URI
        sanitized_relation = relation_name.replace(" ", "_")
        relation_uri = f"<http://kqapro.org/property/{sanitized_relation}>"

        current_var = f"?hop{i}"
        next_var = f"?hop{i+1}"

        if direction == "forward":
            where_clauses.append(f"{current_var} {relation_uri} {next_var} .")
        else:  # backward
            where_clauses.append(f"{next_var} {relation_uri} {current_var} .")

    query_parts.append("WHERE {")
    query_parts.extend([f"  {clause}" for clause in where_clauses])
    query_parts.append("}")
    query_parts.append("LIMIT 100")

    full_query = "\n".join(query_parts)

    try:
        sparql.setQuery(full_query)
        results = sparql.query().convert()
        bindings = results.get("results", {}).get("bindings", [])

        if not bindings:
            logger.info("FindEntitiesByRelationPath: No entities found")
            return {
                "start_node_id": start_node_id,
                "relation_path": relation_path,
                "entities_found": [],
                "path_length": len(relation_path),
                "intermediate_nodes": {},
                "status": "No entities found following this path"
            }

        # Extract entities at each hop
        intermediate_nodes = {f"hop_{i}": set() for i in range(len(relation_path) + 1)}

        for binding in bindings:
            for i in range(len(relation_path) + 1):
                hop_var = f"hop{i}"
                if hop_var in binding:
                    uri = binding[hop_var]["value"]
                    # Extract entity ID from URI
                    if "/entity/" in uri:
                        entity_id = uri.split("/entity/")[-1]
                    else:
                        entity_id = uri
                    intermediate_nodes[f"hop_{i}"].add(entity_id)

        # Convert sets to lists
        intermediate_nodes = {k: list(v) for k, v in intermediate_nodes.items()}

        # Final hop contains the target entities
        final_entities = intermediate_nodes[f"hop_{len(relation_path)}"]

        logger.info(f"FindEntitiesByRelationPath: Found {len(final_entities)} entities")

        # Auto-update journal
        session_journal.verified_facts.append({
            "fact": f"Multi-hop path from {start_node_id}: {len(final_entities)} entities found",
            "path": relation_path,
            "results": final_entities[:10],  # First 10
            "source": "FindEntitiesByRelationPath"
        })

        session_journal.completed_steps.append(
            f"Navigated {len(relation_path)}-hop path from {start_node_id}: found {len(final_entities)} entities"
        )

        return {
            "start_node_id": start_node_id,
            "relation_path": relation_path,
            "entities_found": final_entities,
            "path_length": len(relation_path),
            "intermediate_nodes": intermediate_nodes,
            "status": f"Found {len(final_entities)} entity/entities"
        }

    except Exception as e:
        logger.error(f"FindEntitiesByRelationPath failed: {e}")
        session_journal.failed_attempts.append(
            f"FindEntitiesByRelationPath({start_node_id}): {str(e)[:100]}"
        )
        return {
            "start_node_id": start_node_id,
            "relation_path": relation_path,
            "entities_found": [],
            "path_length": len(relation_path),
            "intermediate_nodes": {},
            "status": f"Error: {str(e)}"
        }


@mcp.tool
@log_tool_duration
def RunSPARQL(query: str, context: Context) -> SPARQLResponse:
    """
    Executes a SPARQL query against the KQAPro knowledge graph.

    ⚠️ IMPORTANT NAMESPACE RULES (REQUIRED FOR ALL QUERIES):
    The system auto-injects these prefixes - you MUST use them correctly:

    - Entities:    ex:Q12345        (e.g., ex:Q23844 for George Clooney)
    - Properties:  prop:P1082       (e.g., prop:P1082 for population relation)
    - Attributes:  attr:population  (e.g., attr:population for population attribute)
    - Qualifiers:  qual:P585        (e.g., qual:P585 for "point in time")
    - Units:       unit:minute      (e.g., unit:square_kilometre)
    - Standard:    rdf:value, rdfs:label

    ❌ DO NOT USE:
    - Bare IDs: Q54089 (WRONG - will cause syntax error)
    - Full URIs: <http://kqapro.org/entity/Q54089> (WRONG - unnecessary)
    - Wrong prefixes: wdt:, wd:, kqapro: (WRONG - these don't exist)

    📚 COMMON QUERY PATTERNS (COPY THESE):

    1️⃣ Get all values for an attribute:
       SELECT ?value WHERE {
         ex:Q54089 attr:population ?value .
       }

    2️⃣ Get attribute values WITH qualifiers (FOR TEMPORAL QUERIES):
       SELECT ?value ?date WHERE {
         ex:Q54089 attr:population ?popNode .
         ?popNode rdf:value ?value .
         ?popNode qual:P585 ?date .
       }

    3️⃣ Filter by date/year:
       SELECT ?value WHERE {
         ex:Q54089 attr:population ?popNode .
         ?popNode rdf:value ?value .
         ?popNode qual:P585 ?date .
         FILTER(YEAR(?date) = 2015)
       }

    4️⃣ Find entities by attribute value:
       SELECT ?entity WHERE {
         ?entity attr:FIPS_6-4_(US_counties) "24031" .
       }

    5️⃣ Multi-hop relation navigation:
       SELECT ?dialingCode WHERE {
         ex:Q678410 prop:location_of_formation ?location .
         ?location attr:local_dialing_code ?dialingCode .
       }

    6️⃣ Count with filters:
       SELECT (COUNT(?business) as ?count) WHERE {
         ?business prop:owned_by ex:Q23844 .
         ?business prop:parent_organization ex:Q1164779 .
       }

    ⚠️ BEFORE USING THIS TOOL:
    - Try GetAttributeDetails, GetRelationDetails, or FindNode first
    - Use GetAttributeWithQualifiers for temporal queries (easier than SPARQL)
    - Only use RunSPARQL for complex queries those tools cannot handle
    - If you get a syntax error, use the simpler tools instead

    🔧 SYNTAX ERROR RECOVERY:
    If you get "Error calling tool 'RunSPARQL'":
    - Your query had syntax issues (usually wrong namespace prefix)
    - Check you're using ex:, prop:, attr:, qual:, unit: correctly
    - Try using GetAttributeDetails or GetRelationDetails instead
    - Don't retry the same failed query - use a different approach

    Args:
        query (str): A valid SPARQL SELECT query using the prefixes above.

    Returns:
        SPARQLResponse: The structured results containing variables and bindings.
    """
    app_context: AppContext = context.request_context.lifespan_context
    sparql: SPARQLWrapper = app_context.sparql

    # Auto-inject prefixes
    full_query = query
    if "PREFIX" not in query:
        full_query = f"{SPARQL_PREFIXES}\n{query}"

    logger.info(f"RunSPARQL: Executing query (length: {len(full_query)} chars)")
    
    # FIX: Sicherer Log-Aufruf für Queries mit geschweiften Klammern
    logger.debug("RunSPARQL: Full query:\n{}", full_query)

    try:
        sparql.setQuery(full_query)
        logger.debug("RunSPARQL: Query set, executing...")
        
        raw_results = sparql.query().convert()
        logger.debug("RunSPARQL: Query executed successfully")

        head_vars = raw_results.get("head", {}).get("vars", [])
        bindings = raw_results.get("results", {}).get("bindings", [])
        logger.info(f"RunSPARQL: Query returned {len(bindings)} result(s) with variables: {head_vars}")

        simplified_rows = []
        for row in bindings:
            simple_row = {}
            for var in head_vars:
                if var in row:
                    simple_row[var] = row[var]["value"]
            simplified_rows.append(simple_row)

        if bindings:
            fact_entry = {
                "query_type": "SPARQL",
                "variables": head_vars,
                "result_count": len(simplified_rows),
                "results": simplified_rows[:10],
                "source": "RunSPARQL"
            }
            session_journal.verified_facts.append(fact_entry)
            session_journal.completed_steps.append(
                f"Executed SPARQL query: {len(simplified_rows)} results"
            )
            logger.info(f"Journal auto-updated: Stored {len(simplified_rows)} SPARQL results")

        return SPARQLResponse(
            vars=head_vars,
            bindings=simplified_rows,
            raw_json=raw_results
        )

    except Exception as e:
        logger.error(f"RunSPARQL: SPARQL Execution Error: {e}")
        # Auch hier sicheres Logging im Fehlerfall
        logger.error("RunSPARQL: Failed query was:\n{}", full_query)

        session_journal.failed_attempts.append(f"RunSPARQL failed: {str(e)[:100]}")

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
    app_context: AppContext = context.request_context.lifespan_context
    sparql: SPARQLWrapper = app_context.sparql

    logger.info(f"FindByAttribute: Searching for entities with {attribute_name}={value}")

    # Sanitize attribute name for URI
    sanitized_attr_name = attribute_name.replace(" ", "_")
    attr_uri = f"<http://kqapro.org/attribute/{sanitized_attr_name}>"

    # Build SPARQL query to find entities with this attribute value
    # We need to handle both direct values and blank nodes

    safe_value = value.replace('"', '\\"')

    query = f"""
    {SPARQL_PREFIXES}

    SELECT DISTINCT ?entity ?entityName WHERE {{
        ?entity {attr_uri} ?attrValue .
        OPTIONAL {{ ?entity rdfs:label ?entityName }}

        FILTER(
            STR(?attrValue) = "{safe_value}" ||  # HIER: Anführungszeichen waren wichtig!
            # ?attrValue = <{value}> ||  <-- DAS LÖSCHEN! Das verursacht den Syntaxfehler bei Strings mit Leerzeichen!
            
            # Nur checken wenn es wie eine URI aussieht (keine Leerzeichen)
            ( !CONTAINS("{safe_value}", " ") && ?attrValue = <{NS_ENTITY}{safe_value}> ) ||
            
            EXISTS {{
                ?attrValue rdf:value ?numVal .
                FILTER(STR(?numVal) = "{safe_value}")
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
    value1: str | int | float,
    operator: Literal["<", ">", "<=", ">=", "==", "!="],
    value2: str | int | float,
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
    # Coerce numeric inputs to str (LLMs often send int/float instead of str)
    value1 = str(value1)
    value2 = str(value2)

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
    app_context: AppContext = context.request_context.lifespan_context
    sparql: SPARQLWrapper = app_context.sparql

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
