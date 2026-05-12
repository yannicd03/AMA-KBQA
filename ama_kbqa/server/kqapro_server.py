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
import re
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
NS_QUALIFIER = "http://kqapro.org/qualifier/"

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


def _resolve_app_context(context: Any) -> AppContext:
    """Accept either a FastMCP Context or the already-unwrapped lifespan context."""
    request_context = getattr(context, "request_context", None)
    if request_context is not None:
        return request_context.lifespan_context
    return context


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


class StringComparisonResponse(BaseModel):
    """Response from deterministic string comparison verification."""
    verdict: Literal["TRUE", "FALSE", "ERROR"] = Field(..., description="The comparison result.")
    explanation: str = Field(..., description="Human-readable explanation of the comparison.")
    value1: str = Field(..., description="First value (as provided).")
    value2: str = Field(..., description="Second value (as provided).")
    mode: str = Field(..., description="Comparison mode used.")


from ama_kbqa.framework.state import JournalState


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


def _attr_condition_sparql(
    attribute_name: str,
    attribute_value: str,
    operator: str,
    suffix: str,
    entity_var: str = "?entity",
) -> str:
    """Build a SPARQL WHERE-block for one attribute condition on `entity_var`.

    Handles the three KQAPro attribute storage shapes:
    - direct literal: `?e attr:k <literal>`
    - quantity blank node: `?e attr:k _:b ; _:b rdf:value ?v ; _:b unit:unit ?u`
    - typed (xsd:date / xsd:gYear / xsd:decimal) literals

    Returns a single block of SPARQL (no surrounding braces) suitable for embedding
    inside a `{ ... }` group. Variable names are uniquified with `suffix` so several
    blocks can be UNION'd safely.
    """
    import re

    sanitized_attr = attribute_name.replace(" ", "_")
    attr_uri = f"<http://kqapro.org/attribute/{sanitized_attr}>"
    safe_value = attribute_value.replace('"', '\\"')

    is_date = bool(re.match(r"^\d{4}-\d{2}-\d{2}$", attribute_value))
    is_year = bool(re.match(r"^\d{4}$", attribute_value)) and not is_date
    is_numeric = False
    if not is_date and not is_year and attribute_value:
        try:
            float(attribute_value.replace(",", ""))
            is_numeric = True
        except ValueError:
            pass

    sparql_op = {"=": "=", "!=": "!=", "<": "<", ">": ">", "<=": "<=", ">=": ">="}.get(operator, "=")

    if not attribute_value:
        # Existence-only check
        return f"{entity_var} {attr_uri} ?attrVal{suffix} .\n"

    if operator == "contains":
        return (
            f"{entity_var} {attr_uri} ?attrVal{suffix} .\n"
            f'FILTER(CONTAINS(LCASE(STR(?attrVal{suffix})), LCASE("{safe_value}")))\n'
        )
    if operator == "=" and not is_numeric and not is_date and not is_year:
        return (
            f"{entity_var} {attr_uri} ?attrVal{suffix} .\n"
            f'FILTER(STR(?attrVal{suffix}) = "{safe_value}")\n'
        )
    if is_date:
        return (
            f"{entity_var} {attr_uri} ?attrVal{suffix} .\n"
            f'FILTER(xsd:date(STR(?attrVal{suffix})) {sparql_op} "{safe_value}"^^xsd:date)\n'
        )

    numeric_val = attribute_value.replace(",", "")
    # A 4-digit value like "1990" is ambiguous: could be a year (xsd:gYear stored
    # attribute, e.g. "release year") or a number (xsd:decimal in a bnode quantity,
    # e.g. small population). We can't tell from the input alone, so emit a UNION
    # of both interpretations and let the data decide. The DATATYPE filter on each
    # branch keeps SPARQL from evaluating the wrong cast on the wrong literal.
    if is_year:
        # Year branch: extract the 4-digit year from xsd:gYear or xsd:date as an integer
        # (works because gYear's lexical form IS the 4-digit string, and xsd:date strings
        # start with YYYY which integer-cast picks up via SUBSTR). For non-year-typed
        # storage (decimal in a bnode), fall back to numeric comparison.
        return (
            f"{{\n"
            f"  {entity_var} {attr_uri} ?attrYear{suffix} .\n"
            f"  FILTER(DATATYPE(?attrYear{suffix}) = xsd:gYear || DATATYPE(?attrYear{suffix}) = xsd:date)\n"
            f"  FILTER(xsd:integer(SUBSTR(STR(?attrYear{suffix}), 1, 4)) {sparql_op} {numeric_val})\n"
            f"}} UNION {{\n"
            f"  {entity_var} {attr_uri} ?attrRaw{suffix} .\n"
            f"  OPTIONAL {{ ?attrRaw{suffix} rdf:value ?blankVal{suffix} }}\n"
            f"  BIND(COALESCE(?blankVal{suffix}, ?attrRaw{suffix}) AS ?attrVal{suffix})\n"
            f"  FILTER(DATATYPE(?attrVal{suffix}) != xsd:gYear && DATATYPE(?attrVal{suffix}) != xsd:date)\n"
            f"  FILTER(xsd:decimal(STR(?attrVal{suffix})) {sparql_op} {numeric_val})\n"
            f"}}\n"
        )
    # Pure numeric — handle bnode (rdf:value) and direct decimal literals
    return (
        f"{entity_var} {attr_uri} ?attrRaw{suffix} .\n"
        f"OPTIONAL {{ ?attrRaw{suffix} rdf:value ?blankVal{suffix} }}\n"
        f"BIND(COALESCE(?blankVal{suffix}, ?attrRaw{suffix}) AS ?attrVal{suffix})\n"
        f"FILTER(xsd:decimal(STR(?attrVal{suffix})) {sparql_op} {numeric_val})\n"
    )


def _concept_clause(concept: str, transitive: bool, entity_var: str = "?entity") -> str:
    """Build the rdf:type clause for a concept filter (direct or transitive).

    `concept` may be either:
    - a Q-ID like "Q5" (matched directly against `ex:Q5`), or
    - a human-readable label like "human" or "county of Pennsylvania" (resolved
      via `rdfs:label` inside the SPARQL).

    Underscores in labels are tolerated (converted back to spaces before label match).
    """
    if not concept:
        return ""
    type_path = "rdf:type/rdf:type*" if transitive else "rdf:type"

    import re as _re
    if _re.match(r"^[QP]\d+$", concept.strip()):
        # Direct Q-ID
        sanitized = concept.strip()
        return f"{entity_var} {type_path} <{NS_ENTITY}{sanitized}> .\n"

    # Label-based: match the concept's rdfs:label. Keep the label as a literal
    # (Virtuoso indexes labels) so we don't pay the FILTER cost for case folding.
    label = concept.replace("_", " ").replace('"', '\\"')
    return (
        f"{entity_var} {type_path} ?conceptIRI .\n"
        f'?conceptIRI rdfs:label "{label}" .\n'
    )


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


# LRU cache for embeddings to avoid redundant API calls
_embedding_cache: Dict[str, list] = {}
_EMBEDDING_CACHE_MAX = 256


def get_embedding(client: OpenAI, text: str) -> list[float]:
    text = text.replace("\n", " ")
    cache_key = text.strip().lower()

    if cache_key in _embedding_cache:
        return _embedding_cache[cache_key]

    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=[text],
        encoding_format="float"
    )
    embedding = response.data[0].embedding

    # Evict oldest entry if cache is full
    if len(_embedding_cache) >= _EMBEDDING_CACHE_MAX:
        oldest_key = next(iter(_embedding_cache))
        del _embedding_cache[oldest_key]

    _embedding_cache[cache_key] = embedding
    return embedding


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

        # Backfill: if any previously-stored relation target references this
        # node_id (e.g. GetRelationDetails stored "Q3012" before we knew it was
        # "Ulm"), upgrade those entries in place so the journal summary shows
        # the human label instead of the opaque ID.
        for _entity_id, _attrs in session_journal.found_values.items():
            for _attr_name, _attr_data in _attrs.items():
                if isinstance(_attr_data, list):
                    for _item in _attr_data:
                        if isinstance(_item, dict) and _item.get("related_id") == node_id:
                            _item["value"] = label
        
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

async def _batch_get_node_labels_impl(app_context: Any, node_ids: list[str]) -> str:
    app = _resolve_app_context(app_context)

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

        app.sparql.setQuery(full_query)
        app.sparql.setReturnFormat(JSON)

        results = app.sparql.query().convert()
        bindings = results.get("results", {}).get("bindings", [])

        resolved = {}
        found_ids = set()

        for binding in bindings:
            entity_uri = binding.get("entity", {}).get("value", "")
            label = binding.get("label", {}).get("value", None)

            entity_id = entity_uri.split("/")[-1]

            if label:
                resolved[entity_id] = label
                found_ids.add(entity_id)
                session_journal.visited_nodes[entity_id] = label
                for _attrs in session_journal.found_values.values():
                    for _attr_data in _attrs.values():
                        if isinstance(_attr_data, list):
                            for _item in _attr_data:
                                if isinstance(_item, dict) and _item.get("related_id") == entity_id:
                                    _item["value"] = label

        not_found = [nid for nid in unique_ids if nid not in found_ids]

        for nid in not_found:
            session_journal.add_failed_attempt(f"BatchGetNodeLabels: {nid} not found")

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
    return await _batch_get_node_labels_impl(app_context, node_ids)


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
        app_context = _resolve_app_context(app_context)
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
    attribute_name: str = "",
    attribute_value: Optional[str] = None,
    predicate: Optional[str] = None,
    target: Optional[str] = None,
    target_node_id: Optional[str] = None,
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
        if predicate and not attribute_name:
            normalized_target = target_node_id or target
            if normalized_target and re.match(r"^Q\d+$", normalized_target.strip()):
                return await _get_qualifiers_by_predicate_impl(
                    app_context,
                    base_node_id=base_node_id,
                    relation_name=predicate,
                    target_node_id=normalized_target.strip(),
                )
            response = {
                "base_node": base_node_id,
                "attribute": attribute_name,
                "relation_hint": predicate,
                "target": normalized_target,
                "qualifiers": {},
                "status": (
                    "GetEdgeQualifiers received relation-style arguments. "
                    "For relation qualifiers, call GetQualifiersByPredicate or "
                    "GetQualifierValue(subject_id, predicate, target_qid, qualifier_name, "
                    "predicate_type='relation')."
                ),
            }
            session_journal.add_failed_attempt(
                f"GetEdgeQualifiers({base_node_id}, {predicate}): relation-style call needs target Q-id"
            )
            return json.dumps(response, indent=2)

        if not attribute_name:
            return json.dumps({
                "error": (
                    "attribute_name is required for attribute qualifiers. "
                    "For relation qualifiers, pass predicate plus target Q-id, or use "
                    "GetQualifiersByPredicate/GetQualifierValue."
                )
            }, indent=2)

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
        app_context = _resolve_app_context(app_context)
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
            session_journal.add_failed_attempt(f"GetEdgeQualifiers({base_node_id}, {attribute_name}): No qualifiers")
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
            batch_result = await _batch_get_node_labels_impl(app_context, entity_ids_to_resolve)
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
        session_journal.add_failed_attempt(f"GetEdgeQualifiers: {str(e)}")
        return json.dumps({"error": error_msg}, indent=2)


# ============================================================================
# TOOL 5: GetQualifiersByPredicate
# ============================================================================

async def _get_qualifiers_by_predicate_impl(
    app_context: Any,
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
            session_journal.add_failed_attempt(f"GetQualifiersByPredicate({base_node_id}, {relation_name}): No qualifiers")
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
            batch_result = await _batch_get_node_labels_impl(app_context, entity_ids_to_resolve)
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
        session_journal.add_failed_attempt(f"GetQualifiersByPredicate: {str(e)}")
        return json.dumps({"error": error_msg}, indent=2)


@mcp.tool()
async def GetQualifiersByPredicate(
    app_context: Context,
    base_node_id: str,
    relation_name: str,
    target_node_id: Optional[str] = None
) -> str:
    """
    Find ALL qualifiers for a specific RELATION statement (not attribute).

    Use this for relationship metadata, for example award `for_work`, sports-team
    `number_of_matches`, or spouse `start_time` qualifiers.
    """
    return await _get_qualifiers_by_predicate_impl(
        app_context,
        base_node_id=base_node_id,
        relation_name=relation_name,
        target_node_id=target_node_id,
    )


# ============================================================================
# TOOL 6: GetQualifierValue
# ============================================================================

@mcp.tool()
async def GetQualifierValue(
    app_context: Context,
    subject_id: str,
    predicate: str,
    target: str,
    qualifier_name: str,
    predicate_type: Literal["auto", "relation", "attribute"] = "auto"
) -> str:
    """
    Project a SINGLE qualifier value from a specific (subject, predicate, target) statement.

    Use this for QueryAttrQualifier / QueryRelationQualifier when you already know which
    qualifier you want (e.g., "start time", "point in time", "location", "for work").
    Returns ONLY the requested qualifier value(s), not the full qualifier dict — fewer
    distractors than GetEdgeQualifiers / GetQualifiersByPredicate.

    **When to prefer this over GetEdgeQualifiers:**
    - "When did Mel Brooks marry Anne Bancroft?" → known statement, want only `start_time`
    - "Where was Sucker Punch released on 2011-03-31?" → known fact, want only `place_of_publication`
    - "For which film did X win award Y?" → known relation, want only `for_work`

    **Direction handling:** Tries forward (subject -predicate-> target) first; if no statement
    is found, automatically retries backward (target -predicate-> subject). The returned
    `direction` field tells you which matched.

    Args:
        subject_id: Source entity ID (e.g., "Q100").
        predicate: Predicate name — relation or attribute (e.g., "spouse", "publication date").
        target: Target — entity ID for a relation (e.g., "Q200"), or literal value for an
            attribute (e.g., "2011-03-31"). Auto-detected by default.
        qualifier_name: Specific qualifier to project (e.g., "start time", "point in time").
        predicate_type: Force "relation" or "attribute"; default "auto" tries relation first
            for entity-shaped targets, attribute first for literal-shaped targets.

    Returns:
        JSON: {subject_id, predicate, target, qualifier_name, values: [...], direction, status}
        where values is a list of {value, type, entity_id?, entity_label?} entries.
    """
    try:
        pred_normalized = predicate.replace(" ", "_")
        qual_normalized = qualifier_name.replace(" ", "_")
        qualifier_aliases = {
            "number_of_matches": ["number_of_matches_played/races/starts"],
            "matches_played": ["number_of_matches_played/races/starts"],
            "appearances": ["number_of_matches_played/races/starts"],
        }
        qualifier_variants = [qual_normalized]
        for alias in qualifier_aliases.get(qual_normalized, []):
            if alias not in qualifier_variants:
                qualifier_variants.append(alias)

        def _qualifier_term(name: str) -> str:
            if name.startswith("http"):
                return f"<{name}>"
            if "/" in name:
                return f"<{NS_QUALIFIER}{name}>"
            return f"qual:{name}"

        # Detect target shape: Q-id ⇒ entity (likely relation); otherwise literal (likely attribute).
        target_is_qid = bool(re.match(r"^Q\d+$", target.strip()))

        # Determine which predicate type(s) to try, and in which order.
        if predicate_type == "relation":
            modes_to_try = ["relation"]
        elif predicate_type == "attribute":
            modes_to_try = ["attribute"]
        else:  # "auto"
            modes_to_try = ["relation", "attribute"] if target_is_qid else ["attribute", "relation"]

        sparql_app_context = app_context.request_context.lifespan_context

        def run_query(mode: str, direction: str, qualifier_variant: str) -> list:
            """Build and run a single SPARQL projection. direction in {"forward","backward"}."""
            if mode == "relation":
                if direction == "forward":
                    subj_uri, obj_uri = f"ex:{subject_id}", f"ex:{target}"
                else:
                    subj_uri, obj_uri = f"ex:{target}", f"ex:{subject_id}"
                stmt_pattern = f"""
                    {subj_uri} prop:{pred_normalized} {obj_uri} .
                    ?stmt rdf:subject {subj_uri} ;
                          rdf:predicate prop:{pred_normalized} ;
                          rdf:object {obj_uri} .
                """
            else:  # attribute — direction is always forward (attributes don't reify backward)
                if direction == "backward":
                    return []
                safe_target = target.replace('"', '\\"')
                stmt_pattern = f"""
                    ex:{subject_id} attr:{pred_normalized} ?bnode .
                    ?bnode rdf:value ?targetValue .
                    FILTER(STR(?targetValue) = "{safe_target}")
                    ?stmt rdf:subject ex:{subject_id} ;
                          rdf:predicate attr:{pred_normalized} ;
                          rdf:object ?bnode .
                """

            query = f"""
            {SPARQL_PREFIXES}
            SELECT DISTINCT ?qval ?qlabel WHERE {{
                {stmt_pattern}
                ?stmt {_qualifier_term(qualifier_variant)} ?qraw .
                OPTIONAL {{ ?qraw rdf:value ?qrawValue . }}
                BIND(COALESCE(?qrawValue, ?qraw) AS ?qval)
                OPTIONAL {{ ?qval rdfs:label ?qlabel . }}
            }}
            LIMIT 50
            """
            sparql_app_context.sparql.setQuery(query)
            sparql_app_context.sparql.setReturnFormat(JSON)
            results = sparql_app_context.sparql.query().convert()
            return results.get("results", {}).get("bindings", [])

        # Try each mode × direction until we get bindings.
        bindings: list = []
        matched_mode = None
        matched_direction = None
        matched_qualifier = qual_normalized
        for mode in modes_to_try:
            for direction in ("forward", "backward"):
                for qualifier_variant in qualifier_variants:
                    bindings = run_query(mode, direction, qualifier_variant)
                    if bindings:
                        matched_mode = mode
                        matched_direction = direction
                        matched_qualifier = qualifier_variant
                        break
                if bindings:
                    break
            if bindings:
                break

        if not bindings:
            response = {
                "subject_id": subject_id,
                "predicate": predicate,
                "target": target,
                "qualifier_name": qualifier_name,
                "tried_qualifier_names": qualifier_variants,
                "values": [],
                "direction": None,
                "status": (
                    f"No '{qualifier_name}' qualifier found on any "
                    f"({subject_id}, {predicate}, {target}) statement (tried "
                    f"{', '.join(modes_to_try)}, both directions). The statement may "
                    f"not exist, or the qualifier name may differ — try "
                    f"GetEdgeQualifiers/GetQualifiersByPredicate to list available qualifiers."
                ),
            }
            session_journal.add_failed_attempt(
                f"GetQualifierValue({subject_id}, {predicate}, {target}, {qualifier_name}): no match"
            )
            return json.dumps(response, indent=2)

        # Parse bindings into typed values; collect entity IDs for batch label resolution.
        values: list = []
        entity_ids_to_resolve: list = []
        for binding in bindings:
            qval_node = binding.get("qval", {})
            qval_uri = qval_node.get("value", "")
            qval_type = qval_node.get("type", "literal")
            qlabel = binding.get("qlabel", {}).get("value")

            if qval_type == "uri":
                entity_id = qval_uri.split("/")[-1]
                entry = {"value": qval_uri, "type": "entity", "entity_id": entity_id}
                if qlabel:
                    entry["entity_label"] = qlabel
                else:
                    entity_ids_to_resolve.append(entity_id)
                values.append(entry)
            else:
                values.append({"value": qval_uri, "type": "literal"})

        if entity_ids_to_resolve:
            batch_result = await _batch_get_node_labels_impl(app_context, entity_ids_to_resolve)
            resolved = json.loads(batch_result).get("resolved", {})
            for entry in values:
                if entry.get("type") == "entity" and "entity_label" not in entry:
                    eid = entry["entity_id"]
                    if eid in resolved:
                        entry["entity_label"] = resolved[eid]

        if subject_id not in session_journal.found_values:
            session_journal.found_values[subject_id] = {}
        found_key = f"{predicate}.{qualifier_name}"
        session_journal.found_values[subject_id][found_key] = [
            {
                "value": entry.get("entity_label") or entry.get("value"),
                "related_id": entry.get("entity_id"),
                "type": entry.get("type"),
                "target": target,
                "direction": matched_direction,
                "matched_as": matched_mode,
                "matched_qualifier": matched_qualifier,
            }
            for entry in values
        ]

        session_journal.verified_facts.append({
            "type": "qualifier_value",
            "subject": subject_id,
            "predicate": predicate,
            "target": target,
            "qualifier": qualifier_name,
            "matched_qualifier": matched_qualifier,
            "value_count": len(values),
            "matched_as": matched_mode,
            "direction": matched_direction,
        })

        response = {
            "subject_id": subject_id,
            "predicate": predicate,
            "target": target,
            "qualifier_name": qualifier_name,
            "values": values,
            "direction": matched_direction,
            "matched_as": matched_mode,
            "status": f"Found {len(values)} value(s) for qualifier '{qualifier_name}'",
        }
        logger.info(
            f"GetQualifierValue: {subject_id}.{predicate}({target}).{qualifier_name} "
            f"-> {len(values)} value(s) [{matched_mode}/{matched_direction}]"
        )
        return json.dumps(response, indent=2)

    except Exception as e:
        error_msg = f"Error projecting qualifier value: {str(e)}"
        logger.error(error_msg)
        session_journal.add_failed_attempt(f"GetQualifierValue: {str(e)}")
        return json.dumps({"error": error_msg}, indent=2)



@mcp.tool
@log_tool_duration
def ManageJournal(
    action: Literal["add_visited", "add_fact", "update_plan", "set_question", "set_qtype", "set_target", "set_partial_answer", "read", "clear"],
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
            - "set_question": Store the original question text
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
        # Split on newlines to support multi-step plans
        session_journal.current_plan = [s.strip() for s in content.split("\n") if s.strip()] if content else []

    elif action == "set_question":
        session_journal.question_text = content

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
def GetJournalStateJSON(context: Context) -> str:
    """
    Return the structured journal state as a JSON string.

    Used by the frontend's live graph view to render the discovered
    subgraph (visited_nodes, verified_facts, found_values). Mirrors the
    contents of `GetJournalSummary` but as machine-readable JSON instead
    of a human-readable summary.

    Returns:
        JSON-encoded dict with keys: question_text, question_type,
        target_entities, visited_nodes, found_values, verified_facts,
        failed_attempts, current_plan, completed_steps, partial_answer,
        kg_name, created_at, updated_at.
    """
    return json.dumps(session_journal.model_dump(), default=str, ensure_ascii=False)


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
                            related_id = val_item.get("related_id")
                            val_str = val_item.get("value", "?")
                            if related_id:
                                label = session_journal.visited_nodes.get(related_id)
                                if label and not label.startswith("("):
                                    val_str = f"{label} ({related_id})"
                                else:
                                    val_str = related_id
                            unit_str = val_item.get("unit", "")
                            summary_lines.append(f"    ✓ {attr_name}: {val_str} {unit_str}".strip())
                        else:
                            summary_lines.append(f"    ✓ {attr_name}: {val_item}")
                else:
                    summary_lines.append(f"    ✓ {attr_name}: {attr_data}")
    else:
        summary_lines.append(f"\n⚠️  NO VALUES DISCOVERED YET - You need to call GetAttributeDetails!")

    # Verified facts (structured triples: subject -rel-> related_id/label)
    if session_journal.verified_facts:
        summary_lines.append(f"\n🔗 VERIFIED FACTS (subject -relation-> target):")
        for fact in session_journal.verified_facts[:15]:
            subj_id = fact.get("subject", "?")
            subj_name = session_journal.visited_nodes.get(subj_id, subj_id)
            rel = fact.get("relation", "?")
            related_id = fact.get("related_id", "?")
            related_name = session_journal.visited_nodes.get(related_id)
            if related_name and not related_name.startswith("("):
                target = f"{related_name} ({related_id})"
            else:
                target = related_id
            direction = fact.get("direction")
            arrow = "->" if direction != "reverse" else "<-"
            summary_lines.append(f"  • {subj_name} ({subj_id}) {arrow}[{rel}]-> {target}")
        if len(session_journal.verified_facts) > 15:
            summary_lines.append(f"  ... and {len(session_journal.verified_facts) - 15} more")

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

    # Dedup: check if this exact entity is already in visited_nodes
    for node_id, node_name in session_journal.visited_nodes.items():
        if node_name.lower() == search_term_clean.lower():
            logger.info(f"FindNode: Already found '{search_term_clean}' as {node_id} — returning cached result")
            return SearchResponse(
                matches=[NodeMatch(
                    original_id=node_id,
                    name=node_name,
                    node_type="entity",
                    relevance_score=1.0,
                    available_attributes=[],
                    available_predicates=[],
                )],
                result_count=1,
            )

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
        search_results = app_context.qdrant.query_points(
            collection_name=COLLECTION_ENTITIES,
            query=vector,
            limit=TOP_N,
            with_payload=True,
            score_threshold=SCORE_THRESHHOLD
        ).points
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
        session_journal.add_completed_step(f"Found {len(matches)} nodes for '{semantic_node_name}'")

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

        session_journal.add_completed_step(
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
        session_journal.add_failed_attempt(
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
            session_journal.add_completed_step(
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
        session_journal.add_failed_attempt(
            f"GetAttributeDetails({base_node_id}, {attribute_name}): {str(e)[:100]}"
        )
        return AttributeDetailsResponse(
            node_id=base_node_id,
            attribute_name=attribute_name,
            values=[],
            status=f"Error: {str(e)}"
        )


def _get_attribute_with_qualifiers_impl(base_node_id: str, attribute_name: str, sparql: SPARQLWrapper) -> Dict[str, Any]:
    """Internal implementation for GetAttributeWithQualifiers (callable from other tools)."""

    # Format the entity URI
    base_uri = format_entity_uri(base_node_id)

    # Construct attribute URI
    sanitized_attr_name = attribute_name.replace(" ", "_")
    attr_uri = f"<http://kqapro.org/attribute/{sanitized_attr_name}>"

    logger.info(f"GetAttributeWithQualifiers: {base_uri} -> {attribute_name}")

    # Comprehensive query that gets values AND their qualifiers in one go
    # Uses two methods: blank-node pattern AND reification pattern (KQAPro standard)
    query = f"""
    {SPARQL_PREFIXES}

    SELECT ?valueNode ?value ?numericValue ?unit ?qualPred ?qualVal ?reifQualPred ?reifQualVal WHERE {{
        {base_uri} {attr_uri} ?valueNode .

        # Resolve blank nodes (for quantities with units)
        OPTIONAL {{
            ?valueNode rdf:value ?numericValue .
            OPTIONAL {{ ?valueNode unit:unit ?unit }}
        }}

        # METHOD 1: Get qualifiers via blank-node pattern
        OPTIONAL {{
            ?valueNode ?qualPred ?qualVal .
            # Filter out structural predicates
            FILTER(?qualPred != rdf:type && ?qualPred != rdf:value && ?qualPred != unit:unit)
        }}

        # METHOD 2: Get qualifiers via RDF reification pattern (KQAPro standard)
        OPTIONAL {{
            ?stmt rdf:subject {base_uri} ;
                  rdf:predicate {attr_uri} ;
                  rdf:object ?valueNode .
            ?stmt ?reifQualPred ?reifQualVal .
            FILTER(STRSTARTS(STR(?reifQualPred), "http://kqapro.org/qualifier/"))
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

            # Add qualifier if present (METHOD 1: blank-node pattern)
            if "qualPred" in binding and "qualVal" in binding:
                qual_pred_uri = binding["qualPred"]["value"]
                qual_val = binding["qualVal"]["value"]

                # Extract readable predicate name
                qual_pred_name = qual_pred_uri.split("/")[-1]
                display_name = qualifier_mappings.get(qual_pred_name, qual_pred_name)
                value_groups[value_node]["qualifiers"][display_name] = qual_val

            # Add qualifier if present (METHOD 2: reification pattern)
            if "reifQualPred" in binding and "reifQualVal" in binding:
                reif_pred_uri = binding["reifQualPred"]["value"]
                reif_val = binding["reifQualVal"]["value"]

                # Extract readable predicate name from qualifier URI
                reif_pred_name = reif_pred_uri.split("/")[-1]
                display_name = qualifier_mappings.get(reif_pred_name, reif_pred_name)
                value_groups[value_node]["qualifiers"][display_name] = reif_val

        # Convert to list format
        values_list = list(value_groups.values())

        logger.info(f"Found {len(values_list)} value(s) with qualifiers for {attribute_name}")

        return {
            "node_id": base_node_id,
            "attribute_name": attribute_name,
            "values": values_list,
            "status": f"Found {len(values_list)} value(s) with qualifiers"
        }

    except Exception as e:
        logger.error(f"GetAttributeWithQualifiers failed: {e}")
        return {
            "node_id": base_node_id,
            "attribute_name": attribute_name,
            "values": [],
            "status": f"Error: {str(e)}"
        }


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

    result = _get_attribute_with_qualifiers_impl(base_node_id, attribute_name, sparql)

    # AUTO-UPDATE JOURNAL
    values_list = result.get("values", [])
    if values_list:
        if base_node_id not in session_journal.found_values:
            session_journal.found_values[base_node_id] = {}

        session_journal.found_values[base_node_id][attribute_name] = values_list

        for val in values_list[:3]:
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

        node_name = session_journal.visited_nodes.get(base_node_id, base_node_id)
        session_journal.add_completed_step(
            f"Retrieved {attribute_name} with qualifiers for {node_name}"
        )
        logger.info(f"Journal auto-updated: Stored {attribute_name} with qualifiers for {base_node_id}")

    if not values_list and "Error" in result.get("status", ""):
        session_journal.add_failed_attempt(
            f"GetAttributeWithQualifiers({base_node_id}, {attribute_name}): {result['status'][:100]}"
        )

    return result


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

    # Get all values with qualifiers using the internal helper (not the MCP-wrapped tool)
    app_context: AppContext = context.request_context.lifespan_context
    sparql: SPARQLWrapper = app_context.sparql
    qualified_response = _get_attribute_with_qualifiers_impl(base_node_id, attribute_name, sparql)

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

            # Also record as DISCOVERED VALUES so synthesis can see them.
            # Relation targets are answers just as much as attribute values are.
            if base_node_id not in session_journal.found_values:
                session_journal.found_values[base_node_id] = {}
            related_entries = []
            for triple in simplified_triples[:5]:
                related_id = triple.get("related_id")
                related_label = session_journal.visited_nodes.get(related_id)
                entry = {
                    "value": related_label if related_label and not related_label.startswith("(") else related_id,
                    "related_id": related_id,
                }
                if triple.get("direction"):
                    entry["direction"] = triple["direction"]
                related_entries.append(entry)
            existing = session_journal.found_values[base_node_id].get(relation_name)
            if isinstance(existing, list):
                session_journal.found_values[base_node_id][relation_name] = existing + related_entries
            else:
                session_journal.found_values[base_node_id][relation_name] = related_entries

            node_name = session_journal.visited_nodes.get(base_node_id, base_node_id)
            session_journal.add_completed_step(
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
        session_journal.add_failed_attempt(
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
    candidates = app_context.qdrant.query_points(
        collection_name=COLLECTION_RELATIONS,
        query=vector,
        limit=TOP_N,
        with_payload=True
    ).points

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
                session_journal.add_completed_step(
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

    session_journal.add_failed_attempt(
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

        session_journal.add_completed_step(
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
        session_journal.add_failed_attempt(
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

            # Store SPARQL results in found_values for persistence across context window
            sparql_count = sum(1 for k in session_journal.found_values if k.startswith("sparql_result_"))
            sparql_key = f"sparql_result_{sparql_count + 1}"
            session_journal.found_values[sparql_key] = {
                "query": query[:200],
                "results": simplified_rows[:10],
            }

            session_journal.add_completed_step(
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

        session_journal.add_failed_attempt(f"RunSPARQL failed: {str(e)[:100]}")

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
        session_journal.add_completed_step(
            f"Found {len(matches)} entities with {attribute_name}={value}"
        )

        logger.info(f"FindByAttribute: Found {len(matches)} entities")
        return SearchResponse(matches=matches, result_count=len(matches))

    except Exception as e:
        logger.error(f"FindByAttribute failed: {e}")
        session_journal.add_failed_attempt(
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
        session_journal.add_completed_step(f"Verified: {explanation}")

        return NumericComparisonResponse(
            verdict=verdict,
            explanation=explanation,
            value1=f"{num1}{unit_str}",
            value2=f"{num2}{unit_str}",
            operator=operator
        )

    except Exception as e:
        logger.error(f"VerifyNumericCondition failed: {e}")
        session_journal.add_failed_attempt(
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

        session_journal.add_completed_step(
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
        session_journal.add_failed_attempt(
            f"CompareEntities({attribute_name}): {str(e)[:100]}"
        )
        return CompareEntitiesResponse(
            attribute_name=attribute_name,
            results=[],
            sorted_by="error",
            status=f"Error: {str(e)}"
        )


@mcp.tool
@log_tool_duration
def FilterEntities(
    context: Context,
    concept: str = "",
    attribute_name: str = "",
    attribute_value: str = "",
    operator: Literal["=", "!=", "<", ">", "<=", ">=", "contains"] = "=",
    entity_ids: list[str] | None = None,
    limit: int = 50,
    or_conditions: list[dict] | None = None,
    transitive_concept: bool = False,
) -> SearchResponse:
    """
    Filter entities by concept type and/or attribute value conditions.

    This is the go-to tool for narrowing down entity sets. It replaces manual SPARQL
    construction for the most common filtering patterns: by type, by string/numeric/date
    attribute, or a combination of both.

    **When to use:**
    - "How many cities in Germany have population > 1M?" → FilterEntities(concept="city", attribute_name="population", attribute_value="1000000", operator=">")
    - "Which humans are members of Duran Duran?" → First get member IDs via GetRelationDetails, then FilterEntities(entity_ids=[...], concept="human")
    - "Find all films with duration > 120" → FilterEntities(concept="film", attribute_name="duration", attribute_value="120", operator=">")
    - "Pennsylvania counties with population > 7800 OR < 40M" → FilterEntities(concept="county of Pennsylvania", attribute_name="population", attribute_value="7800", operator=">", or_conditions=[{"attribute_name": "population", "attribute_value": "40000000", "operator": "<"}])
    - "All mammals" (subclass-aware) → FilterEntities(concept="mammal", transitive_concept=True)

    **Chaining:** Pass entity_ids from a previous tool call to further filter results.

    Args:
        concept: Entity type to filter by (e.g., "human", "city in New Jersey", "film"). Leave empty to skip type filtering.
        attribute_name: Attribute to filter on (e.g., "population", "duration"). Leave empty to skip attribute filtering.
        attribute_value: Value to compare against. Auto-detects type: dates (YYYY-MM-DD), numbers, or strings.
        operator: Comparison operator. Use "contains" for substring matching on strings.
        entity_ids: Optional list of entity IDs to restrict the search to (for chaining with other tools).
        limit: Maximum number of results to return.
        or_conditions: Optional list of additional attribute conditions to OR with the primary one.
            Each item: {"attribute_name": str, "attribute_value": str, "operator": str}.
            Use this for "X OR Y" filters (e.g., "population > 7800 or population < 40000000").
        transitive_concept: If True, match entities whose type is the given concept OR any
            descendant in the concept hierarchy (uses rdf:type/rdf:type*). Default False keeps
            the current direct-type behavior.

    Returns:
        SearchResponse: List of matching entities.
    """
    app_context: AppContext = context.request_context.lifespan_context
    sparql: SPARQLWrapper = app_context.sparql

    if not concept and not attribute_name and not or_conditions:
        return SearchResponse(matches=[], result_count=0)

    logger.info(
        f"FilterEntities: concept={concept}, attr={attribute_name}, val={attribute_value}, "
        f"op={operator}, or_n={len(or_conditions or [])}, transitive={transitive_concept}"
    )

    # Pre-restriction blocks (always apply, never inside the OR)
    pre_blocks: list[str] = []
    if entity_ids:
        entity_uris = " ".join([format_entity_uri(eid) for eid in entity_ids])
        pre_blocks.append(f"VALUES ?entity {{ {entity_uris} }}")
    concept_block = _concept_clause(concept, transitive_concept)
    if concept_block:
        pre_blocks.append(concept_block.rstrip())

    # Attribute condition(s)
    attr_block = ""
    if attribute_name or or_conditions:
        branches: list[str] = []
        if attribute_name:
            branches.append(_attr_condition_sparql(attribute_name, attribute_value, operator, "0"))
        for i, cond in enumerate(or_conditions or [], start=1):
            branches.append(_attr_condition_sparql(
                cond.get("attribute_name", ""),
                cond.get("attribute_value", ""),
                cond.get("operator", "="),
                str(i),
            ))
        if len(branches) == 1:
            attr_block = "{ " + branches[0] + " }"
        elif len(branches) > 1:
            attr_block = " UNION ".join("{ " + b + " }" for b in branches)

    label_block = "OPTIONAL { ?entity rdfs:label ?entityName }"

    where_body = "\n        ".join(filter(None, pre_blocks + [attr_block, label_block]))
    query = f"""
    {SPARQL_PREFIXES}
    SELECT DISTINCT ?entity ?entityName WHERE {{
        {where_body}
    }}
    LIMIT {limit}
    """

    try:
        sparql.setQuery(query)
        results = sparql.query().convert()
        bindings = results.get("results", {}).get("bindings", [])

        matches = []
        for binding in bindings:
            entity_uri = binding.get("entity", {}).get("value", "")
            entity_name = binding.get("entityName", {}).get("value", "Unknown")

            if "/entity/" in entity_uri:
                entity_id = entity_uri.split("/entity/")[-1]
            else:
                entity_id = entity_uri

            matches.append(NodeMatch(
                original_id=entity_id,
                name=entity_name,
                node_type="entity",
                relevance_score=1.0,
                available_attributes=[],
                available_predicates=[]
            ))

        # Auto-update journal
        filter_desc = []
        if concept:
            filter_desc.append(f"type={concept}")
        if attribute_name:
            filter_desc.append(f"{attribute_name}{operator}{attribute_value}")
        desc = ", ".join(filter_desc)

        for m in matches[:5]:
            session_journal.visited_nodes[m.original_id] = m.name
        session_journal.add_completed_step(
            f"FilterEntities({desc}): {len(matches)} results"
        )

        logger.info(f"FilterEntities: Found {len(matches)} entities")
        return SearchResponse(matches=matches, result_count=len(matches))

    except Exception as e:
        logger.error(f"FilterEntities failed: {e}")
        logger.error("FilterEntities: Failed query was:\n{}", query)
        session_journal.add_failed_attempt(
            f"FilterEntities({concept}, {attribute_name}): {str(e)[:100]}"
        )
        return SearchResponse(matches=[], result_count=0)


@mcp.tool
@log_tool_duration
def QualifierFilter(
    entity_ids: list[str],
    relation_or_attribute: str,
    qualifier_name: str,
    qualifier_value: str,
    context: Context,
    operator: Literal["=", "!=", "<", ">", "<=", ">="] = "=",
    limit: int = 50
) -> SearchResponse:
    """
    Filter entities by qualifier values on their statements (facts about facts).

    Use this when you need to narrow down entities based on WHEN, WHERE, or HOW
    a relationship or attribute holds. This replaces manual SPARQL for QFilter operations.

    **When to use:**
    - "Who married Mel Brooks starting in 1964?" → QualifierFilter(entity_ids=[spouse_ids], relation_or_attribute="spouse", qualifier_name="start time", qualifier_value="1964", operator="=")
    - "Which member of Duran Duran joined in 2001?" → QualifierFilter(entity_ids=[member_ids], relation_or_attribute="member of", qualifier_name="start time", qualifier_value="2001")

    **How it works:** Looks up reified statements (rdf:Statement) connecting the entity to the
    base predicate, then filters by the qualifier value on that statement.

    Args:
        entity_ids: List of entity IDs to filter (from a previous relation/filter call).
        relation_or_attribute: The base predicate name (e.g., "spouse", "member of", "population").
        qualifier_name: The qualifier to filter on (e.g., "start time", "point in time", "location").
        qualifier_value: The value to compare against. Supports dates (YYYY-MM-DD), years, and strings.
        operator: Comparison operator for the qualifier value.
        limit: Maximum number of results to return.

    Returns:
        SearchResponse: Entities whose statements match the qualifier condition.
    """
    app_context: AppContext = context.request_context.lifespan_context
    sparql_client: SPARQLWrapper = app_context.sparql

    if not entity_ids:
        return SearchResponse(matches=[], result_count=0)

    logger.info(f"QualifierFilter: {len(entity_ids)} entities, {relation_or_attribute}.{qualifier_name}{operator}{qualifier_value}")

    entity_uris = " ".join([format_entity_uri(eid) for eid in entity_ids])
    sanitized_pred = relation_or_attribute.replace(" ", "_")
    sanitized_qual = qualifier_name.replace(" ", "_")
    safe_value = qualifier_value.replace('"', '\\"')

    # Auto-detect value type for qualifier
    import re
    is_date = bool(re.match(r"^\d{4}-\d{2}-\d{2}$", qualifier_value))
    is_year = bool(re.match(r"^\d{4}$", qualifier_value))

    # Build qualifier filter
    if is_date:
        sparql_op = operator
        qual_filter = f'FILTER(xsd:date(STR(?qualVal)) {sparql_op} "{safe_value}"^^xsd:date)'
    elif is_year:
        sparql_op = operator
        qual_filter = f"FILTER(YEAR(xsd:date(STR(?qualVal))) {sparql_op} {safe_value})"
    else:
        if operator == "=":
            qual_filter = f'FILTER(STR(?qualVal) = "{safe_value}")'
        elif operator == "!=":
            qual_filter = f'FILTER(STR(?qualVal) != "{safe_value}")'
        else:
            # Try numeric for non-string operators
            try:
                float(qualifier_value.replace(",", ""))
                numeric_val = qualifier_value.replace(",", "")
                qual_filter = f"FILTER(xsd:decimal(STR(?qualVal)) {operator} {numeric_val})"
            except ValueError:
                qual_filter = f'FILTER(STR(?qualVal) = "{safe_value}")'

    # Try both prop: (relation) and attr: (attribute) predicates
    # The reified statement pattern uses pred:fact_h, pred:fact_r, pred:fact_t
    # But KQAPro uses standard rdf:Statement reification
    query = f"""
    {SPARQL_PREFIXES}
    SELECT DISTINCT ?entity ?entityName WHERE {{
        VALUES ?entity {{ {entity_uris} }}
        OPTIONAL {{ ?entity rdfs:label ?entityName }}

        ?stmt rdf:type rdf:Statement .
        {{
            ?stmt rdf:subject ?entity .
            ?stmt rdf:predicate prop:{sanitized_pred} .
        }} UNION {{
            ?stmt rdf:object ?entity .
            ?stmt rdf:predicate prop:{sanitized_pred} .
        }} UNION {{
            ?stmt rdf:subject ?entity .
            ?stmt rdf:predicate attr:{sanitized_pred} .
        }}

        ?stmt qual:{sanitized_qual} ?qualVal .
        {qual_filter}
    }}
    LIMIT {limit}
    """

    try:
        sparql_client.setQuery(query)
        results = sparql_client.query().convert()
        bindings = results.get("results", {}).get("bindings", [])

        matches = []
        for binding in bindings:
            entity_uri = binding.get("entity", {}).get("value", "")
            entity_name = binding.get("entityName", {}).get("value", "Unknown")

            if "/entity/" in entity_uri:
                entity_id = entity_uri.split("/entity/")[-1]
            else:
                entity_id = entity_uri

            matches.append(NodeMatch(
                original_id=entity_id,
                name=entity_name,
                node_type="entity",
                relevance_score=1.0,
                available_attributes=[],
                available_predicates=[]
            ))

        for m in matches[:5]:
            session_journal.visited_nodes[m.original_id] = m.name
        session_journal.add_completed_step(
            f"QualifierFilter({relation_or_attribute}.{qualifier_name}{operator}{qualifier_value}): {len(matches)} results"
        )

        logger.info(f"QualifierFilter: Found {len(matches)} matching entities")
        return SearchResponse(matches=matches, result_count=len(matches))

    except Exception as e:
        logger.error(f"QualifierFilter failed: {e}")
        logger.error("QualifierFilter: Failed query was:\n{}", query)
        session_journal.add_failed_attempt(
            f"QualifierFilter({qualifier_name}{operator}{qualifier_value}): {str(e)[:100]}"
        )
        return SearchResponse(matches=[], result_count=0)


@mcp.tool
@log_tool_duration
def VerifyString(
    value1: str,
    value2: str,
    mode: Literal["exact", "case_insensitive", "contains", "normalize"] = "normalize",
    context: Context = None
) -> StringComparisonResponse:
    """
    Performs deterministic string comparison between two values.

    This is the "String Judge" - it returns a definitive TRUE/FALSE verdict for string
    comparisons, analogous to VerifyNumericCondition but for text. Use this instead of
    guessing whether two strings match.

    **When to use:**
    - Any "Verify" question comparing string values (names, labels, categories)
    - "Is X's native name Y?" → VerifyString(actual_name, "Y")
    - "Is the capital of France Paris?" → VerifyString("Paris", "Paris")

    **Modes:**
    - "exact": Strict byte-for-byte equality
    - "case_insensitive": Lowered comparison
    - "contains": TRUE if either string contains the other
    - "normalize" (default): Strips whitespace, lowercases, removes diacritics and
      punctuation — best for KG answer matching where formatting varies

    Args:
        value1: First string value (typically the value retrieved from the KG).
        value2: Second string value (typically the expected/question value).
        mode: Comparison mode.

    Returns:
        StringComparisonResponse: Verdict (TRUE/FALSE/ERROR) with explanation.
    """
    logger.info(f"VerifyString: '{value1}' vs '{value2}' (mode={mode})")

    try:
        if mode == "exact":
            result = value1 == value2
            explanation = f"Exact comparison: '{value1}' == '{value2}' → {result}"

        elif mode == "case_insensitive":
            result = value1.lower() == value2.lower()
            explanation = f"Case-insensitive: '{value1.lower()}' == '{value2.lower()}' → {result}"

        elif mode == "contains":
            v1_lower = value1.lower()
            v2_lower = value2.lower()
            result = v1_lower in v2_lower or v2_lower in v1_lower
            explanation = f"Contains check: '{value1}' ↔ '{value2}' → {result}"

        elif mode == "normalize":
            import unicodedata
            import re

            def normalize(s: str) -> str:
                # Strip whitespace
                s = s.strip()
                # Normalize unicode (decompose diacritics)
                s = unicodedata.normalize("NFKD", s)
                # Remove diacritical marks
                s = "".join(c for c in s if not unicodedata.combining(c))
                # Lowercase
                s = s.lower()
                # Remove punctuation except hyphens (important for compound names)
                s = re.sub(r"[^\w\s-]", "", s)
                # Collapse whitespace
                s = re.sub(r"\s+", " ", s).strip()
                return s

            norm1 = normalize(value1)
            norm2 = normalize(value2)
            result = norm1 == norm2
            explanation = f"Normalized: '{norm1}' == '{norm2}' → {result}"
        else:
            return StringComparisonResponse(
                verdict="ERROR",
                explanation=f"Unknown mode: {mode}",
                value1=value1,
                value2=value2,
                mode=mode
            )

        verdict = "TRUE" if result else "FALSE"

        # Auto-update journal
        session_journal.verified_facts.append({
            "fact": explanation,
            "source": "VerifyString"
        })
        session_journal.add_completed_step(f"Verified: {explanation}")

        logger.info(f"VerifyString result: {verdict}")
        return StringComparisonResponse(
            verdict=verdict,
            explanation=explanation,
            value1=value1,
            value2=value2,
            mode=mode
        )

    except Exception as e:
        logger.error(f"VerifyString failed: {e}")
        session_journal.add_failed_attempt(
            f"VerifyString('{value1}' vs '{value2}'): {str(e)[:100]}"
        )
        return StringComparisonResponse(
            verdict="ERROR",
            explanation=f"Could not compare values: {str(e)}",
            value1=value1,
            value2=value2,
            mode=mode
        )


class CountResponse(BaseModel):
    """Response from a Count tool."""
    count: int = Field(..., description="The exact count (no truncation).")
    description: str = Field(..., description="Human-readable description of what was counted.")
    status: str = Field(..., description="Status message.")


class SelectExtremeResponse(BaseModel):
    """Response from SelectExtreme (argmax/argmin)."""
    mode: Literal["max", "min"] = Field(..., description="Which extreme was selected.")
    attribute_name: str = Field(..., description="The attribute used for ranking.")
    results: list[ComparisonResult] = Field(
        default_factory=list,
        description="Top-k entities ordered by extreme. Index 0 is the winner."
    )
    status: str = Field(..., description="Status message.")


class VerifyFactResponse(BaseModel):
    """Response from VerifyFact (boolean fact existence check)."""
    verdict: Literal["TRUE", "FALSE", "ERROR"] = Field(..., description="Whether the fact exists in the KB.")
    explanation: str = Field(..., description="Human-readable explanation.")
    subject_id: str = Field(..., description="Subject queried.")
    predicate: str = Field(..., description="Predicate (relation or attribute) checked.")
    target: str = Field(..., description="Target entity ID or literal value checked.")
    matched_as: Literal["relation", "attribute", "none", "error"] = Field(
        ..., description="How the predicate was matched (or none if not found)."
    )


class RelationBetweenResponse(BaseModel):
    """Response from GetRelationBetween."""
    subject_id: str = Field(..., description="Question-order subject entity.")
    object_id: str = Field(..., description="Question-order object entity.")
    subject_to_object: list[str] = Field(default_factory=list, description="Predicate labels from subject to object.")
    object_to_subject: list[str] = Field(default_factory=list, description="Predicate labels from object to subject.")
    preferred_answer: str = Field(default="", description="Best predicate label for the question order.")
    status: str = Field(..., description="Status message.")


@mcp.tool
@log_tool_duration
def CountEntities(
    context: Context,
    concept: str = "",
    attribute_name: str = "",
    attribute_value: str = "",
    operator: Literal["=", "!=", "<", ">", "<=", ">=", "contains"] = "=",
    entity_ids: list[str] | None = None,
    or_conditions: list[dict] | None = None,
    not_conditions: list[dict] | None = None,
    transitive_concept: bool = False,
) -> CountResponse:
    """
    Return the EXACT number of entities matching a concept and/or attribute condition(s).

    🎯 Use this for ANY "How many X?" question. It returns a single integer with **no
    truncation** — never use FilterEntities + len() to count, because FilterEntities is
    capped at limit=50 and will silently lie about the true count above that.

    **When to use:**
    - "How many countries are in the EU?" → CountEntities(concept="country", or via member-of relation)
    - "How many films did Nolan direct?" → first get film IDs via GetRelationDetails, then CountEntities(entity_ids=[...])
    - "How many Pennsylvania counties have population > 7800 or < 40M?" → CountEntities(concept="county of Pennsylvania", attribute_name="population", attribute_value="7800", operator=">", or_conditions=[{"attribute_name": "population", "attribute_value": "40000000", "operator": "<"}])
    - "How many mammal species are there?" → CountEntities(concept="mammal", transitive_concept=True)
    - "How many TV series were NOT started in 2005?" → CountEntities(concept="TV series", not_conditions=[{"attribute_name": "start_time", "attribute_value": "2005", "operator": "="}])

    Args:
        concept: Entity type to count. Leave empty to skip type filtering.
        attribute_name: Primary attribute condition (paired with attribute_value/operator).
        attribute_value: Value to compare against (auto-detects date/year/numeric/string).
        operator: Comparison operator for the primary condition.
        entity_ids: Optional list of entity IDs to restrict the count to.
        or_conditions: Additional attribute conditions OR'd with the primary. Same shape as
            FilterEntities. Each item: {"attribute_name", "attribute_value", "operator"}.
        not_conditions: Conditions to EXCLUDE — entities matching any of these are subtracted
            from the count, expressed as `FILTER NOT EXISTS { ... }` per condition. Same item
            shape as `or_conditions`. Use for "how many X NOT Y" / "how many X that are not Z"
            questions instead of writing FILTER NOT EXISTS by hand in RunSPARQL. The semantics
            are "the entity does not have ANY value of attribute_name matching the condition";
            entities that lack the attribute entirely are KEPT (i.e., not excluded).
        transitive_concept: If True, count entities whose type is `concept` OR any descendant
            (rdf:type/rdf:type*). Default False (most "how many X" questions are flat counts
            over a single class — `country`, `province`, `film`). Set True ONLY for genuine
            class hierarchies that the question relies on, e.g. "How many woodwind instruments?"
            (saxophones are a subclass of woodwind instrument). On a 0-result, retrying with
            transitive_concept=True is a cheap pivot.

    Returns:
        CountResponse with the exact integer count.
    """
    app_context: AppContext = context.request_context.lifespan_context
    sparql: SPARQLWrapper = app_context.sparql

    if not concept and not attribute_name and not entity_ids and not or_conditions and not not_conditions:
        return CountResponse(
            count=0,
            description="(no filter provided)",
            status="No concept/attribute/entity_ids supplied — refusing to count the whole KB.",
        )

    pre_blocks: list[str] = []
    if entity_ids:
        entity_uris = " ".join([format_entity_uri(eid) for eid in entity_ids])
        pre_blocks.append(f"VALUES ?entity {{ {entity_uris} }}")
    concept_block = _concept_clause(concept, transitive_concept)
    if concept_block:
        pre_blocks.append(concept_block.rstrip())

    attr_block = ""
    if attribute_name or or_conditions:
        branches: list[str] = []
        if attribute_name:
            branches.append(_attr_condition_sparql(attribute_name, attribute_value, operator, "0"))
        for i, cond in enumerate(or_conditions or [], start=1):
            branches.append(_attr_condition_sparql(
                cond.get("attribute_name", ""),
                cond.get("attribute_value", ""),
                cond.get("operator", "="),
                str(i),
            ))
        if len(branches) == 1:
            attr_block = "{ " + branches[0] + " }"
        elif len(branches) > 1:
            attr_block = " UNION ".join("{ " + b + " }" for b in branches)

    # FILTER NOT EXISTS blocks for each excluded condition
    not_blocks: list[str] = []
    for j, cond in enumerate(not_conditions or [], start=0):
        cond_sparql = _attr_condition_sparql(
            cond.get("attribute_name", ""),
            cond.get("attribute_value", ""),
            cond.get("operator", "="),
            f"n{j}",
        )
        not_blocks.append("FILTER NOT EXISTS { " + cond_sparql.rstrip() + " }")

    where_body = "\n        ".join(filter(None, pre_blocks + [attr_block] + not_blocks))
    query = f"""
    {SPARQL_PREFIXES}
    SELECT (COUNT(DISTINCT ?entity) AS ?c) WHERE {{
        {where_body}
    }}
    """

    # Build a human-readable description of what was counted
    parts = []
    if concept:
        parts.append(f"type={'≤' if transitive_concept else ''}{concept}")
    if attribute_name:
        parts.append(f"{attribute_name}{operator}{attribute_value}")
    for cond in or_conditions or []:
        parts.append(
            f"OR {cond.get('attribute_name','')}{cond.get('operator','=')}{cond.get('attribute_value','')}"
        )
    for cond in not_conditions or []:
        parts.append(
            f"NOT {cond.get('attribute_name','')}{cond.get('operator','=')}{cond.get('attribute_value','')}"
        )
    if entity_ids:
        parts.append(f"in {len(entity_ids)} ids")
    desc = ", ".join(parts) or "(unfiltered)"

    try:
        sparql.setQuery(query)
        results = sparql.query().convert()
        bindings = results.get("results", {}).get("bindings", [])
        count = int(bindings[0]["c"]["value"]) if bindings else 0

        session_journal.verified_facts.append({
            "fact": f"Count[{desc}] = {count}",
            "source": "CountEntities",
        })
        session_journal.add_completed_step(f"CountEntities({desc}) = {count}")
        logger.info(f"CountEntities: {desc} = {count}")
        return CountResponse(count=count, description=desc, status="ok")

    except Exception as e:
        logger.error(f"CountEntities failed: {e}")
        logger.error("CountEntities: Failed query was:\n{}", query)
        session_journal.add_failed_attempt(f"CountEntities({desc}): {str(e)[:120]}")
        return CountResponse(count=0, description=desc, status=f"error: {str(e)[:200]}")


@mcp.tool
@log_tool_duration
def CountUnion(
    branches: list[dict],
    context: Context,
) -> CountResponse:
    """
    Count the DISTINCT union of heterogeneous entity branches.

    Use this for "How many X satisfy A OR are in explicit/relation-derived set B?"
    questions where applying `entity_ids` globally would incorrectly intersect all
    branches. Each branch may contain `concept`, `attribute_name`, `attribute_value`,
    `operator`, `entity_ids`, `or_conditions`, `not_conditions`, and
    `transitive_concept`.

    Example:
      CountUnion(branches=[
        {"concept": "woodwind instrument", "attribute_name": "Hornbostel-Sachs classification",
         "attribute_value": "421.221.12", "transitive_concept": true},
        {"entity_ids": ["Q1414932", "Q1463985"]}
      ])
    """
    app_context: AppContext = context.request_context.lifespan_context
    sparql: SPARQLWrapper = app_context.sparql

    if not branches:
        return CountResponse(count=0, description="(no branches)", status="No branches supplied.")

    branch_blocks: list[str] = []
    desc_parts: list[str] = []

    for i, branch in enumerate(branches):
        if not isinstance(branch, dict):
            continue

        pre_blocks: list[str] = []
        entity_ids = branch.get("entity_ids") or []
        if entity_ids:
            entity_uris = " ".join(format_entity_uri(eid) for eid in entity_ids)
            pre_blocks.append(f"VALUES ?entity {{ {entity_uris} }}")

        concept = branch.get("concept", "")
        transitive = bool(branch.get("transitive_concept", False))
        concept_block = _concept_clause(concept, transitive)
        if concept_block:
            pre_blocks.append(concept_block.rstrip())

        attr_blocks: list[str] = []
        attribute_name = branch.get("attribute_name", "")
        if attribute_name:
            attr_blocks.append(_attr_condition_sparql(
                attribute_name,
                str(branch.get("attribute_value", "")),
                branch.get("operator", "="),
                f"u{i}_0",
            ))
        for j, cond in enumerate(branch.get("or_conditions") or [], start=1):
            attr_blocks.append(_attr_condition_sparql(
                cond.get("attribute_name", ""),
                str(cond.get("attribute_value", "")),
                cond.get("operator", "="),
                f"u{i}_{j}",
            ))

        attr_block = ""
        if len(attr_blocks) == 1:
            attr_block = attr_blocks[0]
        elif len(attr_blocks) > 1:
            attr_block = " UNION ".join("{ " + block + " }" for block in attr_blocks)

        not_blocks: list[str] = []
        for j, cond in enumerate(branch.get("not_conditions") or [], start=0):
            cond_sparql = _attr_condition_sparql(
                cond.get("attribute_name", ""),
                str(cond.get("attribute_value", "")),
                cond.get("operator", "="),
                f"un{i}_{j}",
            )
            not_blocks.append("FILTER NOT EXISTS { " + cond_sparql.rstrip() + " }")

        body = "\n        ".join(filter(None, pre_blocks + [attr_block] + not_blocks))
        if not body.strip():
            continue
        branch_blocks.append("{\n        " + body + "\n      }")

        bits = []
        if concept:
            bits.append(f"type={'≤' if transitive else ''}{concept}")
        if attribute_name:
            bits.append(f"{attribute_name}{branch.get('operator', '=')}{branch.get('attribute_value', '')}")
        if entity_ids:
            bits.append(f"{len(entity_ids)} explicit ids")
        desc_parts.append(" + ".join(bits) or f"branch {i + 1}")

    if not branch_blocks:
        return CountResponse(count=0, description="(empty branches)", status="No usable branch filters supplied.")

    union_body = "\n      UNION\n      ".join(branch_blocks)
    query = f"""
    {SPARQL_PREFIXES}
    SELECT (COUNT(DISTINCT ?entity) AS ?c) WHERE {{
      {union_body}
    }}
    """
    desc = " OR ".join(desc_parts)

    try:
        sparql.setQuery(query)
        results = sparql.query().convert()
        bindings = results.get("results", {}).get("bindings", [])
        count = int(bindings[0]["c"]["value"]) if bindings else 0

        key = "count_union"
        session_journal.found_values.setdefault(key, {})[desc] = count
        session_journal.verified_facts.append({
            "fact": f"CountUnion[{desc}] = {count}",
            "source": "CountUnion",
        })
        session_journal.add_completed_step(f"CountUnion({desc}) = {count}")
        return CountResponse(count=count, description=desc, status="ok")

    except Exception as e:
        logger.error(f"CountUnion failed: {e}")
        logger.error("CountUnion: Failed query was:\n{}", query)
        session_journal.add_failed_attempt(f"CountUnion({desc}): {str(e)[:120]}")
        return CountResponse(count=0, description=desc, status=f"error: {str(e)[:200]}")


@mcp.tool
@log_tool_duration
def GetRelationBetween(
    subject_id: str,
    object_id: str,
    context: Context,
) -> RelationBetweenResponse:
    """
    Return exact predicate labels between two entities in both directions.

    Use this for QueryRelation questions such as "How is A related to B?". The
    preferred answer is the predicate from `subject_id` to `object_id` when it
    exists; inverse predicates are returned separately to prevent direction swaps.
    """
    app_context: AppContext = context.request_context.lifespan_context
    sparql: SPARQLWrapper = app_context.sparql

    def _label_from_uri(uri: str) -> str:
        local = uri.rstrip("/").split("/")[-1]
        return local.replace("_", " ")

    query = f"""
    {SPARQL_PREFIXES}
    SELECT ?direction ?p WHERE {{
      {{
        ex:{subject_id} ?p ex:{object_id} .
        BIND("subject_to_object" AS ?direction)
      }} UNION {{
        ex:{object_id} ?p ex:{subject_id} .
        BIND("object_to_subject" AS ?direction)
      }}
      FILTER(STRSTARTS(STR(?p), "http://kqapro.org/property/"))
    }}
    """

    try:
        sparql.setQuery(query)
        results = sparql.query().convert()
        bindings = results.get("results", {}).get("bindings", [])

        direct: list[str] = []
        inverse: list[str] = []
        for binding in bindings:
            label = _label_from_uri(binding.get("p", {}).get("value", ""))
            direction = binding.get("direction", {}).get("value", "")
            if direction == "subject_to_object" and label not in direct:
                direct.append(label)
            elif direction == "object_to_subject" and label not in inverse:
                inverse.append(label)

        preferred = direct[0] if direct else (inverse[0] if inverse else "")
        status = "ok" if preferred else "No relation found between the two entities."

        if preferred:
            key = f"relation_to_{object_id}"
            session_journal.found_values.setdefault(subject_id, {})[key] = preferred
            session_journal.verified_facts.append({
                "subject": subject_id,
                "relation": preferred,
                "related_id": object_id,
                "direction": "forward" if direct else "inverse_only",
                "source": "GetRelationBetween",
            })
            session_journal.add_completed_step(
                f"GetRelationBetween({subject_id}, {object_id}) -> {preferred}"
            )
        else:
            session_journal.add_failed_attempt(
                f"GetRelationBetween({subject_id}, {object_id}): no relation"
            )

        return RelationBetweenResponse(
            subject_id=subject_id,
            object_id=object_id,
            subject_to_object=direct,
            object_to_subject=inverse,
            preferred_answer=preferred,
            status=status,
        )

    except Exception as e:
        logger.error(f"GetRelationBetween failed: {e}")
        session_journal.add_failed_attempt(
            f"GetRelationBetween({subject_id}, {object_id}): {str(e)[:120]}"
        )
        return RelationBetweenResponse(
            subject_id=subject_id,
            object_id=object_id,
            subject_to_object=[],
            object_to_subject=[],
            preferred_answer="",
            status=f"error: {str(e)[:200]}",
        )


@mcp.tool
@log_tool_duration
def SelectExtreme(
    context: Context,
    attribute_name: str,
    mode: Literal["max", "min"],
    entity_ids: list[str] | None = None,
    concept: str = "",
    transitive_concept: bool = False,
    k: int = 1,
    filter_attribute_name: str = "",
    filter_attribute_value: str = "",
    filter_operator: Literal["=", "!=", "<", ">", "<=", ">=", "contains"] = "=",
) -> SelectExtremeResponse:
    """
    Return the entity (or top-k entities) with the maximum or minimum value of an attribute.

    🎯 The deterministic answer to SelectAmong / SelectBetween questions. Use this instead
    of CompareEntities-then-eyeball-the-table — it returns the actual winner via SPARQL
    ORDER BY, so the LLM doesn't have to do arithmetic.

    **When to use:**
    - "Which European country has the lowest GDP?" → SelectExtreme(concept="country", attribute_name="gross domestic product", mode="min")
    - "Which of these films is longest?" → SelectExtreme(entity_ids=[...], attribute_name="duration", mode="max")
    - "Who is older, A or B?" → SelectExtreme(entity_ids=["A","B"], attribute_name="date of birth", mode="min") — earlier date = older
    - "Top 5 most populous cities" → SelectExtreme(concept="city", attribute_name="population", mode="max", k=5)
    - "Smallest French region with population != 97000" → SelectExtreme(concept="former French region", attribute_name="population", mode="min", filter_attribute_name="population", filter_attribute_value="97000", filter_operator="!=")

    Args:
        attribute_name: Attribute to rank by (e.g., "population", "date of birth", "duration").
        mode: "max" for largest, "min" for smallest. For dates, "min" = earliest, "max" = latest.
        entity_ids: Optional list of candidate entity IDs.
        concept: Optional concept restriction (alternative to entity_ids).
        transitive_concept: If True, include subclasses of `concept` (rdf:type*). Default
            False — most Select questions over a flat concept ("country", "film") shouldn't
            expand. Set True only when the concept is a genuine hierarchy whose subclasses
            you want considered.
        k: How many top entities to return. Default 1 (the winner).
        filter_attribute_name / filter_attribute_value / filter_operator: Optional pre-filter
            applied before ranking (lets you express "smallest X where Y != Z" in one call).

    Returns:
        SelectExtremeResponse with up to k results sorted from extreme to less-extreme.
    """
    app_context: AppContext = context.request_context.lifespan_context
    sparql: SPARQLWrapper = app_context.sparql

    if not entity_ids and not concept:
        return SelectExtremeResponse(
            mode=mode, attribute_name=attribute_name, results=[],
            status="error: provide either entity_ids or concept",
        )

    pre_blocks: list[str] = []
    if entity_ids:
        entity_uris = " ".join([format_entity_uri(eid) for eid in entity_ids])
        pre_blocks.append(f"VALUES ?entity {{ {entity_uris} }}")
    concept_block = _concept_clause(concept, transitive_concept)
    if concept_block:
        pre_blocks.append(concept_block.rstrip())

    # Pre-filter (optional) — applied with AND
    if filter_attribute_name and filter_attribute_value:
        pre_blocks.append("{ " + _attr_condition_sparql(
            filter_attribute_name, filter_attribute_value, filter_operator, "f"
        ) + " }")

    # Ranking attribute — quantity bnodes get unwrapped via OPTIONAL+COALESCE
    sanitized_rank = attribute_name.replace(" ", "_")
    rank_uri = f"<http://kqapro.org/attribute/{sanitized_rank}>"
    pre_blocks.append(
        f"?entity {rank_uri} ?rankRaw .\n"
        f"OPTIONAL {{ ?rankRaw rdf:value ?rankBlank }}\n"
        f"BIND(COALESCE(?rankBlank, ?rankRaw) AS ?rankVal)"
    )

    # Subquery: pick the per-entity best value (MAX or MIN across multiple statements
    # per entity — KQAPro often has several historical values for the same attribute).
    # MAX/MIN are type-aware in SPARQL: numeric for xsd:decimal, lex for xsd:date /
    # xsd:gYear / plain strings — and lex equals chronological for ISO-formatted dates.
    agg = "MAX" if mode == "max" else "MIN"
    inner_where = "\n        ".join(filter(None, pre_blocks))
    query = f"""
    {SPARQL_PREFIXES}
    SELECT ?entity ?entityName ?bestVal WHERE {{
        {{
            SELECT ?entity ({agg}(?rankVal) AS ?bestVal) WHERE {{
                {inner_where}
            }}
            GROUP BY ?entity
        }}
        OPTIONAL {{ ?entity rdfs:label ?entityName }}
    }}
    ORDER BY {"DESC" if mode == "max" else "ASC"}(?bestVal)
    LIMIT {max(1, k)}
    """

    try:
        sparql.setQuery(query)
        results = sparql.query().convert()
        bindings = results.get("results", {}).get("bindings", [])

        out: list[ComparisonResult] = []
        for b in bindings:
            uri = b.get("entity", {}).get("value", "")
            eid = uri.split("/entity/")[-1] if "/entity/" in uri else uri
            name = b.get("entityName", {}).get("value", "Unknown")
            raw = b.get("bestVal", {}).get("value", "")
            normalized = None
            try:
                normalized = float(raw)
            except (TypeError, ValueError):
                pass
            out.append(ComparisonResult(
                entity_id=eid, entity_name=name, value=raw, normalized_value=normalized,
            ))

        for r in out:
            session_journal.visited_nodes[r.entity_id] = r.entity_name
        if out:
            winner = out[0]
            session_journal.verified_facts.append({
                "fact": f"{mode}({attribute_name}) = {winner.entity_name} ({winner.value})",
                "source": "SelectExtreme",
            })
            session_journal.add_completed_step(
                f"SelectExtreme({mode}, {attribute_name}, k={k}): {winner.entity_name}"
            )
        else:
            session_journal.add_failed_attempt(
                f"SelectExtreme({mode}, {attribute_name}): no results"
            )

        return SelectExtremeResponse(
            mode=mode, attribute_name=attribute_name, results=out,
            status="ok" if out else "no results",
        )

    except Exception as e:
        logger.error(f"SelectExtreme failed: {e}")
        logger.error("SelectExtreme: Failed query was:\n{}", query)
        session_journal.add_failed_attempt(
            f"SelectExtreme({mode}, {attribute_name}): {str(e)[:120]}"
        )
        return SelectExtremeResponse(
            mode=mode, attribute_name=attribute_name, results=[],
            status=f"error: {str(e)[:200]}",
        )


@mcp.tool
@log_tool_duration
def VerifyFact(
    context: Context,
    subject_id: str,
    predicate: str,
    target: str,
    predicate_type: Literal["relation", "attribute", "auto"] = "auto",
) -> VerifyFactResponse:
    """
    Return TRUE / FALSE for "does the fact (subject, predicate, target) exist in the KB".

    🎯 Use this for any Verify question ("Is X a Y?", "Did X win Y?", "Is the population
    of X equal to N?"). It is a deterministic SPARQL ASK — the LLM does not have to
    interpret an empty GetAttributeDetails response as "no".

    **When to use:**
    - "Is 129586 the exploitation visa number of Bridget Jones's Diary?" → VerifyFact("Q220678", "exploitation visa number", "129586", predicate_type="attribute")
    - "Is Einstein an instance of human?" → VerifyFact("Q937", "instance of", "Q5") — note: instance-of in KQAPro is rdf:type, see VerifyType for that pattern
    - "Did Nolan direct Inception?" → VerifyFact("Q25191", "director", "Q26956", predicate_type="relation")
    - Mode "auto" tries attribute first, then relation — useful when you're not sure.

    Args:
        subject_id: Entity ID of the subject (e.g., "Q42").
        predicate: Predicate name (relation or attribute, with spaces or underscores).
        target: For relations, the object entity ID (e.g., "Q5"). For attributes, the literal
            value to check ("129586", "1979-01-01", "physicist"). Auto-detected by predicate_type.
        predicate_type: "attribute" → checks Subject attr:predicate "target". "relation" → checks
            Subject prop:predicate <Target>. "auto" → tries attribute, then relation.

    Returns:
        VerifyFactResponse with verdict TRUE/FALSE/ERROR and matched_as=relation|attribute|none.
    """
    app_context: AppContext = context.request_context.lifespan_context
    sparql_client: SPARQLWrapper = app_context.sparql

    subj_uri = format_entity_uri(subject_id)
    sanitized_pred = predicate.replace(" ", "_")
    safe_target = target.replace('"', '\\"')

    def ask(query: str) -> bool:
        sparql_client.setQuery(query)
        res = sparql_client.query().convert()
        return bool(res.get("boolean", False))

    def attr_query() -> str:
        # Handle quantity/bnode + literal-typed attribute values uniformly
        # by checking for a triple whose stringified value equals the target,
        # or whose rdf:value (under a bnode) equals the target.
        return f"""
        {SPARQL_PREFIXES}
        ASK {{
            {{
                {subj_uri} <http://kqapro.org/attribute/{sanitized_pred}> ?v .
                FILTER(STR(?v) = "{safe_target}")
            }} UNION {{
                {subj_uri} <http://kqapro.org/attribute/{sanitized_pred}> ?bn .
                ?bn rdf:value ?bv .
                FILTER(STR(?bv) = "{safe_target}")
            }}
        }}
        """

    def resolve_target_to_qid(label: str) -> Optional[str]:
        """Resolve a free-text label to a Q-id when the agent passes 'Netherlands'
        instead of 'Q55'. Case-insensitive exact-label match; returns the first
        matching Q-id or None. Avoids the silent-FALSE bug where
        format_entity_uri('Netherlands') becomes a non-existent IRI."""
        safe_label = label.replace('"', '\\"')
        q = f"""
        {SPARQL_PREFIXES}
        SELECT ?s WHERE {{
            ?s rdfs:label ?l .
            FILTER(LCASE(STR(?l)) = LCASE("{safe_label}"))
            FILTER(STRSTARTS(STR(?s), "{NS_ENTITY}"))
        }} LIMIT 1
        """
        try:
            sparql_client.setQuery(q)
            res = sparql_client.query().convert()
            bindings = res.get("results", {}).get("bindings", [])
            if not bindings:
                return None
            uri = bindings[0]["s"]["value"]
            return uri.split("/entity/")[-1]
        except Exception as ex:
            logger.debug(f"VerifyFact label resolution failed: {ex}")
            return None

    def rel_query(resolved_target: str) -> str:
        target_uri = format_entity_uri(resolved_target)
        return f"""
        {SPARQL_PREFIXES}
        ASK {{
            {{ {subj_uri} <http://kqapro.org/property/{sanitized_pred}> {target_uri} . }}
            UNION
            {{ {target_uri} <http://kqapro.org/property/{sanitized_pred}> {subj_uri} . }}
        }}
        """

    matched_as = "none"
    verdict = "FALSE"
    explanation = ""
    try:
        if predicate_type in ("attribute", "auto"):
            try:
                if ask(attr_query()):
                    matched_as = "attribute"
                    verdict = "TRUE"
                    explanation = f"{subject_id} has attribute '{predicate}' with value '{target}'"
            except Exception as ex_a:
                if predicate_type == "attribute":
                    raise
                logger.debug(f"VerifyFact attr branch failed: {ex_a}")
        if verdict == "FALSE" and predicate_type in ("relation", "auto"):
            # Auto-resolve label → Q-id when target isn't already an entity ID.
            # Without this, VerifyFact(Q772, country, "Netherlands") silently
            # returns FALSE because <ex:Netherlands> doesn't exist.
            rel_target = target
            if not re.match(r"^Q\d+$", target):
                resolved = resolve_target_to_qid(target)
                if resolved:
                    rel_target = resolved
                    logger.info(f"VerifyFact: resolved relation target '{target}' → {resolved}")
            if ask(rel_query(rel_target)):
                matched_as = "relation"
                verdict = "TRUE"
                explanation = f"({subject_id}) -[{predicate}]-> ({rel_target}) exists"
                if rel_target != target:
                    explanation += f" [resolved '{target}' → {rel_target}]"
        if verdict == "FALSE":
            explanation = f"No '{predicate}' fact found between {subject_id} and {target}"
            matched_as = "none"

        session_journal.verified_facts.append({
            "fact": f"VerifyFact({subject_id}, {predicate}, {target}) = {verdict}",
            "source": "VerifyFact",
        })
        session_journal.add_completed_step(explanation)
        return VerifyFactResponse(
            verdict=verdict, explanation=explanation,
            subject_id=subject_id, predicate=predicate, target=target,
            matched_as=matched_as,
        )
    except Exception as e:
        logger.error(f"VerifyFact failed: {e}")
        session_journal.add_failed_attempt(
            f"VerifyFact({subject_id}, {predicate}, {target}): {str(e)[:120]}"
        )
        return VerifyFactResponse(
            verdict="ERROR",
            explanation=f"Could not verify: {str(e)[:200]}",
            subject_id=subject_id, predicate=predicate, target=target,
            matched_as="error",
        )


if __name__ == "__main__":
    mcp.run(transport="stdio")
