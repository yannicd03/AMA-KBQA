"""
SciQA MCP Server for ORKG Knowledge Graph

This MCP server provides tools for querying the Open Research Knowledge Graph (ORKG)
loaded from the SciQA dataset. It follows the same patterns as kqapro_server.py but
adapted for the ORKG schema and namespaces.

Tools provided:
- FindResource: Semantic vector search for ORKG resources
- FindPredicate: Search for ORKG predicates/relations
- GetResourceDetails: Get full details of a resource
- GetRelationTargets: Get targets of a relation from a resource
- RunORKGSPARQL: Execute raw SPARQL queries
- GetResourceLabel: Quick label lookup for a URI
- BatchGetResourceLabels: Get labels for multiple URIs
- GetPaperContributions: Get contributions for a paper
- GetPaperAuthors: Get authors for a paper
- GetContributionMethods: Get methods used in a contribution
- GetResearchFieldPapers: List papers in a research field
- VerifyNumericCondition: Deterministic numeric/date comparison
- GetResourceSummary: All-in-one resource exploration
- FindByPredicateValue: Reverse lookup by predicate value
- CompareResources: Compare a predicate across multiple resources
- FollowRelationPath: Multi-hop relation navigation
- ManageJournal: Scratchpad for state management
- GetJournalSummary: Format journal state for synthesis
"""

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
    get_sciqa_collection_entities,
    get_sciqa_collection_relations,
    get_sciqa_virtuoso_graph,
    get_sciqa_entity_threshold,
    get_sciqa_relation_threshold,
)
import os
import sys
import time
import json
from contextlib import asynccontextmanager
from typing import AsyncIterator, Any, Dict, List, Optional
from pathlib import Path
from functools import wraps

from fastmcp import FastMCP, Context
from qdrant_client import QdrantClient
from qdrant_client.http import models
from openai import OpenAI
from loguru import logger
from pydantic import BaseModel, Field, ConfigDict
from typing import Literal
from SPARQLWrapper import SPARQLWrapper, JSON
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

REPO_ROOT = Path(__file__).resolve().parents[2]

# Import configuration utilities
sys.path.insert(0, str(REPO_ROOT))

# Configure logger
log_dir = REPO_ROOT / "logs"
log_dir.mkdir(exist_ok=True)
logger.add(
    log_dir / "sciqa_server.log",
    rotation="10 MB",
    retention="7 days",
    level="DEBUG",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"
)

# --- Configuration from config.toml ---
QDRANT_HOST = get_qdrant_host()
QDRANT_PORT = get_qdrant_port()
COLLECTION_ENTITIES = get_sciqa_collection_entities()
COLLECTION_RELATIONS = get_sciqa_collection_relations()
VIRTUOSO_ENDPOINT = get_virtuoso_endpoint()
SCIQA_GRAPH = get_sciqa_virtuoso_graph()
EMBEDDING_MODEL = get_embedding_model_name()
CHAT_MODEL = get_chat_model_name()
CHAT_TEMPERATURE = get_chat_temperature()
CHAT_MAX_TOKENS = get_chat_max_tokens()
TOP_N = 5
ENTITY_THRESHOLD = get_sciqa_entity_threshold()
RELATION_THRESHOLD = get_sciqa_relation_threshold()

# --- ORKG Namespaces ---
NS_RESOURCE = "http://orkg.org/orkg/resource/"
NS_PREDICATE = "http://orkg.org/orkg/predicate/"
NS_CLASS = "http://orkg.org/orkg/class/"

# SPARQL Prefixes for ORKG
SPARQL_PREFIXES = """
PREFIX orkgr: <http://orkg.org/orkg/resource/>
PREFIX orkgp: <http://orkg.org/orkg/predicate/>
PREFIX orkgc: <http://orkg.org/orkg/class/>
PREFIX rdfs:  <http://www.w3.org/2000/01/rdf-schema#>
PREFIX rdf:   <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX xsd:   <http://www.w3.org/2001/XMLSchema#>
PREFIX owl:   <http://www.w3.org/2002/07/owl#>
"""


# ==============================================================================
# Pydantic Models
# ==============================================================================

class AppContext(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    qdrant: QdrantClient
    chat_client: OpenAI
    embedding_client: OpenAI
    sparql: Any


class ResourceMatch(BaseModel):
    """Represents a single resource found in ORKG."""
    original_id: str = Field(..., description="The unique identifier (e.g., 'R12345').")
    name: str = Field(..., description="The human-readable label.")
    node_type: str = Field(..., description="Type of resource (paper, author, contribution, etc.).")
    relevance_score: float = Field(..., description="Vector similarity score (0-1).")
    available_predicates: List[str] = Field(default_factory=list, description="Available predicates for this resource.")


class SearchResponse(BaseModel):
    """The top-level response object for search."""
    matches: List[ResourceMatch] = Field(default_factory=list)
    result_count: int = Field(...)


class PredicateMatch(BaseModel):
    """Represents a predicate found in ORKG."""
    predicate_id: str = Field(..., description="The predicate ID (e.g., 'P30').")
    label: str = Field(..., description="Human-readable label.")
    relevance_score: float = Field(..., description="Vector similarity score.")


class PredicateSearchResponse(BaseModel):
    """Response for predicate search."""
    matches: List[PredicateMatch] = Field(default_factory=list)
    result_count: int = Field(...)


class SPARQLResponse(BaseModel):
    """Raw results from a SPARQL query."""
    vars: List[str] = Field(...)
    bindings: List[Dict[str, Any]] = Field(...)
    raw_json: Dict[str, Any] = Field(...)


class JournalState(BaseModel):
    """The scratchpad state for the current reasoning session."""

    question_text: str = Field(default="", description="The original question")
    question_type: str = Field(default="", description="Question type classification")
    target_entities: List[str] = Field(default_factory=list, description="Entities we're looking for")
    target_attributes: List[str] = Field(default_factory=list, description="Attributes we need to find")

    visited_nodes: Dict[str, str] = Field(default_factory=dict, description="Map of {node_id: node_name}")
    verified_facts: List[Dict] = Field(default_factory=list, description="Verified facts")
    failed_attempts: List[str] = Field(default_factory=list, description="Failed attempts")

    found_values: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Map of {entity_id: {attribute: value}}"
    )

    current_plan: List[str] = Field(default_factory=list, description="Remaining steps")
    completed_steps: List[str] = Field(default_factory=list, description="Completed steps")
    partial_answer: str = Field(default="", description="Intermediate answer")

    def to_str(self) -> str:
        """Format journal state for display."""
        lines = ["=" * 70]
        lines.append("SCIQA SCRATCHPAD STATE")
        lines.append("=" * 70)

        if self.question_type:
            lines.append(f"Question Type: {self.question_type}")
        if self.target_entities:
            lines.append(f"Target Entities: {', '.join(self.target_entities)}")

        if self.visited_nodes:
            lines.append(f"\nEXPLORED NODES ({len(self.visited_nodes)}):")
            for node_id, node_name in list(self.visited_nodes.items())[:5]:
                lines.append(f"  - {node_name} ({node_id})")
            if len(self.visited_nodes) > 5:
                lines.append(f"  ... and {len(self.visited_nodes) - 5} more")

        if self.found_values:
            lines.append(f"\nDISCOVERED VALUES:")
            for entity_id, attrs in self.found_values.items():
                entity_name = self.visited_nodes.get(entity_id, entity_id)
                lines.append(f"  {entity_name}:")
                for attr_name, attr_data in attrs.items():
                    if isinstance(attr_data, list) and attr_data:
                        for val_item in attr_data[:3]:
                            if isinstance(val_item, dict):
                                val_str = val_item.get("value", "?")
                                lines.append(f"    - {attr_name}: {val_str}")
                            else:
                                lines.append(f"    - {attr_name}: {val_item}")
                    else:
                        lines.append(f"    - {attr_name}: {attr_data}")

        if self.completed_steps:
            lines.append(f"\nCOMPLETED STEPS ({len(self.completed_steps)}):")
            for step in self.completed_steps[-3:]:
                lines.append(f"  + {step}")

        if self.current_plan:
            lines.append(f"\nNEXT STEPS:")
            for i, step in enumerate(self.current_plan[:3], 1):
                lines.append(f"  {i}. {step}")

        if self.failed_attempts:
            lines.append(f"\nFAILED ATTEMPTS ({len(self.failed_attempts)}):")
            for attempt in self.failed_attempts[-2:]:
                lines.append(f"  x {attempt}")

        if self.partial_answer:
            lines.append(f"\nPARTIAL ANSWER: {self.partial_answer}")

        lines.append(
            f"\nSTATS: {len(self.visited_nodes)} nodes, {len(self.found_values)} entities with data, "
            f"{len(self.completed_steps)} steps done"
        )
        lines.append("=" * 70)
        return "\n".join(lines)


class NumericComparisonResponse(BaseModel):
    """Response from numeric/date comparison verification."""
    verdict: Literal["TRUE", "FALSE", "ERROR"]
    explanation: str
    value1: str
    value2: str
    operator: str


class ComparisonResult(BaseModel):
    """Single resource's predicate value in a comparison."""
    resource_id: str
    name: str
    value: Any
    normalized_value: Optional[float] = None


class CompareResourcesResponse(BaseModel):
    """Response from comparing a predicate across multiple resources."""
    predicate: str
    results: list[ComparisonResult]
    sorted_by: str
    status: str


# Global state container
session_journal = JournalState()


# ==============================================================================
# Helper Functions
# ==============================================================================

def format_resource_uri(node_id: str) -> str:
    """Format a raw ID into an ORKG Resource URI."""
    node_id = node_id.replace(" ", "_")

    if node_id.startswith("<") and node_id.endswith(">"):
        return node_id
    if "http" in node_id:
        return f"<{node_id}>"

    return f"<{NS_RESOURCE}{node_id}>"


def format_predicate_uri(predicate_id: str) -> str:
    """Format a raw ID into an ORKG Predicate URI."""
    predicate_id = predicate_id.replace(" ", "_")

    if predicate_id.startswith("<") and predicate_id.endswith(">"):
        return predicate_id
    if "http" in predicate_id:
        return f"<{predicate_id}>"

    return f"<{NS_PREDICATE}{predicate_id}>"


def get_embedding(client: OpenAI, text: str) -> List[float]:
    """Get embedding for text."""
    text = text.replace("\n", " ")
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=[text],
        encoding_format="float"
    )
    return response.data[0].embedding


def log_tool_duration(func):
    """Decorator to log tool execution duration."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        tool_name = func.__name__
        start_time = time.time()

        param_info = ""
        if args:
            arg_start = 1 if hasattr(args[0], '__class__') else 0
            display_args = []
            for arg in args[arg_start:arg_start+3]:
                if not isinstance(arg, Context):
                    arg_repr = str(arg)[:50]
                    display_args.append(arg_repr)
            if display_args:
                param_info = f" with params: {', '.join(display_args)}"

        logger.info(f"[{tool_name}] Starting{param_info}")

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


# ==============================================================================
# Lifespan Manager
# ==============================================================================

@asynccontextmanager
async def server_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    """Manage server lifecycle."""
    logger.info("Starting SciQA MCP Server...")

    qdrant = None
    try:
        qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        qdrant.get_collections()

        chat_client = get_chat_client()
        embedding_client = get_embedding_client()

        sparql = SPARQLWrapper(VIRTUOSO_ENDPOINT)
        sparql.setReturnFormat(JSON)

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
        logger.info("Shutting down SciQA MCP Server...")
        if qdrant is not None:
            qdrant.close()


# Initialize FastMCP
mcp = FastMCP("SciQA-ORKG-Server", lifespan=server_lifespan)


# ==============================================================================
# TOOL 1: FindResource
# ==============================================================================

@mcp.tool()
async def FindResource(
    app_context: Context,
    semantic_query: str,
    top_n: int = 5
) -> str:
    """
    Search for ORKG resources (papers, authors, contributions, etc.) by semantic similarity.

    Use this to find entities when you have a natural language description.

    Args:
        semantic_query: Natural language description (e.g., "machine learning paper", "NLP contribution")
        top_n: Number of results to return (default: 5)

    Returns:
        JSON with matching resources and their available predicates
    """
    app = app_context.request_context.lifespan_context

    try:
        query_vector = get_embedding(app.embedding_client, semantic_query)

        search_results = app.qdrant.search(
            collection_name=COLLECTION_ENTITIES,
            query_vector=query_vector,
            limit=top_n,
            score_threshold=ENTITY_THRESHOLD
        )

        matches = []
        for hit in search_results:
            payload = hit.payload or {}
            resource_id = payload.get("uri", "").split("/")[-1]
            name = payload.get("name", "")
            node_type = payload.get("node_type", "resource")

            # Update journal
            session_journal.visited_nodes[resource_id] = name

            # Get available predicates via SPARQL
            predicates = []
            try:
                pred_query = f"""
                SELECT DISTINCT ?p WHERE {{
                    GRAPH <{SCIQA_GRAPH}> {{
                        orkgr:{resource_id} ?p ?o .
                    }}
                }} LIMIT 20
                """
                full_query = SPARQL_PREFIXES + pred_query
                app.sparql.setQuery(full_query)
                results = app.sparql.query().convert()

                for binding in results.get("results", {}).get("bindings", []):
                    pred_uri = binding.get("p", {}).get("value", "")
                    pred_label = pred_uri.split("/")[-1]
                    if pred_label not in ["type", "label"]:
                        predicates.append(pred_label)
            except Exception as e:
                logger.warning(f"Failed to get predicates for {resource_id}: {e}")

            matches.append(ResourceMatch(
                original_id=resource_id,
                name=name,
                node_type=node_type,
                relevance_score=round(hit.score, 4),
                available_predicates=predicates[:10]
            ))

        session_journal.completed_steps.append(f"FindResource('{semantic_query}') -> {len(matches)} results")

        response = SearchResponse(matches=matches, result_count=len(matches))
        return response.model_dump_json(indent=2)

    except Exception as e:
        error_msg = f"Error in FindResource: {str(e)}"
        logger.error(error_msg)
        session_journal.failed_attempts.append(f"FindResource('{semantic_query}'): {str(e)}")
        return json.dumps({"error": error_msg}, indent=2)


# ==============================================================================
# TOOL 2: FindPredicate
# ==============================================================================

@mcp.tool()
async def FindPredicate(
    app_context: Context,
    semantic_query: str,
    top_n: int = 5
) -> str:
    """
    Search for ORKG predicates/relations by semantic similarity.

    Use this to find the correct predicate name when you need to query relationships.

    Args:
        semantic_query: Natural language description (e.g., "has author", "research field")
        top_n: Number of results to return

    Returns:
        JSON with matching predicates
    """
    app = app_context.request_context.lifespan_context

    try:
        query_vector = get_embedding(app.embedding_client, semantic_query)

        search_results = app.qdrant.search(
            collection_name=COLLECTION_RELATIONS,
            query_vector=query_vector,
            limit=top_n,
            score_threshold=RELATION_THRESHOLD
        )

        matches = []
        for hit in search_results:
            payload = hit.payload or {}
            pred_id = payload.get("uri", "").split("/")[-1]
            label = payload.get("predicate", pred_id)

            matches.append(PredicateMatch(
                predicate_id=pred_id,
                label=label,
                relevance_score=round(hit.score, 4)
            ))

        session_journal.completed_steps.append(f"FindPredicate('{semantic_query}') -> {len(matches)} results")

        response = PredicateSearchResponse(matches=matches, result_count=len(matches))
        return response.model_dump_json(indent=2)

    except Exception as e:
        error_msg = f"Error in FindPredicate: {str(e)}"
        logger.error(error_msg)
        session_journal.failed_attempts.append(f"FindPredicate('{semantic_query}'): {str(e)}")
        return json.dumps({"error": error_msg}, indent=2)


# ==============================================================================
# TOOL 3: GetResourceDetails
# ==============================================================================

@mcp.tool()
async def GetResourceDetails(
    app_context: Context,
    resource_id: str
) -> str:
    """
    Get full details of an ORKG resource including label, type, and all outgoing relations.

    Use this after FindResource to get complete information about a specific resource.

    Args:
        resource_id: The resource ID (e.g., "R12345")

    Returns:
        JSON with resource details
    """
    app = app_context.request_context.lifespan_context

    try:
        # Get label and type
        query = f"""
        SELECT ?label ?type WHERE {{
            GRAPH <{SCIQA_GRAPH}> {{
                orkgr:{resource_id} rdfs:label ?label .
                OPTIONAL {{ orkgr:{resource_id} rdf:type ?type . }}
            }}
        }}
        """
        full_query = SPARQL_PREFIXES + query
        app.sparql.setQuery(full_query)
        results = app.sparql.query().convert()
        bindings = results.get("results", {}).get("bindings", [])

        label = ""
        types = []
        for binding in bindings:
            if "label" in binding:
                label = binding["label"]["value"]
            if "type" in binding:
                types.append(binding["type"]["value"].split("/")[-1])

        # Get all outgoing relations
        rel_query = f"""
        SELECT ?p ?o ?oLabel WHERE {{
            GRAPH <{SCIQA_GRAPH}> {{
                orkgr:{resource_id} ?p ?o .
                OPTIONAL {{ ?o rdfs:label ?oLabel . }}
                FILTER(?p != rdf:type && ?p != rdfs:label)
            }}
        }} LIMIT 50
        """
        full_rel_query = SPARQL_PREFIXES + rel_query
        app.sparql.setQuery(full_rel_query)
        rel_results = app.sparql.query().convert()

        relations = {}
        for binding in rel_results.get("results", {}).get("bindings", []):
            pred = binding.get("p", {}).get("value", "").split("/")[-1]
            obj = binding.get("o", {}).get("value", "")
            obj_label = binding.get("oLabel", {}).get("value", "")

            if pred not in relations:
                relations[pred] = []

            if obj_label:
                relations[pred].append({"id": obj.split("/")[-1], "label": obj_label})
            elif obj.startswith("http"):
                relations[pred].append({"id": obj.split("/")[-1]})
            else:
                relations[pred].append({"value": obj})

        # Update journal
        session_journal.visited_nodes[resource_id] = label
        session_journal.found_values[resource_id] = {"_type": types, **relations}
        session_journal.completed_steps.append(f"GetResourceDetails('{resource_id}') -> {label}")

        response = {
            "resource_id": resource_id,
            "label": label,
            "types": types,
            "relations": relations,
            "status": f"Found {len(relations)} relation types for {resource_id}"
        }

        return json.dumps(response, indent=2)

    except Exception as e:
        error_msg = f"Error in GetResourceDetails: {str(e)}"
        logger.error(error_msg)
        session_journal.failed_attempts.append(f"GetResourceDetails('{resource_id}'): {str(e)}")
        return json.dumps({"error": error_msg}, indent=2)


# ==============================================================================
# TOOL 4: GetRelationTargets
# ==============================================================================

@mcp.tool()
async def GetRelationTargets(
    app_context: Context,
    resource_id: str,
    predicate: str
) -> str:
    """
    Get all targets of a specific relation from a resource.

    Use this to follow a specific relation from a known resource.

    Args:
        resource_id: The source resource ID (e.g., "R12345")
        predicate: The predicate/relation name (e.g., "P31", "P6", "has_author")

    Returns:
        JSON with relation targets
    """
    app = app_context.request_context.lifespan_context

    try:
        # Handle both P-style IDs and full names
        if not predicate.startswith("P") and not predicate.startswith("orkgp:"):
            predicate = predicate.replace(" ", "_")

        pred_uri = f"orkgp:{predicate}" if not predicate.startswith("orkgp:") else predicate

        query = f"""
        SELECT ?o ?oLabel WHERE {{
            GRAPH <{SCIQA_GRAPH}> {{
                orkgr:{resource_id} {pred_uri} ?o .
                OPTIONAL {{ ?o rdfs:label ?oLabel . }}
            }}
        }}
        """
        full_query = SPARQL_PREFIXES + query
        app.sparql.setQuery(full_query)
        results = app.sparql.query().convert()
        bindings = results.get("results", {}).get("bindings", [])

        targets = []
        for binding in bindings:
            obj = binding.get("o", {}).get("value", "")
            obj_label = binding.get("oLabel", {}).get("value", "")

            if obj.startswith("http"):
                target = {"id": obj.split("/")[-1]}
                if obj_label:
                    target["label"] = obj_label
                    session_journal.visited_nodes[target["id"]] = obj_label
            else:
                target = {"value": obj}

            targets.append(target)

        # Update journal
        if resource_id not in session_journal.found_values:
            session_journal.found_values[resource_id] = {}
        session_journal.found_values[resource_id][predicate] = targets
        session_journal.completed_steps.append(
            f"GetRelationTargets('{resource_id}', '{predicate}') -> {len(targets)} targets"
        )

        response = {
            "resource_id": resource_id,
            "predicate": predicate,
            "targets": targets,
            "count": len(targets),
            "status": f"Found {len(targets)} targets for {predicate}"
        }

        return json.dumps(response, indent=2)

    except Exception as e:
        error_msg = f"Error in GetRelationTargets: {str(e)}"
        logger.error(error_msg)
        session_journal.failed_attempts.append(
            f"GetRelationTargets('{resource_id}', '{predicate}'): {str(e)}"
        )
        return json.dumps({"error": error_msg}, indent=2)


# ==============================================================================
# TOOL 5: RunORKGSPARQL
# ==============================================================================

@mcp.tool()
async def RunORKGSPARQL(
    app_context: Context,
    query: str
) -> str:
    """
    Execute a raw SPARQL query on the ORKG graph.

    DO NOT include PREFIX declarations - they are auto-injected.
    The query will be executed against the SciQA graph.

    Available prefixes:
    - orkgr: <http://orkg.org/orkg/resource/>
    - orkgp: <http://orkg.org/orkg/predicate/>
    - orkgc: <http://orkg.org/orkg/class/>
    - rdfs:, rdf:, xsd:, owl:

    Args:
        query: SPARQL query WITHOUT PREFIX declarations

    Returns:
        JSON with query results
    """
    app = app_context.request_context.lifespan_context

    try:
        # Wrap query with graph if not already wrapped
        if "GRAPH" not in query.upper():
            # Find WHERE clause and inject GRAPH
            if "WHERE" in query.upper():
                query = query.replace("{", f"{{ GRAPH <{SCIQA_GRAPH}> {{", 1)
                # Find matching closing brace
                query = query.rstrip()
                if query.endswith("}"):
                    query = query[:-1] + "} }"

        full_query = SPARQL_PREFIXES + query
        app.sparql.setQuery(full_query)
        results = app.sparql.query().convert()

        vars_list = results.get("head", {}).get("vars", [])
        bindings = results.get("results", {}).get("bindings", [])

        # Simplify bindings
        simplified = []
        for binding in bindings:
            row = {}
            for var in vars_list:
                if var in binding:
                    val = binding[var].get("value", "")
                    # Shorten URIs
                    if val.startswith("http://orkg.org/orkg/"):
                        val = val.split("/")[-1]
                    row[var] = val
            simplified.append(row)

        session_journal.completed_steps.append(
            f"RunORKGSPARQL() -> {len(simplified)} results"
        )

        response = SPARQLResponse(
            vars=vars_list,
            bindings=simplified,
            raw_json=results
        )
        return response.model_dump_json(indent=2)

    except Exception as e:
        error_msg = f"SPARQL Error: {str(e)}"
        logger.error(error_msg)
        session_journal.failed_attempts.append(f"RunORKGSPARQL: {str(e)}")
        return json.dumps({"error": error_msg, "query": query}, indent=2)


# ==============================================================================
# TOOL 6: GetResourceLabel
# ==============================================================================

@mcp.tool()
async def GetResourceLabel(
    app_context: Context,
    resource_id: str
) -> str:
    """
    Quick lookup of a resource's human-readable label.

    Args:
        resource_id: The resource ID (e.g., "R12345")

    Returns:
        JSON with resource label
    """
    app = app_context.request_context.lifespan_context

    try:
        query = f"""
        SELECT ?label WHERE {{
            GRAPH <{SCIQA_GRAPH}> {{
                orkgr:{resource_id} rdfs:label ?label .
            }}
        }}
        """
        full_query = SPARQL_PREFIXES + query
        app.sparql.setQuery(full_query)
        results = app.sparql.query().convert()
        bindings = results.get("results", {}).get("bindings", [])

        if bindings:
            label = bindings[0].get("label", {}).get("value", "")
            session_journal.visited_nodes[resource_id] = label
            return json.dumps({
                "resource_id": resource_id,
                "label": label,
                "status": "Found"
            }, indent=2)
        else:
            return json.dumps({
                "resource_id": resource_id,
                "label": None,
                "status": "Not found"
            }, indent=2)

    except Exception as e:
        error_msg = f"Error in GetResourceLabel: {str(e)}"
        logger.error(error_msg)
        return json.dumps({"error": error_msg}, indent=2)


# ==============================================================================
# TOOL 7: BatchGetResourceLabels
# ==============================================================================

@mcp.tool()
async def BatchGetResourceLabels(
    app_context: Context,
    resource_ids: List[str]
) -> str:
    """
    Efficiently resolve multiple resource IDs to labels in a single SPARQL call.

    Args:
        resource_ids: List of resource IDs (e.g., ["R12345", "R67890"])

    Returns:
        JSON with resolved labels
    """
    app = app_context.request_context.lifespan_context

    try:
        if not resource_ids:
            return json.dumps({"status": "No IDs provided"}, indent=2)

        unique_ids = list(set(resource_ids))
        values_clause = " ".join([f"orkgr:{rid}" for rid in unique_ids])

        query = f"""
        SELECT ?resource ?label WHERE {{
            GRAPH <{SCIQA_GRAPH}> {{
                VALUES ?resource {{ {values_clause} }}
                OPTIONAL {{ ?resource rdfs:label ?label . }}
            }}
        }}
        """
        full_query = SPARQL_PREFIXES + query
        app.sparql.setQuery(full_query)
        results = app.sparql.query().convert()
        bindings = results.get("results", {}).get("bindings", [])

        resolved = {}
        for binding in bindings:
            resource_uri = binding.get("resource", {}).get("value", "")
            label = binding.get("label", {}).get("value", None)
            resource_id = resource_uri.split("/")[-1]

            if label:
                resolved[resource_id] = label
                session_journal.visited_nodes[resource_id] = label

        not_found = [rid for rid in unique_ids if rid not in resolved]

        return json.dumps({
            "resolved": resolved,
            "not_found": not_found,
            "status": f"Resolved {len(resolved)}/{len(unique_ids)} resources"
        }, indent=2)

    except Exception as e:
        error_msg = f"Error in BatchGetResourceLabels: {str(e)}"
        logger.error(error_msg)
        return json.dumps({"error": error_msg}, indent=2)


# ==============================================================================
# TOOL 8: GetPaperContributions
# ==============================================================================

@mcp.tool()
async def GetPaperContributions(
    app_context: Context,
    paper_id: str
) -> str:
    """
    Get all contributions for a paper.

    Common pattern for ORKG: Papers have contributions via P31 (has_contribution).

    Args:
        paper_id: The paper resource ID (e.g., "R12345")

    Returns:
        JSON with paper contributions
    """
    app = app_context.request_context.lifespan_context

    try:
        query = f"""
        SELECT ?contrib ?label ?problem ?method WHERE {{
            GRAPH <{SCIQA_GRAPH}> {{
                orkgr:{paper_id} orkgp:P31 ?contrib .
                OPTIONAL {{ ?contrib rdfs:label ?label . }}
                OPTIONAL {{ ?contrib orkgp:P32 ?problem . }}
                OPTIONAL {{ ?contrib orkgp:P2 ?method . }}
            }}
        }}
        """
        full_query = SPARQL_PREFIXES + query
        app.sparql.setQuery(full_query)
        results = app.sparql.query().convert()
        bindings = results.get("results", {}).get("bindings", [])

        contributions = []
        for binding in bindings:
            contrib_uri = binding.get("contrib", {}).get("value", "")
            contrib_id = contrib_uri.split("/")[-1]
            label = binding.get("label", {}).get("value", "")

            contrib = {"id": contrib_id, "label": label}

            if "problem" in binding:
                contrib["problem"] = binding["problem"]["value"].split("/")[-1]
            if "method" in binding:
                contrib["method"] = binding["method"]["value"].split("/")[-1]

            contributions.append(contrib)
            session_journal.visited_nodes[contrib_id] = label

        session_journal.completed_steps.append(
            f"GetPaperContributions('{paper_id}') -> {len(contributions)} contributions"
        )

        return json.dumps({
            "paper_id": paper_id,
            "contributions": contributions,
            "count": len(contributions),
            "status": f"Found {len(contributions)} contributions"
        }, indent=2)

    except Exception as e:
        error_msg = f"Error in GetPaperContributions: {str(e)}"
        logger.error(error_msg)
        return json.dumps({"error": error_msg}, indent=2)


# ==============================================================================
# TOOL 9: GetPaperAuthors
# ==============================================================================

@mcp.tool()
async def GetPaperAuthors(
    app_context: Context,
    paper_id: str
) -> str:
    """
    Get all authors of a paper.

    Common pattern for ORKG: Papers have authors via P27 or P6.

    Args:
        paper_id: The paper resource ID (e.g., "R12345")

    Returns:
        JSON with paper authors
    """
    app = app_context.request_context.lifespan_context

    try:
        # Try both P27 and P6 predicates
        query = f"""
        SELECT DISTINCT ?author ?label WHERE {{
            GRAPH <{SCIQA_GRAPH}> {{
                {{
                    orkgr:{paper_id} orkgp:P27 ?author .
                }} UNION {{
                    orkgr:{paper_id} orkgp:P6 ?author .
                }}
                OPTIONAL {{ ?author rdfs:label ?label . }}
            }}
        }}
        """
        full_query = SPARQL_PREFIXES + query
        app.sparql.setQuery(full_query)
        results = app.sparql.query().convert()
        bindings = results.get("results", {}).get("bindings", [])

        authors = []
        for binding in bindings:
            author_uri = binding.get("author", {}).get("value", "")
            author_id = author_uri.split("/")[-1]
            label = binding.get("label", {}).get("value", author_id)

            authors.append({"id": author_id, "name": label})
            session_journal.visited_nodes[author_id] = label

        session_journal.completed_steps.append(
            f"GetPaperAuthors('{paper_id}') -> {len(authors)} authors"
        )

        return json.dumps({
            "paper_id": paper_id,
            "authors": authors,
            "count": len(authors),
            "status": f"Found {len(authors)} authors"
        }, indent=2)

    except Exception as e:
        error_msg = f"Error in GetPaperAuthors: {str(e)}"
        logger.error(error_msg)
        return json.dumps({"error": error_msg}, indent=2)


# ==============================================================================
# TOOL 10: GetContributionMethods
# ==============================================================================

@mcp.tool()
async def GetContributionMethods(
    app_context: Context,
    contribution_id: str
) -> str:
    """
    Get methods/approaches used in a contribution.

    Common pattern: Contributions employ methods via P2.

    Args:
        contribution_id: The contribution resource ID

    Returns:
        JSON with methods used
    """
    app = app_context.request_context.lifespan_context

    try:
        query = f"""
        SELECT ?method ?label WHERE {{
            GRAPH <{SCIQA_GRAPH}> {{
                orkgr:{contribution_id} orkgp:P2 ?method .
                OPTIONAL {{ ?method rdfs:label ?label . }}
            }}
        }}
        """
        full_query = SPARQL_PREFIXES + query
        app.sparql.setQuery(full_query)
        results = app.sparql.query().convert()
        bindings = results.get("results", {}).get("bindings", [])

        methods = []
        for binding in bindings:
            method_uri = binding.get("method", {}).get("value", "")
            method_id = method_uri.split("/")[-1]
            label = binding.get("label", {}).get("value", method_id)

            methods.append({"id": method_id, "label": label})

        session_journal.completed_steps.append(
            f"GetContributionMethods('{contribution_id}') -> {len(methods)} methods"
        )

        return json.dumps({
            "contribution_id": contribution_id,
            "methods": methods,
            "count": len(methods),
            "status": f"Found {len(methods)} methods"
        }, indent=2)

    except Exception as e:
        error_msg = f"Error in GetContributionMethods: {str(e)}"
        logger.error(error_msg)
        return json.dumps({"error": error_msg}, indent=2)


# ==============================================================================
# TOOL 11: GetResearchFieldPapers
# ==============================================================================

@mcp.tool()
async def GetResearchFieldPapers(
    app_context: Context,
    field_name: str,
    limit: int = 10
) -> str:
    """
    List papers in a specific research field.

    Args:
        field_name: Research field name (e.g., "Natural Language Processing")
        limit: Maximum number of papers to return (default: 10)

    Returns:
        JSON with papers in the field
    """
    app = app_context.request_context.lifespan_context

    try:
        # First find the field by label
        query = f"""
        SELECT ?paper ?paperLabel ?year WHERE {{
            GRAPH <{SCIQA_GRAPH}> {{
                ?field rdfs:label ?fieldLabel .
                FILTER(CONTAINS(LCASE(STR(?fieldLabel)), LCASE("{field_name}")))
                ?paper orkgp:P30 ?field .
                OPTIONAL {{ ?paper rdfs:label ?paperLabel . }}
                OPTIONAL {{ ?paper orkgp:P29 ?year . }}
            }}
        }} LIMIT {limit}
        """
        full_query = SPARQL_PREFIXES + query
        app.sparql.setQuery(full_query)
        results = app.sparql.query().convert()
        bindings = results.get("results", {}).get("bindings", [])

        papers = []
        for binding in bindings:
            paper_uri = binding.get("paper", {}).get("value", "")
            paper_id = paper_uri.split("/")[-1]
            label = binding.get("paperLabel", {}).get("value", "")
            year = binding.get("year", {}).get("value", "")

            papers.append({
                "id": paper_id,
                "title": label,
                "year": year
            })
            if label:
                session_journal.visited_nodes[paper_id] = label

        session_journal.completed_steps.append(
            f"GetResearchFieldPapers('{field_name}') -> {len(papers)} papers"
        )

        return json.dumps({
            "field_name": field_name,
            "papers": papers,
            "count": len(papers),
            "status": f"Found {len(papers)} papers in {field_name}"
        }, indent=2)

    except Exception as e:
        error_msg = f"Error in GetResearchFieldPapers: {str(e)}"
        logger.error(error_msg)
        return json.dumps({"error": error_msg}, indent=2)


# ==============================================================================
# TOOL 12: VerifyNumericCondition
# ==============================================================================

@mcp.tool()
async def VerifyNumericCondition(
    app_context: Context,
    value1: str,
    operator: Literal["<", ">", "<=", ">=", "==", "!="],
    value2: str,
    unit: str = ""
) -> str:
    """
    Performs deterministic mathematical comparison between two values.

    Returns a definitive TRUE/FALSE verdict for numeric comparisons.
    Use this whenever the question involves numeric conditions, thresholds,
    or comparisons (e.g., "more than 10,000", "before 2020", "greater than X").

    Supported formats:
    - Plain numbers: "150", "1500000"
    - Numbers with multipliers: "150 million", "1.5k"
    - Dates (ISO format): "2019-05-15"
    - Numbers with commas: "1,500,000"

    Args:
        value1: First value to compare
        operator: Comparison operator: <, >, <=, >=, ==, !=
        value2: Second value to compare
        unit: Optional unit description for context (e.g., "questions", "papers")

    Returns:
        JSON with verdict (TRUE/FALSE/ERROR) and explanation
    """
    logger.info(f"VerifyNumericCondition: {value1} {operator} {value2} ({unit})")

    def parse_numeric(val: str) -> float:
        """Parse numeric value, handling common formats."""
        val = val.strip().lower()

        # Handle dates (convert to timestamp for comparison)
        if "-" in val and len(val) >= 10:
            try:
                from datetime import datetime
                dt = datetime.fromisoformat(val.split("T")[0])
                return dt.timestamp()
            except Exception:
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
        num1 = parse_numeric(value1)
        num2 = parse_numeric(value2)

        comparisons = {
            "<": num1 < num2,
            ">": num1 > num2,
            "<=": num1 <= num2,
            ">=": num1 >= num2,
            "==": abs(num1 - num2) < 1e-9,
            "!=": abs(num1 - num2) >= 1e-9
        }

        result = comparisons[operator]
        verdict = "TRUE" if result else "FALSE"

        unit_str = f" {unit}" if unit else ""
        explanation = f"{value1}{unit_str} {operator} {value2}{unit_str} → {num1} {operator} {num2} = {verdict}"

        logger.info(f"VerifyNumericCondition result: {verdict}")

        session_journal.verified_facts.append({
            "fact": explanation,
            "source": "VerifyNumericCondition"
        })
        session_journal.completed_steps.append(f"Verified: {explanation}")

        response = NumericComparisonResponse(
            verdict=verdict,
            explanation=explanation,
            value1=f"{num1}{unit_str}",
            value2=f"{num2}{unit_str}",
            operator=operator
        )
        return response.model_dump_json(indent=2)

    except Exception as e:
        logger.error(f"VerifyNumericCondition failed: {e}")
        session_journal.failed_attempts.append(
            f"VerifyNumericCondition({value1} {operator} {value2}): {str(e)[:100]}"
        )
        response = NumericComparisonResponse(
            verdict="ERROR",
            explanation=f"Could not compare values: {str(e)}",
            value1=value1,
            value2=value2,
            operator=operator
        )
        return response.model_dump_json(indent=2)


# ==============================================================================
# TOOL 13: GetResourceSummary
# ==============================================================================

@mcp.tool()
async def GetResourceSummary(
    app_context: Context,
    resource_id: str
) -> str:
    """
    Get a comprehensive summary of an ORKG resource with ALL predicates and values in ONE call.

    This is more efficient than GetResourceDetails + multiple GetRelationTargets calls.
    Returns all predicates grouped by type (literal values vs. resource links).

    Use this when:
    - You need to explore what data a resource has
    - You need multiple predicate values from the same resource
    - A predicate search returned empty and you need to discover available predicates

    Args:
        resource_id: The resource ID (e.g., "R12345")

    Returns:
        JSON with resource_id, label, predicates (grouped), summary_stats, and status
    """
    app = app_context.request_context.lifespan_context

    try:
        query = f"""
        SELECT ?pred ?predLabel ?obj ?objLabel WHERE {{
            GRAPH <{SCIQA_GRAPH}> {{
                orkgr:{resource_id} ?pred ?obj .
                OPTIONAL {{ ?pred rdfs:label ?predLabel }}
                OPTIONAL {{ ?obj rdfs:label ?objLabel }}
            }}
        }}
        """
        full_query = SPARQL_PREFIXES + query
        app.sparql.setQuery(full_query)
        results = app.sparql.query().convert()
        bindings = results.get("results", {}).get("bindings", [])

        # Get resource label
        label = ""
        literal_values = {}
        resource_links = {}

        for binding in bindings:
            pred_uri = binding.get("pred", {}).get("value", "")
            pred_label = binding.get("predLabel", {}).get("value", "")
            obj_value = binding.get("obj", {}).get("value", "")
            obj_label = binding.get("objLabel", {}).get("value", "")
            obj_type = binding.get("obj", {}).get("type", "")

            # Extract predicate ID
            pred_id = pred_uri.split("/")[-1]

            # Skip RDF system predicates
            if pred_uri.startswith("http://www.w3.org/1999/02/22-rdf-syntax-ns#") or \
               pred_uri.startswith("http://www.w3.org/2000/01/rdf-schema#label"):
                if pred_id == "label":
                    label = obj_value
                continue

            if pred_id == "label":
                label = obj_value
                continue

            display_name = pred_label if pred_label else pred_id

            if obj_type == "literal" or not obj_value.startswith("http"):
                # Literal value
                if display_name not in literal_values:
                    literal_values[display_name] = []
                entry = {"value": obj_value, "predicate_id": pred_id}
                if entry not in literal_values[display_name]:
                    literal_values[display_name].append(entry)
            else:
                # Resource link
                obj_id = obj_value.split("/")[-1]
                if display_name not in resource_links:
                    resource_links[display_name] = []
                entry = {"id": obj_id, "predicate_id": pred_id}
                if obj_label:
                    entry["label"] = obj_label
                    session_journal.visited_nodes[obj_id] = obj_label
                if entry not in resource_links[display_name]:
                    resource_links[display_name].append(entry)

        # Update journal
        session_journal.visited_nodes[resource_id] = label or resource_id
        if literal_values or resource_links:
            if resource_id not in session_journal.found_values:
                session_journal.found_values[resource_id] = {}
            for attr_name, attr_vals in literal_values.items():
                session_journal.found_values[resource_id][attr_name] = attr_vals
            for rel_name, rel_vals in resource_links.items():
                session_journal.found_values[resource_id][rel_name] = rel_vals

        session_journal.completed_steps.append(
            f"GetResourceSummary('{resource_id}') -> {label}: "
            f"{len(literal_values)} literal predicates, {len(resource_links)} resource links"
        )

        response = {
            "resource_id": resource_id,
            "label": label,
            "literal_values": literal_values,
            "resource_links": resource_links,
            "summary_stats": {
                "literal_predicate_count": len(literal_values),
                "resource_link_count": len(resource_links),
                "total_literal_values": sum(len(v) for v in literal_values.values()),
                "total_resource_links": sum(len(v) for v in resource_links.values())
            },
            "status": f"Found {len(literal_values)} literal predicates and {len(resource_links)} resource links"
        }

        return json.dumps(response, indent=2)

    except Exception as e:
        error_msg = f"Error in GetResourceSummary: {str(e)}"
        logger.error(error_msg)
        session_journal.failed_attempts.append(f"GetResourceSummary('{resource_id}'): {str(e)}")
        return json.dumps({"error": error_msg}, indent=2)


# ==============================================================================
# TOOL 14: FindByPredicateValue
# ==============================================================================

@mcp.tool()
async def FindByPredicateValue(
    app_context: Context,
    predicate_id: str,
    value: str,
    match_type: Literal["exact", "contains", "greater", "less"] = "exact"
) -> str:
    """
    Reverse lookup: find resources by a predicate's value.

    Use this to find resources that have a specific value for a given predicate.
    E.g., "find all resources where P41923 > 10000" or "find resources with P30 = 'Computer Science'".

    Args:
        predicate_id: The predicate ID (e.g., "P30", "P41923")
        value: The value to search for
        match_type: How to match - "exact" (string equals), "contains" (substring),
                    "greater" (numeric >), "less" (numeric <)

    Returns:
        JSON with matching resources
    """
    app = app_context.request_context.lifespan_context

    try:
        safe_value = value.replace('"', '\\"')

        # Build FILTER clause based on match_type
        if match_type == "exact":
            filter_clause = f'FILTER(STR(?obj) = "{safe_value}")'
        elif match_type == "contains":
            filter_clause = f'FILTER(CONTAINS(LCASE(STR(?obj)), LCASE("{safe_value}")))'
        elif match_type == "greater":
            filter_clause = f'FILTER(xsd:decimal(?obj) > {safe_value})'
        elif match_type == "less":
            filter_clause = f'FILTER(xsd:decimal(?obj) < {safe_value})'
        else:
            filter_clause = f'FILTER(STR(?obj) = "{safe_value}")'

        query = f"""
        SELECT DISTINCT ?resource ?label WHERE {{
            GRAPH <{SCIQA_GRAPH}> {{
                ?resource orkgp:{predicate_id} ?obj .
                {filter_clause}
                OPTIONAL {{ ?resource rdfs:label ?label }}
            }}
        }} LIMIT 50
        """
        full_query = SPARQL_PREFIXES + query
        app.sparql.setQuery(full_query)
        results = app.sparql.query().convert()
        bindings = results.get("results", {}).get("bindings", [])

        matches = []
        for binding in bindings:
            resource_uri = binding.get("resource", {}).get("value", "")
            resource_label = binding.get("label", {}).get("value", "")
            rid = resource_uri.split("/")[-1]

            matches.append(ResourceMatch(
                original_id=rid,
                name=resource_label or rid,
                node_type="resource",
                relevance_score=1.0,
                available_predicates=[]
            ))
            if resource_label:
                session_journal.visited_nodes[rid] = resource_label

        session_journal.completed_steps.append(
            f"FindByPredicateValue('{predicate_id}', '{value}', '{match_type}') -> {len(matches)} results"
        )

        response = SearchResponse(matches=matches, result_count=len(matches))
        return response.model_dump_json(indent=2)

    except Exception as e:
        error_msg = f"Error in FindByPredicateValue: {str(e)}"
        logger.error(error_msg)
        session_journal.failed_attempts.append(
            f"FindByPredicateValue('{predicate_id}', '{value}'): {str(e)}"
        )
        return json.dumps({"error": error_msg}, indent=2)


# ==============================================================================
# TOOL 15: CompareResources
# ==============================================================================

@mcp.tool()
async def CompareResources(
    app_context: Context,
    resource_ids: List[str],
    predicate_id: str
) -> str:
    """
    Compare a specific predicate across multiple resources in one call.

    Returns sorted results (descending by numeric value if parseable).
    Use this for comparison questions like "which has more X" or "rank these by Y".

    Args:
        resource_ids: List of resource IDs to compare (e.g., ["R12345", "R67890"])
        predicate_id: The predicate to compare (e.g., "P41923")

    Returns:
        JSON with comparison results sorted by value
    """
    app = app_context.request_context.lifespan_context

    try:
        resource_uris = " ".join([f"orkgr:{rid}" for rid in resource_ids])

        query = f"""
        SELECT ?resource ?resourceLabel ?value WHERE {{
            GRAPH <{SCIQA_GRAPH}> {{
                VALUES ?resource {{ {resource_uris} }}
                ?resource orkgp:{predicate_id} ?value .
                OPTIONAL {{ ?resource rdfs:label ?resourceLabel }}
            }}
        }}
        """
        full_query = SPARQL_PREFIXES + query
        app.sparql.setQuery(full_query)
        results = app.sparql.query().convert()
        bindings = results.get("results", {}).get("bindings", [])

        # Process results
        resource_values = {}
        for binding in bindings:
            resource_uri = binding.get("resource", {}).get("value", "")
            resource_label = binding.get("resourceLabel", {}).get("value", "")
            value = binding.get("value", {}).get("value", "")
            rid = resource_uri.split("/")[-1]

            if rid not in resource_values:
                # Try to parse as float for sorting
                try:
                    normalized = float(value.replace(",", ""))
                except (ValueError, AttributeError):
                    normalized = None

                resource_values[rid] = {
                    "name": resource_label or rid,
                    "value": value,
                    "normalized": normalized
                }

        # Build comparison results
        comparison_results = []
        for rid in resource_ids:
            if rid in resource_values:
                data = resource_values[rid]
                comparison_results.append(ComparisonResult(
                    resource_id=rid,
                    name=data["name"],
                    value=data["value"],
                    normalized_value=data["normalized"]
                ))
            else:
                comparison_results.append(ComparisonResult(
                    resource_id=rid,
                    name=session_journal.visited_nodes.get(rid, rid),
                    value="N/A",
                    normalized_value=None
                ))

        # Sort by normalized value (descending)
        sorted_results = sorted(
            comparison_results,
            key=lambda x: x.normalized_value if x.normalized_value is not None else float('-inf'),
            reverse=True
        )

        # Update journal
        for result in sorted_results:
            if result.resource_id not in session_journal.found_values:
                session_journal.found_values[result.resource_id] = {}
            session_journal.found_values[result.resource_id][predicate_id] = result.value

        sorted_by = "numeric value (descending)" if any(
            r.normalized_value for r in sorted_results) else "order provided"

        session_journal.completed_steps.append(
            f"CompareResources({predicate_id}) for {len(resource_ids)} resources"
        )

        response = CompareResourcesResponse(
            predicate=predicate_id,
            results=sorted_results,
            sorted_by=sorted_by,
            status=f"Compared {len(sorted_results)} resources"
        )
        return response.model_dump_json(indent=2)

    except Exception as e:
        error_msg = f"Error in CompareResources: {str(e)}"
        logger.error(error_msg)
        session_journal.failed_attempts.append(
            f"CompareResources({predicate_id}): {str(e)}"
        )
        response = CompareResourcesResponse(
            predicate=predicate_id,
            results=[],
            sorted_by="error",
            status=f"Error: {str(e)}"
        )
        return response.model_dump_json(indent=2)


# ==============================================================================
# TOOL 16: FollowRelationPath
# ==============================================================================

@mcp.tool()
async def FollowRelationPath(
    app_context: Context,
    start_resource_id: str,
    relation_path: List[Dict[str, str]]
) -> str:
    """
    Navigate multi-hop relation paths to find connected resources in one SPARQL call.

    Instead of chaining multiple GetRelationTargets calls, this follows a chain of
    predicates from a starting resource in a single query.

    Common pattern: Paper --P31--> Contribution --domain_predicate--> Value

    Args:
        start_resource_id: Starting resource ID (e.g., "R12345")
        relation_path: List of steps, each with:
          - "predicate": Predicate ID (e.g., "P31")
          - "direction": "forward" (subject→object) or "backward" (object→subject)

    Returns:
        JSON with entities found at the end of the path and intermediate nodes
    """
    app = app_context.request_context.lifespan_context

    try:
        if not relation_path:
            return json.dumps({"error": "relation_path cannot be empty"}, indent=2)

        # Build SPARQL query dynamically
        hop_vars = [f"?hop{i}" for i in range(len(relation_path) + 1)]
        label_vars = [f"?hop{i}Label" for i in range(len(relation_path) + 1)]
        select_clause = "SELECT DISTINCT " + " ".join(hop_vars + label_vars)

        where_clauses = [f"BIND(orkgr:{start_resource_id} AS ?hop0)"]

        for i, step in enumerate(relation_path):
            predicate = step["predicate"]
            direction = step.get("direction", "forward")

            pred_uri = f"orkgp:{predicate}"
            current_var = f"?hop{i}"
            next_var = f"?hop{i+1}"

            if direction == "forward":
                where_clauses.append(f"{current_var} {pred_uri} {next_var} .")
            else:
                where_clauses.append(f"{next_var} {pred_uri} {current_var} .")

        # Add OPTIONAL labels for all hops
        for i in range(len(relation_path) + 1):
            where_clauses.append(f"OPTIONAL {{ ?hop{i} rdfs:label ?hop{i}Label }}")

        query = f"""
        {select_clause} WHERE {{
            GRAPH <{SCIQA_GRAPH}> {{
                {chr(10).join('    ' + c for c in where_clauses)}
            }}
        }} LIMIT 100
        """
        full_query = SPARQL_PREFIXES + query
        app.sparql.setQuery(full_query)
        results = app.sparql.query().convert()
        bindings = results.get("results", {}).get("bindings", [])

        if not bindings:
            session_journal.failed_attempts.append(
                f"FollowRelationPath({start_resource_id}): No results for path"
            )
            return json.dumps({
                "start_resource_id": start_resource_id,
                "relation_path": relation_path,
                "entities_found": [],
                "path_length": len(relation_path),
                "intermediate_nodes": {},
                "status": "No entities found following this path"
            }, indent=2)

        # Extract entities at each hop
        intermediate_nodes = {f"hop_{i}": {} for i in range(len(relation_path) + 1)}

        for binding in bindings:
            for i in range(len(relation_path) + 1):
                hop_var = f"hop{i}"
                label_var = f"hop{i}Label"
                if hop_var in binding:
                    uri = binding[hop_var]["value"]
                    obj_type = binding[hop_var].get("type", "")

                    if uri.startswith("http"):
                        entity_id = uri.split("/")[-1]
                    else:
                        entity_id = uri

                    hop_label = binding.get(label_var, {}).get("value", "")
                    intermediate_nodes[f"hop_{i}"][entity_id] = hop_label or entity_id

                    if hop_label:
                        session_journal.visited_nodes[entity_id] = hop_label

        # Final hop contains target entities
        final_hop = intermediate_nodes[f"hop_{len(relation_path)}"]
        final_entities = [{"id": eid, "label": elabel} for eid, elabel in final_hop.items()]

        # Simplify intermediate_nodes for output
        intermediate_summary = {
            k: list(v.keys()) for k, v in intermediate_nodes.items()
        }

        session_journal.verified_facts.append({
            "fact": f"Multi-hop path from {start_resource_id}: {len(final_entities)} entities found",
            "path": relation_path,
            "results": [e["id"] for e in final_entities[:10]],
            "source": "FollowRelationPath"
        })
        session_journal.completed_steps.append(
            f"FollowRelationPath({start_resource_id}, {len(relation_path)} hops) -> {len(final_entities)} entities"
        )

        return json.dumps({
            "start_resource_id": start_resource_id,
            "relation_path": relation_path,
            "entities_found": final_entities,
            "path_length": len(relation_path),
            "intermediate_nodes": intermediate_summary,
            "status": f"Found {len(final_entities)} entities"
        }, indent=2)

    except Exception as e:
        error_msg = f"Error in FollowRelationPath: {str(e)}"
        logger.error(error_msg)
        session_journal.failed_attempts.append(
            f"FollowRelationPath({start_resource_id}): {str(e)}"
        )
        return json.dumps({"error": error_msg}, indent=2)


# ==============================================================================
# TOOL 17: ManageJournal
# ==============================================================================

@mcp.tool()
async def ManageJournal(
    app_context: Context,
    action: Literal["read", "write", "clear", "add_step", "add_fact", "set_answer"],
    content: str = ""
) -> str:
    """
    Manage the agent's scratchpad/journal for state tracking.

    Actions:
    - read: Get current journal state
    - write: Overwrite partial_answer
    - clear: Reset entire journal
    - add_step: Add a completed step
    - add_fact: Add a verified fact (JSON format)
    - set_answer: Set the partial answer

    Args:
        action: The action to perform
        content: Content for write/add operations

    Returns:
        Current journal state as string
    """
    global session_journal

    try:
        if action == "read":
            return session_journal.to_str()

        elif action == "write":
            session_journal.partial_answer = content
            return f"Updated partial_answer to: {content[:100]}..."

        elif action == "clear":
            session_journal = JournalState()
            return "Journal cleared."

        elif action == "add_step":
            session_journal.completed_steps.append(content)
            return f"Added step: {content}"

        elif action == "add_fact":
            try:
                fact = json.loads(content)
                session_journal.verified_facts.append(fact)
                return f"Added verified fact: {content[:100]}..."
            except json.JSONDecodeError:
                session_journal.verified_facts.append({"raw": content})
                return f"Added raw fact: {content[:100]}..."

        elif action == "set_answer":
            session_journal.partial_answer = content
            return f"Set partial answer: {content[:100]}..."

        else:
            return f"Unknown action: {action}"

    except Exception as e:
        error_msg = f"Error in ManageJournal: {str(e)}"
        logger.error(error_msg)
        return error_msg


# ==============================================================================
# TOOL 18: GetJournalSummary
# ==============================================================================

@mcp.tool()
async def GetJournalSummary(
    app_context: Context
) -> str:
    """
    Get a formatted summary of all discoveries in the journal.

    CALL THIS BEFORE PROVIDING YOUR FINAL ANSWER to ensure you've captured all findings.

    Returns:
        Formatted summary of visited nodes, discovered values, and completed steps
    """
    return session_journal.to_str()


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":
    mcp.run()
