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
- GetComparisonContributions: Navigate Comparison -> Contribution -> Value pattern
- FindAuthorPapers: Find papers by author name (SPARQL-based, better than vector for names)
- AggregateComparisonValues: AVG/SUM/MIN/MAX/COUNT/MODE_TOP over a Comparison's
  contributions, with optional GROUP BY and prefilter (handles HAS_VALUE indirection)
- FindFrequentValues: Cross-resource aggregation without a single Comparison anchor
- FindCoAuthors: Find co-authors of an author across all their papers in one call
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
import re
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
    top_n: int = 5,
    node_type_filter: str = "",
) -> str:
    """
    Search for ORKG resources (papers, authors, contributions, etc.) by semantic similarity.

    Use this to find entities when you have a natural language description.

    Args:
        semantic_query: Natural language description (e.g., "machine learning paper", "NLP contribution")
        top_n: Number of results to return (default: 5)
        node_type_filter: Optional ORKG class name (e.g. "Comparison", "Paper",
            "Contribution", "Author", "Dataset", "Model", "Metric"). When set,
            only results with `rdf:type orkgc:<class>` are returned. Use this
            for SciQA aggregation questions where you specifically need a
            Comparison (set to "Comparison") rather than the highest-scored
            semantic hit, which is often a Paper or Contribution. The vector
            search runs with a larger candidate pool (4×top_n), then filters
            to the requested class via a single SPARQL ASK-batch.

    Returns:
        JSON with matching resources and their available predicates
    """
    app = app_context.request_context.lifespan_context

    try:
        query_vector = get_embedding(app.embedding_client, semantic_query)

        # When filtering by class, over-fetch and prune so we still return
        # roughly top_n hits of the desired type.
        candidate_limit = max(top_n * 4, 20) if node_type_filter else top_n
        search_results = app.qdrant.query_points(
            collection_name=COLLECTION_ENTITIES,
            query=query_vector,
            limit=candidate_limit,
            score_threshold=ENTITY_THRESHOLD
        ).points

        # If a node_type_filter is supplied, batch-check rdf:type via SPARQL
        # and keep only candidates matching the requested ORKG class.
        if node_type_filter and search_results:
            wanted_class = node_type_filter.strip()
            if wanted_class.startswith("orkgc:"):
                wanted_class = wanted_class[6:]
            candidate_ids = [
                (hit.payload or {}).get("uri", "").split("/")[-1]
                for hit in search_results
            ]
            candidate_ids = [cid for cid in candidate_ids if cid]
            if candidate_ids:
                values = " ".join(f"orkgr:{cid}" for cid in candidate_ids)
                type_query = f"""
                SELECT ?resource WHERE {{
                    GRAPH <{SCIQA_GRAPH}> {{
                        VALUES ?resource {{ {values} }}
                        ?resource rdf:type orkgc:{wanted_class} .
                    }}
                }}
                """
                full = SPARQL_PREFIXES + type_query
                app.sparql.setQuery(full)
                tres = app.sparql.query().convert()
                matched_ids = {
                    b["resource"]["value"].split("/")[-1]
                    for b in tres.get("results", {}).get("bindings", [])
                }
                search_results = [
                    hit for hit in search_results
                    if (hit.payload or {}).get("uri", "").split("/")[-1] in matched_ids
                ][:top_n]
            else:
                search_results = []

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

        search_results = app.qdrant.query_points(
            collection_name=COLLECTION_RELATIONS,
            query=query_vector,
            limit=top_n,
            score_threshold=RELATION_THRESHOLD
        ).points

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

        # Fallback: if no direct targets found, try reverse lookup
        # (resource might be a value within a comparison contribution)
        if not targets:
            fallback_query = f"""
            SELECT ?o ?oLabel WHERE {{
                GRAPH <{SCIQA_GRAPH}> {{
                    ?contrib ?somePred orkgr:{resource_id} .
                    ?contrib {pred_uri} ?o .
                    OPTIONAL {{ ?o rdfs:label ?oLabel . }}
                }}
            }}
            """
            full_fallback = SPARQL_PREFIXES + fallback_query
            app.sparql.setQuery(full_fallback)
            fb_results = app.sparql.query().convert()
            fb_bindings = fb_results.get("results", {}).get("bindings", [])

            for binding in fb_bindings:
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

    Auto-wrapping behavior: If the query does not already contain a GRAPH clause,
    the server automatically wraps it with GRAPH <http://sciqa.org/kg> { ... }.
    Solution modifiers (LIMIT, ORDER BY, OFFSET, GROUP BY, HAVING) are extracted
    before wrapping and re-appended outside the GRAPH block (per SPARQL spec).

    Available prefixes:
    - orkgr: <http://orkg.org/orkg/resource/>
    - orkgp: <http://orkg.org/orkg/predicate/>
    - orkgc: <http://orkg.org/orkg/class/>
    - rdfs:, rdf:, xsd:, owl:

    Args:
        query: SPARQL query WITHOUT PREFIX declarations. Can include trailing
               solution modifiers (LIMIT, ORDER BY, OFFSET, GROUP BY).

    Returns:
        JSON with query results
    """
    app = app_context.request_context.lifespan_context

    try:
        # Detect if this is an ASK query
        is_ask = query.strip().upper().startswith("ASK")

        # Wrap query with graph if not already wrapped
        if "GRAPH" not in query.upper():
            if "WHERE" in query.upper() or is_ask:
                # Extract trailing solution modifiers (LIMIT, ORDER BY, OFFSET, GROUP BY)
                # that must stay outside the WHERE/GRAPH blocks
                modifier_pattern = r'(\})\s*((?:ORDER\s+BY|GROUP\s+BY|HAVING|LIMIT|OFFSET)\b.*)$'
                modifier_match = re.search(modifier_pattern, query, re.IGNORECASE | re.DOTALL)

                if modifier_match:
                    # Split query body from trailing modifiers
                    modifier_start = modifier_match.start(2)
                    trailing_modifiers = query[modifier_start:].strip()
                    query_body = query[:modifier_start].strip()
                else:
                    trailing_modifiers = ""
                    query_body = query.rstrip()

                # Wrap with GRAPH clause
                query_body = query_body.replace("{", f"{{ GRAPH <{SCIQA_GRAPH}> {{", 1)
                if query_body.endswith("}"):
                    query_body = query_body[:-1] + "} }"

                query = query_body + ("\n" + trailing_modifiers if trailing_modifiers else "")

        full_query = SPARQL_PREFIXES + query
        app.sparql.setQuery(full_query)
        results = app.sparql.query().convert()

        # Handle ASK queries (return boolean result)
        if is_ask:
            boolean_result = results.get("boolean", False)
            session_journal.completed_steps.append(
                f"RunORKGSPARQL(ASK) -> {boolean_result}"
            )
            response = SPARQLResponse(
                vars=["result"],
                bindings=[{"result": str(boolean_result)}],
                raw_json=results
            )
            return response.model_dump_json(indent=2)

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
# TOOL 17: GetComparisonContributions
# ==============================================================================

@mcp.tool()
async def GetComparisonContributions(
    app_context: Context,
    comparison_id: str,
    domain_predicate: str = "",
    filter_value: str = "",
    filter_type: str = "contains"
) -> str:
    """
    Navigate the Comparison -> compareContribution -> Contribution pattern.

    Many SciQA questions involve Comparison resources that link to multiple
    Contributions via the compareContribution predicate. Each Contribution
    then has domain-specific predicates (e.g., HAS_VALUE, P43156).

    **Usage modes:**
    1. Without domain_predicate: Returns contributions with their available predicates
       (schema discovery mode - use this first to see what predicates exist)
    2. With domain_predicate: Returns contributions filtered by that predicate's values
    3. With domain_predicate + filter_value: Returns only contributions matching the filter

    Args:
        comparison_id: The Comparison resource ID (e.g., "R44073")
        domain_predicate: Optional predicate to retrieve from contributions (e.g., "HAS_VALUE", "P43156")
        filter_value: Optional value to filter by (requires domain_predicate)
        filter_type: Filter mode: "exact", "contains", "greater", "less" (default: "contains")

    Returns:
        JSON with contributions and their domain predicate values
    """
    app = app_context.request_context.lifespan_context

    try:
        if domain_predicate:
            # Sanitize: LLM may pass comma/space-separated predicates like "P43156, P43133"
            raw_preds = [p.strip() for p in re.split(r'[,\s]+', domain_predicate) if p.strip()]
            # Filter to valid predicate-like tokens (alphanumeric + underscore)
            preds = [p for p in raw_preds if re.match(r'^[A-Za-z_]\w*$', p)]
            if not preds:
                return json.dumps({"error": f"Invalid domain_predicate: {domain_predicate!r}. Provide a single predicate ID like 'P43156'."}, indent=2)

            if len(preds) == 1:
                pred_filter = f"?contribution orkgp:{preds[0]} ?value ."
            else:
                # Multiple predicates: use VALUES to match any of them
                values_list = " ".join(f"orkgp:{p}" for p in preds)
                pred_filter = f"VALUES ?pred {{ {values_list} }}\n        ?contribution ?pred ?value ."

            # Mode 2/3: Get contributions with specific domain predicate values
            # Also follow HAS_VALUE indirection for nested values
            query = f"""
SELECT ?contribution ?contribLabel ?value ?valueLabel ?nestedValue WHERE {{
    GRAPH <{SCIQA_GRAPH}> {{
        orkgr:{comparison_id} orkgp:compareContribution ?contribution .
        {pred_filter}
        OPTIONAL {{ ?contribution rdfs:label ?contribLabel }}
        OPTIONAL {{ ?value rdfs:label ?valueLabel }}
        OPTIONAL {{ ?value orkgp:HAS_VALUE ?nestedValue }}
    }}
}}
"""
            full_query = SPARQL_PREFIXES + query
            app.sparql.setQuery(full_query)
            results = app.sparql.query().convert()

            bindings = results.get("results", {}).get("bindings", [])
            contributions = {}

            for b in bindings:
                contrib_uri = b.get("contribution", {}).get("value", "")
                contrib_id = contrib_uri.split("/")[-1] if "/" in contrib_uri else contrib_uri
                contrib_label = b.get("contribLabel", {}).get("value", contrib_id)

                val_raw = b.get("value", {}).get("value", "")
                val_id = val_raw.split("/")[-1] if val_raw.startswith("http://") else val_raw
                val_label = b.get("valueLabel", {}).get("value", val_id)

                if contrib_id not in contributions:
                    contributions[contrib_id] = {
                        "id": contrib_id,
                        "label": contrib_label,
                        "values": []
                    }
                value_entry = {
                    "id": val_id,
                    "label": val_label,
                    "raw": val_raw
                }
                # Include nested value from HAS_VALUE indirection if present
                nested = b.get("nestedValue", {}).get("value")
                if nested:
                    value_entry["nested_value"] = nested
                contributions[contrib_id]["values"].append(value_entry)

            # Apply filter if specified
            if filter_value and contributions:
                filtered = {}
                for cid, cdata in contributions.items():
                    matching_values = []
                    for v in cdata["values"]:
                        match = False
                        check_str = v["label"].lower()
                        fv = filter_value.lower()
                        if filter_type == "exact":
                            match = check_str == fv
                        elif filter_type == "contains":
                            match = fv in check_str
                        elif filter_type == "greater":
                            try:
                                match = float(check_str) > float(fv)
                            except ValueError:
                                pass
                        elif filter_type == "less":
                            try:
                                match = float(check_str) < float(fv)
                            except ValueError:
                                pass
                        if match:
                            matching_values.append(v)
                    if matching_values:
                        filtered[cid] = {**cdata, "values": matching_values}
                contributions = filtered

            result_list = list(contributions.values())

            # Store values in journal so they persist across context trimming
            for contrib in result_list:
                cid = contrib["id"]
                if cid not in session_journal.found_values:
                    session_journal.found_values[cid] = {}
                session_journal.found_values[cid][domain_predicate] = [
                    v["label"] for v in contrib.get("values", [])
                ]

            session_journal.completed_steps.append(
                f"GetComparisonContributions({comparison_id}, {domain_predicate}) -> {len(result_list)} contributions"
            )
            session_journal.visited_nodes[comparison_id] = f"Comparison (queried {domain_predicate})"

            return json.dumps({
                "comparison_id": comparison_id,
                "domain_predicate": domain_predicate,
                "contributions": result_list,
                "count": len(result_list),
                "status": f"Found {len(result_list)} contributions with {domain_predicate}"
            }, indent=2)

        else:
            # Mode 1: Schema discovery - get contributions and their available predicates
            query = f"""
SELECT ?contribution ?contribLabel ?pred ?predLabel WHERE {{
    GRAPH <{SCIQA_GRAPH}> {{
        orkgr:{comparison_id} orkgp:compareContribution ?contribution .
        ?contribution ?pred ?obj .
        OPTIONAL {{ ?contribution rdfs:label ?contribLabel }}
        OPTIONAL {{ ?pred rdfs:label ?predLabel }}
    }}
}}
"""
            full_query = SPARQL_PREFIXES + query
            app.sparql.setQuery(full_query)
            results = app.sparql.query().convert()

            bindings = results.get("results", {}).get("bindings", [])
            contributions = {}

            for b in bindings:
                contrib_uri = b.get("contribution", {}).get("value", "")
                contrib_id = contrib_uri.split("/")[-1] if "/" in contrib_uri else contrib_uri
                contrib_label = b.get("contribLabel", {}).get("value", contrib_id)

                pred_uri = b.get("pred", {}).get("value", "")
                pred_id = pred_uri.split("/")[-1] if "/" in pred_uri else pred_uri
                pred_label = b.get("predLabel", {}).get("value", pred_id)

                if contrib_id not in contributions:
                    contributions[contrib_id] = {
                        "id": contrib_id,
                        "label": contrib_label,
                        "predicates": {}
                    }
                contributions[contrib_id]["predicates"][pred_id] = pred_label

            result_list = list(contributions.values())

            session_journal.completed_steps.append(
                f"GetComparisonContributions({comparison_id}, discovery) -> {len(result_list)} contributions"
            )
            session_journal.visited_nodes[comparison_id] = "Comparison (schema discovery)"

            return json.dumps({
                "comparison_id": comparison_id,
                "mode": "schema_discovery",
                "contributions": result_list,
                "count": len(result_list),
                "status": f"Found {len(result_list)} contributions. Use domain_predicate parameter to query specific values."
            }, indent=2)

    except Exception as e:
        error_msg = f"Error in GetComparisonContributions: {str(e)}"
        logger.error(error_msg)
        session_journal.failed_attempts.append(
            f"GetComparisonContributions({comparison_id}): {str(e)}"
        )
        return json.dumps({"error": error_msg}, indent=2)


# ==============================================================================
# TOOL 18: FindAuthorPapers
# ==============================================================================

@mcp.tool()
async def FindAuthorPapers(
    app_context: Context,
    author_name: str
) -> str:
    """
    Find all papers by an author name (case-insensitive partial match).

    Use this instead of FindResource when searching for papers by a specific
    author name, since vector similarity is poor for proper nouns/person names.

    Args:
        author_name: The author's name or partial name (e.g., "Kurt Thomas")

    Returns:
        JSON with matching papers and their authors
    """
    app = app_context.request_context.lifespan_context

    try:
        query = f"""
SELECT DISTINCT ?paper ?paperLabel ?author ?authorLabel WHERE {{
    GRAPH <{SCIQA_GRAPH}> {{
        {{
            {{ ?paper orkgp:P27 ?author }} UNION {{ ?paper orkgp:P6 ?author }}
            ?author rdfs:label ?authorLabel .
            FILTER(CONTAINS(LCASE(?authorLabel), LCASE("{author_name}")))
        }}
        UNION
        {{
            {{ ?paper orkgp:P27 ?authorLabel }} UNION {{ ?paper orkgp:P6 ?authorLabel }}
            FILTER(isLiteral(?authorLabel))
            FILTER(CONTAINS(LCASE(?authorLabel), LCASE("{author_name}")))
            BIND(?authorLabel AS ?author)
        }}
        OPTIONAL {{ ?paper rdfs:label ?paperLabel . }}
    }}
}}
"""
        full_query = SPARQL_PREFIXES + query
        app.sparql.setQuery(full_query)
        results = app.sparql.query().convert()
        bindings = results.get("results", {}).get("bindings", [])

        papers = {}
        for b in bindings:
            paper_uri = b.get("paper", {}).get("value", "")
            paper_id = paper_uri.split("/")[-1] if "/" in paper_uri else paper_uri
            paper_label = b.get("paperLabel", {}).get("value", paper_id)
            author_uri = b.get("author", {}).get("value", "")
            author_id = author_uri.split("/")[-1] if "/" in author_uri else author_uri
            author_label = b.get("authorLabel", {}).get("value", author_id)

            if paper_id not in papers:
                papers[paper_id] = {
                    "id": paper_id,
                    "label": paper_label,
                    "authors": []
                }
            papers[paper_id]["authors"].append({
                "id": author_id,
                "label": author_label
            })

        result_list = list(papers.values())

        # Update journal
        for p in result_list:
            session_journal.visited_nodes[p["id"]] = p["label"]
        session_journal.completed_steps.append(
            f"FindAuthorPapers('{author_name}') -> {len(result_list)} papers"
        )

        return json.dumps({
            "author_query": author_name,
            "papers": result_list,
            "count": len(result_list),
            "status": f"Found {len(result_list)} papers by authors matching '{author_name}'"
        }, indent=2)

    except Exception as e:
        error_msg = f"Error in FindAuthorPapers: {str(e)}"
        logger.error(error_msg)
        session_journal.failed_attempts.append(
            f"FindAuthorPapers('{author_name}'): {str(e)}"
        )
        return json.dumps({"error": error_msg}, indent=2)


# ==============================================================================
# TOOL 19: AggregateComparisonValues
# ==============================================================================

@mcp.tool()
async def AggregateComparisonValues(
    app_context: Context,
    comparison_id: str,
    value_predicate: str,
    agg: Literal["avg", "sum", "min", "max", "count", "count_distinct", "mode_top", "all_values"] = "avg",
    group_by_predicate: str = "",
    filter_predicate: str = "",
    filter_value: str = "",
    filter_match: Literal["exact", "contains", "regex"] = "contains",
    top_n: int = 1,
    value_via_group: bool = False,
    comparison_ids: str = "",
    value_parser: str = "leading_number",
    return_predicate: str = "",
    intermediate_predicate: str = "",
) -> str:
    """
    Aggregate values across the contributions of one or more Comparison resources.

    This is the right tool whenever a question asks for AVG/SUM/MIN/MAX/COUNT,
    "most common X", "frequency of X", or per-group min/max over the rows of a
    Featured Comparison. It encapsulates the common
    ``orkgr:RXXX orkgp:compareContribution ?contrib`` pattern, including the
    ``HAS_VALUE`` indirection that wraps numeric measurements, so you do not
    have to hand-write SPARQL and reason about ``xsd:decimal`` casts.

    Use this INSTEAD OF RunORKGSPARQL whenever the underlying pattern is
    "for each contribution of a Comparison, take the value of <predicate>
    and aggregate it (optionally grouped by another predicate)".

    For questions that span MULTIPLE related Comparisons (e.g. "across all
    energy-supply studies", "in the patient cohort comparisons"), pass the
    list via ``comparison_ids`` (comma-separated). The tool unions
    contributions across all listed Comparisons before aggregating — which
    is the correct shape when no single Comparison covers the question's
    scope. For graph-wide aggregation that has no Comparison anchor at all,
    use FindFrequentValues instead.

    Args:
        comparison_id: Comparison resource ID (e.g., "R44930"). Used when
            ``comparison_ids`` is empty. Pass "" to use ``comparison_ids`` only.
        value_predicate: Predicate ID of the value to aggregate
            (e.g., "P23140", "P43133"). Numeric predicates often store their
            literal under HAS_VALUE; this tool tries direct, HAS_VALUE, and
            label fallbacks automatically.
        agg: Aggregation:
            - avg / sum / min / max: numeric aggregate of parsed values
            - count: total contributions matching the filter
            - count_distinct: distinct values for value_predicate
            - mode_top: most frequent value (returns top_n)
            - all_values: list raw values per contribution (no aggregation)
        group_by_predicate: Optional predicate to GROUP BY. When set, the
            aggregate is computed per distinct value of this predicate
            (e.g., "extreme values per energy source").
        filter_predicate: Optional predicate used to subset contributions.
        filter_value: Value the filter_predicate must match.
        filter_match: "exact" (literal equals), "contains" (substring on
            label/value), or "regex" (full SPARQL regex, case-insensitive).
        top_n: For mode_top, how many top entries to return (default 1).
        value_via_group: If True (and group_by_predicate is set), the value path
            is ``?contrib group_pred ?group . ?group value_pred ?valueObj`` —
            i.e. the value lives on the GROUP node, not on the contribution.
            Use this for "extreme values of installed capacity grouped by
            energy source"-style questions where the measurement hangs off the
            grouping node.
        intermediate_predicate: Optional predicate for nested values that should
            be aggregated overall rather than returned per group. The value path
            becomes ``?contrib intermediate_pred ?intermediate .
            ?intermediate value_pred ?valueObj``. Use this for questions like
            "average energy generation of all energy sources", where energy
            sources are rows below each contribution.
        comparison_ids: Optional comma-separated list of Comparison IDs
            (e.g. "R153801,R155266,R44073"). When non-empty this OVERRIDES
            ``comparison_id`` and the aggregate is computed over the UNION of
            contributions from every listed Comparison. Use for cross-comparison
            questions where the relevant studies are split across multiple
            Featured Comparisons in the same domain.
        value_parser: Numeric parsing mode for numeric aggregations. Use
            ``embedded_number`` for values like ``n=54`` or ``86 %`` where the
            number does not necessarily start at character 0.
        return_predicate: Optional companion predicate to return for min/max rows
            (e.g. metric = geographic scale, return = studied location).

    Returns:
        JSON with `comparison_id` (or `comparison_ids` when multi-source),
        `value_predicate`, `agg`, optional `group_by`,
        `result` (scalar or list of {group, value, count}), `n_contributions`,
        and `status`.
    """
    app = app_context.request_context.lifespan_context

    def _norm_pred(p: str) -> str:
        p = p.strip()
        if not p:
            return ""
        if p.startswith("orkgp:"):
            return p
        if p.startswith("http"):
            return f"<{p}>"
        return f"orkgp:{p}"

    # Resolve scope: list of comparisons takes precedence
    cmp_id_list = [c.strip() for c in (comparison_ids or "").split(",") if c.strip()]
    multi_mode = bool(cmp_id_list)
    if not multi_mode and not (comparison_id or "").strip():
        return json.dumps(
            {"error": "Either comparison_id or comparison_ids must be set"},
            indent=2,
        )

    try:
        value_pred = _norm_pred(value_predicate)
        if not value_pred:
            return json.dumps({"error": "value_predicate is required"}, indent=2)
        group_pred = _norm_pred(group_by_predicate) if group_by_predicate else ""
        flt_pred = _norm_pred(filter_predicate) if filter_predicate else ""
        return_pred = _norm_pred(return_predicate) if return_predicate else ""
        intermediate_pred = _norm_pred(intermediate_predicate) if intermediate_predicate else ""
        if value_parser not in {"leading_number", "embedded_number", "auto"}:
            value_parser = "auto" if value_parser in {"string", "text", "literal"} else "leading_number"

        # Build optional filter clause
        filter_block = ""
        if flt_pred and filter_value:
            safe = filter_value.replace('"', '\\"')
            if filter_match == "exact":
                filter_block = (
                    f"?contrib {flt_pred} ?fobj .\n"
                    f"        OPTIONAL {{ ?fobj rdfs:label ?flbl }}\n"
                    f'        FILTER( STR(?fobj) = "{safe}" || STR(?flbl) = "{safe}" )\n'
                )
            elif filter_match == "regex":
                filter_block = (
                    f"?contrib {flt_pred} ?fobj .\n"
                    f"        OPTIONAL {{ ?fobj rdfs:label ?flbl }}\n"
                    f'        FILTER( REGEX(STR(?fobj), "{safe}", "i") '
                    f'|| REGEX(STR(?flbl), "{safe}", "i") )\n'
                )
            else:  # contains
                filter_block = (
                    f"?contrib {flt_pred} ?fobj .\n"
                    f"        OPTIONAL {{ ?fobj rdfs:label ?flbl }}\n"
                    f'        FILTER( CONTAINS(LCASE(STR(?fobj)), LCASE("{safe}")) '
                    f'|| CONTAINS(LCASE(STR(?flbl)), LCASE("{safe}")) )\n'
                )

        # Body retrieving raw rows: contribution, group, raw value (with
        # HAS_VALUE/label indirection lifted to ?val).
        group_select = "?group ?groupLabel " if group_pred else ""
        intermediate_select = "?intermediate ?intermediateLabel " if intermediate_pred else ""
        return_select = "?returnObj ?returnLabel ?returnNested " if return_pred else ""
        return_block = (
            f"OPTIONAL {{\n"
            f"          ?contrib {return_pred} ?returnObj .\n"
            f"          OPTIONAL {{ ?returnObj rdfs:label ?returnLabel }}\n"
            f"          OPTIONAL {{ ?returnObj orkgp:HAS_VALUE ?returnNested }}\n"
            f"        }}\n"
            if return_pred else ""
        )
        if group_pred and value_via_group:
            # contrib -- group_pred --> ?group -- value_pred --> ?valueObj
            value_block = (
                f"?contrib {group_pred} ?group .\n"
                f"        OPTIONAL {{ ?group rdfs:label ?groupLabel }}\n"
                f"        ?group {value_pred} ?valueObj .\n"
                f"        OPTIONAL {{ ?valueObj rdfs:label ?valueLabel }}\n"
                f"        OPTIONAL {{ ?valueObj orkgp:HAS_VALUE ?nestedValue }}\n"
            )
        elif intermediate_pred:
            group_clause = (
                f"?contrib {group_pred} ?group .\n"
                f"        OPTIONAL {{ ?group rdfs:label ?groupLabel }}\n"
                if group_pred else ""
            )
            value_block = (
                f"?contrib {intermediate_pred} ?intermediate .\n"
                f"        OPTIONAL {{ ?intermediate rdfs:label ?intermediateLabel }}\n"
                f"        ?intermediate {value_pred} ?valueObj .\n"
                f"        OPTIONAL {{ ?valueObj rdfs:label ?valueLabel }}\n"
                f"        OPTIONAL {{ ?valueObj orkgp:HAS_VALUE ?nestedValue }}\n"
                f"        {group_clause}"
            )
        else:
            group_clause = (
                f"?contrib {group_pred} ?group .\n"
                f"        OPTIONAL {{ ?group rdfs:label ?groupLabel }}\n"
                if group_pred else ""
            )
            value_block = (
                f"?contrib {value_pred} ?valueObj .\n"
                f"        OPTIONAL {{ ?valueObj rdfs:label ?valueLabel }}\n"
                f"        OPTIONAL {{ ?valueObj orkgp:HAS_VALUE ?nestedValue }}\n"
                f"        {group_clause}"
            )
        if multi_mode:
            cmp_values = " ".join(f"orkgr:{c}" for c in cmp_id_list)
            scope_clause = (
                f"VALUES ?cmp {{ {cmp_values} }}\n"
                f"        ?cmp orkgp:compareContribution ?contrib .\n"
            )
        else:
            scope_clause = (
                f"orkgr:{comparison_id} orkgp:compareContribution ?contrib .\n"
            )
        body_query = f"""
SELECT DISTINCT ?contrib {intermediate_select}{group_select}?valueObj ?valueLabel ?nestedValue {return_select}WHERE {{
    GRAPH <{SCIQA_GRAPH}> {{
        {scope_clause}
        {value_block}
        {return_block}
        {filter_block}
    }}
}}
"""
        full_query = SPARQL_PREFIXES + body_query
        app.sparql.setQuery(full_query)
        results = app.sparql.query().convert()
        bindings = results.get("results", {}).get("bindings", [])

        # Choose the most-literal representation per row
        def _row_value(b):
            for key in ("nestedValue", "valueLabel", "valueObj"):
                v = b.get(key, {}).get("value")
                if v is not None and v != "":
                    if v.startswith("http://orkg.org/orkg/"):
                        # If it's still a URI at this layer, fall back to its label
                        continue
                    return v
            v = b.get("valueObj", {}).get("value", "")
            return v.split("/")[-1] if v.startswith("http") else v

        def _row_group(b):
            if not group_pred:
                return None
            v = b.get("groupLabel", {}).get("value") or b.get("group", {}).get("value", "")
            if v.startswith("http://orkg.org/orkg/"):
                v = v.split("/")[-1]
            return v

        def _row_return(b):
            if not return_pred:
                return None
            for key in ("returnNested", "returnLabel", "returnObj"):
                v = b.get(key, {}).get("value")
                if v is not None and v != "":
                    if v.startswith("http://orkg.org/orkg/"):
                        continue
                    return v
            v = b.get("returnObj", {}).get("value", "")
            return v.split("/")[-1] if v.startswith("http") else (v or None)

        def _row_intermediate(b):
            if not intermediate_pred:
                return None
            v = b.get("intermediateLabel", {}).get("value") or b.get("intermediate", {}).get("value", "")
            if v.startswith("http://orkg.org/orkg/"):
                v = v.split("/")[-1]
            return v or None

        def _to_float(s):
            if s is None:
                return None
            t = str(s).strip().replace(",", "")
            pattern = r"[+-]?\d+(\.\d+)?([eE][+-]?\d+)?"
            if value_parser == "embedded_number":
                m = re.search(pattern, t)
            elif value_parser == "auto":
                m = re.match(pattern, t) or re.search(pattern, t)
            else:
                m = re.match(pattern, t)
            if not m:
                return None
            try:
                return float(m.group(0))
            except ValueError:
                return None

        rows = []
        for b in bindings:
            contrib = b.get("contrib", {}).get("value", "").split("/")[-1]
            rows.append({
                "contrib": contrib,
                "intermediate": _row_intermediate(b),
                "group": _row_group(b),
                "value": _row_value(b),
                "return_value": _row_return(b),
            })

        n_rows = len(rows)
        scope_label = (
            ",".join(cmp_id_list) if multi_mode else comparison_id
        )
        if n_rows == 0:
            session_journal.failed_attempts.append(
                f"AggregateComparisonValues({scope_label}, {value_predicate}): "
                f"no contributions matched"
            )
            return json.dumps({
                ("comparison_ids" if multi_mode else "comparison_id"): scope_label,
                "value_predicate": value_predicate,
                "agg": agg,
                "group_by": group_by_predicate or None,
                "result": None,
                "n_contributions": 0,
                "status": "No contributions matched the comparison/filter.",
            }, indent=2)

        # Group rows
        from collections import defaultdict, Counter
        groups: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
        for r in rows:
            groups[r["group"]].append(r)

        def _aggregate(items: List[Dict[str, Any]]):
            vals = [it["value"] for it in items]
            if agg == "all_values":
                return vals
            if agg == "count":
                return len(items)
            if agg == "count_distinct":
                return len({v for v in vals if v is not None})
            if agg == "mode_top":
                counter = Counter(v for v in vals if v is not None)
                top = counter.most_common(top_n)
                return [{"value": v, "count": c} for v, c in top]
            # numeric aggregates
            nums = [n for n in (_to_float(v) for v in vals) if n is not None]
            if not nums:
                return None
            if agg == "avg":
                return sum(nums) / len(nums)
            if agg == "sum":
                return sum(nums)
            if agg == "min":
                return min(nums)
            if agg == "max":
                return max(nums)
            return None

        if group_pred:
            grouped_result = []
            for g, items in groups.items():
                grouped_result.append({
                    "group": g,
                    "value": _aggregate(items),
                    "n": len(items),
                })
            # For numeric grouping, sort descending by value when comparable
            if agg in ("avg", "sum", "min", "max"):
                grouped_result.sort(
                    key=lambda x: x["value"] if isinstance(x["value"], (int, float)) else float("-inf"),
                    reverse=True,
                )
            elif agg in ("count", "count_distinct"):
                grouped_result.sort(key=lambda x: x["value"] or 0, reverse=True)
            result_payload: Any = grouped_result
        else:
            result_payload = _aggregate(rows)

        if return_pred and agg in ("min", "max"):
            scored_rows = [(r, _to_float(r["value"])) for r in rows]
            scored_rows = [(r, n) for r, n in scored_rows if n is not None]
            if scored_rows:
                extreme_value = (
                    max(n for _, n in scored_rows)
                    if agg == "max"
                    else min(n for _, n in scored_rows)
                )
                extreme_rows = [r for r, n in scored_rows if n == extreme_value]
                return_values = []
                for r in extreme_rows:
                    rv = r.get("return_value")
                    if rv is not None and rv not in return_values:
                        return_values.append(rv)
                result_payload = {
                    "extreme_value": extreme_value,
                    "return_predicate": return_predicate,
                    "return_values": return_values,
                    "rows": extreme_rows[:50],
                }

        # Journal — bucket per scope
        journal_key = scope_label
        if journal_key not in session_journal.found_values:
            session_journal.found_values[journal_key] = {}
        key = f"{agg}({value_predicate})" + (f" by {group_by_predicate}" if group_by_predicate else "")
        if intermediate_predicate:
            key += f" via {intermediate_predicate}"
        if return_predicate:
            key += f" return {return_predicate}"
        session_journal.found_values[journal_key][key] = result_payload
        session_journal.completed_steps.append(
            f"AggregateComparisonValues({scope_label}, {value_predicate}, {agg}) "
            f"-> {n_rows} contributions"
        )
        if multi_mode:
            for cid in cmp_id_list:
                session_journal.visited_nodes.setdefault(cid, "Comparison")
        else:
            session_journal.visited_nodes.setdefault(comparison_id, "Comparison")

        return json.dumps({
            ("comparison_ids" if multi_mode else "comparison_id"): scope_label,
            "value_predicate": value_predicate,
            "agg": agg,
            "group_by": group_by_predicate or None,
            "intermediate_predicate": intermediate_predicate or None,
            "value_parser": value_parser,
            "return_predicate": return_predicate or None,
            "filter": (
                {"predicate": filter_predicate, "value": filter_value, "match": filter_match}
                if filter_predicate else None
            ),
            "result": result_payload,
            "n_contributions": n_rows,
            "status": (
                f"Aggregated {n_rows} contribution rows with {agg}"
                + (f" grouped by {group_by_predicate}" if group_by_predicate else "")
                + (f" across {len(cmp_id_list)} comparisons" if multi_mode else "")
            ),
        }, indent=2, default=str)

    except Exception as e:
        error_msg = f"Error in AggregateComparisonValues: {str(e)}"
        logger.error(error_msg)
        scope_for_log = ",".join(cmp_id_list) if cmp_id_list else comparison_id
        session_journal.failed_attempts.append(
            f"AggregateComparisonValues({scope_for_log}, "
            f"{value_predicate}): {str(e)}"
        )
        return json.dumps({"error": error_msg}, indent=2)


# ==============================================================================
# TOOL 20: FindFrequentValues — cross-resource aggregation (no Comparison anchor)
# ==============================================================================

@mcp.tool()
async def FindFrequentValues(
    app_context: Context,
    value_predicate: str,
    agg: Literal["mode_top", "count", "count_distinct", "sum", "avg", "min", "max", "all_values"] = "mode_top",
    research_field_id: str = "",
    comparison_ids: str = "",
    group_by_predicate: str = "",
    filter_predicate: str = "",
    filter_value: str = "",
    filter_match: Literal["exact", "contains", "regex"] = "contains",
    top_n: int = 5,
    limit_subjects: int = 5000,
    value_parser: str = "leading_number",
    return_predicate: str = "",
    scope: Literal["comparisons", "papers"] = "comparisons",
    split_values: bool = False,
) -> str:
    """
    Cross-resource aggregation: rank/count/sum values for a predicate across
    contributions in MULTIPLE Comparisons or across an entire research field —
    when no single Comparison anchors the question.

    Use this for global-scope superlative/aggregation questions like:
      - "Most popular X" / "most frequent Y" / "top N Z"  → agg="mode_top"
      - "Largest sample size" / "highest score" / "lowest value" → agg="max" / "min"
      - "Total / sum across studies"                       → agg="sum"
      - "Number of studies that report X"                  → agg="count"
    where the relevant data is spread across many papers/contributions and
    AggregateComparisonValues with a single comparison_id cannot capture the
    full scope.

    Scope precedence:
      1. ``research_field_id`` — restricts to contributions of papers in
         that research field (orkgp:P30 → field).
      2. ``comparison_ids`` — comma-separated list of Comparison resource IDs;
         unions over all listed Comparisons.
      3. (default, none set) — every Contribution that is part of any
         Comparison (subject-of-orkgp:compareContribution). This is the
         broadest reasonable scope without scanning every triple in the graph.
      4. ``scope="papers"`` — every paper contribution (`?paper P31 ?contrib`),
         for questions phrased as "throughout/across the papers" rather than
         "across featured comparisons".

    Args:
        value_predicate: Predicate ID of the value to count/aggregate
            (e.g., "P15585", "P43133"). HAS_VALUE/label fallbacks applied.
        agg: One of mode_top (default — frequency table), count, count_distinct,
            sum, avg, min, max, all_values.
        research_field_id: Optional Research Field resource (e.g. "R132").
            When set, only contributions whose paper has P30→<field> are scanned.
        comparison_ids: Optional comma-separated list (e.g. "R153801,R155266").
            Union of compareContribution rows across all listed Comparisons.
        group_by_predicate: Optional grouping predicate. With agg="mode_top"
            this gives per-group frequency tables; with numeric aggs it gives
            per-group min/max/sum/avg.
        filter_predicate / filter_value / filter_match: Optional pre-filter
            on contributions (same shape as AggregateComparisonValues).
        top_n: For mode_top, how many top values to return (default 5).
        limit_subjects: Hard cap on contributions scanned (default 5000) to
            keep the SPARQL bounded; raise if a research field is large.
        value_parser: Numeric parsing mode. Use ``embedded_number`` for values
            like ``n=54`` or ``86 %`` where the number may not be at the start.
        return_predicate: Optional companion predicate to return for min/max rows.
        scope: Default global scope when no research_field_id or comparison_ids
            is supplied: ``comparisons`` (backwards-compatible) or ``papers``.
        split_values: If true, split semicolon/pipe-delimited categorical values
            before counting/mode aggregation. Use for packed values such as
            ``Plants;Insects`` in species/category predicates.

    Returns:
        JSON with `scope`, `value_predicate`, `agg`, optional `group_by`,
        `result` (scalar / list of {value,count} / list of {group,value,n}),
        `n_contributions`, and `status`.
    """
    from collections import Counter, defaultdict

    app = app_context.request_context.lifespan_context

    def _norm_pred(p: str) -> str:
        p = p.strip()
        if not p:
            return ""
        if p.startswith("orkgp:"):
            return p
        if p.startswith("http"):
            return f"<{p}>"
        return f"orkgp:{p}"

    try:
        value_pred = _norm_pred(value_predicate)
        if not value_pred:
            return json.dumps({"error": "value_predicate is required"}, indent=2)
        group_pred = _norm_pred(group_by_predicate) if group_by_predicate else ""
        flt_pred = _norm_pred(filter_predicate) if filter_predicate else ""
        return_pred = _norm_pred(return_predicate) if return_predicate else ""
        if value_parser not in {"leading_number", "embedded_number", "auto"}:
            value_parser = "auto" if value_parser in {"string", "text", "literal"} else "leading_number"

        # Build scope clause
        if research_field_id and research_field_id.strip():
            rf = research_field_id.strip()
            scope_clause = (
                f"?paper orkgp:P30 orkgr:{rf} .\n"
                f"        ?paper orkgp:P31 ?contrib .\n"
            )
            scope_label = f"research_field:{rf}"
        elif comparison_ids and comparison_ids.strip():
            ids = [c.strip() for c in comparison_ids.split(",") if c.strip()]
            if not ids:
                return json.dumps({"error": "comparison_ids was non-empty but parsed to no IDs"}, indent=2)
            cmp_values = " ".join(f"orkgr:{c}" for c in ids)
            scope_clause = (
                f"VALUES ?cmp {{ {cmp_values} }}\n"
                f"        ?cmp orkgp:compareContribution ?contrib .\n"
            )
            scope_label = f"comparisons:{','.join(ids)}"
        else:
            if scope == "papers":
                scope_clause = "?paper orkgp:P31 ?contrib .\n"
                scope_label = "all_paper_contributions"
            else:
                scope_clause = "?cmp orkgp:compareContribution ?contrib .\n"
                scope_label = "all_comparisons"

        # Build optional filter clause
        filter_block = ""
        if flt_pred and filter_value:
            safe = filter_value.replace('"', '\\"')
            if filter_match == "exact":
                filter_block = (
                    f"?contrib {flt_pred} ?fobj .\n"
                    f"        OPTIONAL {{ ?fobj rdfs:label ?flbl }}\n"
                    f'        FILTER( STR(?fobj) = "{safe}" || STR(?flbl) = "{safe}" )\n'
                )
            elif filter_match == "regex":
                filter_block = (
                    f"?contrib {flt_pred} ?fobj .\n"
                    f"        OPTIONAL {{ ?fobj rdfs:label ?flbl }}\n"
                    f'        FILTER( REGEX(STR(?fobj), "{safe}", "i") || REGEX(STR(?flbl), "{safe}", "i") )\n'
                )
            else:
                filter_block = (
                    f"?contrib {flt_pred} ?fobj .\n"
                    f"        OPTIONAL {{ ?fobj rdfs:label ?flbl }}\n"
                    f'        FILTER( CONTAINS(LCASE(STR(?fobj)), LCASE("{safe}")) '
                    f'|| CONTAINS(LCASE(STR(?flbl)), LCASE("{safe}")) )\n'
                )

        group_select = "?group ?groupLabel " if group_pred else ""
        return_select = "?returnObj ?returnLabel ?returnNested " if return_pred else ""
        group_clause = (
            f"?contrib {group_pred} ?group .\n"
            f"        OPTIONAL {{ ?group rdfs:label ?groupLabel }}\n"
            if group_pred else ""
        )
        return_block = (
            f"OPTIONAL {{\n"
            f"          ?contrib {return_pred} ?returnObj .\n"
            f"          OPTIONAL {{ ?returnObj rdfs:label ?returnLabel }}\n"
            f"          OPTIONAL {{ ?returnObj orkgp:HAS_VALUE ?returnNested }}\n"
            f"        }}\n"
            if return_pred else ""
        )
        body = f"""
SELECT DISTINCT ?contrib {group_select}?valueObj ?valueLabel ?nestedValue {return_select}WHERE {{
    GRAPH <{SCIQA_GRAPH}> {{
        {scope_clause}
        ?contrib {value_pred} ?valueObj .
        OPTIONAL {{ ?valueObj rdfs:label ?valueLabel }}
        OPTIONAL {{ ?valueObj orkgp:HAS_VALUE ?nestedValue }}
        {group_clause}
        {return_block}
        {filter_block}
    }}
}} LIMIT {limit_subjects}
"""
        full_query = SPARQL_PREFIXES + body
        app.sparql.setQuery(full_query)
        results = app.sparql.query().convert()
        bindings = results.get("results", {}).get("bindings", [])

        def _row_value(b):
            for key in ("nestedValue", "valueLabel", "valueObj"):
                v = b.get(key, {}).get("value")
                if v is not None and v != "":
                    if v.startswith("http://orkg.org/orkg/"):
                        continue
                    return v
            v = b.get("valueObj", {}).get("value", "")
            return v.split("/")[-1] if v.startswith("http") else v

        def _row_group(b):
            if not group_pred:
                return None
            v = b.get("groupLabel", {}).get("value") or b.get("group", {}).get("value", "")
            if v.startswith("http://orkg.org/orkg/"):
                v = v.split("/")[-1]
            return v

        def _row_return(b):
            if not return_pred:
                return None
            for key in ("returnNested", "returnLabel", "returnObj"):
                v = b.get(key, {}).get("value")
                if v is not None and v != "":
                    if v.startswith("http://orkg.org/orkg/"):
                        continue
                    return v
            v = b.get("returnObj", {}).get("value", "")
            return v.split("/")[-1] if v.startswith("http") else (v or None)

        def _to_float(s):
            if s is None:
                return None
            t = str(s).strip().replace(",", "")
            pattern = r"[+-]?\d+(\.\d+)?([eE][+-]?\d+)?"
            if value_parser == "embedded_number":
                m = re.search(pattern, t)
            elif value_parser == "auto":
                m = re.match(pattern, t) or re.search(pattern, t)
            else:
                m = re.match(pattern, t)
            if not m:
                return None
            try:
                return float(m.group(0))
            except ValueError:
                return None

        rows = []
        for b in bindings:
            contrib = b.get("contrib", {}).get("value", "").split("/")[-1]
            rows.append({
                "contrib": contrib,
                "group": _row_group(b),
                "value": _row_value(b),
                "return_value": _row_return(b),
            })

        if split_values and agg in {"count", "count_distinct", "mode_top", "all_values"}:
            expanded_rows: list[dict[str, Any]] = []
            for row in rows:
                value = row.get("value")
                if isinstance(value, str):
                    parts = [p.strip() for p in re.split(r"\s*[;|]\s*", value) if p.strip()]
                else:
                    parts = []
                if len(parts) > 1:
                    for part in parts:
                        expanded = dict(row)
                        expanded["value"] = part
                        expanded_rows.append(expanded)
                else:
                    expanded_rows.append(row)
            rows = expanded_rows

        n_rows = len(rows)
        if n_rows == 0:
            session_journal.failed_attempts.append(
                f"FindFrequentValues({scope_label}, {value_predicate}): no rows matched"
            )
            return json.dumps({
                "scope": scope_label,
                "value_predicate": value_predicate,
                "agg": agg,
                "group_by": group_by_predicate or None,
                "result": None,
                "n_contributions": 0,
                "status": "No contributions matched the scope/filter.",
            }, indent=2)

        groups: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
        for r in rows:
            groups[r["group"]].append(r)

        def _aggregate(items: List[Dict[str, Any]]):
            vals = [it["value"] for it in items]
            if agg == "all_values":
                return vals
            if agg == "count":
                return len(items)
            if agg == "count_distinct":
                return len({v for v in vals if v is not None})
            if agg == "mode_top":
                counter = Counter(v for v in vals if v is not None)
                top = counter.most_common(top_n)
                return [{"value": v, "count": c} for v, c in top]
            nums = [n for n in (_to_float(v) for v in vals) if n is not None]
            if not nums:
                return None
            if agg == "avg":
                return sum(nums) / len(nums)
            if agg == "sum":
                return sum(nums)
            if agg == "min":
                return min(nums)
            if agg == "max":
                return max(nums)
            return None

        if group_pred:
            grouped_result = []
            for g, items in groups.items():
                grouped_result.append({
                    "group": g,
                    "value": _aggregate(items),
                    "n": len(items),
                })
            if agg in ("avg", "sum", "min", "max"):
                grouped_result.sort(
                    key=lambda x: x["value"] if isinstance(x["value"], (int, float)) else float("-inf"),
                    reverse=True,
                )
            elif agg in ("count", "count_distinct"):
                grouped_result.sort(key=lambda x: x["value"] or 0, reverse=True)
            result_payload: Any = grouped_result
        else:
            result_payload = _aggregate(rows)

        if return_pred and agg in ("min", "max"):
            scored_rows = [(r, _to_float(r["value"])) for r in rows]
            scored_rows = [(r, n) for r, n in scored_rows if n is not None]
            if scored_rows:
                extreme_value = (
                    max(n for _, n in scored_rows)
                    if agg == "max"
                    else min(n for _, n in scored_rows)
                )
                extreme_rows = [r for r, n in scored_rows if n == extreme_value]
                return_values = []
                for r in extreme_rows:
                    rv = r.get("return_value")
                    if rv is not None and rv not in return_values:
                        return_values.append(rv)
                result_payload = {
                    "extreme_value": extreme_value,
                    "return_predicate": return_predicate,
                    "return_values": return_values,
                    "rows": extreme_rows[:50],
                }

        journal_key = scope_label
        if journal_key not in session_journal.found_values:
            session_journal.found_values[journal_key] = {}
        key = f"{agg}({value_predicate})" + (f" by {group_by_predicate}" if group_by_predicate else "")
        if return_predicate:
            key += f" return {return_predicate}"
        session_journal.found_values[journal_key][key] = result_payload
        session_journal.completed_steps.append(
            f"FindFrequentValues({scope_label}, {value_predicate}, {agg}) -> {n_rows} rows"
        )

        return json.dumps({
            "scope": scope_label,
            "value_predicate": value_predicate,
            "agg": agg,
            "group_by": group_by_predicate or None,
            "value_parser": value_parser,
            "return_predicate": return_predicate or None,
            "scope_mode": scope,
            "split_values": split_values,
            "filter": (
                {"predicate": filter_predicate, "value": filter_value, "match": filter_match}
                if filter_predicate else None
            ),
            "result": result_payload,
            "n_contributions": n_rows,
            "limit_subjects": limit_subjects,
            "status": (
                f"Aggregated {n_rows} contribution rows with {agg}"
                + (f" grouped by {group_by_predicate}" if group_by_predicate else "")
                + f" across {scope_label}"
            ),
        }, indent=2, default=str)

    except Exception as e:
        error_msg = f"Error in FindFrequentValues: {str(e)}"
        logger.error(error_msg)
        session_journal.failed_attempts.append(
            f"FindFrequentValues({value_predicate}): {str(e)}"
        )
        return json.dumps({"error": error_msg}, indent=2)


# ==============================================================================
# TOOL 21: FindCoAuthors
# ==============================================================================

@mcp.tool()
async def FindCoAuthors(
    app_context: Context,
    author_name: str,
    top_n: int = 50,
) -> str:
    """
    Find co-authors of an author across all their papers in ORKG.

    The query first locates papers authored by anyone matching ``author_name``
    (substring, case-insensitive) using P27/P6 — including both
    resource-typed authors (with rdfs:label) and literal authors. It then
    returns every other author of those same papers, ranked by how many
    shared papers they have with the seed.

    Use this for "who has X written papers with?" / "co-authors of X" /
    "collaborators of X" questions, where chaining FindAuthorPapers and
    GetPaperAuthors would otherwise require N+1 calls and brittle bookkeeping.

    Args:
        author_name: Author name or partial name (e.g., "Kurt Thomas").
        top_n: Maximum number of co-authors to return (default 50).

    Returns:
        JSON with `seed_author`, `papers` (list of {id, title}), `coauthors`
        (sorted list of {name, id, shared_paper_count, papers}), and `status`.
    """
    app = app_context.request_context.lifespan_context

    try:
        safe = author_name.replace('"', '\\"')
        query = f"""
SELECT DISTINCT ?paper ?paperLabel ?seedAuthor ?seedLabel ?coAuthor ?coLabel WHERE {{
    GRAPH <{SCIQA_GRAPH}> {{
        {{
            {{ ?paper orkgp:P27 ?seedAuthor }} UNION {{ ?paper orkgp:P6 ?seedAuthor }}
            ?seedAuthor rdfs:label ?seedLabel .
            FILTER( CONTAINS(LCASE(?seedLabel), LCASE("{safe}")) )
        }}
        UNION
        {{
            {{ ?paper orkgp:P27 ?seedAuthor }} UNION {{ ?paper orkgp:P6 ?seedAuthor }}
            FILTER( isLiteral(?seedAuthor) )
            FILTER( CONTAINS(LCASE(STR(?seedAuthor)), LCASE("{safe}")) )
            BIND( STR(?seedAuthor) AS ?seedLabel )
        }}
        OPTIONAL {{ ?paper rdfs:label ?paperLabel }}
        {{
            {{ ?paper orkgp:P27 ?coAuthor }} UNION {{ ?paper orkgp:P6 ?coAuthor }}
            FILTER( ?coAuthor != ?seedAuthor )
        }}
        OPTIONAL {{ ?coAuthor rdfs:label ?coLabel }}
    }}
}}
"""
        full_query = SPARQL_PREFIXES + query
        app.sparql.setQuery(full_query)
        results = app.sparql.query().convert()
        bindings = results.get("results", {}).get("bindings", [])

        papers: Dict[str, str] = {}
        co_index: Dict[str, Dict[str, Any]] = {}

        for b in bindings:
            paper_uri = b.get("paper", {}).get("value", "")
            paper_id = paper_uri.split("/")[-1] if "/" in paper_uri else paper_uri
            paper_title = b.get("paperLabel", {}).get("value", paper_id)
            papers[paper_id] = paper_title

            co_uri = b.get("coAuthor", {}).get("value", "")
            co_label = b.get("coLabel", {}).get("value", "")
            if co_uri.startswith("http"):
                co_id = co_uri.split("/")[-1]
                co_name = co_label or co_id
            else:
                co_id = ""
                co_name = co_uri  # literal author
            if not co_name:
                continue

            key = co_id or f"_lit:{co_name}"
            entry = co_index.setdefault(key, {
                "name": co_name,
                "id": co_id,
                "papers": set(),
            })
            entry["papers"].add(paper_id)

        coauthors = []
        for entry in co_index.values():
            shared = entry["papers"]
            coauthors.append({
                "name": entry["name"],
                "id": entry["id"] or None,
                "shared_paper_count": len(shared),
                "papers": sorted(shared),
            })
        coauthors.sort(key=lambda x: (-x["shared_paper_count"], x["name"]))
        coauthors = coauthors[:top_n]

        # Journal
        for pid, plbl in papers.items():
            session_journal.visited_nodes[pid] = plbl
        session_journal.completed_steps.append(
            f"FindCoAuthors('{author_name}') -> {len(coauthors)} co-authors "
            f"across {len(papers)} papers"
        )

        return json.dumps({
            "seed_author": author_name,
            "papers": [{"id": pid, "title": papers[pid]} for pid in sorted(papers)],
            "coauthors": coauthors,
            "n_papers": len(papers),
            "n_coauthors": len(coauthors),
            "status": (
                f"Found {len(coauthors)} co-authors across {len(papers)} papers"
                if papers else f"No papers found for author matching '{author_name}'"
            ),
        }, indent=2)

    except Exception as e:
        error_msg = f"Error in FindCoAuthors: {str(e)}"
        logger.error(error_msg)
        session_journal.failed_attempts.append(
            f"FindCoAuthors('{author_name}'): {str(e)}"
        )
        return json.dumps({"error": error_msg}, indent=2)


# ==============================================================================
# TOOL 21: ManageJournal
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
# TOOL 19: GetJournalSummary
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
# TOOL 20: GetJournalStateJSON
# ==============================================================================

@mcp.tool()
async def GetJournalStateJSON(
    app_context: Context
) -> str:
    """
    Return the structured journal state as a JSON string.

    Used by the frontend's live graph view to render the discovered
    subgraph (visited_nodes, verified_facts, found_values). Machine-readable
    counterpart to `GetJournalSummary`.

    Returns:
        JSON-encoded dict mirroring the JournalState pydantic model.
    """
    return json.dumps(session_journal.model_dump(), default=str, ensure_ascii=False)


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":
    mcp.run()
