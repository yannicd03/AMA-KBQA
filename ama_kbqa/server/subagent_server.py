import os
import sys
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import AsyncIterator

from fastmcp import FastMCP, Context
from qdrant_client import QdrantClient
from openai import OpenAI
from loguru import logger
from pydantic import BaseModel, Field, ConfigDict
from typing import Literal, Any, Optional
from SPARQLWrapper import SPARQLWrapper, JSON
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

# --- Configuration ---
QDRANT_HOST = "localhost"  # TODO Get config parameters from config.toml instead of here
QDRANT_PORT = 6333
COLLECTION_ENTITIES = "kqapro-entities"
COLLECTION_RELATIONS = "kqapro-relations"
VIRTUOSO_ENDPOINT = "http://localhost:8890/sparql"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODEL = "qwen/qwen3-embedding-8b"
TOP_N = 10
SCORE_THRESHHOLD = 0.7
OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]

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
    openai: OpenAI
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
    metadata: list[str] = Field(
        default_factory=list,
        description="The keys of the node payload containing attributes, relations, and schema info."
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
    logger.info("Starting up: Connecting to Qdrant & OpenAI...")

    try:
        # Initialize Clients
        qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)  # TODO Use Async Client instead?

        # Quick connectivity check
        qdrant.get_collections()

        openai = OpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=os.environ['OPENROUTER_API_KEY']
        )

        sparql = SPARQLWrapper(VIRTUOSO_ENDPOINT)
        sparql.setReturnFormat(JSON)
        # Yield the context so tools can access it
        logger.debug("Initialisation was successful")
        yield AppContext(qdrant=qdrant, openai=openai, sparql=sparql)

    except Exception as e:
        logger.error(f"Server Initialisatuion failed: {e}")

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
        model=OPENROUTER_MODEL,
        input=[text],
        encoding_format="float"
    )
    return response.data[0].embedding

# --- 5. Refactored Tool using Context ---


@mcp.tool
def FindNode(semantic_node_name: str, context: Context) -> SearchResponse:
    """
    Performs a semantic vector search to identify relevant nodes (Entities or Concepts) within the Knowledge Graph.

    Use this tool to resolve natural language descriptions into concrete Knowledge Graph nodes. 
    It retrieves the top matches based on vector similarity.

    Key Features:
    - **Semantic Resolution:** Can find nodes even without exact name matches (e.g., inputting "The capital of France" will find "Paris").
    - **Shape Retrieval:** Returns the **available keys** of the metadata payload. This provides the "Shape" (available attributes and relations) required for generating SPARQL queries without retrieving the heavy data values immediately.

    Args:
        semantic_node_name (str): The search query. This can be a specific entity name (e.g., "Berlin") or a descriptive phrase (e.g., "German cities with a population over 3 million").
        context (Context): The FastMCP request context containing the active database connections.

    Returns:
        SearchResponse: A structured object containing a list of `NodeMatch` items, each with its original ID, relevance score, and a list of available metadata keys.
    """
    # 1. Get Context
    logger.info(f"Ran FindNode Tool with semantic_node_name: {semantic_node_name} and context: {context}")
    app_context: AppContext = context.request_context.lifespan_context

    try:
        # 2. Generate Embedding
        vector = get_embedding(app_context.openai, semantic_node_name)

        # 3. Search Qdrant
        # We request the payload to extract keys, even though we won't return values
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

            # EXTRACT KEYS ONLY
            # We convert the keys to a list to show the schema without the data overhead
            payload_keys = list(payload.keys())

            match = NodeMatch(
                original_id=payload.get("original_id", "N/A"),
                name=payload.get("name", "Unknown"),
                node_type=payload.get("node_type", "unknown"),
                relevance_score=point.score,
                # Pass only the list of keys instead of the full dictionary
                metadata=payload_keys
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
def ExploreNeighborhood(base_node_id: str, semantic_relation_name: str, context: Context) -> NeighborhoodResponse:
    """
    Finds specific facts about a node by semantically matching relations and verifying them in the Graph DB.

    Args:
        base_node_id (str): The unique ID of the node (e.g., "Q64") found via FindNode.
        semantic_relation_name (str): The relation to find (e.g., "population", "born in").

    Returns:
        NeighborhoodResponse: Verified triples found in the Virtuoso database.
    """
    logger.info(
        f"Ran ExploreNeighborhood Tool with base_node_id: {base_node_id}, semantic_relation_name: {semantic_relation_name} and context: {context}")

    ctx: AppContext = context.request_context.lifespan_context
    sparql: SPARQLWrapper = ctx.sparql

    # 1. Embed the relation query
    vector = get_embedding(ctx.openai, semantic_relation_name)

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
    logger.info(f"Ran ExploreNeighborhood Tool with query: {query} and context: {context}")
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
