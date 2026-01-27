"""
Populate Qdrant with SciQA/ORKG entities and relations for semantic search.

This script:
1. Queries Virtuoso for ORKG resources (papers, contributions, research fields, etc.)
2. Extracts labels for embedding
3. Creates embeddings using OpenRouter
4. Uploads to Qdrant collections: sciqa-entities, sciqa-relations

Configuration:
- LIMIT: Set to an integer for testing, None for full run
- RESOURCES_ONLY: Set to True to skip classes

Run after loading 'ORKG RDF dump 14.02.2023.nt' into Virtuoso graph http://sciqa.org/kg
"""
import os
import sys
from tqdm import tqdm
from typing import List
from qdrant_client import QdrantClient, models
from openai import OpenAI
from SPARQLWrapper import SPARQLWrapper, JSON
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

# --- Configuration ---
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
VIRTUOSO_ENDPOINT = "http://localhost:8890/sparql"
SCIQA_GRAPH = "http://sciqa.org/kg"

COLLECTION_ENTITIES = "sciqa-entities"
COLLECTION_RELATIONS = "sciqa-relations"

# --- OpenRouter Configuration ---
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODEL = "qwen/qwen3-embedding-8b"
VECTOR_DIMENSION = 4096
DISTANCE_MEASURE = models.Distance.COSINE

# Get API Key
try:
    OPENROUTER_API_KEY = os.environ['OPENROUTER_API_KEY']
except KeyError:
    print("Error: OPENROUTER_API_KEY environment variable not set.")
    sys.exit(1)

# Initialize clients
openai_client = OpenAI(
    base_url=OPENROUTER_BASE_URL,
    api_key=OPENROUTER_API_KEY,
)

sparql = SPARQLWrapper(VIRTUOSO_ENDPOINT)
sparql.setReturnFormat(JSON)


def setup_collections(client: QdrantClient) -> bool:
    """Creates or recreates the Qdrant collections for SciQA/ORKG.

    Returns:
        True if setup should proceed, False if user cancelled.
    """
    collections = [COLLECTION_ENTITIES, COLLECTION_RELATIONS]

    # Check if any collections exist and have data
    existing_collections = []
    for col_name in collections:
        if client.collection_exists(collection_name=col_name):
            info = client.get_collection(col_name)
            point_count = info.points_count
            existing_collections.append((col_name, point_count))

    # Prompt user if collections exist with data
    if existing_collections:
        print("\n⚠️  Existing collections found:")
        for col_name, count in existing_collections:
            print(f"   - {col_name}: {count} vectors")

        response = input("\nOverwrite existing collections? [y/N]: ").strip().lower()
        if response != 'y':
            print("Cancelled.")
            return False

    # Create/recreate collections
    for col_name in collections:
        print(f"Setting up collection: {col_name}")

        if client.collection_exists(collection_name=col_name):
            client.delete_collection(collection_name=col_name)
            print(f"Deleted existing collection: {col_name}")

        client.create_collection(
            collection_name=col_name,
            vectors_config=models.VectorParams(
                size=VECTOR_DIMENSION,
                distance=DISTANCE_MEASURE
            )
        )
    print("Collections created successfully.")
    return True


def get_embeddings(texts: List[str], batch_size: int = 100) -> List[List[float]]:
    """Generates embeddings using OpenRouter with batching."""
    if not texts:
        return []

    all_embeddings = []
    total = len(texts)

    with tqdm(total=total, desc="Generating Embeddings", unit="vec") as pbar:
        for i in range(0, total, batch_size):
            batch = texts[i: i + batch_size]
            try:
                clean_batch = [t.replace("\n", " ") for t in batch]

                response = openai_client.embeddings.create(
                    model=OPENROUTER_MODEL,
                    input=clean_batch,
                    encoding_format="float"
                )
                batch_embeddings = [data.embedding for data in response.data]
                all_embeddings.extend(batch_embeddings)
                pbar.update(len(batch))

            except Exception as e:
                print(f"\nError generating embeddings at index {i}: {e}")
                sys.exit(1)

    return all_embeddings


def query_sparql(query: str) -> list:
    """Execute SPARQL query and return results."""
    sparql.setQuery(query)
    try:
        results = sparql.query().convert()
        return results["results"]["bindings"]
    except Exception as e:
        print(f"SPARQL error: {e}")
        return []


def extract_resources(limit: int = None) -> List[dict]:
    """Extract all resources with labels from ORKG."""
    print("\n--- Extracting ORKG Resources ---")

    limit_clause = f"LIMIT {limit}" if limit else ""

    query = f"""
    SELECT DISTINCT ?resource ?label ?type
    FROM <{SCIQA_GRAPH}>
    WHERE {{
        ?resource <http://www.w3.org/2000/01/rdf-schema#label> ?label .
        OPTIONAL {{ ?resource a ?type }}
        FILTER(STRSTARTS(STR(?resource), "http://orkg.org/orkg/resource/"))
    }}
    {limit_clause}
    """

    results = query_sparql(query)
    resources = []
    seen = set()

    for r in results:
        uri = r["resource"]["value"]
        if uri in seen:
            continue
        seen.add(uri)

        # Determine node type from class URI
        type_uri = r.get("type", {}).get("value", "")
        if "Paper" in type_uri:
            node_type = "paper"
        elif "Author" in type_uri:
            node_type = "author"
        elif "Contribution" in type_uri:
            node_type = "contribution"
        elif "ResearchField" in type_uri:
            node_type = "research_field"
        elif "Venue" in type_uri:
            node_type = "venue"
        elif "Problem" in type_uri:
            node_type = "problem"
        elif "Comparison" in type_uri:
            node_type = "comparison"
        else:
            node_type = "resource"

        resources.append({
            "uri": uri,
            "name": r["label"]["value"],
            "type": node_type,
            "class_uri": type_uri
        })

    print(f"Found {len(resources)} resources")
    return resources


def extract_classes(limit: int = None) -> List[dict]:
    """Extract ORKG classes."""
    print("\n--- Extracting ORKG Classes ---")

    limit_clause = f"LIMIT {limit}" if limit else ""

    query = f"""
    SELECT DISTINCT ?class ?label
    FROM <{SCIQA_GRAPH}>
    WHERE {{
        ?class a <http://www.w3.org/2002/07/owl#Class> .
        ?class <http://www.w3.org/2000/01/rdf-schema#label> ?label .
        FILTER(STRSTARTS(STR(?class), "http://orkg.org/orkg/class/"))
    }}
    {limit_clause}
    """

    results = query_sparql(query)
    classes = []

    for r in results:
        classes.append({
            "uri": r["class"]["value"],
            "name": r["label"]["value"],
            "type": "class"
        })

    print(f"Found {len(classes)} classes")
    return classes


def extract_predicates(limit: int = None) -> List[dict]:
    """Extract ORKG predicates/properties."""
    print("\n--- Extracting ORKG Predicates ---")

    limit_clause = f"LIMIT {limit}" if limit else ""

    query = f"""
    SELECT DISTINCT ?predicate ?label
    FROM <{SCIQA_GRAPH}>
    WHERE {{
        ?predicate a <http://orkg.org/orkg/class/Predicate> .
        ?predicate <http://www.w3.org/2000/01/rdf-schema#label> ?label .
    }}
    {limit_clause}
    """

    results = query_sparql(query)
    predicates = []

    for r in results:
        predicates.append({
            "uri": r["predicate"]["value"],
            "name": r["label"]["value"],
            "type": "predicate"
        })

    print(f"Found {len(predicates)} predicates")
    return predicates


def process_entities(client: QdrantClient, entities: List[dict]):
    """Embed and upload entities to Qdrant."""
    if not entities:
        print("No entities to process.")
        return

    print(f"\n--- Processing {len(entities)} entities ---")

    # Get names for embedding
    names_to_embed = [e["name"] for e in entities]

    # Get embeddings
    vectors = get_embeddings(names_to_embed)

    # Build points
    points = []
    for i, entity in enumerate(entities):
        points.append(
            models.PointStruct(
                id=i,
                vector=vectors[i],
                payload={
                    "uri": entity["uri"],
                    "name": entity["name"],
                    "node_type": entity["type"],
                    "class_uri": entity.get("class_uri", "")
                }
            )
        )

    # Upload in batches
    print(f"Uploading {len(points)} items to '{COLLECTION_ENTITIES}'...")
    batch_size = 500
    for i in range(0, len(points), batch_size):
        client.upload_points(
            collection_name=COLLECTION_ENTITIES,
            points=points[i: i + batch_size],
            wait=True
        )


def process_relations(client: QdrantClient, relations: List[dict]):
    """Embed and upload relations/predicates to Qdrant."""
    if not relations:
        print("No relations to process.")
        return

    print(f"\n--- Processing {len(relations)} relations ---")

    names_to_embed = [r["name"] for r in relations]
    vectors = get_embeddings(names_to_embed)

    points = []
    for i, rel in enumerate(relations):
        points.append(
            models.PointStruct(
                id=i,
                vector=vectors[i],
                payload={
                    "uri": rel["uri"],
                    "predicate": rel["name"],
                    "type": "relation_schema"
                }
            )
        )

    print(f"Uploading {len(points)} relations to '{COLLECTION_RELATIONS}'...")
    client.upload_points(
        collection_name=COLLECTION_RELATIONS,
        points=points,
        wait=True
    )


def main():
    # Configuration - modify these as needed
    LIMIT = None           # Set to an integer (e.g., 1000) for testing, None for full run
    RESOURCES_ONLY = False  # Set to True to skip classes

    # 1. Connect to Qdrant
    try:
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        client.get_collections()
        print(f"Connected to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}")
    except Exception as e:
        print(f"Error connecting to Qdrant: {e}")
        return

    # 2. Setup collections (with overwrite prompt)
    if not setup_collections(client):
        return

    # 3. Extract entities from Virtuoso
    all_entities = []

    # Resources (papers, contributions, research fields, etc.)
    resources = extract_resources(limit=LIMIT)
    all_entities.extend(resources)

    # Classes (optional)
    if not RESOURCES_ONLY:
        classes = extract_classes(limit=LIMIT)
        all_entities.extend(classes)

    # 4. Extract predicates (relations)
    predicates = extract_predicates(limit=LIMIT)

    # 5. Process and upload
    process_entities(client, all_entities)
    process_relations(client, predicates)

    print("\n--- SciQA Vector Population Complete ---")
    print(f"Total entities: {len(all_entities)}")
    print(f"Total relations: {len(predicates)}")


if __name__ == "__main__":
    main()
