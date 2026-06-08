import json
import os
import sys
from tqdm import tqdm
from typing import Dict, List, Any
from qdrant_client import QdrantClient, models
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
from pathlib import Path

load_dotenv(find_dotenv())

# --- Configuration ---
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_ENTITIES = "kqapro-entities"
COLLECTION_RELATIONS = "kqapro-relations"
KB_FILE_NAME = Path("db/datasets/kqapro/kb.json")

# --- OpenRouter / OpenAI Configuration ---
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODEL = "qwen/qwen3-embedding-8b"
VECTOR_DIMENSION = 4096
# Can also be models.Distance.EUCLID or models.Distance.DOT. This is defined in the beginning, when setting up the collection and cannot be changed without recreating the collection.
# The Distance Measure has to fir the Embedding Model used, in most modern models like Qwen3-Embedding Cosine Similarity is the standard.
DISTANCE_MEASURE = models.Distance.COSINE
# BM25 sparse index for hybrid search. Collections are always created
# hybrid-capable; the runtime [retrieval].hybrid_enabled toggle decides
# whether the sparse branch is actually queried. Sparse vectors are computed
# SERVER-SIDE by Qdrant (>=1.15.2) from the Document objects in the upserts.
# Names must match ama_kbqa/retrieval/search.py.
BM25_SPARSE_VECTOR_NAME = "bm25"
BM25_MODEL = "Qdrant/bm25"

# TODO: Modify this seeding script to no longer add all of the metadata to the payload, only the available keys to save storage space.

# Get API Key
try:
    OPENROUTER_API_KEY = os.environ['OPENROUTER_API_KEY']
except KeyError:
    print("Error: OPENROUTER_API_KEY environment variable not set. Please set it in the .env File")
    sys.exit(1)

# Initialize OpenAI Client
openai_client = OpenAI(
    base_url=OPENROUTER_BASE_URL,
    api_key=OPENROUTER_API_KEY,
)

# --- Qdrant Setup ---


def setup_collections(client: QdrantClient) -> bool:
    """Creates or recreates the Qdrant collections.

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
            ),
            sparse_vectors_config={
                BM25_SPARSE_VECTOR_NAME: models.SparseVectorParams(
                    modifier=models.Modifier.IDF
                )
            }
        )
    print("Collections created successfully.")
    return True


# --- Embedding Helper ---


def get_embeddings(texts: List[str], batch_size: int = 100) -> List[List[float]]:
    """
    Generates embeddings using the OpenAI SDK with batching to avoid API limits.
    Includes a tqdm progress bar.
    """
    if not texts:
        return []

    all_embeddings = []
    total = len(texts)

    # Initialize the progress bar
    # unit="vec" labels the count (e.g., "100/500 vec")
    with tqdm(total=total, desc="Generating Embeddings", unit="vec") as pbar:
        for i in range(0, total, batch_size):
            batch = texts[i: i + batch_size]
            try:
                # Replace newlines to avoid negative effects on embedding performance
                clean_batch = [t.replace("\n", " ") for t in batch]

                response = openai_client.embeddings.create(
                    model=OPENROUTER_MODEL,
                    input=clean_batch,
                    encoding_format="float"
                )
                # Extract embeddings in order
                batch_embeddings = [data.embedding for data in response.data]
                all_embeddings.extend(batch_embeddings)

                # Update the progress bar by the number of items in the current batch
                pbar.update(len(batch))

            except Exception as e:
                # We use \n to ensure the error prints on a new line below the progress bar
                print(f"\nError generating embeddings at index {i}: {e}")
                sys.exit(1)

    return all_embeddings

# --- Processing Helper 1: Entities & Concepts ---


def process_entities_and_concepts(client: QdrantClient, kb_data: Dict):
    """
    Combines concepts and entities, tags them with 'node_type', 
    embeds their names, and uploads to Qdrant.
    """
    print("\n--- Processing Entities and Concepts ---")

    concepts = kb_data.get('concepts', {})
    entities = kb_data.get('entities', {})

    # Merge for processing, but keep track of IDs
    all_data = {**concepts, **entities}

    if not all_data:
        print("No concepts or entities found.")
        return

    # Map JSON string IDs to Integer IDs for Qdrant
    # We use a stable sort to ensure IDs remain consistent across runs if data doesn't change
    sorted_ids = sorted(all_data.keys())
    json_id_to_int_id = {json_id: i for i, json_id in enumerate(sorted_ids)}

    names_to_embed = []
    points = []

    # Prepare data
    for json_id in sorted_ids:
        data = all_data[json_id]
        names_to_embed.append(data['name'])

    # Get Vectors
    vectors = get_embeddings(names_to_embed)

    # Build Points
    for i, json_id in enumerate(sorted_ids):
        data = all_data[json_id]

        # Create Payload
        payload = data.copy()
        payload['original_id'] = json_id

        # --- ADDING NODE_TYPE FLAG ---
        # Check membership to determine type
        if json_id in entities:
            payload['node_type'] = 'entity'
        else:
            payload['node_type'] = 'concept'

        points.append(
            models.PointStruct(
                id=json_id_to_int_id[json_id],
                # "" addresses the unnamed default dense vector; the BM25
                # sparse vector is inferred server-side from the name text.
                vector={
                    "": vectors[i],
                    BM25_SPARSE_VECTOR_NAME: models.Document(
                        text=data['name'], model=BM25_MODEL
                    ),
                },
                payload=payload
            )
        )

    # Upload
    print(f"Uploading {len(points)} items to '{COLLECTION_ENTITIES}'...")
    batch_size = 500
    for i in range(0, len(points), batch_size):
        client.upload_points(
            collection_name=COLLECTION_ENTITIES,
            points=points[i: i + batch_size],
            wait=True
        )

# --- Processing Helper 2: Unique Relations ---


def process_unique_relations(client: QdrantClient, kb_data: Dict):
    """
    Finds UNIQUE predicates (relations) across the entire KB,
    embeds them as schema items for SPARQL mapping.
    """
    print("\n--- Processing Unique Relations (Schema) ---")

    unique_predicates = set()

    # Scan all entities to find predicates
    for entity in kb_data.get('entities', {}).values():
        for relation in entity.get('relations', []):
            unique_predicates.add(relation['predicate'])

    if not unique_predicates:
        print("No relations found.")
        return

    predicate_list = sorted(list(unique_predicates))
    print(f"Found {len(predicate_list)} unique predicates.")

    # Get Vectors
    vectors = get_embeddings(predicate_list)

    points = []
    for i, predicate in enumerate(predicate_list):
        points.append(
            models.PointStruct(
                id=i,  # Simple incremental ID
                # "" addresses the unnamed default dense vector; the BM25
                # sparse vector is inferred server-side from the predicate.
                vector={
                    "": vectors[i],
                    BM25_SPARSE_VECTOR_NAME: models.Document(
                        text=predicate, model=BM25_MODEL
                    ),
                },
                payload={
                    "predicate": predicate,
                    "type": "relation_schema"
                }
            )
        )

    # Upload
    print(f"Uploading {len(points)} predicates to '{COLLECTION_RELATIONS}'...")
    client.upload_points(
        collection_name=COLLECTION_RELATIONS,
        points=points,
        wait=True
    )

# --- Main Execution ---


def main():
    # 1. Connect to Qdrant
    try:
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        client.get_collections()

        print(f"Successfully connected to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}")
    except Exception as e:
        print(f"Error connecting to Qdrant: {e}")
        print("Please ensure Qdrant is running in Docker on port 6333.")
        return

    # 2. Setup Collections (with overwrite prompt)
    if not setup_collections(client):
        return

    # 3. Load Data
    if not os.path.exists(KB_FILE_NAME):
        print(f"Error: File '{KB_FILE_NAME}' not found.")
        return

    with open(KB_FILE_NAME, 'r') as f:
        kb_data = json.load(f)

    # 4. Process Data
    process_entities_and_concepts(client, kb_data)
    process_unique_relations(client, kb_data)

    print("\n--- Process Complete ---")


if __name__ == "__main__":
    main()
