# Guide: Integrating New Datasets

This guide explains how to integrate a new knowledge graph dataset into the AMA-KBQA system, including loading RDF data into Virtuoso and creating vector embeddings in Qdrant.

## Overview

Integrating a new dataset involves these main steps:

1. **Prepare the data** - Convert your knowledge graph to N-Triples format
2. **Load into Virtuoso** - Upload RDF triples to the graph database
3. **Create embeddings** - Generate and upload vector embeddings to Qdrant
4. **Create MCP server** - Build tools for agent interaction (optional)
5. **Verify the setup** - Test queries and semantic search

## Prerequisites

- Docker Desktop running with Virtuoso and Qdrant containers
- Python 3.12+ with project dependencies (`pip install -e .`)
- `.env` file with `OPENROUTER_API_KEY` configured
- Your knowledge graph data (JSON, RDF, or other format)

## Step 1: Prepare Your Data

### Supported Input Formats

The system can work with various input formats:

| Format | Extension | Notes |
|--------|-----------|-------|
| N-Triples | `.nt` | Ready for Virtuoso (preferred) |
| RDF/XML | `.rdf`, `.xml` | Convert to N-Triples first |
| Turtle | `.ttl` | Convert to N-Triples first |
| JSON-LD | `.jsonld` | Convert to N-Triples first |
| Custom JSON | `.json` | Write converter script |

### Converting to N-Triples

If your data is not in N-Triples format, convert it using RDFLib:

```python
from rdflib import Graph

# Load source file
g = Graph()
g.parse("source_data.ttl", format="turtle")  # or "xml", "json-ld"

# Serialize to N-Triples
g.serialize("output.nt", format="nt")
print(f"Converted {len(g)} triples")
```

### Converting from Custom JSON (KQAPro Example)

For custom JSON formats, create a converter script. See `db/datasets/kqapro/convert_kb_to_nt.py` for reference.

Key conversion patterns:

```python
from rdflib import Graph, Namespace, Literal, URIRef, BNode
from rdflib.namespace import RDF, RDFS, XSD

# Define your namespaces
EX = Namespace("http://example.org/entity/")
PROP = Namespace("http://example.org/property/")
ATTR = Namespace("http://example.org/attribute/")

g = Graph()

# Add entity with label
entity_uri = EX["Entity_Name"]
g.add((entity_uri, RDFS.label, Literal("Entity Name")))

# Add typed relation
g.add((entity_uri, PROP["relation_name"], EX["Other_Entity"]))

# Add typed attribute
g.add((entity_uri, ATTR["birth_date"], Literal("1990-01-15", datatype=XSD.date)))

# Add quantity with unit
bnode = BNode()
g.add((entity_uri, ATTR["height"], bnode))
g.add((bnode, RDF.value, Literal("180.5", datatype=XSD.decimal)))
g.add((bnode, URIRef("http://example.org/unit/unit"), URIRef("http://example.org/unit/centimetre")))

# Serialize
g.serialize("output.nt", format="nt")
```

### URI Sanitization

Ensure URIs are valid (no spaces or special characters):

```python
import re

def sanitize_id(id_str: str) -> str:
    """Convert string to valid URI component."""
    # Replace spaces with underscores
    sanitized = id_str.replace(" ", "_")
    # Remove invalid characters
    sanitized = re.sub(r'[^\w\-_]', '', sanitized)
    return sanitized

# Usage
entity_uri = EX[sanitize_id("Albert Einstein")]  # -> ex:Albert_Einstein
```

### Data Placement

Place your dataset in the appropriate location:

```
db/datasets/<dataset_name>/
├── <dataset>.nt           # N-Triples file for Virtuoso
├── convert_to_nt.py       # Optional: converter script
├── train.json             # Optional: Q&A training data
├── test.json              # Optional: Q&A test data
└── README.md              # Optional: dataset documentation
```

## Step 2: Load Data into Virtuoso

### Directory Mapping

The Docker compose file maps `db/datasets` to `/usr/share/proj` in the Virtuoso container:

```yaml
# db/docker-compose.yml
volumes:
  - ./datasets:/usr/share/proj:rw
```

### Choose a Graph URI

Select a unique graph URI for your dataset:

```
http://<domain>/<dataset_name>

# Examples:
http://kqapro.org/kb
http://orkg.org/kg
http://example.org/mydata
```

### Loading via isql (Interactive)

Connect to Virtuoso isql:

```powershell
docker exec -it virtuoso_ama_kbqa isql 1111 dba kit_ama_kbqa
```

Execute load commands:

```sql
-- Register the file for loading
ld_dir('/usr/share/proj/<dataset_folder>', '<filename>.nt', '<graph_uri>');

-- Execute the loader
rdf_loader_run();

-- Commit changes
checkpoint;

-- Exit
exit;
```

**Example:**

```sql
ld_dir('/usr/share/proj/mydata', 'knowledge_graph.nt', 'http://example.org/mydata');
rdf_loader_run();
checkpoint;
```

### Loading via One-liner (Automated)

For scripts and automation:

```powershell
docker exec -i virtuoso_ama_kbqa isql 1111 dba kit_ama_kbqa "EXEC=ld_dir('/usr/share/proj/<folder>', '<file>.nt', '<graph_uri>'); rdf_loader_run(); checkpoint;"
```

### Handling Files with Spaces

For filenames with spaces, the `ld_dir` function handles them correctly:

```sql
ld_dir('/usr/share/proj/sciqa', 'ORKG RDF dump 14.02.2023.nt', 'http://orkg.org/kg');
```

### Verify Load Success

Check triple count:

```powershell
docker exec -i virtuoso_ama_kbqa isql 1111 dba kit_ama_kbqa "EXEC=SPARQL SELECT COUNT(*) FROM <http://example.org/mydata> WHERE { ?s ?p ?o };"
```

Or use the web interface at http://localhost:8890/sparql:

```sparql
SELECT COUNT(*) FROM <http://example.org/mydata> WHERE { ?s ?p ?o }
```

## Step 3: Create Vector Embeddings

### Understanding the Embedding Strategy

The system creates two Qdrant collections per dataset:

1. **`<dataset>-entities`** - Entity/resource embeddings for semantic search
2. **`<dataset>-relations`** - Predicate/relation embeddings for relation matching

### Create a Population Script

Create `db/populate_<dataset>_vectors.py` based on existing scripts:

```python
"""
Populate Qdrant with <dataset> entities and relations.
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
GRAPH_URI = "http://example.org/mydata"  # Your graph URI

COLLECTION_ENTITIES = "mydata-entities"
COLLECTION_RELATIONS = "mydata-relations"

# --- Embedding Configuration ---
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODEL = "qwen/qwen3-embedding-8b"
VECTOR_DIMENSION = 4096
DISTANCE_MEASURE = models.Distance.COSINE

# Get API Key
OPENROUTER_API_KEY = os.environ.get('OPENROUTER_API_KEY')
if not OPENROUTER_API_KEY:
    print("Error: OPENROUTER_API_KEY not set")
    sys.exit(1)

# Initialize clients
openai_client = OpenAI(
    base_url=OPENROUTER_BASE_URL,
    api_key=OPENROUTER_API_KEY,
)

sparql = SPARQLWrapper(VIRTUOSO_ENDPOINT)
sparql.setReturnFormat(JSON)


def setup_collections(client: QdrantClient):
    """Create or recreate Qdrant collections."""
    for col_name in [COLLECTION_ENTITIES, COLLECTION_RELATIONS]:
        if client.collection_exists(collection_name=col_name):
            client.delete_collection(collection_name=col_name)

        client.create_collection(
            collection_name=col_name,
            vectors_config=models.VectorParams(
                size=VECTOR_DIMENSION,
                distance=DISTANCE_MEASURE
            )
        )
    print("Collections created.")


def get_embeddings(texts: List[str], batch_size: int = 100) -> List[List[float]]:
    """Generate embeddings using OpenRouter with batching."""
    all_embeddings = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        batch = [t.replace("\n", " ") for t in texts[i:i + batch_size]]

        response = openai_client.embeddings.create(
            model=OPENROUTER_MODEL,
            input=batch,
            encoding_format="float"
        )
        all_embeddings.extend([d.embedding for d in response.data])

    return all_embeddings


def query_sparql(query: str) -> list:
    """Execute SPARQL query and return results."""
    sparql.setQuery(query)
    results = sparql.query().convert()
    return results["results"]["bindings"]


def extract_entities(limit: int = None) -> List[dict]:
    """Extract entities with labels from your knowledge graph."""
    limit_clause = f"LIMIT {limit}" if limit else ""

    # Customize this query for your dataset structure
    query = f"""
    SELECT DISTINCT ?entity ?label ?type
    FROM <{GRAPH_URI}>
    WHERE {{
        ?entity <http://www.w3.org/2000/01/rdf-schema#label> ?label .
        OPTIONAL {{ ?entity a ?type }}
    }}
    {limit_clause}
    """

    results = query_sparql(query)
    entities = []
    seen = set()

    for r in results:
        uri = r["entity"]["value"]
        if uri in seen:
            continue
        seen.add(uri)

        entities.append({
            "uri": uri,
            "name": r["label"]["value"],
            "type": r.get("type", {}).get("value", "entity")
        })

    print(f"Found {len(entities)} entities")
    return entities


def extract_predicates(limit: int = None) -> List[dict]:
    """Extract unique predicates from your knowledge graph."""
    limit_clause = f"LIMIT {limit}" if limit else ""

    query = f"""
    SELECT DISTINCT ?predicate
    FROM <{GRAPH_URI}>
    WHERE {{
        ?s ?predicate ?o .
    }}
    {limit_clause}
    """

    results = query_sparql(query)
    predicates = []

    for r in results:
        uri = r["predicate"]["value"]
        # Extract local name from URI
        name = uri.split("/")[-1].split("#")[-1].replace("_", " ")

        predicates.append({
            "uri": uri,
            "name": name
        })

    print(f"Found {len(predicates)} predicates")
    return predicates


def upload_entities(client: QdrantClient, entities: List[dict]):
    """Embed and upload entities to Qdrant."""
    if not entities:
        return

    names = [e["name"] for e in entities]
    vectors = get_embeddings(names)

    points = [
        models.PointStruct(
            id=i,
            vector=vectors[i],
            payload={
                "uri": e["uri"],
                "name": e["name"],
                "node_type": e["type"]
            }
        )
        for i, e in enumerate(entities)
    ]

    # Upload in batches
    batch_size = 500
    for i in range(0, len(points), batch_size):
        client.upload_points(
            collection_name=COLLECTION_ENTITIES,
            points=points[i:i + batch_size],
            wait=True
        )
    print(f"Uploaded {len(points)} entities")


def upload_relations(client: QdrantClient, predicates: List[dict]):
    """Embed and upload predicates to Qdrant."""
    if not predicates:
        return

    names = [p["name"] for p in predicates]
    vectors = get_embeddings(names)

    points = [
        models.PointStruct(
            id=i,
            vector=vectors[i],
            payload={
                "uri": p["uri"],
                "predicate": p["name"],
                "type": "relation_schema"
            }
        )
        for i, p in enumerate(predicates)
    ]

    client.upload_points(
        collection_name=COLLECTION_RELATIONS,
        points=points,
        wait=True
    )
    print(f"Uploaded {len(points)} relations")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, help="Limit items (for testing)")
    args = parser.parse_args()

    # Connect to Qdrant
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    print(f"Connected to Qdrant")

    # Setup collections
    setup_collections(client)

    # Extract and upload
    entities = extract_entities(limit=args.limit)
    predicates = extract_predicates(limit=args.limit)

    upload_entities(client, entities)
    upload_relations(client, predicates)

    print(f"\nDone! Entities: {len(entities)}, Relations: {len(predicates)}")


if __name__ == "__main__":
    main()
```

### Run the Population Script

```powershell
# Test with limited data first
python db/populate_mydata_vectors.py --limit 1000

# Full run
python db/populate_mydata_vectors.py
```

### Verify Qdrant Collections

```python
from qdrant_client import QdrantClient

client = QdrantClient(host="localhost", port=6333)

# Check collections exist
for col in ["mydata-entities", "mydata-relations"]:
    info = client.get_collection(col)
    print(f"{col}: {info.points_count} vectors")

# Test semantic search
results = client.query_points(
    collection_name="mydata-entities",
    query=[0.1] * 4096,  # Replace with actual embedding
    limit=5
)
print(results)
```

## Step 4: Create MCP Server (Optional)

For full agent integration, create an MCP server with tools. See `ama_kbqa/server/kqapro_server.py` for reference.

### Minimal MCP Server Template

Create `ama_kbqa/server/<dataset>_server.py`:

```python
"""MCP Server for <dataset> knowledge graph."""
from fastmcp import FastMCP
from SPARQLWrapper import SPARQLWrapper, JSON
from qdrant_client import QdrantClient

mcp = FastMCP("<dataset>-server")

# Configuration
SPARQL_ENDPOINT = "http://localhost:8890/sparql"
GRAPH_URI = "http://example.org/mydata"
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333

sparql = SPARQLWrapper(SPARQL_ENDPOINT)
sparql.setReturnFormat(JSON)
qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


@mcp.tool()
def execute_sparql(query: str) -> dict:
    """Execute a SPARQL query against the knowledge graph."""
    full_query = f"""
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

    {query}
    """
    sparql.setQuery(full_query)
    results = sparql.query().convert()
    return results


@mcp.tool()
def search_entities(query: str, limit: int = 5) -> list:
    """Search for entities by semantic similarity."""
    # Generate embedding for query
    # ... embedding code ...

    results = qdrant.query_points(
        collection_name="mydata-entities",
        query=query_embedding,
        limit=limit
    )
    return [r.payload for r in results.points]


if __name__ == "__main__":
    mcp.run()
```

## Step 5: Verify the Setup

### Test SPARQL Queries

Visit http://localhost:8890/sparql:

```sparql
# Count triples
SELECT COUNT(*) FROM <http://example.org/mydata> WHERE { ?s ?p ?o }

# Sample entities
SELECT ?s ?label FROM <http://example.org/mydata>
WHERE {
    ?s <http://www.w3.org/2000/01/rdf-schema#label> ?label
}
LIMIT 10

# Sample predicates
SELECT DISTINCT ?p FROM <http://example.org/mydata>
WHERE { ?s ?p ?o }
LIMIT 20
```

### Test Qdrant Search

```python
from qdrant_client import QdrantClient
from openai import OpenAI

# Setup
qdrant = QdrantClient(host="localhost", port=6333)
openai = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="your_key"
)

# Generate test embedding
response = openai.embeddings.create(
    model="qwen/qwen3-embedding-8b",
    input="test query"
)
query_vector = response.data[0].embedding

# Search
results = qdrant.query_points(
    collection_name="mydata-entities",
    query=query_vector,
    limit=5
)

for r in results.points:
    print(f"{r.score:.3f} - {r.payload['name']}")
```

## Troubleshooting

### Virtuoso Load Issues

**File not found:**
```sql
-- Check directory contents
SELECT * FROM sys_dir_ls WHERE path = '/usr/share/proj/';
```

**Loader stuck:**
```sql
-- Check loader status
SELECT * FROM DB.DBA.load_list;

-- Clear and retry
DELETE FROM DB.DBA.load_list;
ld_dir('/path', 'file.nt', 'graph');
rdf_loader_run();
```

**Invalid RDF:**
Check N-Triples syntax - each line must be:
```
<subject> <predicate> <object> .
```

### Qdrant Issues

**Connection refused:**
```powershell
docker ps  # Check if qdrant_ama_kbqa is running
docker logs qdrant_ama_kbqa  # Check for errors
```

**Collection not found:**
Re-run the population script to create collections.

### Embedding API Errors

**Rate limits:**
- Reduce `batch_size` in `get_embeddings()`
- Add delays between batches

**Invalid API key:**
- Verify `.env` has correct `OPENROUTER_API_KEY`
- Check API key permissions at openrouter.ai

## Quick Reference

### Load RDF into Virtuoso

```powershell
docker exec -i virtuoso_ama_kbqa isql 1111 dba kit_ama_kbqa "EXEC=ld_dir('/usr/share/proj/<folder>', '<file>.nt', '<graph_uri>'); rdf_loader_run(); checkpoint;"
```

### Clear a Graph

```sql
SPARQL CLEAR GRAPH <http://example.org/mydata>;
```

### Delete Qdrant Collections

```python
client.delete_collection("mydata-entities")
client.delete_collection("mydata-relations")
```

### Full Reset

```powershell
cd db
docker compose down -v
docker compose up -d
# Wait for containers to start, then reload data
```

## Checklist

- [ ] Data converted to N-Triples format
- [ ] N-Triples file placed in `db/datasets/<name>/`
- [ ] RDF triples loaded into Virtuoso
- [ ] Triple count verified via SPARQL
- [ ] Population script created
- [ ] Qdrant collections populated
- [ ] Vector search tested
- [ ] (Optional) MCP server created
- [ ] Documentation updated

## Related Documentation

- [KQAPro Dataset](../datasets/kqapro.md) - KQAPro structure reference
- [SciQA Dataset](../datasets/sciqa.md) - SciQA structure reference
- [Database Schema](../../.agent/System/database_schema.md) - Full schema details
- [Database Setup SOP](../../.agent/SOP/database_setup.md) - Step-by-step setup
