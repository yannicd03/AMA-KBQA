"""Populate a Qdrant collection with all Wikidata properties for dense/hybrid
property linking on the WikiKGQA without-mentions track.

Mirrors ``populate_kqapro_vectors.py`` (same 4096-dim / COSINE / bm25-sparse
conventions so the shared ``ama_kbqa.retrieval.search`` layer works), but:
  * source is ``data/wikikgqa/properties.json`` (downloaded from the challenge
    endpoint: {id, label, description, aliases} per property),
  * the dense vector embeds "label: description" (semantic), while the bm25
    sparse text is "label | aliases" (lexical recall on names/aliases),
  * payload keeps the PID + label (label is the reranker doc key) + description.

Unlike the KQAPro script this uses config accessors (so it targets the running
Qdrant on the configured port, not a hardcoded 6333) and the shared embedding
client, and runs non-interactively (safe for a background/batch run).

Usage:  uv run python db/populate_wikidata_properties.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from qdrant_client import QdrantClient, models
from tqdm import tqdm

from ama_kbqa.config import (
    get_embedding_client,
    get_embedding_model_name,
    get_qdrant_host,
    get_qdrant_port,
)

PROPERTIES_FILE = Path("data/wikikgqa/properties.json")
COLLECTION = "wikidata-properties"
VECTOR_DIMENSION = 4096
DISTANCE = models.Distance.COSINE
BM25_SPARSE_VECTOR_NAME = "bm25"
BM25_MODEL = "Qdrant/bm25"


def _dense_text(p: dict) -> str:
    """Semantic text for the dense vector: label plus description when present."""
    label, desc = p.get("label", ""), p.get("description", "")
    return f"{label}: {desc}" if desc else (label or p["id"])


def _bm25_text(p: dict) -> str:
    """Lexical text for BM25: label plus aliases (so alias spellings match)."""
    label, aliases = p.get("label", ""), p.get("aliases", "")
    return f"{label} | {aliases}" if aliases else (label or p["id"])


def get_embeddings(client, model: str, texts: list[str], batch_size: int = 100) -> list[list[float]]:
    out: list[list[float]] = []
    with tqdm(total=len(texts), desc="Embedding properties", unit="prop") as pbar:
        for i in range(0, len(texts), batch_size):
            batch = [t.replace("\n", " ") for t in texts[i : i + batch_size]]
            resp = client.embeddings.create(model=model, input=batch, encoding_format="float")
            out.extend(d.embedding for d in resp.data)
            pbar.update(len(batch))
    return out


def main() -> int:
    if not PROPERTIES_FILE.exists():
        print(f"ERROR: {PROPERTIES_FILE} not found (run the property download first).")
        return 1
    props = json.load(open(PROPERTIES_FILE))
    props = [p for p in props if p.get("label") or p.get("description")]  # skip empties
    # Drop ExternalId properties (~75% of all properties): they are database identifiers,
    # never a KBQA answer relation, and their descriptions ("... identifier for X ...")
    # pollute semantic relation search. Everything else (WikibaseItem, Quantity, Time,
    # Monolingualtext, String, ...) is a plausible relation/attribute.
    before = len(props)
    props = [p for p in props if p.get("ptype") != "ExternalId"]
    print(f"Loaded {before} properties; kept {len(props)} after dropping ExternalId.")

    qc = QdrantClient(host=get_qdrant_host(), port=get_qdrant_port(), timeout=60)
    if qc.collection_exists(COLLECTION):
        print(f"Recreating existing collection '{COLLECTION}'...")
        qc.delete_collection(COLLECTION)
    qc.create_collection(
        collection_name=COLLECTION,
        vectors_config=models.VectorParams(size=VECTOR_DIMENSION, distance=DISTANCE),
        sparse_vectors_config={
            BM25_SPARSE_VECTOR_NAME: models.SparseVectorParams(modifier=models.Modifier.IDF)
        },
    )
    print(f"Created collection '{COLLECTION}' (dim={VECTOR_DIMENSION}, {DISTANCE}, +bm25).")

    emb_client = get_embedding_client()
    model = get_embedding_model_name()
    vectors = get_embeddings(emb_client, model, [_dense_text(p) for p in props])

    points = [
        models.PointStruct(
            id=i,
            vector={
                "": vectors[i],
                BM25_SPARSE_VECTOR_NAME: models.Document(text=_bm25_text(p), model=BM25_MODEL),
            },
            payload={
                "pid": p["id"],
                "label": p.get("label", ""),        # reranker doc key
                "description": p.get("description", ""),
                "aliases": p.get("aliases", ""),
                "ptype": p.get("ptype", ""),
                "type": "property_schema",
            },
        )
        for i, p in enumerate(props)
    ]
    print(f"Upserting {len(points)} properties into '{COLLECTION}'...")
    for i in range(0, len(points), 500):
        qc.upload_points(collection_name=COLLECTION, points=points[i : i + 500], wait=True)

    info = qc.get_collection(COLLECTION)
    print(f"DONE: '{COLLECTION}' now has {info.points_count} points.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
