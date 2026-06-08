"""Upgrade existing dense-only Qdrant collections to BM25-hybrid-capable.

Adds a named "bm25" sparse vector (IDF modifier, server-side inference) to
each collection WITHOUT re-embedding: existing dense vectors and payloads
are preserved; only the sparse side is added, computed by the Qdrant server
from each point's name/predicate text.

Strategy (crash-safe: data always exists in at least one collection):
  1. Scroll the original collection (dense vectors + payloads).
  2. Build a verified copy at {name}__bm25_tmp with the hybrid schema;
     the server computes the BM25 sparse vectors during these upserts.
  3. Delete + recreate the original with the hybrid schema and copy the
     points back from the tmp (sparse vectors are scrolled back as-is, so
     no second inference pass), then delete the tmp.

Idempotent: collections that already have the "bm25" sparse vector are
skipped. Fresh populations via db/populate_*_vectors.py create
hybrid-capable collections directly and don't need this script.

Usage:
    uv run python db/migrate_add_bm25.py [--dry-run] [--yes]
        [--collections kqapro-entities sciqa-entities ...]
        [--host HOST] [--port PORT] [--batch-size N] [--keep-tmp]
"""

import argparse
import sys

from qdrant_client import QdrantClient, models

from ama_kbqa.config import (
    get_collection_entities,
    get_collection_relations,
    get_qdrant_host,
    get_qdrant_port,
    get_sciqa_collection_entities,
    get_sciqa_collection_relations,
)
from ama_kbqa.retrieval.search import (
    BM25_MODEL,
    BM25_SPARSE_VECTOR_NAME,
    _doc_text,
)

TMP_SUFFIX = "__bm25_tmp"


def default_collections() -> list[str]:
    return [
        get_collection_entities(),
        get_collection_relations(),
        get_sciqa_collection_entities(),
        get_sciqa_collection_relations(),
    ]


def has_bm25(client: QdrantClient, collection_name: str) -> bool:
    info = client.get_collection(collection_name)
    sparse = info.config.params.sparse_vectors or {}
    return BM25_SPARSE_VECTOR_NAME in sparse


def dense_params(client: QdrantClient, collection_name: str) -> models.VectorParams:
    """Read the unnamed dense vector config of an existing collection."""
    vectors = client.get_collection(collection_name).config.params.vectors
    if not isinstance(vectors, models.VectorParams):
        raise RuntimeError(
            f"Collection '{collection_name}' does not use a single unnamed "
            f"dense vector (found: {type(vectors).__name__}); this migration "
            "only handles the layout created by db/populate_*_vectors.py."
        )
    return vectors


def create_hybrid_collection(
    client: QdrantClient, collection_name: str, dense: models.VectorParams
) -> None:
    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=dense.size, distance=dense.distance),
        sparse_vectors_config={
            BM25_SPARSE_VECTOR_NAME: models.SparseVectorParams(
                modifier=models.Modifier.IDF
            )
        },
    )


def scroll_all(client: QdrantClient, collection_name: str, batch_size: int):
    """Yield batches of points (with vectors and payloads)."""
    offset = None
    while True:
        records, offset = client.scroll(
            collection_name=collection_name,
            limit=batch_size,
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )
        if records:
            yield records
        if offset is None:
            return


def copy_points(
    client: QdrantClient,
    source: str,
    target: str,
    batch_size: int,
    *,
    add_bm25_documents: bool,
) -> int:
    """Copy all points from source to target.

    With add_bm25_documents=True the source is dense-only and the sparse
    vector is attached as a Document for server-side inference. With False
    the source already carries the computed sparse vectors and they are
    copied through verbatim.
    """
    copied = 0
    for records in scroll_all(client, source, batch_size):
        points = []
        for record in records:
            if add_bm25_documents:
                vector = {
                    # "" addresses the unnamed default dense vector.
                    "": record.vector,
                    BM25_SPARSE_VECTOR_NAME: models.Document(
                        text=_doc_text(record.payload), model=BM25_MODEL
                    ),
                }
            else:
                vector = record.vector
            points.append(
                models.PointStruct(id=record.id, vector=vector, payload=record.payload)
            )
        client.upsert(collection_name=target, points=points, wait=True)
        copied += len(points)
        print(f"  ... {copied} points copied to '{target}'")
    return copied


def assert_count(client: QdrantClient, collection_name: str, expected: int) -> None:
    actual = client.get_collection(collection_name).points_count
    if actual != expected:
        raise RuntimeError(
            f"Point count mismatch in '{collection_name}': "
            f"expected {expected}, found {actual}"
        )


def migrate_collection(
    client: QdrantClient, collection_name: str, batch_size: int, keep_tmp: bool
) -> None:
    tmp_name = collection_name + TMP_SUFFIX
    source_count = client.get_collection(collection_name).points_count
    dense = dense_params(client, collection_name)
    print(
        f"\nMigrating '{collection_name}' "
        f"({source_count} points, dim={dense.size}, {dense.distance})"
    )

    # Stage 1: verified hybrid copy at the tmp name (server computes BM25).
    if client.collection_exists(tmp_name):
        print(f"  Removing leftover tmp collection '{tmp_name}'")
        client.delete_collection(tmp_name)
    create_hybrid_collection(client, tmp_name, dense)
    copy_points(client, collection_name, tmp_name, batch_size, add_bm25_documents=True)
    assert_count(client, tmp_name, source_count)
    print(f"  Verified tmp copy '{tmp_name}' ({source_count} points)")

    # Stage 2: recreate the original with the hybrid schema and copy back.
    # The tmp holds a full verified copy, so the original is recoverable at
    # every step; the scrolled sparse vectors are reused (no re-inference).
    client.delete_collection(collection_name)
    create_hybrid_collection(client, collection_name, dense)
    copy_points(client, tmp_name, collection_name, batch_size, add_bm25_documents=False)
    assert_count(client, collection_name, source_count)
    print(f"  Rebuilt '{collection_name}' with BM25 sparse index")

    if keep_tmp:
        print(f"  Keeping tmp collection '{tmp_name}' (--keep-tmp)")
    else:
        client.delete_collection(tmp_name)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Add a BM25 sparse index to existing dense-only collections."
    )
    parser.add_argument(
        "--collections",
        nargs="+",
        default=None,
        help="Collections to migrate (default: the four configured ones)",
    )
    parser.add_argument("--host", default=None, help="Qdrant host (default: config.toml)")
    parser.add_argument("--port", type=int, default=None, help="Qdrant port (default: config.toml)")
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--dry-run", action="store_true", help="Report only, change nothing")
    parser.add_argument("--yes", action="store_true", help="Skip the confirmation prompt")
    parser.add_argument("--keep-tmp", action="store_true", help="Keep the tmp copies for inspection")
    args = parser.parse_args()

    host = args.host or get_qdrant_host()
    port = args.port or get_qdrant_port()
    client = QdrantClient(host=host, port=port, timeout=120)
    client.get_collections()
    print(f"Connected to Qdrant at {host}:{port}")

    collections = args.collections or default_collections()
    todo = []
    for name in collections:
        if name.endswith(TMP_SUFFIX):
            print(f"- '{name}': SKIP (tmp collection)")
            continue
        if not client.collection_exists(name):
            print(f"- '{name}': SKIP (does not exist)")
            continue
        if has_bm25(client, name):
            print(f"- '{name}': SKIP (already has '{BM25_SPARSE_VECTOR_NAME}' sparse index)")
            continue
        count = client.get_collection(name).points_count
        print(f"- '{name}': MIGRATE ({count} points)")
        todo.append(name)

    if not todo:
        print("\nNothing to do.")
        return 0
    if args.dry_run:
        print(f"\nDry run: would migrate {len(todo)} collection(s).")
        return 0

    if not args.yes:
        response = input(
            f"\nMigrate {len(todo)} collection(s) in place "
            "(originals are rebuilt; a verified tmp copy exists throughout)? [y/N]: "
        ).strip().lower()
        if response != "y":
            print("Cancelled.")
            return 1

    for name in todo:
        migrate_collection(client, name, args.batch_size, args.keep_tmp)

    print("\n--- Migration complete ---")
    return 0


if __name__ == "__main__":
    sys.exit(main())
