"""Add a keyword payload index on the label field to existing collections.

Why: neither `kqapro-entities` nor `sciqa-entities` carries ANY payload
index today (verified via `GET /collections/{name}` -> `payload_schema: {}`
on both). Every exact/lexical label lookup this codebase runs — KQAPro
FindNode's Phase 1, SciQA FindResource's equivalent phase, and both
LookupEntityByName/LookupResourceByLabel tools (see
ama_kbqa/retrieval/lookup.py) — issues a Qdrant `scroll` with a payload
`Filter`. WITHOUT a payload index, Qdrant answers that filter with a full
O(collection) server-side scan: every point's payload is inspected, on
every single lookup call, whether or not it matches. WITH a keyword index
on the filtered field, the same filter becomes an index lookup: Qdrant
resolves candidate point IDs directly from the index instead of scanning.

This is an OPTIMISATION, not a prerequisite: every automatic phase and tool
call this feature adds already works correctly without this index, just
by paying an unindexed scan. Concretely, what that costs today (no index)
and what changes (with index):

- kqapro-entities (17,754 points): FindNode's automatic Phase 1 already
  pays for one O(17,754) scan on EVERY call. This has been accepted since
  before this migration existed (measured median ~109.8ms per filtered
  scroll at limit=50). With the index: an index lookup instead — the scan
  cost drops out of the FindNode critical path entirely.
- sciqa-entities (171,588 points, ~10x kqapro-entities BY POINT COUNT):
  FindResource's new automatic exact-match phase (this feature branch)
  pays for one unindexed scan on EVERY call too — but measured directly
  against the live collection, this is ~36.5ms median, CHEAPER than
  KQAPro's already-accepted phase above, not ~10x more expensive as point
  count alone would suggest. Scan cost is dominated by payload SIZE, not
  point count: KQAPro payloads carry full `attributes`/`relations` arrays,
  sciqa-entities' are four small fields. So this one scan is an accepted,
  cheap trade already, with or without this migration; see
  sciqa_server.py's EXACT_MATCH_SCROLL_LIMIT comment for the full numbers.
- LookupEntityByName / LookupResourceByLabel in "contains"/"prefix" mode
  are where this migration actually matters: they issue UP TO
  `ama_kbqa.retrieval.lookup.MAX_NGRAM_PROBES` (12) separate exact-match
  probes per call (see that module's docstring for why: the
  over-specified-mention case needs "is the label a substring of the
  query", which is the opposite of what a text/full-text index answers, so
  it's solved by probing n-grams of the query with cheap exact lookups
  instead, and — per a since-fixed defect — every probe that hits must be
  kept, not just the first, which means MORE probes run per call on
  average than a "stop at first hit" design would). Without a payload
  index, that is up to 12 full scans per call instead of 1 — the single
  biggest latency multiplier this migration removes.

What this script does: `client.create_payload_index(collection_name,
field_name="name", field_schema=models.PayloadSchemaType.KEYWORD)` for both
entity collections. This is an IN-PLACE Qdrant operation — unlike
db/migrate_add_bm25.py, it does NOT recreate the collection, delete points,
or touch vectors; Qdrant builds the index from the existing payload values
in the background. Idempotent: `create_payload_index` on a field that
already has an index is a no-op (Qdrant returns success without rebuilding).

What this does NOT do: it does not index `original_id` or
`attributes.value.value` (the two extra fields KQAPro's Phase 1 / the
lookup tools' `extra_fields=` also filter on). Those `should` conditions
remain unindexed scans even after this script runs, UNLESS `--fields` is
used to add them too. The "name"/label field is the one this task's
briefing and deliverable explicitly called out, and it's the field every
new automatic phase and tool call in this feature filters on; widening to
the other fields is a follow-up, not bundled into this default run.

Cost of running it (so the operator can judge the trade-off before typing
--yes): building a keyword index requires Qdrant to read the field out of
every existing point once, which is a one-time O(collection) pass PER
collection (comparable cost to one of the scans this script exists to
eliminate) plus ongoing memory for the index structure itself (small for a
single low-cardinality-per-point string field). It does not require
downtime; the collection stays queryable while the index builds, though
filtered queries on that field are unindexed (same as today) until the
build completes.

THIS SCRIPT IS WRITE-ONLY IN THE SENSE OF INTENT — IT IS NOT RUN AS PART OF
THIS CHANGE. Creating a payload index mutates the live collections, which
is the orchestrator's call, not something to do silently as a side effect
of shipping the lookup tooling.

Usage:
    uv run python db/migrate_add_label_index.py [--dry-run] [--yes]
        [--collections kqapro-entities sciqa-entities ...]
        [--fields name] [--host HOST] [--port PORT]
"""

import argparse
import sys

from qdrant_client import QdrantClient, models

from ama_kbqa.config import (
    get_collection_entities,
    get_qdrant_host,
    get_qdrant_port,
    get_sciqa_collection_entities,
)

DEFAULT_FIELDS = ("name",)


def default_collections() -> list[str]:
    """The two entity collections every lexical lookup path filters against.

    Deliberately narrower than db/migrate_add_bm25.py's four collections:
    the relation collections (kqapro-relations / sciqa-relations) are not
    targets of any exact/lexical label lookup in this feature, so indexing
    them here would be scope creep beyond what this migration is for.
    """
    return [get_collection_entities(), get_sciqa_collection_entities()]


def existing_index_fields(client: QdrantClient, collection_name: str) -> set[str]:
    info = client.get_collection(collection_name)
    schema = info.payload_schema or {}
    return set(schema.keys())


def add_indexes(
    client: QdrantClient, collection_name: str, fields: list[str], dry_run: bool
) -> None:
    have = existing_index_fields(client, collection_name)
    for field in fields:
        if field in have:
            print(f"  '{collection_name}'.{field}: SKIP (index already exists)")
            continue
        if dry_run:
            print(f"  '{collection_name}'.{field}: WOULD CREATE keyword index")
            continue
        print(f"  '{collection_name}'.{field}: creating keyword index...")
        client.create_payload_index(
            collection_name=collection_name,
            field_name=field,
            field_schema=models.PayloadSchemaType.KEYWORD,
        )
        print(f"  '{collection_name}'.{field}: done")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Add a keyword payload index on the label field to entity collections."
    )
    parser.add_argument(
        "--collections",
        nargs="+",
        default=None,
        help="Collections to index (default: kqapro-entities + sciqa-entities from config.toml)",
    )
    parser.add_argument(
        "--fields",
        nargs="+",
        default=list(DEFAULT_FIELDS),
        help=f"Payload fields to index per collection (default: {list(DEFAULT_FIELDS)})",
    )
    parser.add_argument("--host", default=None, help="Qdrant host (default: config.toml)")
    parser.add_argument("--port", type=int, default=None, help="Qdrant port (default: config.toml)")
    parser.add_argument("--dry-run", action="store_true", help="Report only, change nothing")
    parser.add_argument("--yes", action="store_true", help="Skip the confirmation prompt")
    args = parser.parse_args()

    host = args.host or get_qdrant_host()
    port = args.port or get_qdrant_port()
    client = QdrantClient(host=host, port=port, timeout=120)
    client.get_collections()
    print(f"Connected to Qdrant at {host}:{port}")

    collections = args.collections or default_collections()
    todo = []
    for name in collections:
        if not client.collection_exists(name):
            print(f"- '{name}': SKIP (does not exist)")
            continue
        count = client.get_collection(name).points_count
        have = existing_index_fields(client, name)
        missing = [f for f in args.fields if f not in have]
        if not missing:
            print(f"- '{name}' ({count} points): SKIP (all requested fields already indexed)")
            continue
        print(f"- '{name}' ({count} points): INDEX fields {missing}")
        todo.append(name)

    if not todo:
        print("\nNothing to do.")
        return 0
    if args.dry_run:
        print(f"\nDry run: would index {len(todo)} collection(s).")
        for name in todo:
            add_indexes(client, name, args.fields, dry_run=True)
        return 0

    if not args.yes:
        response = input(
            f"\nCreate payload index(es) on {args.fields} for {len(todo)} collection(s) "
            "(in-place; vectors/points untouched, one-time O(collection) build pass "
            "per collection)? [y/N]: "
        ).strip().lower()
        if response != "y":
            print("Cancelled.")
            return 1

    for name in todo:
        add_indexes(client, name, args.fields, dry_run=False)

    print("\n--- Index migration complete ---")
    return 0


if __name__ == "__main__":
    sys.exit(main())
