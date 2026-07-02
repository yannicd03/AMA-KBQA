"""Download all Wikidata properties (id + type + en label/description/aliases) from
the challenge endpoint, for the embedded property-linking index.

Includes ``wikibase:propertyType`` so the ingest can drop ExternalId properties
(~75% of all properties, and never a KBQA answer relation) that otherwise pollute
relation search.

Usage:  uv run python db/download_wikidata_properties.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from ama_kbqa.wikikgqa.endpoint import execute, resolve_endpoint

OUT = Path("data/wikikgqa/properties.json")
_ENT = "http://www.wikidata.org/entity/"

QUERY = (
    "SELECT ?p ?pt ?pLabel ?pDesc "
    '(GROUP_CONCAT(DISTINCT ?alt; SEPARATOR=" | ") AS ?aliases) WHERE { '
    "?p wikibase:propertyType ?pt . "
    'OPTIONAL { ?p rdfs:label ?pLabel . FILTER(LANG(?pLabel)="en") } '
    'OPTIONAL { ?p schema:description ?pDesc . FILTER(LANG(?pDesc)="en") } '
    'OPTIONAL { ?p skos:altLabel ?alt . FILTER(LANG(?alt)="en") } '
    "} GROUP BY ?p ?pt ?pLabel ?pDesc"
)


def main() -> int:
    print(f"endpoint: {resolve_endpoint(None)}", flush=True)
    t0 = time.monotonic()
    r = execute(QUERY, timeout=300, retries=4)
    dt = time.monotonic() - t0
    if not r.ok or r.json is None:
        print(f"FAILED after {dt:.1f}s: {r.error}", flush=True)
        return 1
    rows = r.json.get("results", {}).get("bindings", [])
    props = [
        {
            "id": b["p"]["value"].replace(_ENT, ""),
            "ptype": b["pt"]["value"].rsplit("#", 1)[-1].rsplit("/", 1)[-1],
            "label": b.get("pLabel", {}).get("value", ""),
            "description": b.get("pDesc", {}).get("value", ""),
            "aliases": b.get("aliases", {}).get("value", ""),
        }
        for b in rows
    ]
    props.sort(key=lambda x: int(x["id"][1:]) if x["id"][1:].isdigit() else 0)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(props, open(OUT, "w"), ensure_ascii=False, indent=1)
    from collections import Counter

    top = Counter(p["ptype"] for p in props).most_common(5)
    print(f"DONE in {dt:.1f}s: {len(props)} properties -> {OUT}", flush=True)
    print(f"  top types: {top}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
