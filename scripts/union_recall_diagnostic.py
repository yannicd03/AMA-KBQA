"""Per-query union-recall diagnostic: does the BM25 branch retrieve gold
entities that the dense branch misses?

Runs ENTIRELY against local resources so the Hetzner experiments are not
perturbed: local Qdrant on :6335, embeddings from OpenRouter (not the shared
KIT endpoint the benchmark is using).

Two query conditions are measured, because they bound the answer from both
sides:

  name     - query IS the gold entity's own label. Upper bound; measures
             whether the index can find an entity given its exact name.
  question - query is the raw natural-language question. Lower bound; the
             agent actually searches with a mention it extracts from this,
             so the truth sits between the two.

For each condition and each gold entity we record whether it appears in the
top-k of:
  dense_gated   - dense branch WITH the production cosine gate (0.6)
  dense_ungated - dense branch with no gate  (separates "dense cannot find
                  it" from "our threshold discards it")
  bm25          - sparse branch, which production never gates
"""
import argparse
import json
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
import urllib.request

sys.path.insert(0, "/home/yannic/code/AMAKBQA")
from qdrant_client import QdrantClient, models
from ama_kbqa.retrieval.search import BM25_MODEL, BM25_SPARSE_VECTOR_NAME

QDRANT_URL = "http://localhost:6335"
EMBED_MODEL = "qwen/qwen3-embedding-8b"
THRESHOLD = 0.6          # config.toml [search] score_threshold
PREFETCH_K = 20          # config.toml [retrieval] prefetch_limit

_key = os.environ["OR_KEY"]
_cache, _lock = {}, threading.Lock()


def embed(text):
    with _lock:
        if text in _cache:
            return _cache[text]
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/embeddings",
        data=json.dumps({"model": EMBED_MODEL, "input": text}).encode(),
        headers={"Authorization": "Bearer " + _key, "Content-Type": "application/json"},
    )
    for attempt in range(4):
        try:
            v = json.load(urllib.request.urlopen(req, timeout=60))["data"][0]["embedding"]
            break
        except Exception:
            if attempt == 3:
                raise
    with _lock:
        _cache[text] = v
    return v


def branches(client, collection, query_text):
    """Return (dense_gated, dense_ungated, bm25) payload lists at top-k."""
    vec = embed(query_text)
    dg = client.query_points(collection_name=collection, query=vec, limit=PREFETCH_K,
                             score_threshold=THRESHOLD, with_payload=True).points
    du = client.query_points(collection_name=collection, query=vec, limit=PREFETCH_K,
                             with_payload=True).points
    bm = client.query_points(
        collection_name=collection,
        query=models.Document(text=query_text, model=BM25_MODEL),
        using=BM25_SPARSE_VECTOR_NAME, limit=PREFETCH_K, with_payload=True).points
    return dg, du, bm


def norm(s):
    return (s or "").strip().lower()


def load_kqapro(n, seed):
    from ama_kbqa.benchmark_agents import stratified_sample, load_raw_dataset
    raw = load_raw_dataset("kqapro")
    sample = stratified_sample(raw, 500, seed, "kqapro")[:n]
    items = []
    for q in sample:
        golds = [s["inputs"][0] for s in q["program"]
                 if s["function"] == "Find" and s.get("inputs")]
        if golds:
            items.append({"question": q["question"], "golds": golds})
    return items, "kqapro-entities"


def load_sciqa(n):
    import csv
    # The repo copy is root-owned/absent locally; SCIQA_CSV points at a local copy.
    path = os.environ.get(
        "SCIQA_CSV",
        "/home/yannic/code/AMAKBQA/db/datasets/SciQA/Handcrafted/full dataset.csv")
    items = []
    with open(path, encoding="utf-8") as f:
        for row in list(csv.DictReader(f))[:n]:
            q = (row.get("Paraphrase") or "").strip() or \
                (row.get("Question without context (comparison)") or "").strip()
            sparql = row.get("Machine-readable query") or ""
            ids = sorted(set(re.findall(r"\bR[0-9]{3,}\b", sparql)))
            if q and ids:
                items.append({"question": q, "golds": ids})
    return items, "sciqa-entities"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["kqapro", "sciqa"], required=True)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    if a.dataset == "kqapro":
        items, collection = load_kqapro(a.n, a.seed)
        by_uri = False
    else:
        items, collection = load_sciqa(a.n)
        by_uri = True

    client = QdrantClient(url=QDRANT_URL)
    print(f"[{a.dataset}] {len(items)} questions with gold entities -> {collection}")

    # For SciQA the gold is an R-id; resolve its label so the "name" condition
    # can be run the same way as KQAPro's.
    labels = {}
    if by_uri:
        for it in items:
            for g in it["golds"]:
                if g not in labels:
                    got = client.scroll(
                        collection_name=collection, limit=1, with_payload=True,
                        scroll_filter=models.Filter(must=[models.FieldCondition(
                            key="uri", match=models.MatchValue(
                                value=f"http://orkg.org/orkg/resource/{g}"))]))[0]
                    labels[g] = got[0].payload.get("name") if got else None

    def hit(points, gold):
        for p in points:
            if by_uri:
                if (p.payload.get("uri") or "").rsplit("/", 1)[-1] == gold:
                    return True
            elif norm(p.payload.get("name")) == norm(gold):
                return True
        return False

    def work(it):
        rows = []
        for gold in it["golds"]:
            label = labels.get(gold) if by_uri else gold
            conds = {"question": it["question"]}
            if label:
                conds["name"] = label
            for cond, qtext in conds.items():
                dg, du, bm = branches(client, collection, qtext)
                rows.append({"cond": cond, "gold": gold,
                             "dense_gated": hit(dg, gold),
                             "dense_ungated": hit(du, gold),
                             "bm25": hit(bm, gold)})
        return rows

    results = []
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for i, rows in enumerate(ex.map(work, items), 1):
            results.extend(rows)
            if i % 25 == 0:
                print(f"  {i}/{len(items)}", flush=True)

    print()
    for cond in ("name", "question"):
        rs = [r for r in results if r["cond"] == cond]
        if not rs:
            continue
        n = len(rs)
        dg = sum(r["dense_gated"] for r in rs)
        du = sum(r["dense_ungated"] for r in rs)
        bm = sum(r["bm25"] for r in rs)
        only_g = sum(r["bm25"] and not r["dense_gated"] for r in rs)
        only_u = sum(r["bm25"] and not r["dense_ungated"] for r in rs)
        dense_only = sum(r["dense_ungated"] and not r["bm25"] for r in rs)
        neither = sum(not r["dense_ungated"] and not r["bm25"] for r in rs)
        def pct(x, _n=n):
            return f"{x/_n*100:5.1f}%"
        print(f"=== condition: {cond}   (gold entities measured: {n}, top-k={PREFETCH_K})")
        print(f"  dense recall, gated @{THRESHOLD} : {pct(dg)} ({dg})")
        print(f"  dense recall, ungated          : {pct(du)} ({du})")
        print(f"  bm25 recall                    : {pct(bm)} ({bm})")
        print("  ---")
        print(f"  BM25-only vs GATED dense       : {pct(only_g)} ({only_g})   <- recall the gate discards")
        print(f"  BM25-only vs UNGATED dense     : {pct(only_u)} ({only_u})   <- true unique contribution")
        print(f"  dense-only (ungated)           : {pct(dense_only)} ({dense_only})")
        print(f"  neither branch found gold      : {pct(neither)} ({neither})")
        print()

    if a.out:
        with open(a.out, "w") as f:
            json.dump(results, f, indent=1)
        print("wrote", a.out)


if __name__ == "__main__":
    main()
