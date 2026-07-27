"""Condition 3: realistic agent search keywords.

Conditions 1 and 2 (union_recall.py) bracket the answer but neither is the
agent's operating point:
  - "name"     queries with the gold label itself  -> trivially easy
  - "question" queries with the raw question text  -> unrealistically hard

The agent actually runs an LLM classification/extraction pass first and then
searches FindNode with the mentions it extracted. This script reproduces that
exact step: it imports the agent's OWN prompt from the repo and calls the same
model the benchmark uses (gemma-4-31b-it), via OpenRouter rather than the KIT
endpoint so the running Hetzner benchmark is not perturbed.

A gold entity counts as retrieved if it appears in the top-k of ANY of the
agent's extracted-mention queries, which is how the agent actually behaves.
"""
import argparse
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
import urllib.request

sys.path.insert(0, "/home/yannic/code/AMAKBQA")
from qdrant_client import QdrantClient, models
from ama_kbqa.retrieval.search import BM25_MODEL, BM25_SPARSE_VECTOR_NAME
from ama_kbqa.agents.kqapro_agent.prompts import CLASSIFICATION_AND_EXTRACTION_PROMPT

QDRANT_URL = "http://localhost:6335"
EMBED_MODEL = "qwen/qwen3-embedding-8b"
CHAT_MODEL = "google/gemma-4-31b-it"
THRESHOLD, PREFETCH_K = 0.6, 20

_key = os.environ["OR_KEY"]
_cache, _lock = {}, threading.Lock()


def _post(url, body, timeout=90):
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(),
        headers={"Authorization": "Bearer " + _key, "Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def embed(text):
    with _lock:
        if text in _cache:
            return _cache[text]
    for attempt in range(4):
        try:
            v = _post("https://openrouter.ai/api/v1/embeddings",
                      {"model": EMBED_MODEL, "input": text})["data"][0]["embedding"]
            break
        except Exception:
            if attempt == 3:
                raise
    with _lock:
        _cache[text] = v
    return v


def extract(question):
    """Run the agent's real classification+extraction step."""
    prompt = CLASSIFICATION_AND_EXTRACTION_PROMPT.format(question=question)
    for attempt in range(4):
        try:
            r = _post("https://openrouter.ai/api/v1/chat/completions", {
                "model": CHAT_MODEL,
                "messages": [{"role": "system", "content": prompt}],
                "temperature": 1.0,          # config.toml chat_temperature
                "response_format": {"type": "json_object"},
                "max_tokens": 1500,
            })
            txt = r["choices"][0]["message"]["content"]
            obj = json.loads(txt[txt.index("{"):txt.rindex("}") + 1])
            ents = [e for e in obj.get("entities", []) if isinstance(e, str) and e.strip()]
            return ents, obj.get("question_type")
        except Exception:
            if attempt == 3:
                return [], None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-golds", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    from ama_kbqa.benchmark_agents import stratified_sample, load_raw_dataset
    raw = load_raw_dataset("kqapro")
    sample = stratified_sample(raw, 500, a.seed, "kqapro")

    items, total = [], 0
    for q in sample:
        golds = [s["inputs"][0] for s in q["program"]
                 if s["function"] == "Find" and s.get("inputs")]
        if not golds:
            continue
        items.append({"question": q["question"], "golds": golds})
        total += len(golds)
        if total >= a.target_golds:
            break

    client = QdrantClient(url=QDRANT_URL)
    print(f"{len(items)} questions / {total} gold entities (target {a.target_golds})")

    def norm(s):
        return (s or "").strip().lower()

    def work(it):
        ents, qtype = extract(it["question"])
        # Union over every mention the agent would search with.
        dg, du, bm = set(), set(), set()
        for e in ents[:6]:
            v = embed(e)
            for p in client.query_points(collection_name="kqapro-entities", query=v,
                                         limit=PREFETCH_K, score_threshold=THRESHOLD,
                                         with_payload=True).points:
                dg.add(norm(p.payload.get("name")))
            for p in client.query_points(collection_name="kqapro-entities", query=v,
                                         limit=PREFETCH_K, with_payload=True).points:
                du.add(norm(p.payload.get("name")))
            for p in client.query_points(
                    collection_name="kqapro-entities",
                    query=models.Document(text=e, model=BM25_MODEL),
                    using=BM25_SPARSE_VECTOR_NAME, limit=PREFETCH_K,
                    with_payload=True).points:
                bm.add(norm(p.payload.get("name")))
        rows = []
        for g in it["golds"]:
            rows.append({
                "gold": g, "qtype": qtype, "extracted": ents,
                "exact_mention": any(norm(e) == norm(g) for e in ents),
                "dense_gated": norm(g) in dg,
                "dense_ungated": norm(g) in du,
                "bm25": norm(g) in bm,
            })
        return rows

    results = []
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for i, rows in enumerate(ex.map(work, items), 1):
            results.extend(rows)
            if i % 10 == 0:
                print(f"  {i}/{len(items)}", flush=True)

    n = len(results)
    dg = sum(r["dense_gated"] for r in results)
    du = sum(r["dense_ungated"] for r in results)
    bm = sum(r["bm25"] for r in results)
    only_g = sum(r["bm25"] and not r["dense_gated"] for r in results)
    only_u = sum(r["bm25"] and not r["dense_ungated"] for r in results)
    dense_only = sum(r["dense_ungated"] and not r["bm25"] for r in results)
    neither = sum(not r["dense_ungated"] and not r["bm25"] for r in results)
    exact = sum(r["exact_mention"] for r in results)
    def pct(x):
        return f"{x/n*100:5.1f}%"

    print()
    print(f"=== condition: agent-keywords  (gold entities: {n}, top-k={PREFETCH_K}, model={CHAT_MODEL})")
    print(f"  agent extracted the gold label verbatim : {pct(exact)} ({exact})")
    print("  ---")
    print(f"  dense recall, gated @{THRESHOLD} : {pct(dg)} ({dg})")
    print(f"  dense recall, ungated          : {pct(du)} ({du})")
    print(f"  bm25 recall                    : {pct(bm)} ({bm})")
    print("  ---")
    print(f"  BM25-only vs GATED dense       : {pct(only_g)} ({only_g})")
    print(f"  BM25-only vs UNGATED dense     : {pct(only_u)} ({only_u})   <- unique contribution")
    print(f"  dense-only (ungated)           : {pct(dense_only)} ({dense_only})")
    print(f"  neither branch found gold      : {pct(neither)} ({neither})")

    if a.out:
        with open(a.out, "w") as f:
            json.dump(results, f, indent=1)
        print("\nwrote", a.out)


if __name__ == "__main__":
    main()
