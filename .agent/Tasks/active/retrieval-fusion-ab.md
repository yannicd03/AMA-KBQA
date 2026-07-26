# Retrieval fusion A/B: RRF vs DBSF (and whether BM25 earns its place)

**Status:** 📋 Planned — blocking validation for a default that has already been changed
**Raised:** 2026-07-26
**Why it exists:** `config.toml` now ships `fusion = "dbsf"`, switched on a hypothesis
that has never been tested. This document is what makes that provisional rather than
silent. Until this A/B runs, nothing may cite DBSF as validated.

## The hypothesis

`ama_kbqa/retrieval/search.py:19-21` states it in its own module docstring: the caller's
cosine `score_threshold` gates **only the dense branch**. At the BM25 prefetch the code
says `# No score_threshold: BM25 scores are not cosine similarities.` So with
`hybrid_enabled = true`, the `score_threshold = 0.6` that protects precision applies to
half the pipeline. BM25 contributes up to `prefetch_limit = 20` candidates with no
quality gate, and RRF fuses by **rank**, discarding magnitude. A lexically top-ranked but
semantically weak candidate therefore receives a top fused score even when its cosine
similarity is far below the threshold that would have excluded it in dense-only mode.

Entity labels are short ("Wonder Woman", "CINIC-10"), so BM25 over them rewards rare-token
overlap and surfaces near-homographs.

Predicted symptom: more plausible-but-wrong entry points reach the agent, which spends
turns investigating and discarding them. **This matches the observed data** — turns rose
in every hybrid/reranked arm of `rag-rerank-100q-2026-06-07` (KQAPro 8.50 → 8.75, SciQA
7.77 → 8.19).

DBSF normalises each branch's score distribution before combining, which preserves
confidence and acts as a soft gate. If the hypothesis is right, DBSF should recover the
precision that RRF throws away without giving up BM25's recall.

## The critical point this experiment exists to settle

**The existing evidence cannot distinguish two very different explanations:**

1. BM25 is useless for short entity labels, or
2. RRF over an ungated BM25 branch is a bad way to use BM25.

Both predict the observed null result, and only configuration (2) has ever been tested.
Dropping the sparse branch now would be deciding on evidence that cannot support the
conclusion. That matters because BM25's theoretical strength is exactly rare tokens and
identifiers (`CINIC-10`, metric and dataset names), which is a real share of SciQA — so
hybrid failing to help SciQA is *surprising* and "we fused it badly" is a more plausible
reading than "lexical matching has no value for scientific entity names".

## Arms

All at commit ≥ `5fdca62`, gemma-4-31b-kit, seed 42, orca stack stopped (memory pressure
costs ~0.92 s/q of tool time and must be held constant).

| arm | `hybrid_enabled` | `fusion` | `reranker_enabled` | tests |
|---|---|---|---|---|
| A `dense_rerank` | false | n/a | true | "drop BM25" — **already running 2026-07-26** as `densererank-{kqapro-n500,sciqa-n100}-2026-07-26` |
| B `hybrid_rerank_rrf` | true | rrf | true | the old default; baseline = `full-kqapro-n500-seed42-2026-07-25` |
| C `hybrid_rerank_dbsf` | true | dbsf | true | **the new default — the one that must be validated** |

Set arms via env, not by editing `config.toml`: `AMA_RETRIEVAL_HYBRID_ENABLED`,
`AMA_RETRIEVAL_FUSION`, `AMA_RETRIEVAL_RERANKER_ENABLED` (prefix `AMA_RETRIEVAL_`,
`ama_kbqa/config.py` ~:662). Verify the live process actually got them by reading
`/proc/<pid>/environ` — env overrides do NOT appear in `run_manifest.json`'s `config_toml`
snapshot, which is why past arm configs are unrecoverable from manifests.

## Sizing, and an honest limit

KQAPro at n=500 gives roughly ±3 pp, tight enough to resolve this. **SciQA cannot be
powered further**: the handcrafted split is exactly 100 questions
(`db/datasets/SciQA/Handcrafted/full dataset.csv`), so its accuracy comparison stays at
about ±8 pp and will not reach significance no matter what. Judge SciQA on turns and
latency, which are far more sensitive, and treat its accuracy as directional only.

## Metrics, in priority order

1. **Turns/question** (`avg_tool_calls_per_question`) — the most sensitive signal, and the
   one the hypothesis makes a directional prediction about. DBSF should reduce it relative
   to RRF if precision improves.
2. **Accuracy** — KQAPro only for significance; run McNemar against the RRF arm.
3. **`FindNode` avg duration** — DBSF should not change retrieval cost materially; if it
   does, that is a separate finding.
4. Tokens/question and wall clock, reported but not decisive (wall clock on this host is
   confounded by shared-endpoint variance).

## Decision rule, fixed in advance

- **C ≥ B on accuracy and C < B on turns** → keep `dbsf`, mark validated, update the
  config comment.
- **C ≈ B on both** → the fusion method is not the lever; revert to `rrf` and treat the
  BM25 branch itself as the suspect, i.e. promote arm A.
- **A ≥ C** → BM25 earns nothing even when fused well; set `hybrid_enabled = false` and
  consider removing the branch.
- **C < B** → revert immediately; the provisional default was wrong.

Writing the rule down before running the arms is deliberate: the whole reason this file
exists is that a default was previously justified by numbers belonging to a different arm.

## Related

- `.agent/Tasks/active/prompt-cache-utilization.md` — same host, overlapping latency work.
- Wiki `0_Claude/AMA-KBQA/ablation-ab-testing-programme.md` owns the cross-session
  experiment ledger.
- `GetSchemaForAttribute` (`ama_kbqa/server/kqapro_server.py:957-972`) does bespoke
  in-process numpy cosine and never touches Qdrant, so it is governed by none of these
  flags. Out of scope here, but it means "our retrieval is configured as X" always has an
  exception.
