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

## Sizing: run these at n=50–100, and what that costs you

**Decision (2026-07-26): the fusion arms run at n=50–100, not n=500.** A full KQAPro sweep
is ~7 h per arm; three arms of that is not worth spending to test a hypothesis. Accept the
smaller set and adjust what you claim from it.

What n≈100 can and cannot do:

- **Turns/question: yes.** This is the primary metric and it is far more sensitive than
  accuracy. The June sweep separated arms at 8.48 / 8.50 / 8.68 / 8.75 turns at n=100, and
  the hypothesis makes a *directional* prediction here, so this is the metric the
  experiment actually turns on.
- **`FindNode` avg duration: yes.** Rests on ~3 calls/question, so ~300 samples at n=100.
- **Accuracy: no.** At n≈100 the interval is roughly ±8–10 pp. Every accuracy difference
  in the June four-arm sweep was inside it and McNemar found nothing. Do not expect this
  run to resolve accuracy, and do not report an accuracy delta from it as a finding.

**Pin the same questions across arms.** `--question-indices` (`benchmark_agents.py:2068`,
comma-separated 0-based indices applied after sampling) makes every arm answer the
*identical* question set. This matters for two reasons: stratified sampling at n=100 draws
a different sample than at n=500, so unpinned arms of different sizes are not comparable at
all; and McNemar requires **paired** observations, which only pinned indices provide. Use
one fixed index list for all fusion arms and record it in the run notes.

Caveat this creates: **arm A is already running at n=500 and is therefore NOT paired with
the fusion arms.** Compare A against them on per-tool durations and turns as a rough
reference only, or re-run A at the pinned n≈100 if a clean three-way is wanted.

**SciQA cannot be powered further regardless**: the handcrafted split is exactly 100
questions (`db/datasets/SciQA/Handcrafted/full dataset.csv`), so its accuracy stays at
about ±8 pp no matter what. Judge SciQA on turns and latency; treat its accuracy as
directional only.

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

At n≈100 accuracy is **not** a gate (see sizing above); it is a guardrail. Turns is the
deciding metric, with accuracy used only to veto an obvious regression.

- **C < B on turns, and accuracy not visibly worse (no drop beyond ~8 pp)** → keep `dbsf`.
  Mark it *provisionally supported*, not validated — a turns win at n=100 justifies keeping
  the default, not claiming an accuracy result.
- **C ≈ B on turns** → the fusion method is not the lever. Revert to `rrf`, and treat the
  BM25 branch itself as the suspect (promote arm A).
- **C > B on turns** → revert to `rrf` immediately; the provisional default was wrong.
- **A ≤ C on turns and A not worse on accuracy** → BM25 earns nothing even when fused
  well; set `hybrid_enabled = false` and consider removing the branch. Note A is currently
  unpaired (n=500), so confirm at the pinned n≈100 before acting on this one.
- **Any accuracy drop larger than ~8 pp** → treat as real despite the small n and revert,
  since an effect that large would be visible even underpowered.

Writing the rule down before running the arms is deliberate: the whole reason this file
exists is that a default was previously justified by numbers belonging to a different arm.
The rule is also deliberately weaker than the earlier draft, because n≈100 cannot support
the accuracy-based conclusions that draft assumed.

## Related

- `.agent/Tasks/active/prompt-cache-utilization.md` — same host, overlapping latency work.
- Wiki `0_Claude/AMA-KBQA/ablation-ab-testing-programme.md` owns the cross-session
  experiment ledger.
- `GetSchemaForAttribute` (`ama_kbqa/server/kqapro_server.py:957-972`) does bespoke
  in-process numpy cosine and never touches Qdrant, so it is governed by none of these
  flags. Out of scope here, but it means "our retrieval is configured as X" always has an
  exception.
