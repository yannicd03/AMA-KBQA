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

## Queue status (2026-07-26)

Arms B and C are **queued and armed**, not yet running: `~/fusion-ab-queue.sh` on
Hetzner, tmux session `fusionab`, launched 16:23 UTC. It blocks on the `SWEEP COMPLETE`
marker in `~/densererank-sweep.log`, then fast-forwards the checkout to `origin/dev`,
then runs B (rrf) followed by C (dbsf). Expected start ~01:30 UTC, ~5.5 h for both arms.

Two deliberate choices in that script:

- **B runs before C.** If the shared KIT endpoint degrades overnight, the later slot is
  the worse one, and C is the arm we hypothesise in favour of. Ordering it second means
  the schedule works against our own hypothesis rather than for it.
- **Arms are set by `AMA_RETRIEVAL_*` env only**, so the deployed `config.toml` value is
  irrelevant to arm assignment. All four override combinations were verified against
  `get_retrieval_config()` before arming.

**Confound to watch:** orca is stopped. Restarting it *between* arms rather than before
or after both invalidates B vs C (~0.92 s/q of tool time). Each arm logs `free -g` at
start so this is detectable after the fact.

## What the literature says, and how it changes the reading (2026-07-26)

A web-research pass ran while these arms were being sized. It does not stop the
experiment, but it should temper what a win for C would mean.

- **The choice of normaliser is a smaller lever than whether you normalise at all.**
  Bruch, Gai & Ingber, *An Analysis of Fusion Functions for Hybrid Retrieval* (ACM TOIS
  2023, arXiv 2210.11934) find convex combination beats RRF in- and out-of-domain, and
  that "there always exist convex combinations of scores normalized by min-max, standard
  score, or any other linear transformation that are rank-equivalent." So DBSF is one
  arbitrary point in a family, not the principled answer. Qdrant simply does not offer
  the peer-reviewed one (convex combination with tuned α).
- **DBSF is the least-validated method in the field**: no peer review, and Qdrant's own
  docs warn its per-query mean±3σ statistics come from the prefetch top-k, where a single
  dominant outlier skews the normalisation for that query.
- **RRF's documented worst case is exactly our configuration**: it is at its best fusing
  retrievers of comparable strength (its 2009 TREC setting) and at its worst when one
  branch is clearly better. *Balancing the Blend* (PVLDB, arXiv 2508.01405) measured a
  strong lexical path at 0.650 nDCG@10 fused with a weak dense path at 0.390 producing
  0.604, i.e. **below the strong path alone**.
- **The most on-point study for our corpus shape argues the sparse branch is the
  problem, not the fusion.** Hebert et al., *Robust Candidate Generation for Entity
  Linking on Short Social Media Texts* (W-NUT @ EMNLP 2022) measured, on short
  mention→entity matching, BM25 recall@16 of **0.221** (academic) / 0.556 (OOD) against
  dense at 0.78. Their winning "hybrid" was dense ∪ **alias lookup**, not dense ∪ BM25.
  Mechanically this is unsurprising: on 1-4 token labels, tf saturation is inert and
  length normalisation is near-degenerate, so BM25 reduces to IDF-weighted token overlap.
- **There is no consensus to appeal to on the specific question.** No controlled
  dense-vs-hybrid comparison on a KG entity-label corpus appears to exist, nor any study
  of cross-encoder rerankers where the document is a bare entity label. Our null result
  is not anomalous against that backdrop; it is what the nearest study predicts.

**Consequence for the decision rule:** a C-beats-B result should be read as "DBSF is a
better way to fuse this particular weak branch", never as "distribution-preserving fusion
is validated". A C≈B result is the outcome the literature predicts and should move weight
onto arm A rather than onto tuning fusion further.

## Correction to a premise this document was built on

`search.py`'s BM25 prefetch carries the comment `# No score_threshold: BM25 scores are
not cosine similarities.` That is true about the *scale*, but it has been read here as
implying the API cannot gate that branch. It can: Qdrant's `Prefetch` object accepts
`score_threshold` (confirmed in the query-points API reference; the narrative
hybrid-queries doc page omits the field, which is likely why it looked unavailable).
Gating the sparse branch is therefore a config-level change, not a code change. The open
question is only what threshold value means anything on an unbounded BM25 scale.

Note also that under RRF, gating the sparse branch and shrinking its `limit` are nearly
the same intervention, because RRF consumes only ranks.

## Cheaper and more decisive than any of these arms

**Per-query union-recall diagnostic.** For each query, is the gold entity retrieved by
the BM25 branch but *not* by the dense branch within k? This needs no benchmark run and
no LLM calls, just a sweep over the evaluation questions against Qdrant. It answers the
actual question the A/B only approaches indirectly:

- BM25-only slice under ~2% → the sparse branch contributes nothing; drop it, and no
  fusion method can change that.
- Above ~5% → a lexical branch is earning its place and fusion is worth tuning.
- Composition matters as much as size: if that slice is dominated by identifier-like
  labels (`CINIC-10`, metric names), the indicated fix is exact/prefix/alias lookup
  rather than BM25 ranking, which is what actually won in the W-NUT hybrid.

This should probably run before or alongside the arms. It is minutes of compute.

## Related

- `.agent/Tasks/active/prompt-cache-utilization.md` — same host, overlapping latency work.
- Wiki `0_Claude/AMA-KBQA/ablation-ab-testing-programme.md` owns the cross-session
  experiment ledger.
- `GetSchemaForAttribute` (`ama_kbqa/server/kqapro_server.py:957-972`) does bespoke
  in-process numpy cosine and never touches Qdrant, so it is governed by none of these
  flags. Out of scope here, but it means "our retrieval is configured as X" always has an
  exception.
