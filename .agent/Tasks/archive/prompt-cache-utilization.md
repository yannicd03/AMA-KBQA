# Prompt-Cache Utilization on the KIT Endpoint

**Status:** 📋 Planned (analysis only — no application code changed by this doc)
**Identified:** 2026-07-26, live-endpoint investigation against `kit.gemma4-31b-it`
**Scope:** `BaseKBQAAgent` tool loop (`ama_kbqa/framework/base_agent.py`) and `KQAProAgent`
(`ama_kbqa/agents/kqapro_agent/`), the path exercised by the `full-kqapro-n500-seed42-2026-07-25`
gemma leg. SciQA shares the same base class; findings generalize but were not separately measured.

## Inputs taken as given (not re-derived here)

- Decode ~47–58 tok/s, stable — not the bottleneck.
- Cold prefill ~2,200–3,000 tok/s; used as **2,200 tok/s** in the arithmetic below per instruction.
- A "fully warm" ceiling of **21,600 tok/s** (used in the arithmetic below).
- Real run (`full-kqapro-n500-seed42-2026-07-25`, gemma leg): ~10.3 LLM calls/question, ~6,752
  prompt tokens/call, 447 completion tokens total, 11.35s tool exec. All-cold predicts 52.1s/q,
  all-warm predicts 23.5s/q, measured **46.31s/q** — close to the cold end, i.e. poor cache
  utilization in production.
- OpenRouter ruled out already (0/18 endpoints support implicit caching for this model).

All new measurements below are in
`/tmp/claude-1000/-home-yannic-code-AMAKBQA/5af3f564-7e37-4c40-b3d6-3920c82adcd4/scratchpad/`:
`prefix_cache_test.py` / `prefix_cache_results.json`, `prefix_cache_variance.py` /
`prefix_cache_variance_results.json`, `tool_choice_cache_test.py` /
`tool_choice_cache_results.json`. Total new requests against KIT: 24 (well under "a few dozen").

---

## 1. The critical unknown, settled: prefix caching, not exact-match

**Verdict: the KIT endpoint does genuine, position-anchored prefix caching (vLLM/SGLang-style),
not exact-request-hash caching.** Two independent tests confirm this, and one of them
(MODEND vs MODSTART) is the clean discriminator the two hypotheses disagree on.

### Test A — grow the prompt (what the agent loop actually does)

Built prompt `A` (~29k tokens), sent cold, then warm-repeated it, then sent `A + unique_tail`
(~7k more tokens appended, exactly the shape of the real agent loop), and compared against a
**fresh unique prompt of the same total length** (the correct cold reference, not `A`'s shorter
cold time):

| case | prompt tokens | TTFT | interpretation |
|---|---|---|---|
| `cold_A` | 29,027 | 8.89s | cold reference at A's length |
| `warm_A` (median of 4 reps) | 29,027 | 2.15s | exact-match cache, works but noisy (1.43s–6.39s range) |
| `grown` = A + new tail | 35,957 | 3.81s | **A's prefix reused; only the new tail cost anything** |
| `fresh_ctrl` (same length as grown, no shared prefix) | 36,560 | 12.22s | cold reference at grown's length |

`grown`/`fresh_ctrl` TTFT ratio = **0.31**. If only exact-match caching existed, `grown` (a
never-before-seen byte string) would cost the same as `fresh_ctrl`. It costs less than a third —
the shared leading ~29k tokens were reused, only the new ~7k tail was paid for.

### Test B — the discriminating case: where does the change sit?

Same total length as `A`, but change ~50 tokens either at the very **start** or right at the
**end**:

| case | TTFT | vs. baseline |
|---|---|---|
| `modstart` (first ~50 tok differ, rest identical to A) | 9.10s | ≈ `cold_A` (8.89s) — fully cold |
| `modend` (last ~50 tok differ, rest identical to A) | 2.18s | ≈ `warm_A` median (2.15s) — fully warm |

`modend`/`modstart` ratio = **0.24**. Exact-match caching predicts these two should cost the
*same* (any single differing byte anywhere is a full miss). Prefix caching predicts `modstart`
is fully cold (divergence at token 0 kills the whole lookup) and `modend` is almost fully warm
(everything before the change point still matches from position 0). The measurement matches the
prefix-caching prediction exactly. **This is decisive.**

### Corroborating: warm-repeat variance (10 back-to-back identical requests)

Cold: 9.67s. Ten immediate repeats: 1.39s–4.00s (median 1.59s, mean 2.06s, stdev 0.90s) — all
well below cold, so caching reliably engages, but with real jitter (2.5x spread among "warm"
hits) even when nothing about the request changes. A separate 4-rep run earlier showed one outlier
at 6.39s (74% of cold). This variance, on a client making back-to-back identical requests with
zero other traffic in between, is the most direct evidence that **cache hit quality is not fully
under this codebase's control** — see §4.

### Ruled out along the way

- **`tool_choice` ("required"→"auto", `base_agent.py:1354`) does not affect the prefix.**
  Flipping it with identical `messages`/`tools` gave TTFT ratios of 0.97 and 1.17 (noise-level) —
  not the ~3–5x hit a real cache break would cause. This was a plausible hazard from reading the
  code alone; live testing rules it out.
- **The `tools=` payload sits very early in the rendered prompt and is expensive to change.**
  Adding one extra dummy tool definition (+45 tokens) to an otherwise-identical request raised
  TTFT from 1.54s (warm) to 4.66s (ratio 3.03, close to that request's own cold time of 7.76s) —
  i.e. a 45-token change invalidated nearly the *entire* prompt's cache. This confirms tool
  schemas are rendered near the front of the sequence (before or interleaved with the system
  message), which matters for §2 item 4 and for the "not worth it" note on cross-qtype tool
  filtering below.

---

## 2. Ranked improvement plan

Arithmetic convention for all rows: cold = 2,200 tok/s (0.4545 ms/token), warm = 21,600 tok/s
(0.0463 ms/token), so each token moved from cold to warm saves **0.408 ms**. "Applies to N
calls/question" multiplies accordingly. All estimates are the reasoning behind the ranking, not
guarantees — see uncertainty notes per row and §4.

| # | Change | File:line | Cached-token gain | Est. savings | Risk | Effort |
|---|---|---|---|---|---|---|
| 1 | Stop shuffling the qtype-stratified sample before execution; schedule consecutive questions by (gold) qtype so back-to-back conversations share an identical leading `system_prompt + qtype-tool-schema` byte sequence | `ama_kbqa/benchmark_agents.py:672` (`random.shuffle(result)`) and `get_question_type()` at `:687` (gold `program`-derived qtype, available **before** any LLM call) | ~2,000 tok (SYSTEM_PROMPT 1,448 + qtype-specific tool schema increment, estimated 500–750) shareable across ~(k-1)/k of same-qtype-cluster questions | ≈2,000 tok × 0.408 ms ≈ **0.82s/question**, on ~90%+ of questions given 9 qtypes over ~500 questions | Low — pure processing-order change; doesn't alter which questions are sampled or their independent evaluation | Low — reorder/group after stratified sampling instead of the final `random.shuffle` |
| 2 | Give the classification call (`base_agent.py:635`, system=`CLASSIFICATION_AND_EXTRACTION_PROMPT`, entirely disjoint from the tool-loop's `SYSTEM_PROMPT`) and the synthesis call (`base_agent.py:2163-2165`, system=`_get_synthesis_system_prompt()`, also disjoint) a **shared static leading prefix** with the main `SYSTEM_PROMPT` (1,448 tok), instead of two unrelated short system messages | `base_agent.py:635`, `base_agent.py:2163` | Up to 1,448 tok reusable on 2 of ~10.3 calls/question, *if* that segment is still warm (see §4) | ≈1,448 tok × 0.408 ms × 2 calls ≈ **1.18s/question** (best case) | Medium — adds ~1,450 irrelevant tokens of tool-tier/rule content to the classifier and synthesizer's context; needs a benchmark re-run to confirm no accuracy regression before shipping | Medium — prompt restructuring + validation pass (test-agent territory) |
| 3 | Shrink the always-resent tool-schema payload so a cache **miss** (common — see §4) costs less regardless of whether caching helps: lower `_compress_description`'s `max_chars=300` (`ama_kbqa/framework/mcp_client.py:226`), and/or trim `CORE_TOOLS` (`ama_kbqa/agents/kqapro_agent/agent.py:41-45`, 11 tools always included regardless of qtype) | `mcp_client.py:226`, `kqapro_agent/agent.py:41-45` | Est. 500–750 tok reduction out of an est. 1,300–1,500 tok tool-schema block, unconditionally (doesn't depend on a cache hit) | ≈600 tok × 0.4545 ms (cold rate, since this saves on the *miss* path) × ~10.3 calls ≈ **2.5s/question upper bound** if most calls are misses in production (matches the observed near-cold average); less if caching is actually working better than it appears | Low-medium — verify tool selection accuracy doesn't regress from shorter descriptions | Low-medium — tune constant, re-run benchmark |
| 4 | Reorder `CLASSIFICATION_AND_EXTRACTION_PROMPT` (`ama_kbqa/agents/kqapro_agent/prompts.py:297-326`) so the static JSON-schema trailer (currently **after** `Question: {question}`, `:319-326`) moves **before** the question, making the full ~358-token static block cacheable instead of only the ~273 tokens preceding the question | `prompts.py:319-326` | 85 additional static tokens shareable across all classification calls in the run | ≈85 tok × 0.408 ms ≈ **0.035s/question** — negligible in isolation, but free and compounds with #2 | Low-medium — reordering instructions vs. the input is a common, generally safe pattern, but is a semantic change to a call that affects tool-set selection; validate before shipping | Low — pure text reorder |

**Combined optimistic estimate:** items 1+2+4 together ≈ **2.0s/question** of genuine cache-hit
gains, plus item 3's ≈2.5s/question hedge on miss cost ⇒ roughly **3.5–4.5s/question** of the
current 22.81s gap (46.31s measured − 23.5s all-warm ceiling) is realistically addressable from
the client side.

> **CORRECTED 2026-07-26 (later, after this doc was written).** The sentence that
> stood here claimed "the remaining ~18–19s/question gap most plausibly sits outside
> this codebase's control", attributing it to KV-cache eviction. **Both halves of that
> are now falsified.**
>
> 1. **Eviction does not happen.** A paired 15-trial experiment (unique 12k-token prompt
>    per trial, send / wait G / resend, gaps 0/5/15/45/90s, interleaved order, plus a
>    never-repeated control arm) gave median warm/cold ratios of 0.28 / 0.38 / 0.24 /
>    0.32 / 0.56 and a gap-vs-ratio Pearson r of **+0.27 on n=15**, with within-gap
>    spread swamping it. The cache survives **at least 45 seconds**; the real inter-call
>    gap is ~1.2s. Item 1's rationale should NOT be read as "reducing round-trips
>    exposed to eviction" — it is simply a longer shared prefix. See also §4 below,
>    whose eviction discussion is superseded by this note.
> 2. **The 23.5s "all-warm ceiling" was too optimistic**, so the 22.81s gap it implies
>    is overstated. That ceiling priced warm tokens at 1/21,600 s each and assumed **no
>    fixed per-call cost**. Measured directly: an immediate repeat of a 12k prompt still
>    costs **1.30s at best, 2.90s median** (queue + scheduling + network). At 10.35
>    calls/question that is 13.5–30s irreducible, leaving real headroom of roughly
>    **4–12s/question**, not 22.8s.
>
> **What the gap actually is.** Decomposing `summary.json` against the June baseline
> (`rag-rerank-100q-2026-06-07/hybrid_rerank`, 30.57s/q) shows tool time rose 4.72 →
> 11.35 s/q (+141%) while LLM time rose 25.85 → 34.96 s/q (+35%), on flat token volume
> (71,667 → 70,331) and only +7% tool calls. Tool execution is local Virtuoso/Qdrant and
> never contacts the inference endpoint, so **~42% of the regression is local
> infrastructure, not the LLM provider.**
>
> **Consequence for this doc's ranking.** Items 1–4 remain net-positive and are worth
> landing, but they are ~1–2s/question of polish, not the front door to a large win. The
> dominant term is **call count × per-call floor** (10.35 × 1.3–2.9s). Cutting average
> tool calls from 9.35 to ~6 is worth ~9–12s/question and is the primary lever.

---

## 3. Not worth it (settled, don't relitigate)

- **`auto_inject_journal` / journal refresh injection (`base_agent.py:1913-1971`).** This was the
  prime suspect going in. It turns out to have **already been fixed**: the code and its own
  comment (`base_agent.py:1915-1926`) explicitly describe switching from a replace-in-place
  refresh (which busted the cache every 5 iterations) to an append-only design specifically for
  this reason. Nothing to change here.
- **Few-shot examples sitting before stable content.** Checked `KQAProAgent._build_analysis_context`
  (`kqapro_agent/agent.py:392-460`): the fewshot block is appended as part of the analysis-context
  message, itself the **second** message (after system prompt and the raw query), i.e. at the tail,
  not the head. The "fewshot before static content" antipattern the task description flagged as a
  lead does not exist in this code.
- **`tool_choice` "required"→"auto" transition (`base_agent.py:1354`).** Looked like a plausible
  cache-busting transition point from reading the code. Directly measured (§1) to have no effect
  (TTFT ratios 0.97, 1.17 — noise level). Do not spend effort on this.
- **Tool-definition ordering / MCP server dict-ordering concerns.** `kqapro_server.py` registers
  tools via `@mcp.tool()` decorators in fixed source order (FastMCP preserves insertion order), and
  `_ask_impl` fetches+filters the tool list **once per question**, reusing it for the whole
  conversation (`base_agent.py:911-912`, `1020-1027`). No evidence of instability. (We *did* prove
  that IF this were unstable, it would be very costly — §1's tools-list test — so this is a "leave
  it alone, but don't regress it" item, not a "fix it" item.)
- **Further tuning `_manage_context_window`'s hysteresis (`base_agent.py:1974-2105`).** Already a
  deliberate one-shot-per-threshold-crossing design with an explicit rationale in the code comment.
  Marginal additional tuning (moving the initial 50% trigger) has uncertain payoff and risks context
  overflow on long questions. Not a good use of effort relative to items 1–4 above.
- **Reasoning about `seed` (`AMA_LLM_SEED`) as a cache-key input.** It's a sampling parameter, held
  constant for an entire run (set once from `--seed` at process start), never varies call-to-call,
  and (being a decode-time knob rather than prompt content) shouldn't participate in a KV-cache key
  in the first place. Nothing to do here.

---

## 4. Honest uncertainty

**What we verified directly, with live measurements:** prefix caching exists and is real
(§1, decisive `modend`/`modstart` test); `tool_choice` doesn't matter; the `tools=` payload sits
early and is expensive to perturb; warm-repeat TTFT has real variance (1.4x–4x cold-adjusted
spread) even on a client issuing perfectly identical, back-to-back requests with no other traffic
interleaved.

> **SUPERSEDED 2026-07-26 (later).** The eviction hypothesis set out in the paragraph
> below was tested directly and **refuted** (see the correction note in §2). The cache
> survives at least 45s against a ~1.2s real inter-call gap. The genuinely unverified
> items in this paragraph are still open (KIT topology, replica routing, KV-pool size);
> only the eviction conclusion drawn from them is wrong.

**What we could not verify:** we have no visibility into the KIT deployment's topology (single
replica vs. load-balanced pool), whether it has session/cache affinity, how large its KV-cache
pool is, or how much concurrent load from other tenants it carries at any given moment. This
matters because it's our leading hypothesis for the gap between (a) our isolated test, which
showed strong and fairly reliable caching with zero interleaved foreign traffic, and (b) the real
benchmark run's near-cold average despite the agent's already-good append-only message design. If
the endpoint spreads requests across replicas without sticky routing, or if other tenants' traffic
evicts our KV blocks in the ~1–2s gap between our own calls (tool execution + decode time), no
client-side prompt-shape fix can close most of that gap — only shrinking total bytes at risk
(item 3) and reducing the number of round-trips exposed to eviction (item 1, by keeping matching
prefixes temporally adjacent) meaningfully help, and neither is a full fix.

**What would falsify our conclusions:** if KIT ops confirms a single-replica deployment with a
large KV-cache pool and no other-tenant contention during the benchmark window, then the observed
poor cache utilization must instead come from something in the code we didn't catch — worth
re-auditing `_manage_context_window`'s trigger math against the *actual* per-question token
trajectories (we used the domain-settings default of 100k context / 50k trigger and the *average*
6,752 tok/call, but didn't inspect real per-question message-length curves, which could compact
earlier/more often than assumed for above-average-length questions). Conversely, if a future
change to KQAPro's tool set introduces per-iteration or per-refresh tool-list rebuilding, §1's
finding that a 45-token tools-list change costs ~3x TTFT means that change would be far more
damaging than it looks from a token-count diff alone — worth flagging to anyone touching
`_get_allowed_tools_for_qtype` or the MCP tool-fetch path in the future.

---

## Implemented (2026-07-26)

Items **1** and **2** from the ranked plan above were implemented (ordering-only changes; no
information added or removed, only where it sits):

- **Item 1** — `ama_kbqa/benchmark_agents.py::stratified_sample`: replaced the trailing
  `random.shuffle(result)` with `result.sort(key=lambda item: get_question_type(item, agent_name))`.
  Groups are ordered by qtype name (deterministic); within-group order still comes from the
  `random.sample` calls already using the function's seeded RNG (`random.seed(seed)` at the top),
  so no fresh randomness was introduced and the sampled *set* is unchanged for a given seed.
- **Item 2** — split the per-question message stack into a STATIC block (qtype strategy +
  few-shot + general guidance + tool tips, ~2,900–3,000 tok) followed by a PER-QUESTION block
  (query + entities + relations + exact constraints), instead of the reverse. Implementation:
  - `BaseKBQAAgent._build_analysis_context` (`ama_kbqa/framework/base_agent.py`) split into
    `_build_static_qtype_context` and `_build_question_context`; the old method kept as a thin
    wrapper for backward compatibility (tests still call it directly).
  - `KQAProAgent` and `SciQAAgent` both override the two new methods (equivalent split applied to
    both agents — SciQA's shape was simple enough for a clean split too, so both are covered, not
    just KQAPro).
  - `BaseKBQAAgent._ask_impl`: `is_followup` detection kept at its original position (before any
    append for the turn); on a fresh turn, classification now runs before any message is appended
    (it only needs the raw query string), so the static context can be appended first, then the
    raw query, then the per-question context. Follow-up turns are unchanged (only the raw query is
    appended, no static/per-question context, matching prior behavior).
  - Verified end-to-end with a live smoke-test question ("Who is the director of Inception?")
    against the real KIT endpoint: message order came out as
    `[system, STATIC_QTYPE_CONTEXT, query, PER-QUESTION_CONTEXT, ...]` and the agent answered
    correctly ("Christopher Nolan").
- Full test suite: 505 passed (494 before + 11 new/updated tests covering the new ordering and
  follow-up-skip behavior), `ruff check` clean.
- Not attempted: items 3 and 4 (tool-schema trimming, classification-prompt reorder) — out of
  scope for this pass, left for a future change.
