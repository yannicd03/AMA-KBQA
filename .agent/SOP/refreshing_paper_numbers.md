# SOP: Refreshing Paper Numbers

Runbook for pulling fresh benchmark numbers into the SEMANTiCS 2026 paper (Table 1, abstract, any other results claim). Follow in order; each section is a checkpoint, not a suggestion.

## Related Docs
- [System/research_corpus.md](../System/research_corpus.md) — how research artifacts (datasets, paper) relate to code
- [SOP/running_batch_processing.md](running_batch_processing.md) — how to actually launch the benchmark runs this SOP consumes
- [Decisions/root-cause-tool-generalization-2026-05-14.md](../Decisions/root-cause-tool-generalization-2026-05-14.md) — fix ledger; check a run's commit against this before treating its numbers as current
- [Decisions/relation-path-step-schema-hardening.md](../Decisions/relation-path-step-schema-hardening.md) — one of the fixes that postdates the 921a04d numbers below

## 0. Where the paper lives

The paper is **not in this repo**. The local `paper/` directory was deleted 2026-06-06. Overleaf is the sole source of truth:

- Project ID `69dfa31055ece0bf3db25723` (name "AMAKBQA"), main file `main.tex`, bibliography `references_new.bib` (**not** `references.bib`).
- Access via the `overleaf` MCP server, which maintains a real git checkout at `/tmp/overleaf-<projectId>/` whose `origin` is the Overleaf git remote.
- `write_section` replaces one named `\section{...}` — the safe default for targeted edits. It **cannot** reach the abstract (sits before `\section{Introduction}`). For pre-section content (abstract, title block), edit the local checkout directly and `git commit` + `git push origin master` — this is the same path the MCP tool itself uses, not an out-of-band hotfix.
- **Always re-read `main.tex` fresh immediately before writing.** Co-authors edit directly in Overleaf between sessions; a stale read will silently clobber their edits.
- **After pushing, verify**: compare `git rev-parse HEAD` against `git rev-parse origin/master`. A push exit code is not evidence — reconcile before moving on if they diverge.

> Note: this repo's own `.agent/README.md` still lists `paper/main.tex` as a project-root doc — that row is now stale per the above; flag it next time you touch the README index.

## 1. Which fields feed which paper number

Table 1 (`\label{tab:eval}`) columns map to `summary.json` under `statistics`:

| Column | Source field |
|---|---|
| Accuracy | `statistics.accuracy` |
| Avg. time | `statistics.avg_time_seconds` |
| Avg. tokens | `statistics.avg_tokens` |
| #Q | `statistics.total_questions` |
| Caption cost claim | computed from `statistics.prompt_tokens` and `statistics.completion_tokens`, divided by `total_questions`, priced at the OpenRouter rates named in the caption ($0.12 / $0.35 per 1M input/output tokens, as of 2026-06) |

Do **not** use `statistics.estimated_cost_usd` for the cost claim — it reads `0.0` because the KIT endpoint is free, which is not the number the paper claims.

## 1a. Why the cost claim is a repricing, not a measured spend (verified 2026-07-26)

**Provider that actually served the runs.** The benchmark does not run on OpenRouter. `config.toml` sets `[llm] chat_provider = "kit"`, and `config.toml:35-37` points `[kit]` at `base_url = "https://ki-toolbox.scc.kit.edu/api/v1"` / `chat_model = "kit.gemma4-31b-it"`. Sweeping `resolved_models[].model_id` across every `benchmark_results/*/run_manifest.json`: the KIT endpoint switch happened mid-day on 2026-04-29 (`2026-04-29-1` is still `openrouter`; `2026-04-29-2` onward is `kit`), and every Gemma run from that switch through the 2026-07-25 n=500 run resolves to provider `kit`. Earlier runs (`2026-04-20-1` through `-7`, `2026-04-29-1`) predate the switch and used `openrouter` directly — several under the *same* model id (`google/gemma-4-31b-it`) the paper cites, so don't assume a manifest is safe just because the model id matches; check `resolved_models[].provider` too.

Consequences:

1. The KIT endpoint is free, which is why `statistics.estimated_cost_usd` reads `0.0` on every run past the switch. The paper's cost figure is a **hypothetical repricing** of the measured token volumes at commercial OpenRouter rates, not a measured spend. Caption wording must keep saying "At \<rates\>, a question costs under \$X" — never "we spent."
2. **OpenRouter IS used elsewhere in this pipeline** — the LLM judge (`[postprocessing] judge_provider = "openrouter"`, `deepseek/deepseek-v4-pro`, `config.toml:86-87`), synthesis (`[synthesis] synthesis_provider = "openrouter"`), and the demo's embeddings (`config.docker.toml:3`). Seeing "openrouter" anywhere in `config.toml` is not evidence the *benchmark* ran there. This exact confusion produced a wrong hypothesis on 2026-07-26 that a latency regression was caused by a hosting switch that never happened — check `resolved_models[].provider` on the actual run manifest before reaching for a provider-change explanation.
3. **Latency is not a stable point estimate.** The KIT endpoint is a shared community deployment whose throughput varies with load — the same model at the same commit has produced 30.8, 46.3, and 47.8 s/q on KQAPro across different runs. Hedge latency claims in the paper (a range, "roughly", or a specific run + date) rather than presenting a single figure as exact.

**Demo vs. benchmark are different configs, not different providers.** The demo also runs on the KIT endpoint (`config.docker.toml:1-2`, `chat_provider = "kit"`), but `config.docker.toml:27-29` points it at `kit.qwen3.5-397b-A17b` — a different, larger KIT model than the benchmark's `kit.gemma4-31b-it`. Any demo-vs-benchmark latency gap the paper's Demonstration section discusses is model size + Orchestrator routing overhead + the demo's smaller VM (see §1c), not a provider difference — both hit the same KIT endpoint.

## 1b. Decompose wall-clock before attributing it (added 2026-07-26)

Item 3 above says latency isn't a stable point estimate — this is the technique that makes a latency-regression claim checkable instead of just hedged. `summary.json` carries `total_tool_duration_seconds` alongside `avg_time_seconds` and `total_questions`. Subtracting gives the split between local tool execution and everything else (LLM round trips). **Always compute this split before attributing a timing change to any single cause** — provider, code change, or otherwise.

Worked example, both runs KQAPro/gemma on the same Hetzner host:

| | June (`rag-rerank-100q-2026-06-07/hybrid_rerank/kqapro/default`, commit `ba780af`) | July (`full-kqapro-n500-seed42-2026-07-25` gemma leg, commit `921a04d`) |
|---|---|---|
| avg_tokens | 71,666.9 | 70,331 |
| avg_tool_calls_per_question | 8.75 | 9.35 |
| tool time per question | 4.72 s | 11.35 s |
| LLM (non-tool) time per question | 25.85 s | 34.96 s |
| avg_time_seconds | 30.57 s | 46.31 s |

Token volume flat (down 2%), tool calls up only 7%, yet wall clock up 51%. The decomposition shows tool time rose 141% (0.54 s → 1.21 s per tool call) while LLM time rose 35%. Since tool execution is local Virtuoso/Qdrant querying that never contacts the inference endpoint, at least 42% of that regression cannot be the LLM provider. Three separate investigations had attributed the whole slowdown to the shared inference endpoint before this one-minute subtraction falsified it.

## 1c. The benchmark host is not dedicated (corrected 2026-07-26)

State plainly, because the paper asserted the opposite until it was corrected today: the Hetzner benchmark host is **4 cores / 7.7 GB RAM** and concurrently runs roughly a dozen Docker containers, including an unrelated MAS production stack (mas-api, mas-mongodb, mas-mongo-express, mas-qdrant, mas-frontend-react, five mas-in-production MCP servers) alongside an AMA-KBQA stack of its own (`qdrant_ama_kbqa`, `virtuoso_ama_kbqa`, `frontend_ama_kbqa`).

**The public demo is NOT on this host.** Both machines run a container set with identical names, which is exactly how that error was made and re-made on 2026-07-26. Resolve it from the tunnel config, not from `docker ps`:

| | host | spec | role |
|---|---|---|---|
| benchmark | `hetzner` | 4 cores / 7.7 GB | runs `benchmark_agents.py` out of `~/AMAKBQA-main`; also hosts an unrelated MAS production stack |
| public demo | `bwcloud` (hostname `seminar`) | 2 cores / 3 GB | serves `amakbqa.yanlab.de`, the target of the paper's `purl.archive.org/ama-kbqa-demo` |

Proof, since DNS is Cloudflare-proxied (`amakbqa.yanlab.de` → 104.21.32.5) and cannot identify the origin: `bwcloud`'s `/etc/cloudflared/config.yml` maps `hostname: amakbqa.yanlab.de` → `service: http://localhost:8502`. Hetzner's `cloudflared` runs a different tunnel (`scoresheet.yml`). Both bind their frontend to `127.0.0.1:8502`, so the port tells you nothing.

So the paper's "dedicated host" wording was wrong about the benchmark (it is shared and co-tenanted), but its benchmark-vs-demo distinction was right: those genuinely are two different machines, and the demo's is the smaller of the two.

Consequence for anyone refreshing paper numbers: tool-time figures are sensitive to co-tenancy on this box and are not reproducible across months as new services are deployed to it. Check `docker ps` start dates and `/proc/loadavg` when a timing regression appears, and record what else was running. Caveat: retroactive contention cannot be proven without historical metrics, so co-tenancy is a motivated hypothesis for such regressions, not a demonstrated cause — what the §1b decomposition proves is only which side of the split the regression sits on.

## 2. Pre-flight checks before trusting a run

Run these against every leg (`run_manifest.json` / `summary.json` / `results.json`) before citing a number:

1. **Commit identity.** `run_manifest.json` → `git_commit`. State which commit the numbers describe and check whether it predates fixes that would change them (cross-reference `Decisions/root-cause-tool-generalization-2026-05-14.md` and any recent ADR).
2. **Model identity.** `run_manifest.json` → `resolved_models[].model_id`. Never infer the model from the directory name — a run launched without `--models` is written to a directory literally called `default`, which has caused prior audits to walk straight past the canonical run.
3. **Retrieval config comparability.** `run_manifest.json` → `config_toml.retrieval`. Numbers are only comparable across runs when `hybrid_enabled`, `fusion`, and `reranker_enabled` match. The paper's config is hybrid + rrf + reranker (`gte-reranker-modernbert-base`).
4. **Fewshot pool stability.** `config_toml.postprocessing.generate_fewshot` MUST be `false`. If `true`, the fewshot pool mutates mid-run and later questions (and later models) see a different pool than earlier ones — this confounds any model-to-model comparison.
5. **Completeness.** `summary.json` → `is_complete` must be `true` on every leg.
6. **Error rows.** Count `error` rows in `results.json`. Timeouts are scored as incorrect and silently depress accuracy — a nonzero count needs to be called out alongside the number, not buried.

## 3. Dataset size trap (read this before sizing a "sample")

`--n-questions 500` yields only **100** questions for SciQA, because the entire SciQA handcrafted split IS 100 questions.

- Verified: the run log prints `"Loaded 100 questions from SciQA handcrafted dataset"`.
- Source: `db/datasets/SciQA/Handcrafted/full dataset.csv`, read by `load_raw_dataset()` in `ama_kbqa/benchmark_agents.py:542` (SciQA branch around `:562-598`).
- KQAPro by contrast loads 11,797 from `db/datasets/kqapro/val.json`.

**Consequence:** a SciQA result is the COMPLETE split, not a sample — the paper must not describe it as "sampled." This error persisted in the paper text until 2026-07-26; check current wording before assuming it's still fixed.

## 4. Page-budget verification

Camera-ready cap: 6 pages including references and the GenAI declaration (see `Decisions/` / wiki paper-budget note if one exists).

A local `latexmk -pdf` in the Overleaf checkout **fails to resolve the bibliography** (missing `elsarticle-num-names.bst` locally — present only on Overleaf's compile servers), so the absolute local page count is not trustworthy.

Workaround: compile the pre-change commit and the post-change commit locally and **compare the delta**, not the absolute count:
1. Clone the checkout to a scratch dir for the baseline compile (do not check out an old commit in place in the working checkout — that's how you lose track of HEAD vs. origin).
2. Compile both.
3. Compare page count delta and `Overfull \hbox` warning delta between the two. The delta is meaningful even though the absolute count isn't.

## 5. Worked example (2026-07-26)

Runs `full-kqapro-n500-seed42-2026-07-25` and `full-sciqa-n500-seed42-2026-07-25` on the Hetzner host under `~/AMAKBQA-main/benchmark_results/`, commit `921a04d`, seed 42.

| Dataset | Model | Accuracy | Avg time | Avg tokens | Errors |
|---|---|---|---|---|---|
| KQAPro | gemma-4-31b-kit | 427/500 = 85.4% | 46.31 s/q | 70,331 tok/q | 1 timeout |
| SciQA | gemma-4-31b-kit | 73/100 = 73.0% | 51.24 s/q | 114,461 tok/q | 1 timeout |

Also captured but not used in the paper: `minimax-m2.7-kit` at 86.2% KQAPro / 66.0% SciQA.

**Cost repricing (checked 2026-07-26):** current OpenRouter price for `google/gemma-4-31b-it` is $0.10 / $0.34 per 1M input/output tokens. The n=500 Gemma workload (KQAPro + SciQA combined) used 46,343,899 prompt + 267,830 completion tokens → $4.73 total, ~$4.99 after OpenRouter's ~5.5% credit fee — $0.0071/question on KQAPro, $0.0116/question on SciQA (input tokens are ~98% of spend). The paper caption still cites the older $0.12 / $0.35 rate (checked 2026-06) — that's more conservative than the current $0.10 / $0.34, so its "under $0.02" claim still holds; re-check both if the caption is ever updated to the current rate.

Table 1 and the abstract were updated; Overleaf commits `a3cf560` and `61ca779`.

**Comparability caveat:** `921a04d` predates the five verified tool fixes in `8d519ca` and the relation-path hardening in `195ce31` (see `Decisions/relation-path-step-schema-hardening.md`) — treat these published numbers as a **floor** for current code, not a ceiling. Re-verify before citing them as representative of the current codebase.
