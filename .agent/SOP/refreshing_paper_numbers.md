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

Table 1 and the abstract were updated; Overleaf commits `a3cf560` and `61ca779`.

**Comparability caveat:** `921a04d` predates the five verified tool fixes in `8d519ca` and the relation-path hardening in `195ce31` (see `Decisions/relation-path-step-schema-hardening.md`) — treat these published numbers as a **floor** for current code, not a ceiling. Re-verify before citing them as representative of the current codebase.
