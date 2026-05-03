# Research Corpus

AMA KBQA is an active research project. This document maps the research artifacts — the conference paper, dataset documentation, and evaluation results — to the code they describe.

## Related Docs
- [Project Architecture](./project_architecture.md) — overall system overview
- [Agent System](./agent_system.md) — agent lifecycle and loop detection

---

## Conference Paper

**File:** `paper/main.tex` (compiled PDF: `paper/main.pdf`)

**Title:** *AMA-KBQA: An Adaptable Multi-Agent Framework for Generalizable Knowledge Base Question Answering*

**Venue:** SEMANTiCS 2026 — Posters & Demos track (Ghent, 15–17 Sept 2026)  
**Format:** CEUR-ART single-column, 5 pages including references

**Paper structure:**
- Introduction — motivation for multi-KG generalization
- Background — RDF/SPARQL context, KBQA landscape
- System Overview:
  - Layered architecture → `ama_kbqa/framework/` (BaseKBQAAgent, adapters)
  - Configuration-driven KG integration → `framework/adapters/` + `framework/config.py`
  - Agent lifecycle → `base_agent.py` (pre-hook → tool loop → post-hook)
- Evaluation:
  - Setup — KQAPro and SciQA benchmarks, LLM judge
  - Results — accuracy metrics across models
  - Comparison with Interactive-KBQA
  - Cross-domain generalisation
- Demonstration — Streamlit frontend + CLI
- Limitations and Conclusion

**Supporting files in `paper/`:**
- `main.bib` / `sample-ceur.bib` — bibliography
- `lncs_full.tex`, `sample-1col.tex` — template reference files
- `CEURART-README.md` — submission instructions
- `main.abs`, `main.xmpdata` — PDF/A metadata

---

## Dataset Documentation

Located in `docs/`:

### `docs/datasets/kqapro.md`

KQAPro dataset — Wikidata-derived factoid QA benchmark.

| Property | Value |
|----------|-------|
| Entities | ~16,960 entities + 794 concepts |
| RDF triples | ~1.6M (after JSON→N-Triples conversion) |
| Q&A pairs | ~94K across train/val/test |
| File on disk | `db/datasets/kqapro/kb.json` (76 MB), `kb.nt` (1.6M lines) |
| Benchmark file | `db/datasets/kqapro/val.json` (11 MB) |
| Question types | 10 types (Count, Verify, Select, SelectBetween, SelectAmong, QueryAttr, QueryAttrQualifier, QueryRelation, QueryRelationQualifier, QueryName) |

**Code connection:** `ama_kbqa/agents/kqapro_agent/`, `ama_kbqa/server/kqapro_server.py` (22 tools), Virtuoso graph `http://kqapro.org/kg`

### `docs/datasets/sciqa.md`

SciQA/ORKG dataset — scientific research paper QA benchmark.

| Property | Value |
|----------|-------|
| RDF triples | ~1.1M (ORKG dump 14.02.2023) |
| Q&A pairs | 468 (100 handcrafted + 368 auto-generated) |
| File on disk | `db/datasets/SciQA/ORKG RDF dump 14.02.2023.nt` (~160 MB) |
| Question types | 8 types (Factoid, Count, List, Boolean, Comparison, Superlative, Aggregation, General) |

**Code connection:** `ama_kbqa/agents/sciqa_agent/`, `ama_kbqa/server/sciqa_server.py` (18 tools), Virtuoso graph `http://sciqa.org/kg`

### `docs/guides/dataset_integration.md`

Step-by-step guide for adding a **new knowledge graph** to the system:
1. Prepare data (convert to N-Triples)
2. Load into Virtuoso
3. Create Qdrant embeddings
4. Create MCP server
5. Verify setup

This is the canonical reference if a third KG (beyond KQAPro and SciQA) is being integrated.

---

## Benchmark Results

Live evaluation output lives in `benchmark_results/`:

```
benchmark_results/
└── <YYYY-MM-DD-N>/          # Date + run number (e.g. 2026-02-08_kqapro-minimax-m2.1-n20-90pct)
    ├── overview.json         # Multi-model leaderboard
    ├── benchmark_results.csv # CSV export (if --export-csv)
    └── <agent>/<model>/
        ├── results.json      # Per-question results
        ├── summary.json      # Aggregate accuracy/tokens/cost
        ├── judgments.json    # LLM judge evaluations
        └── tool_traces/      # Full conversation traces (question_NNN.json)
```

Models evaluated in `benchmark_results/` (based on directory names):
- `minimax-m2.1`, `qwen3-32b`, `gpt-oss-120b`, `gpt-4.1-mini` via OpenRouter and KIT
- Accuracy varied from ~40% to ~90% across KQAPro runs

**Old-style results** (pre-naming-change) live in `batch_results/` with pattern `kqapro-n10-<pct>pct-batch<N>/`.

---

## Research Context

The system was built to demonstrate that a **single agent architecture** can generalize across structurally different RDF knowledge graphs without per-KG re-training. The paper's contribution is the adapter pattern (`framework/adapters/`) and the scratchpad-enforced agent loop (described in `System/agent_system.md`), which together allow the same base agent to answer questions over both Wikidata-style factoid KGs and ORKG-style scientific KGs.

The LLM judge (DeepSeek v3.2 via OpenRouter) is used for evaluation rather than exact-string matching, as many KBQA answers admit equivalent phrasings.
