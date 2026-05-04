# AMA KBQA — Documentation Index

**Project:** Multi-agent Knowledge Base Question Answering system over heterogeneous RDF/SPARQL knowledge graphs.  
**Research venue:** SEMANTiCS 2026 Posters & Demos track (`paper/main.tex`).  
**Quick orientation:** See [System/project_architecture.md](System/project_architecture.md) first.

---

## System

| Document | Description |
|----------|-------------|
| [System/project_architecture.md](System/project_architecture.md) | Start here. Full structure, tech stack, MCP servers, CLI, frontend, batch pipeline, config system |
| [System/agent_system.md](System/agent_system.md) | Agent lifecycle, question classification, loop detection (6 layers), journal/scratchpad data model |
| [System/research_corpus.md](System/research_corpus.md) | Paper (`paper/`), dataset docs (`docs/`), and how research artifacts relate to code |

> `System/database_schema.md` — referenced in older READMEs but file is absent; Qdrant/Virtuoso schema is documented inline in `System/project_architecture.md` and `System/agent_system.md`.

---

## SOP

| Document | Description |
|----------|-------------|
| [SOP/running_batch_processing.md](SOP/running_batch_processing.md) | Run single-model or multi-model benchmarks; LLM judge; fewshot generation; CSV export; resume |
| [SOP/adding_new_kqapro_tools.md](SOP/adding_new_kqapro_tools.md) | Checklist for adding KQAPro MCP tools that produce answer values (journal write invariants) |
| [SOP/hetzner_deployment.md](SOP/hetzner_deployment.md) | Hetzner VPS deployment runbook: 3-service compose stack, Qdrant port, frontend container, RDF bootstrap, SSH tunnel |

> `SOP/database_setup.md` and `SOP/changing_llm_provider.md` — referenced in earlier docs but files are absent; Virtuoso/Qdrant setup is in the root `README.md`; LLM provider config is in `docs/guides/dataset_integration.md` and `config.toml`.

---

## Decisions

| Document | Description |
|----------|-------------|
| [Decisions/kqapro-tool-surface-expansion.md](Decisions/kqapro-tool-surface-expansion.md) | Why CountEntities, SelectExtreme, VerifyFact were added + FilterEntities extensions (closes 42% coverage gap) |
| [Decisions/text-mode-tool-calls.md](Decisions/text-mode-tool-calls.md) | Why and how a client-side `<tool_call>` parser was built for models that can't emit native function calls (minimax-m2.7) |
| [Decisions/llm-judge-concept-extraction-rubric.md](Decisions/llm-judge-concept-extraction-rubric.md) | LLM judge rubric rewrite: concept-extraction first, per-type rules, hard-negatives for tool-call fragments and max-iteration errors (+~2 pp, removes verbose-answer bias) |
| [Decisions/transitive-concept-default-false.md](Decisions/transitive-concept-default-false.md) | Why `transitive_concept` default stays False: n=100 benchmark showed -28 pp Count regression on flat concepts; retry-on-empty guidance keeps hierarchy cases recoverable |
| [Decisions/qualifier-and-format-prompt-hardening.md](Decisions/qualifier-and-format-prompt-hardening.md) | 5-prompt-fix bundle + iter-cap 25→40 targeting n=100 → 0.86–0.90: GetNodeSummary gate, QueryAttr attribute-first fallback, QueryAttrQualifier direction rule + QualifierFilter, QueryRelation format + passive-voice trap |

---

## Tasks

| Document | Status | Description |
|----------|--------|-------------|
| [Tasks/benchmark_persistence_refactor.md](Tasks/benchmark_persistence_refactor.md) | Planned | Decouple benchmark runs from Streamlit session; incremental disk writes + job_id reconnect (see Orca's `batch_queue.py` for the reference pattern) |
| [Tasks/archive/generic-framework-implementation.md](Tasks/archive/generic-framework-implementation.md) | Archived | Generic KBQA framework with BaseKBQAAgent, adapters, 97 tests |
| [Tasks/archive/sciqa-agent-implementation.md](Tasks/archive/sciqa-agent-implementation.md) | Archived | SciQA/ORKG agent with 468-pair ground truth benchmark |
| [Tasks/archive/scratchpad-enforced-agent-loop.md](Tasks/archive/scratchpad-enforced-agent-loop.md) | Archived | Scratchpad-first loop, tool response truncation, journal reflection |

---

## Project Root Documentation

| Document | Description |
|----------|-------------|
| [README.md](../README.md) | Setup instructions, environment variables, database bootstrap |
| [AGENT_ARCHITECTURE.md](../AGENT_ARCHITECTURE.md) | Detailed agent architecture reference (1000+ lines) |
| [TOOLS_REFERENCE.md](../TOOLS_REFERENCE.md) | Complete KQAPro MCP tool reference (25 tools) |
| [CLAUDE.md](../CLAUDE.md) | AI assistant instructions and codebase context |
| [docs/datasets/kqapro.md](../docs/datasets/kqapro.md) | KQAPro dataset structure (~94K Q&A, 1.6M RDF triples) |
| [docs/datasets/sciqa.md](../docs/datasets/sciqa.md) | SciQA/ORKG dataset structure (468 Q&A, 1.1M triples) |
| [docs/guides/dataset_integration.md](../docs/guides/dataset_integration.md) | How to integrate a new KG into the system |
| [paper/main.tex](../paper/main.tex) | SEMANTiCS 2026 submission — AMA-KBQA system paper |

---

## CHANGELOG

See [CHANGELOG.md](CHANGELOG.md).
