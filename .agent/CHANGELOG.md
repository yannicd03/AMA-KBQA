# Documentation Changelog

Newest first. One line per doc-affecting change.

---

| Date | Change |
|------|--------|
| 2026-05-03 | Created `Decisions/transitive-concept-default-false.md` — ADR: `transitive_concept` default reverted to False after n=100 benchmark regression (-28 pp Count for minimax); retry-on-empty guidance added to `TOOL_LOOP_GUIDANCE` (commits `1383a1d`, `5a5090f`) |
| 2026-05-03 | Created `Decisions/llm-judge-concept-extraction-rubric.md` — ADR: LLM judge rubric rewritten from one-liner to structured concept-extraction rubric; fixes ~3 pp verbose-answer bias; adds hard-negative rules for tool-call fragments, max-iteration errors, and KG-miss answers (commit `9822bda`) |
| 2026-04-29 | Appended to `Decisions/kqapro-tool-surface-expansion.md` — "Follow-up: Prompt Hardening to Drive Adoption" section: root cause of 0 CountEntities calls in n=20 benchmark; changes to `prompts.py` FORBIDDEN clauses and six fewshot JSON files (commits `08cadeb`, `d0978c9`) |
| 2026-04-29 | Created `Decisions/text-mode-tool-calls.md` — ADR for client-side `<tool_call>` parser + `tools=None` suppression for models that can't emit native function calls; covers minimax-m2.7 failure mode and smoke-test validation (commits `08cadeb`, `d0978c9`) |
| 2026-04-29 | Updated `System/agent_system.md` — added `text_tool_calls.py` to framework file table; added "Text-Tool-Call Mode" section under BaseKBQAAgent |
| 2026-04-29 | Updated `README.md` — added `Decisions/text-mode-tool-calls.md` to Decisions index |
| 2026-04-29 | Created `Decisions/kqapro-tool-surface-expansion.md` — ADR for CountEntities / SelectExtreme / VerifyFact addition + FilterEntities extensions (commit `9f677ac`); covers coverage gap rationale, two pre-existing bugs fixed, and validation results |
| 2026-04-29 | Updated `System/agent_system.md` — kqapro_server tool count 22 → 25 |
| 2026-04-29 | Updated `README.md` — added Decisions section with ADR pointer; TOOLS_REFERENCE count 22 → 25 |
| 2026-04-29 | Appended to `SOP/adding_new_kqapro_tools.md` — added "Aggregating Tools: Always Use Deterministic SPARQL" rule citing limit=50 footgun |
| 2026-04-20 | Added `Tasks/benchmark_persistence_refactor.md` — planned decoupling of Streamlit benchmark runs from session lifecycle (job_id + incremental disk writes, reference pattern in Orca) |
| 2026-04-20 | Created `SOP/hetzner_deployment.md` — full runbook for Hetzner VPS deployment (3-service compose stack, Qdrant port offset, frontend container, RDF bootstrap, SSH tunnel) |
| 2026-04-20 | Updated `System/project_architecture.md` — added frontend container to service table and diagram; changed Qdrant port reference 6333→6335; corrected platform from "Windows" to "Linux (Hetzner VPS)" |
| 2026-04-20 | Updated `README.md` — added Docker-based frontend startup as alternative to host install; noted Qdrant port 6335 for Hetzner |
| 2026-04-20 | Updated `.agent/README.md` — added `SOP/hetzner_deployment.md` and `CHANGELOG.md` entries; flagged missing files referenced from index |
| 2026-04-20 | Initialized `.agent/CHANGELOG.md` |
