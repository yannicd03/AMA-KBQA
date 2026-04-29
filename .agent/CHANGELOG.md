# Documentation Changelog

Newest first. One line per doc-affecting change.

---

| Date | Change |
|------|--------|
| 2026-04-20 | Added `Tasks/benchmark_persistence_refactor.md` — planned decoupling of Streamlit benchmark runs from session lifecycle (job_id + incremental disk writes, reference pattern in Orca) |
| 2026-04-20 | Created `SOP/hetzner_deployment.md` — full runbook for Hetzner VPS deployment (3-service compose stack, Qdrant port offset, frontend container, RDF bootstrap, SSH tunnel) |
| 2026-04-20 | Updated `System/project_architecture.md` — added frontend container to service table and diagram; changed Qdrant port reference 6333→6335; corrected platform from "Windows" to "Linux (Hetzner VPS)" |
| 2026-04-20 | Updated `README.md` — added Docker-based frontend startup as alternative to host install; noted Qdrant port 6335 for Hetzner |
| 2026-04-20 | Updated `.agent/README.md` — added `SOP/hetzner_deployment.md` and `CHANGELOG.md` entries; flagged missing files referenced from index |
| 2026-04-20 | Initialized `.agent/CHANGELOG.md` |
