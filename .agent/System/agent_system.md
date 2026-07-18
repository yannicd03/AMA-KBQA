# Agent System — Index

The AMA KBQA agent system uses a multi-agent architecture:

1. **Orchestrator Agent** — routes queries to appropriate sub-agents
2. **KQAPro Agent** — general-knowledge KBQA (Wikidata-derived) ✅ Active
3. **SciQA Agent** — scientific-research KBQA (ORKG) ✅ Active
4. Placeholder agents (code, math) — not implemented

All agents inherit shared lifecycle mechanics from `BaseKBQAAgent` and communicate
with their MCP servers via stdio.

```
┌─────────────────────────────────────────────────────────────┐
│                    Orchestrator Agent                        │
│   ama_kbqa/agents/orchestrator_agent/agent.py                │
│   - Routes queries based on LLM classification (1 round-trip)│
│   - Falls back to KQAProAgent on unrecognised routing         │
└─────────────────────────────────────────────────────────────┘
                              │
    ┌─────────────────────────┼─────────────────────────┐
    ▼                         ▼                         ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│  KQAPro Agent   │  │   SciQA Agent   │  │   Code Agent    │
│  (General KB)   │  │  (Scientific)   │  │  (Placeholder)  │
└─────────────────┘  └─────────────────┘  └─────────────────┘
        │                     │
        ▼                     ▼
┌─────────────────┐  ┌─────────────────┐
│ kqapro_server   │  │ sciqa_server    │
│ (29 tools)      │  │ (28 tools)      │
└─────────────────┘  └─────────────────┘
```

This file is a pure index. Detailed, current-state documentation for each part
of the agent system was split out on 2026-07-18 (the original monolithic file
had grown to 1275 lines, past the ~800-line split threshold) — read the linked
doc for the topic you need, not this file, for anything beyond orientation:

## Related Docs
- [Agent Framework](agent_framework.md) — `BaseKBQAAgent` shared mechanics: tool gating, synthesis funnel, trace instrumentation, text-tool-call mode, MCP client pattern, token/duration tracking, synthesis config, reset patterns, LLM sampling seed
- [Orchestrator Routing](orchestrator_routing.md) — evidence-based two-step routing flow, `analyze_query_recommend_db` contract, `route_reason` span
- [KQAProAgent](kqapro_agent.md) — lifecycle, 10-type classification, loop detection, journal data model, message history format
- [SciQAAgent](sciqa_agent.md) — lifecycle, 8-type classification, multi-label classifier tolerance, raw-SPARQL denylist gate, ORKG predicate reference
- [Project Architecture](project_architecture.md) — repo structure, full MCP tool reference, config system, CI, frontend
- [Research Corpus](research_corpus.md) — paper and dataset docs
