[demo-video.webm](https://github.com/user-attachments/assets/54464342-4ba0-4b2f-96e8-46569adae5ac)

# AMA KBQA

A multi-agent **Knowledge Base Question Answering** system that answers natural
language questions over **multiple knowledge graphs**, demonstrating
generalization across different RDF/SPARQL databases.

Most KBQA systems are built and tuned for a single knowledge graph. AMA KBQA
instead pairs a shared agent framework with per-graph tool servers, so the same
reasoning machinery answers questions over very different graphs (a broad
Wikidata-style factoid KG and a domain-specific scientific KG) without
graph-specific glue in the agent logic.

## How It Works

- **Orchestrator + specialist agents.** An orchestrator classifies each question
  and routes it to the specialist agent best suited to the relevant knowledge
  graph (KQAPro or SciQA), or you can call a specialist directly.
- **ReAct tool-use loop.** Each agent runs an LLM reasoning loop that calls
  knowledge-graph tools (entity lookup, neighborhood exploration, attribute and
  relation retrieval, SPARQL, fact verification) until it has grounded evidence
  for an answer.
- **Tools over MCP.** Knowledge-graph capabilities are exposed as
  [MCP (Model Context Protocol)](https://modelcontextprotocol.io) servers, one
  per graph, which keeps the agent framework graph-agnostic and makes adding a
  new graph a matter of standing up a new tool server.
- **Hybrid retrieval.** Tools combine vector semantic search
  ([Qdrant](https://qdrant.tech)) for fuzzy entity/relation lookup with exact
  SPARQL queries ([Virtuoso](https://virtuoso.openlinksw.com)) for precise
  graph traversal.
- **LLM-backed reasoning.** Models are served via OpenRouter or the KIT
  endpoint and are configurable per role (chat, synthesis, judge).

## Features

- **Streamlit web UI** with a live, step-by-step visualization of the agent's
  reasoning loop, plus chat, batch processing, evaluation, and settings pages.
- **Command-line interface** for one-off questions and reproducible benchmarks.
- **Benchmarking harness** for single- and multi-model evaluation with
  configurable postprocessing and CSV export.

## Supported Knowledge Graphs

- **KQAPro** - Wikidata-derived factoid Q&A (~47K entities).
- **SciQA/ORKG** - Scientific research papers and contributions (~1.1M triples,
  468 Q&A pairs).

## Getting Started

Installation, dataset setup, and usage are covered in the
**[Getting Started guide](docs/guides/getting-started.md)**.

Once set up, a quick taste:

```bash
# Web UI
ama-kbqa-frontend

# Ask a question from the CLI (orchestrator auto-routes)
ama-kbqa ask "Who directed Inception?"
```

## Usage Observations (Informal)

AMA-KBQA has not been evaluated in a formal user study. The notes below are
informal observations from real, unscripted usage of the public demo, primarily
by the SEMANTiCS 2026 reviewers during the review period and by the authors
during development. They are anecdotes rather than measurements, and are shared
to set honest expectations; the quantitative evaluation is the KQAPro/SciQA
benchmark reported in the paper.

**What holds up in practice**

- Multi-hop counting and comparison questions resolve correctly end-to-end,
  e.g. "How many heavy metal groups are in the genre of Queen?" (KQAPro)
  returns the correct count of 41 with a complete reasoning trace.
- The "How I found this" trace is the feature unassisted users lean on: every
  claim in an answer links back to the exact tool calls and triples that
  produced it, so answers can be checked instead of trusted. Because synthesis
  draws only on the scratchpad's verified facts, a wrong answer is visible as a
  wrong trace, not a silent hallucination.

**Known failure modes** (also documented in the demo's About dialog under
"What can go wrong")

- *Entity mislinks.* Vector-based entity linking occasionally resolves a name
  to a similar but wrong entity; one reviewer's question about the designer
  Franco Moschino was answered about Travis Banton.
- *Definitional questions.* Questions whose answer is not explicit in the graph
  fail; two reviewers independently asked "What is Artificial Intelligence?"
  against SciQA and got no answer, because ORKG stores research contributions,
  not textbook definitions.
- *Honest refusals.* When the graph lacks the fact, the agent answers "I do not
  know" rather than guessing. This is by design, but it surprises users who
  expect chatbot-style improvisation.
- *Slow superlatives.* Aggregation and superlative questions ("the most",
  "how many ... in total") trigger many tool-calling turns and are the slowest
  question class.
- *Latency.* The public demo is noticeably slower than the paper's benchmark
  averages: it defaults to Orchestrator mode (adding a routing probe and
  per-question agent start-up), runs a larger backbone model, and sits on a
  small shared VM behind a shared inference endpoint. One to two minutes per
  question under load is normal.
- *Follow-ups.* Stateful multi-turn follow-ups are supported only in specialist
  mode; the default Orchestrator treats each question independently.

**Changes made in response to observed usage**

- The failure modes above were added to the demo's About dialog so first-time
  users see them before asking.
- A configuration crash that reviewers hit on the live demo (missing API key
  after a provider switch) was fixed and redeployed, and the entity-linking and
  definitional-question failures are tracked as open issues.

## Project Structure

| Path | Purpose |
|------|---------|
| `config.toml` | Central LLM/database configuration |
| `.env` | API keys (not committed) |
| `ama_kbqa/agents/` | KQAPro and SciQA agents |
| `ama_kbqa/framework/` | Generic, graph-agnostic KBQA agent framework |
| `ama_kbqa/server/` | MCP servers exposing knowledge-graph tools |
| `ama_kbqa/frontend/` | Streamlit multi-page UI |
| `db/` | Dataset preparation and vector-population scripts |
| `docs/` | Setup guide and dataset documentation |
| `.agent/` | Developer documentation (architecture, decisions, SOPs) |
