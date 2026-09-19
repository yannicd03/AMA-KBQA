# ADR: Live graph panel built from two sources merged by node id

## Related Docs
- [System/demo_bwcloud_frontend.md](../System/demo_bwcloud_frontend.md): "Live Graph Panel" section, data flow, config, file map
- [Tasks/active/live-graph-panel.md](../Tasks/active/live-graph-panel.md): PRD and implementation outcome
- [Decisions/trace-inspector-frontend-architecture.md](trace-inspector-frontend-architecture.md): earlier trace/snapshot architecture decisions this feature builds on (journal snapshots on mutating tools only, vis-network via CDN)
- [Decisions/live-trace-and-chat-unification.md](live-trace-and-chat-unification.md): the worker-thread + queue + fragment-polling pipeline this feature's `"__graph__"` delta rides on

## Context

The demo wanted a side panel showing, live, the subgraph an agent has
gathered while it answers: entities found or visited, relations seen between
them, literal values retrieved, growing during the run and highlighted
against the final answer once it completes.

`ama_kbqa/frontend/utils/graph_html.py` already had a `journal_to_graph`
function rendering a vis-network graph from a `JournalState` snapshot, built
for a post-hoc scrubber and unused in the demo. It could not simply be
switched on:

1. **Edges were mostly lost.** The old `journal_to_graph` only read
   `verified_facts` entries shaped `{subject, predicate, object}`. The
   servers write at least seven other shapes: `relation`/`related_id`,
   `attribute`/`value`, `predicate`/`objects`, `type`-tagged qualifier
   entries, free-text `fact` entries, and RunSPARQL summaries
   (`kqapro_server.py:1804-1873`).
2. **The main exploration tool never reached the journal as edges.**
   `GetNodeSummary` (`kqapro_server.py:2438-2447`) journals attributes into
   `found_values` and drops its `relations` dict entirely. `FindNode`
   (`kqapro_server.py:2147`) journals nothing at all. A journal-only graph
   would show the visited entity and its literals but no neighbours and no
   search candidates. For the KQAPro fast path specifically, only one
   journal snapshot is ever taken (right after `FindNode`), so the gap is
   total, not partial.
3. **It was not live.** Journal snapshots were only harvested after the run
   completed; nothing fed the panel while the agent was still working.

## Options considered

**A. Journal-only, close the gaps by widening what `journal_to_graph` reads.**
Rejected: even reading every `verified_facts` shape, `GetNodeSummary`'s
relations and `FindNode`'s candidates are never written to the journal at
all, so there is nothing there to widen. This option cannot reach the KQAPro
fast-path case (one snapshot, no neighbours) no matter how the journal
parser is improved.

**B. Modify the MCP servers to journal more (e.g. have `GetNodeSummary` write
its relations, have `FindNode` write its candidates).** Rejected. The
journal is not a display-only data structure: `GetJournalSummary` feeds it
back to the LLM as working memory. Changing what gets journaled changes what
the agent sees mid-run, which changes agent behaviour and would require
re-running the benchmark to confirm no regression. The PRD's non-goals are
explicit about this: "No server-side journal changes (would alter what the
LLM sees in `GetJournalSummary`)." Out of scope for what is meant to be a
frontend-only, zero-extra-cost feature.

**C. Parse tool-call results in the frontend for a fixed allow-list of
tools, merged with the journal by node id.** Chosen. The frontend already
has full tool-call results on the trace/span payload
(`base_agent.py:1913-1940`), so nothing new needs to reach the browser. An
allow-list (`live_graph_data.GRAPH_TOOLS`) keeps the cost bounded: only
search and exploration tools are parsed (`FindNode`, `GetNodeSummary`,
`GetRelationDetails`, their SciQA counterparts, label lookups, raw SPARQL),
so a call to `VerifyFact` or `GetJournalSummary` costs nothing. The journal
still supplies what tool results alone cannot: which entities were actually
*visited* (vs. merely returned as a search candidate) and the literal values
already extracted into `found_values`. Both live and frozen views run
through the same normaliser (`graph_from_journal_state` /
`graph_from_tool_result`), so they can never disagree with each other.

## Decision

Build the panel's subgraph from two sources merged by node id via
`GraphData.merge`:

- **Journal state**: entities visited, literal values found, and edges from
  the `verified_facts` shapes the normaliser now understands.
- **Tool-call results**, parsed on the worker thread for the `GRAPH_TOOLS`
  allow-list: search candidates and the neighbour relations the journal
  drops.

On merge, an entity kind beats a candidate kind, and a real label beats an
id-as-label (`_merge_nodes`'s field-wise precedence), so a neighbour first
seen as a bare id via `GetNodeSummary` upgrades to its real name once (and
if) the agent later resolves it via `GetNodeLabel`.

No new Python dependency: the browser-side rendering stays vis-network via
CDN, unchanged from the existing trace-inspector graph view
(`Decisions/trace-inspector-frontend-architecture.md` decision 5).

## Consequences

- **Per-tool extractors must track server response shapes.** Each entry in
  `GRAPH_TOOLS` has a dedicated `_extract_*` function written against the
  actual Pydantic response models in `kqapro_server.py`/`sciqa_server.py`,
  not against an assumed generic shape. `GetRelationDetails`' triples, for
  example, are `{related_id, related_uri, direction}` with subject/predicate implied
  by the call arguments, not `{subject, predicate, object}` triples. A
  server-side response shape change silently breaks the corresponding
  extractor unless a test catches it.
- **Tests build fixtures from the server Pydantic models.** Per the PRD's
  test plan, extractor tests instantiate the actual response models and
  `model_dump_json()` them rather than hand-writing JSON fixtures, so a
  schema drift between the server and the frontend's assumption breaks the
  test instead of silently producing an empty or wrong graph.
- **No new Python deps; no extra tool calls or tokens.** The feature is
  strictly a frontend read of data the agent already produced. This is also
  why label lookups are opportunistic rather than eager: the frontend never
  issues its own KG queries to pre-resolve a neighbour's label (see the
  System doc's "Known limitation": an unresolved neighbour stays an id on
  the canvas). Doing otherwise would violate the "costs no extra tool calls
  or tokens" property that makes this safe to ship without a benchmark
  re-run.
- **Parse-on-worker-thread / stamp-source-on-main-thread split.** Tool
  results can be large (a raw SPARQL result, a hub entity's full relation
  list), so `json.loads` and the extraction pass happen inside
  `lifecycle_runner.make_listener`'s `listener` callback, which already runs
  on the background worker thread as part of the existing trace pipeline
  (`Decisions/live-trace-and-chat-unification.md`). The resulting `GraphData`
  delta is built with an empty `source` field at that point, because owner
  resolution (which specialist made this call, in a router/federated run)
  depends on walking `parent_span_id` through `delegate` spans, state that
  is easiest and safest to read on the main/Streamlit thread, where
  `_owner_for` already runs for every other span kind. `drain_into` /
  `_absorb_graph_delta` therefore stamps the source after the delta crosses
  the queue, immediately before merging it into `LiveGraphState.by_owner`.
  This keeps the worker thread doing the expensive parsing and the main
  thread doing only the cheap, already-established owner walk; no new
  cross-thread contract is needed beyond the queue that already carries
  every other trace notification.
- **Version-gated redraw.** `_absorb_graph_delta` only bumps
  `LiveGraphState.version` when a merge actually adds a node or edge, so an
  agent re-checking a node it already visited does not force the panel to
  redraw every second regardless of the 1 Hz fragment cadence.
