# PRD: Live knowledge-graph panel for the demo frontend

**Status:** ✅ Implemented and merged into `demo-v2-int` @ `040b71d` (2026-09-14); staging on Hetzner per SOP §9 pending
**Owner:** Yannic · **Written:** 2026-09-14
**Scope:** demo frontend only (`ama_kbqa/frontend/`), plus one config flag and one read-only accessor on the orchestrator. No MCP-server, prompt, or agent-loop behaviour changes.

## 1. Goal

While an agent answers a question, a side panel next to the chat shows the
subgraph the agent has gathered so far: the entities it found or visited, the
relations it has seen between them, and the literal values it retrieved. The
graph grows live during the run, is highlighted against the final answer once
the run completes, and resets for the next question. The feature is gated by a
config flag so the benchmark/dev line and any deployment can turn it off.

Decisions taken with the user (2026-09-14):

| Question | Decision |
|---|---|
| Placement | Persistent side panel (right column), not per-message and not a bottom expander |
| Scope | One question at a time; the panel resets on each new question |
| Content | Entities from lookup/search tools; relations between entities; literal attribute values as leaf nodes; highlight nodes used in the final answer |
| Rendering | Interactive (zoom, pan, drag, hover tooltips) via the existing vis-network iframe; **no new Python dependency** |
| Config | `[frontend] live_graph` in `config.toml`, with an env override; plus a sidebar toggle while it is enabled |

## 2. Why the existing code is not enough

`utils/graph_html.py` + `utils/graph_panel.py` already render a vis-network
graph from a `JournalState` snapshot, but they are unused in the demo and were
written for a post-hoc scrubber. Three gaps, all verified in code:

1. **Edges are mostly lost.** `graph_html.journal_to_graph` only accepts
   `verified_facts` entries shaped `{subject, predicate, object}`. The servers
   write at least seven other shapes (`kqapro_server.py:1804-1873` documents
   them: `relation/related_id`, `attribute/value`, `predicate/objects`,
   `type`-tagged qualifier entries, `fact` prose, RunSPARQL summaries).
2. **The main KQAPro exploration tool never reaches the journal as edges.**
   `GetNodeSummary` (`kqapro_server.py:2438-2447`) writes attributes into
   `found_values` and drops its `relations` dict entirely. `FindNode`
   (`kqapro_server.py:2147`) journals nothing at all. A journal-only graph
   therefore shows the visited entity and its literals but no neighbours and
   no search candidates.
3. **It is not live.** Journal snapshots are harvested only after the run
   (`chat.py:254-274`); the template remounts and re-lays-out on every render.

Therefore the graph is built from **two sources merged by node id**:

- **A. Journal state** (`agent.journal_snapshots[-1]["state"]`, a
  `JournalState` dump): `visited_nodes` → entity nodes with labels;
  `found_values` → literal leaf nodes and, where a value looks like an id,
  entity edges; `verified_facts` → edges after shape normalisation.
- **B. Tool-call results** for a fixed allow-list of tools, parsed from the
  `tool_call` span's `payload["result"]` (a JSON string; see
  `base_agent.py:1913-1940`). This supplies search candidates and the
  neighbour relations that the journal drops.

Both live and frozen views use the same normaliser so they never disagree.

## 3. UX specification

### 3.1 Layout

- `app.py` sets `layout="wide"` when `get_live_graph_enabled()` is true,
  otherwise stays `centered` (unchanged behaviour when the flag is off).
- `chat.py` wraps the page body in `chat_col, graph_col = st.columns([0.58, 0.42], gap="medium")`
  only when the panel is active (flag on **and** sidebar toggle on **and** not
  Simplified view). All existing chat content (history, composer, thinking
  bubble, inspector tabs) renders inside `chat_col`; nothing else moves.
  When the panel is inactive the page renders exactly as today (no columns).
- The sidebar gets a `st.toggle("Live graph", value=True, key="live_graph_view")`
  directly under "Simplified view", rendered only when the config flag is on.
  Help text: "Show the subgraph the agent gathers while it answers."
- The initial landing view (no messages, no live run) does not show the panel.

### 3.2 Panel content

Top to bottom inside `graph_col`:

1. Header line: `st.markdown("**Explored subgraph**")` plus a caption that
   states the phase: `building…` (live), `final · N nodes · M edges`
   (frozen), or `no graph data for this run`.
2. Compact stats row (`st.columns(3)` metrics): Entities, Literals, Edges.
3. The vis-network iframe (`components.html`), height 560 px, full column width.
4. A legend is drawn inside the iframe (HTML), not in Streamlit.
5. Frozen view only: a `st.expander("Nodes in the answer")` listing the
   highlighted node labels (max 20), and an `st.expander("Raw journal")`
   with `st.json` of the last journal state per source (kept small; this is
   the debugging hatch).

### 3.3 Live behaviour

- A dedicated `@st.fragment(run_every=1.0)` inside `graph_col` re-reads the
  live graph state every second and rewrites the iframe **only when the graph
  changed** (a version counter; see §5). The iframe is written into a
  `st.empty()` placeholder created in `graph_col` outside the fragment so an
  unchanged tick does not remount it.
- The panel must not drain the run queue; `_live_tick` in `chat.py` remains
  the only consumer of `live_run["queue"]`.
- Node positions persist across remounts (localStorage, keyed by trace id) so
  the graph grows in place instead of re-scattering. Newly added nodes get an
  "new" ring for one render.
- Nodes are coloured by source knowledge graph: KQAPro blue, SciQA green
  (federated runs show both). Literal nodes are small grey boxes. Search
  candidates that were never visited are drawn hollow (dashed border).
  Answer nodes (frozen only) get an orange fill and a thicker border.
- The palette must fit the demo's light theme (`.streamlit/config.toml`:
  background `#fafafa`, text `#3f3f46`, primary `#60a5fa`, Inter font). The
  current dark template is replaced, not kept as an option.

### 3.4 Frozen behaviour

- After `_persist_completed_run`, the full rerun renders the graph of the trace
  selected in the inspector (`st.session_state.get("chat_panel_trace_selector")`),
  falling back to `latest_trace_id`. Built with `graph_from_trace(trace)`
  (§5), so it works for traces recorded before the panel was enabled in the
  session as long as they carry events/journal snapshots.
- Answer highlighting: `answer_node_ids(graph, answer)` marks a node when its
  id appears verbatim in the answer, or its label (≥ 3 chars, case-insensitive,
  word-boundary match) appears, or for literals when the literal's value string
  appears. Cap: 50 highlighted nodes.
- Error runs (no answer) render whatever was gathered, with no highlight.
- Restart button and a new question clear the panel (they already clear
  `live_run`/`latest_trace_id` flow; the panel simply follows those).

### 3.5 Limits

- Node cap 300 per graph, edge cap 600. Entities are kept before literals,
  literals before search-only candidates; the caption says `showing 300 of N`.
- Labels truncated to 40 chars in the graph; full text in the tooltip.
- RunSPARQL / RunORKGSPARQL: at most 25 entity nodes per call, no edges.

## 4. Config

`config.toml` and `config.docker.toml` gain a new section (first `[frontend]`
section in the repo; keep the comment):

```toml
[frontend]
# Live "explored subgraph" side panel in the demo chat page. Frontend-only:
# reads journal snapshots and tool results the agent already produces, so it
# costs no extra tool calls or tokens. Rendering is client-side (vis-network
# from CDN). Off = the page is exactly the pre-feature single-column layout.
live_graph = true
```

`ama_kbqa/config.py`:

```python
FRONTEND_LIVE_GRAPH_ENV = "AMA_FRONTEND_LIVE_GRAPH"

def get_frontend_config() -> dict: ...          # section dict, {} if absent

def get_live_graph_enabled() -> bool:
    """[frontend].live_graph, overridable by AMA_FRONTEND_LIVE_GRAPH=0/1
    (same truthy parsing as the AMA_RETRIEVAL_* overrides). Default False."""
```

Default in the getter is **False** (feature-off when the section is missing,
e.g. old configs); both shipped toml files set it `true` on this demo line.
`docker-compose.hetzner-next.yml` gets no change (the mounted
`config.docker.toml` carries the flag); document the env override in the
compose comment only.

## 5. Data model and module APIs

### 5.1 `ama_kbqa/frontend/utils/live_graph_data.py` (new, Streamlit-free)

```python
@dataclass(frozen=True)
class GraphNode:
    id: str                      # entity id ("Q937", "R12345") or literal key
    label: str
    kind: Literal["entity", "literal", "candidate"]
    source: str                  # "kqapro" | "sciqa" | "" (unknown)
    title: str                   # HTML tooltip (escaped)
    order: int                   # insertion order, for stable "new" detection

@dataclass(frozen=True)
class GraphEdge:
    id: str                      # deterministic: f"{src}|{pred}|{dst}"
    src: str
    dst: str
    label: str                   # predicate, truncated
    title: str                   # full predicate + provenance (tool/source)

@dataclass
class GraphData:
    nodes: dict[str, GraphNode]  # insertion-ordered
    edges: dict[str, GraphEdge]
    truncated_nodes: int = 0

    def merge(self, other: "GraphData") -> "GraphData"   # union by id; entity kind wins over candidate; longer label wins over id-as-label
    def stats(self) -> dict                              # {"entities", "literals", "candidates", "edges"}
    def is_empty(self) -> bool
```

Builders (all pure, all tolerant of malformed input: never raise on a bad
fact, skip it):

```python
def graph_from_journal_state(state: dict, *, source: str) -> GraphData
def graph_from_tool_result(tool_name: str, arguments: dict, result_text: str, *, source: str) -> GraphData
def graph_from_trace(trace: dict) -> GraphData       # frozen view: events + journal_snapshots, owner attribution via parent chain (same walk as lifecycle_runner.reconstruct_orchestrator)
def answer_node_ids(graph: GraphData, answer: str | None) -> set[str]
def to_vis_payload(graph: GraphData, *, highlight: set[str], new_ids: set[str]) -> dict   # {"nodes": [...], "edges": [...]} with vis fields: id, label, title, group ("kqapro"|"sciqa"|"literal"|"candidate"), highlighted: bool, new: bool
def apply_caps(graph: GraphData, *, max_nodes=300, max_edges=600) -> GraphData
```

Source attribution: `"kqapro"` for `kqapro_agent`/`KQAProAgent`, `"sciqa"`
for `sciqa_agent`/`SciQAAgent`; `JournalState.kg_name` is a fallback; a
helper `source_for_agent(agent_or_name) -> str` centralises this.

**Journal normalisation rules** (`graph_from_journal_state`):

- `visited_nodes: {id: label}` → entity node; a label of `"(resolving label)"`
  or equal to the id counts as unlabelled (label = id, tooltip says
  "label pending").
- `found_values: {entity_id: {attr: value}}`: for each value item (scalar, or
  list of scalars/dicts): if the item (or its `related_id`/`entity_id`/`id`
  field) looks like an entity id → entity node + edge `entity -attr-> id`;
  otherwise → literal node with id `lit:{entity_id}:{attr}:{n}` and label
  `value [unit]`, plus edge `entity -attr-> literal`. `_type` keys (SciQA) are
  rendered as a tooltip line, not nodes. Dict items use `value`/`label`
  fields (see `kqapro_server.py:1664-1678`, `sciqa_server.py:1810-1860`).
  Cap 8 literal children per (entity, attr).
- `verified_facts` shapes (each maps to zero or more edges; unknown shapes are
  ignored, never rendered as "?"):
  - `{subject, predicate, object}` → edge; object literal → literal node.
  - `{subject, relation, related_id, direction?}` → edge, reversed when
    `direction in ("reverse", "inverse_only")`.
  - `{subject, attribute, value, unit?}` → literal node + edge.
  - `{subject, predicate, objects: [...]}` → one edge per object (ids or
    `{id,label}` dicts), cap 5.
  - `{type: "relation_qualifiers", node, relation, target}` → edge node→target.
  - `{type: "edge_qualifiers"|"qualifier_value", ...}` → tooltip line on the
    subject node only (qualifier names), no new nodes.
  - `{fact: str}` / `{raw: str}` / RunSPARQL summaries → ignored.
- `id`-shape heuristic: extend `_looks_like_id` from `graph_html.py`
  (`Q\d+`, `P\d+`, `L\d+`, `R\d+`, `C\d+`, http(s) URIs, `orkgr:`/`orkgp:`
  prefixes); URIs are shortened to their last path segment for the id and
  keep the full URI in the tooltip.

**Tool-result extraction allow-list** (`graph_from_tool_result`; anything
else returns an empty graph; result_text is parsed with `json.loads`, and a
non-JSON result returns empty):

| Tool (server) | Nodes | Edges |
|---|---|---|
| `FindNode`, `LookupEntityByName`, `FindByAttribute` (KQAPro) | `matches[].original_id` / `.name` as **candidate** nodes (cap 10) | none |
| `FindResource`, `LookupResourceByLabel` (SciQA) | same, from `matches[]` | none |
| `GetNodeSummary` (KQAPro) | `node_id`/`name` entity; `relations: {pred: [ids]}` targets as entity nodes (label = id until the journal supplies one) | `node -pred-> target` per relation (cap 10 per pred); `attributes: {attr: [literal]}` → literal nodes + edges (cap 5 per attr) |
| `GetRelationDetails` (KQAPro) | endpoints of `triples[]` | one edge per triple; read `subject`/`object`/`predicate`-like keys defensively (`related_id`, `target`, `entity_id` accepted) |
| `GetRelationTargets` (SciQA) | `resource_id`; `targets[].id` (+`label`) | `resource -predicate-> target`; `targets[].value` → literal |
| `GetResourceDetails` (SciQA) | `resource_id`/`label`; relation targets with `id` | edges per relation; literal values → literal nodes |
| `RunSPARQL`, `RunORKGSPARQL` | URI bindings → entity nodes (label from a sibling `<var>Label` binding if present), cap 25 | none |

Each extractor must be written against the actual response models in
`ama_kbqa/server/kqapro_server.py` (`:117-290`) and
`ama_kbqa/server/sciqa_server.py` (`:159-230`); read them before coding, and
build the unit-test fixtures from those models (instantiate the Pydantic
models and `model_dump_json()` them) so a schema drift breaks the test.

### 5.2 `ama_kbqa/frontend/utils/lifecycle_runner.py` (edit)

- `LiveLifecycleState` gains `graph: LiveGraphState` where

  ```python
  @dataclass
  class LiveGraphState:
      by_owner: dict[Optional[str], GraphData]   # None = orchestrator level (unused for nodes), "kqapro_agent", "sciqa_agent"
      version: int = 0                           # bumped whenever any node/edge is added
      def merged(self) -> GraphData
  ```

- The worker `listener` additionally, for `phase == "close"` and
  `info["kind"] == "tool_call"` whose `name` is in the allow-list, parses the
  payload **on the worker thread** and enqueues
  `("__graph__", {"span_id", "parent_span_id", "tool": name, "graph": GraphData})`
  right after the regular close notification. Parsing errors are swallowed
  (logged at debug). The regular notification tuple is unchanged (payload is
  still stripped from it).
- `drain_into` handles `"__graph__"`: resolves the owner with the existing
  `_owner_for(state, "tool_call", "close", span_id, parent)`; for a
  single-agent run the owner is `None` and the source comes from the agent
  (`state.graph_source_default`, set by `chat.py` from `source_for_agent`);
  merges the delta into `state.graph.by_owner[owner]` and bumps `version`
  when anything was new.
- A new helper `live_graph_snapshot(state, agent) -> tuple[GraphData, int]`
  returns the merged graph of (a) `state.graph` and (b) the latest journal
  state of the agent, or, for an orchestrator, of each specialist from
  `agent.live_journal_snapshots()` (§5.3), and a version key
  `(state.graph.version, journal_lengths...)` the fragment compares to decide
  whether to re-render. Reading the agent's lists from the main thread is
  intentional (append-only lists, CPython GIL; copy with `list(...)` before use).

### 5.3 `ama_kbqa/agents/orchestrator_agent/agent.py` (edit, read-only accessor)

```python
def live_journal_snapshots(self) -> list[dict]:
    """Snapshots from every specialist loaded so far, tagged with
    source_agent, including specialists that are still running. Read-only;
    used by the frontend's live graph. Never raises."""
```

Implementation: iterate `self._agents.items()`, `list(agent.journal_snapshots)`,
tag `{**snap, "source_agent": name}`. Unit test in
`tests/agents/test_orchestrator_routing.py` style (fake agents).

### 5.4 `ama_kbqa/frontend/utils/graph_html.py` (edit)

- Keep `build_graph_html(nodes, edges, *, view_state_key, height_px, highlight_node_ids)`
  as the public name but extend the payload/template:
  - light-theme palette (§3.3); groups `kqapro`, `sciqa`, `literal`,
    `candidate`; `highlighted` and `new` node flags.
  - **Position persistence**: on `stabilized` and `dragEnd`, save
    `network.getPositions()` to `localStorage[view_state_key + ":pos"]`; on
    mount, seed `x`/`y` for nodes with a saved position, set
    `physics.stabilization = {iterations: 80, fit: !hasSavedPositions}`, and
    restore the viewport as today. Clear both keys when the payload's
    `reset: true` flag is set (first render of a new trace).
  - Legend inside the iframe with the four groups plus "in answer".
  - Vendor pin stays `vis-network@9.1.9` on the current CDN.
- `journal_to_graph(state)` stays as a thin compatibility wrapper that calls
  `graph_from_journal_state` and returns the old `(nodes, edges)` dict lists,
  so `graph_panel.py` (unused here, used on `dev`) keeps working.

### 5.5 `ama_kbqa/frontend/utils/live_graph_panel.py` (new)

```python
def render_live_graph_panel(*, live_run: dict | None, trace: dict | None, key_prefix: str = "chat") -> None
```

Renders §3.2 into the current container. Live mode: fragment + placeholder as
in §3.3; frozen mode: `graph_from_trace(trace)` + highlight. Store the last
rendered version key in `st.session_state[f"{key_prefix}:live_graph_version"]`
and the last node-id set for "new" detection. Handles the empty case with a
caption, never an exception.

### 5.6 `ama_kbqa/frontend/chat.py` / `app.py` (edit)

Wire-up per §3.1; keep every existing behaviour, including the
`_live_tick_silent` path. `chat.py` decides `panel_active` once near the top
and uses `contextlib.nullcontext()` when inactive so the body code is not
duplicated. Set `state.graph_source_default = source_for_agent(agent)` when
creating `LiveLifecycleState`.

## 6. Non-goals

- No server-side journal changes (would alter what the LLM sees in
  `GetJournalSummary`).
- No accumulation across follow-up turns; no per-message graphs.
- No new Python dependency; no offline bundling of vis-network.
- Not porting to `feat/langgraph-rewrite` or `dev` in this task.

## 7. Tests (must pass with `uv run pytest tests/frontend tests/agents tests/test_config*`)

1. `tests/frontend/test_live_graph_data.py`: every journal fact shape in §5.1
   (one test each, both KGs), found_values scalar/list/dict cases, id
   heuristic, caps and ordering, merge precedence, `answer_node_ids`
   (id match, label word-boundary, literal value, cap), `to_vis_payload`
   groups/flags, and one extractor test per allow-listed tool using fixtures
   built from the server Pydantic models. Malformed inputs (non-JSON result,
   fact with `None` object) return empty/partial graphs without raising.
2. `tests/frontend/test_lifecycle_runner.py`: `__graph__` items are attributed
   to the right owner in single, router and federated shapes; version bumps
   only on new content; `live_graph_snapshot` merges journal + deltas.
3. `tests/agents/`: `live_journal_snapshots` tags and never raises.
4. `tests/test_config_live_graph.py`: getter default False, toml true, env
   override `0`/`false` wins.
5. `tests/frontend/test_agent_picker_apptest.py` must still pass; add an
   AppTest that loads `chat.py` with the flag on and asserts no exception and
   that the sidebar toggle exists, and one with `AMA_FRONTEND_LIVE_GRAPH=0`
   asserting the toggle is absent.
6. `tests/frontend/test_graph_html.py`: template contains the four groups, the
   position-persistence keys, and no dark-theme colours; payload JSON is
   embedded intact (HTML-escaping of labels).

## 8. Acceptance (manual, local containers `qdrant_ama_kbqa` + `virtuoso_ama_kbqa`)

Run the demo locally (`uv run ama-kbqa-frontend`) against the local KG
containers with a KIT model and check, for each picker entry:

- KQAPro direct: nodes appear within a few seconds of the first `FindNode`
  (candidates), the chosen entity fills in, `GetNodeSummary` adds neighbour
  edges and literal leaves; the final answer highlights the answer entity.
- SciQA direct: same with `FindResource` / `GetRelationTargets`.
- Router: one colour; Federated: both colours, nodes from both specialists
  while both are running.
- Toggle off (sidebar) → single-column page; `AMA_FRONTEND_LIVE_GRAPH=0` →
  no toggle, `centered` layout, identical to `demo-v2-int`.
- Playwright screenshot of the live panel mid-run and after completion,
  saved under the scratchpad, attached to the task report.

## 9. Implementation plan

Two sequential Opus implementation agents on branch `demo-v2-graph`
(worktree `.claude/worktrees/demo-v2-graph`), reviewed by the orchestrator
after each; then `test-agent` (Verify), local acceptance run, docs capture.

| Phase | Agent | Deliverables | Done when |
|---|---|---|---|
| 1 · core | Opus "graph-core" | §4 config + getters; §5.1 `live_graph_data.py`; §5.3 accessor; §5.4 template rework; tests 1, 3, 4, 6 | `uv run pytest tests/frontend tests/agents tests/test_config_live_graph.py` green; `ruff check` clean |
| 2 · ui | Opus "graph-ui" | §5.2 runner changes; §5.5 panel; §5.6 wiring; tests 2, 5 | full `tests/frontend` green; AppTest flag on/off; page renders in a browser with a fake run |
| 3 · verify | orchestrator + `test-agent` | review diffs against this PRD, run full suite, §8 acceptance with screenshots | all §8 items checked or explicitly listed as failed |
| 4 · capture | `docs-agent`, `wiki` | System doc update (`System/demo_bwcloud_frontend.md`), ADR for the two-source graph, CHANGELOG; wiki note | links in this file |

Commit granularity: one commit per phase, message prefixed `demo v2 graph:`.

## Outcome (2026-09-14)

Shipped as three commits on `demo-v2-graph` (off `demo-v2-int` @ `1a9072d`):

- `f97842a` (phase 1 · core): `[frontend].live_graph` config flag +
  `AMA_FRONTEND_LIVE_GRAPH` override; the Streamlit-free
  `live_graph_data.py` normaliser (`GraphData`/`GraphNode`/`GraphEdge`,
  `graph_from_journal_state`, `graph_from_tool_result` over the
  `GRAPH_TOOLS` allow-list, `graph_from_trace`, `answer_node_ids`,
  `to_vis_payload`, `apply_caps`); the read-only
  `Orchestrator.live_journal_snapshots()` accessor; `graph_html.py` reworked
  for the light theme with localStorage position persistence.
- `7615190` (phase 2 · ui): `lifecycle_runner.py`'s worker-thread listener
  parses allow-listed tool results into `"__graph__"` deltas and
  `drain_into`/`_absorb_graph_delta` attribute them to the owning specialist
  via the existing span-owner walk; `live_graph_panel.py` (new) renders the
  1 Hz live fragment and the frozen/highlighted view; `chat.py` wires
  `panel_active` / `chat_col` / `graph_col` and the sidebar toggle; `app.py`
  goes wide when the flag is on.
- `a7f027c` (phase 3 · label-lookup fix found during acceptance): the local
  acceptance run showed the KQAPro fast path resolves the answer entity via
  `GetNodeLabel` from a single journal snapshot, so the highlighted node
  stayed a bare id. Added `GetNodeLabel`/`GetResourceLabel` and their batch
  variants to `GRAPH_TOOLS` (a real label now beats id-as-label on merge),
  and extraction for `GetResourceSummary` (SciQA's `GetNodeSummary`
  counterpart, previously missing from the allow-list).

Deviations from the plan as written: none structural; phase 3 was an
acceptance-driven addition the plan's §9 table anticipated only as "review
diffs against this PRD" in phase 3/verify, not as a named deliverable. It
became its own commit instead of a fixup because it changed the allow-list
contract (§5.1) enough to need its own tests.

Docs: `.agent/System/demo_bwcloud_frontend.md` (new "Live Graph Panel"
section), `.agent/Decisions/live-graph-two-source-subgraph.md` (new ADR),
`.agent/SOP/hetzner_demo_deployment.md` §9 (staging branch note, default-on
flag, reranker-off local acceptance gotcha).

Verification (orchestrator, 2026-09-14, at `a7f027c`): `uv run pytest -q`
887 passed, `ruff check .` clean. §8 acceptance against the local
`qdrant_ama_kbqa` / `virtuoso_ama_kbqa` containers and the KIT endpoint
(`kit.mistral-small-4-119b-a8b`, `AMA_RETRIEVAL_RERANKER_ENABLED=false`),
one agent per process, live pipeline (`make_listener` + `drain_into` +
`live_graph_snapshot`) and frozen pipeline (`graph_from_trace`) compared:

| Entry | Time | Entities / literals / edges | Sources | Answer nodes |
|---|---|---|---|---|
| KQAPro | 26 s | 55 / 9 / 67 | kqapro | Christopher Nolan, Inception |
| SciQA | 68 s | 28 / 0 / 0 | sciqa | one ORKG resource (the local ORKG data returned no links or contributions for the visited resources) |
| Orchestrator (Router) | 99 s | 64 / 54 / 119 | kqapro | Albert Einstein, Ulm |
| Orchestrator (Federated) | 129 s | 87 / 14 / 75 | kqapro 62+14, sciqa 25 | Christopher Nolan, Heterogeneous benchmark, ... |

Live and frozen graphs were identical in all four runs. Known limitation
confirmed: neighbours from `GetNodeSummary` stay id-labelled unless the
agent resolves them (49 to 53 of the KQAPro entities per run). Browser
screenshots of the live and final panel are in the session scratchpad and
the wiki note. Staging deployment remains pending until `demo-v2-graph`
merges into `demo-v2-int`.
