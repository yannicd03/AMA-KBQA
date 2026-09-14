# Demo Frontend (filename retained from the `demo-bwcloud` branch)

> **Naming note (2026-09-14):** this file was created on the `demo-bwcloud`
> branch and originally described that branch's frontend only. It now
> describes the **demo frontend as carried forward onto the `demo-v2` /
> `demo-hetzner` line** (this worktree: `demo-v2-int`). The filename is kept
> for link stability — every ADR and SOP in `.agent/` that references
> "demo_bwcloud_frontend.md" still points at the right file. The frontend
> **code itself** has not been renamed; `ama_kbqa/frontend/chat.py` etc. still
> live at the paths below.
>
> The single biggest change since the original `demo-bwcloud` version of this
> doc: the demo build is no longer branched off an old pre-hybrid snapshot of
> `main`. It is now built on top of **`dev`** — the same codebase documented
> in [System/project_architecture.md](project_architecture.md) — which brings
> the `ama_kbqa/retrieval/` hybrid-search package, the `framework/adapters/`
> KG-adapter layer, the published `chatkit` LLM-retry dependency, and the
> current one-round-trip Orchestrator routing. The demo-only pieces described
> below (single-page shell, About dialog, KIT-only chat_controls sidebar,
> illustrative pricing) are layered on top of that, unchanged in spirit from
> the original `demo-bwcloud` design. See
> [Decisions/demo-bwcloud-frontend-divergence.md](../Decisions/demo-bwcloud-frontend-divergence.md)
> for why these divergences exist, and its 2026-09-14 addendum for what
> changed in the port.

---

## Overview

The demo frontend is a **single-page, chat-only** Streamlit app stripped of
all research-tooling pages. It serves anonymous visitors at the public
Hetzner-hosted endpoint (`amakbqa.yanlab.de`, staging `amakbqa-next.yanlab.de`
— see [SOP/hetzner_demo_deployment.md](../SOP/hetzner_demo_deployment.md)).

**Key differences from the full `dev`/`main` build:**

| Aspect | Full build (`dev`/`main`) | Demo build (this line) |
|--------|----------------------------|-------------------------|
| Pages | 5 pages (Chat, Batch, Eval, Settings, Trace Inspector) | 1 page (Chat only) |
| Entry point | `ama_kbqa/frontend/app.py` + `pages/` dir | `ama_kbqa/frontend/app.py` + `chat.py` (no `pages/`) |
| Agent picker | Sidebar | Chat-bar popover pill, 4 entries (see below) |
| Orchestrator | One Orchestrator, per-call/config `federation` behaviour | Two picker entries — "Orchestrator (Router)" and "Orchestrator (Federated)" — each a differently-constructed `Orchestrator` instance |
| Model/temp controls | Dedicated Settings page | Sidebar (KIT-only, session-only) |
| Provider selection | User-configurable | Pinned to KIT, never user-selectable |
| Cost display | None | Illustrative `~$X est.` per answer |
| Onboarding | Sidebar landing markdown | First-visit About dialog + "?" button |

---

## File Structure

```
ama_kbqa/
├── pricing.py                        # Cost estimation (DEMO-ONLY)
├── retrieval/                        # Hybrid dense/BM25 search — shared with the full build (from dev)
├── framework/adapters/               # KG adapter layer — shared with the full build (from dev)
├── data/
│   └── model_pricing.json            # KIT model OpenRouter list prices (DEMO-ONLY)
├── frontend/
│   ├── app.py                        # Single entry point: About dialog + st.navigation
│   ├── chat.py                       # Chat page (root URL, no pages/ dir)
│   └── utils/
│       ├── agent_factory.py          # AGENT_INFO, ORCHESTRATOR_MODES, tagline field (DEMO-ONLY)
│       ├── chat_controls.py          # KIT model/temperature controls (DEMO-ONLY)
│       ├── orchestrator_svg.py       # Orchestrator lifecycle figure; lighting comes from lifecycle_mapping (shared with full build's trace panel)
│       ├── styling.py                # + .st-key-about_help_btn CSS (DEMO-ONLY)
│       └── (all other utils shared with full build: lifecycle_runner, trace_panel, graph_panel, config_editor, …)
scripts/
└── fetch_model_pricing.py            # Re-runnable pricing JSON generator (DEMO-ONLY)
tests/
├── test_pricing.py                   # (DEMO-ONLY)
└── frontend/
    └── test_chat_controls.py         # (DEMO-ONLY)
```

Files marked `(DEMO-ONLY)` do not exist on `dev`/`main`. Everything else in
`frontend/utils/` (lifecycle runner, trace panel, graph panel, config editor)
is shared with the full build. There is no `pages/` directory on this line —
Settings, Batch Processing, Evaluation, and the standalone Trace Inspector
page are all absent, same as on the original `demo-bwcloud` branch; the
in-page Lifecycle/Trace/Graph tab panel (rendered after each answer, inside
`chat.py`) is retained and shared with the full build.

---

## Entry Point: `ama_kbqa/frontend/app.py`

```bash
streamlit run ama_kbqa/frontend/app.py
```

Responsibilities:
1. `st.set_page_config(...)` and `inject_css()`.
2. Define and conditionally open the "About this demo" `st.dialog` (first visit only,
   keyed on `st.session_state["_about_seen"]`). The About text now explains both
   Orchestrator modes (Router = the paper's single-dispatch behaviour; Federated =
   experimental, may query both specialists and fuse their answers, ~2x tokens).
3. Render the floating "?" button (CSS target: `.st-key-about_help_btn`).
4. Call `st.navigation([chat_page], position="hidden").run()` — this serves
   `chat.py` at the root URL and disables automatic `pages/` discovery and the
   page-switcher UI element.

---

## Chat Page: `ama_kbqa/frontend/chat.py`

The main interactive page. Differences from the full-build `pages/1_Chat.py`:

- **Agent picker** is a chat-bar popover pill next to the input, not a sidebar widget.
  Each option shows the agent's `tagline` (from `AGENT_INFO`) and a checkmark on the
  active agent. Switching agents clears `st.session_state["persistent_agent"]`.
- **Four picker entries**: "Orchestrator (Router)", "Orchestrator (Federated)",
  "KQAPro", "SciQA". The two Orchestrator entries are not two agent classes — both
  resolve to `ama_kbqa.agents.orchestrator_agent.agent.Orchestrator`, constructed
  with `federation=False` (Router) or `federation=True` (Federated) via
  `ORCHESTRATOR_MODES` in `agent_factory.py`. See
  [System/orchestrator_routing.md](orchestrator_routing.md) for the Router vs.
  Federated dispatch mechanics.
- **Model/temperature controls** are in the sidebar via `render_chat_sidebar()` from
  `chat_controls.py`. These are KIT-only and session-only.
- **Cost footer** renders `~$X est.` after each answer using `estimate_cost_usd` /
  `format_cost_usd` from `ama_kbqa/pricing.py`.
- **No Trace Inspector button** (standalone page removed).
- **Lifecycle/Trace/Graph tab panel** (in-page, after run) is retained unchanged. In
  a Federated run the lifecycle figure lights both specialist dispatch edges and the
  Answer Combination node. That lighting is driven entirely by the trace:
  `lifecycle_mapping` maps the fusion span (`kind="synthesis"`, `name="fuse"`) to
  `orch_combine`. `render_orchestrator_svg`'s `mode="router"|"federated"` argument is
  cosmetic and only changes the figure's `aria-label`.
- Multiturn conversation for directly-selected sub-agents (KQAPro, SciQA) is retained
  unchanged; the Orchestrator (either mode) stays stateless.

---

## Module: `ama_kbqa/frontend/utils/chat_controls.py`

Sidebar model and temperature controls for the demo. All purely functional helpers
are unit-tested in `tests/frontend/test_chat_controls.py`.

The model picker is populated by **auto-discovery**, not a hardcoded whitelist: it
queries the live KIT `/models` endpoint and keeps whatever local chat LLMs are
currently advertised, instead of a fixed list that goes stale whenever KIT retires or
adds a model (see the SOP's "Model changes" section for the incident that motivated
this).

| Function | Signature | Purpose |
|----------|-----------|---------|
| `_is_local_chat_model` | `(entry: dict) -> bool` | True for a selectable local chat LLM: `connection_type == "local"`, `kit.*` id, not a preset/alias/hidden entry, non-null `capabilities`, id doesn't match a non-chat keyword (embedding/rerank/flux/voxtral/tts/whisper/image) |
| `default_model` | `(models: list[str]) -> str` | First of `DEFAULT_MODEL_PREFERENCE` present in `models`, else `models[0]`, else the first preference as a last-resort placeholder |
| `filter_selectable_models` | `(entries: list[dict]) -> list[str]` | Sorted local KIT chat model ids from a raw `/models` catalog, via `_is_local_chat_model` |
| `_fetch_kit_models_meta` | `() -> list[dict]` | Cached (`st.cache_data`, 5-min TTL) wrapper around `config_editor.fetch_provider_models_meta("kit")` |
| `available_models` | `() -> list[str]` | Live KIT models via `_fetch_kit_models_meta` + `filter_selectable_models`; falls back to a small static `_OFFLINE_FALLBACK_MODELS` list on any fetch error |
| `display_model_name` | `(model: str) -> str` | Endpoint-provided display name if known, else the id with the `kit.` prefix stripped |
| `price_caption` | `(model: str \| None) -> str \| None` | One-line `$/1M input · $/1M output` string for the sidebar |
| `apply_chat_settings` | `(model: str, temperature: float) -> None` | Mutates `cfg_module._config_cache` **and** sets `AMA_KBQA_CHAT_MODEL`/`AMA_KBQA_CHAT_TEMPERATURE` in `os.environ` (session-only; never writes to disk) |

`apply_chat_settings` always sets `chat_provider = "kit"`, enforcing the KIT-only
constraint at the config level. The env-var half matters because the Orchestrator and
specialist MCP servers run as **subprocesses** that load their own `config.toml`
independently — mutating this process's in-memory config alone never reaches them
(see `SOP/hetzner_demo_deployment.md` §4 for the "Model not found" incident this
fixed).

**Why KIT-only:** The public demo is KIT-funded; routing traffic through OpenRouter
or other providers must not be possible.

**Why session-only settings:** The demo runs as a shared public service. Writing
`config.toml` would persist one visitor's choices for all subsequent visitors.

---

## Module: `ama_kbqa/pricing.py`

Illustrative cost estimation. Data loaded from `ama_kbqa/data/model_pricing.json`
via `lru_cache` (parsed once per process). Unchanged from the original `demo-bwcloud`
design.

| Function | Signature | Purpose |
|----------|-----------|---------|
| `known_models` | `() -> list[str]` | Sorted KIT model IDs in the pricing table |
| `get_model_pricing` | `(model: str | None) -> dict | None` | Pricing entry or `None` if unknown |
| `estimate_cost_usd` | `(model, prompt_tokens, completion_tokens) -> float | None` | Illustrative USD estimate |
| `format_cost_usd` | `(cost: float | None) -> str | None` | Magnitude-aware USD string |

Prices are OpenRouter list prices for the equivalent model. KIT does not bill.
The UI labels this "illustrative" explicitly.

**Known gap (as of 2026-09-14):** `model_pricing.json` has no rows yet for the current
KIT chat models (`kit.mistral-small-4-119b-a8b`, `kit.deepseek-v4-flash`,
`kit.glm-5.3`) — `price_caption` returns `None` for them (hidden, not wrong) until the
table is regenerated. See `SOP/hetzner_demo_deployment.md` §8.

---

## Data: `ama_kbqa/data/model_pricing.json`

Schema:

```json
{
  "generated_at": "ISO-8601 timestamp",
  "models": {
    "<kit-model-id>": {
      "openrouter_slug": "<vendor/name>",
      "prompt_usd_per_token": 0.000000xxx,
      "completion_usd_per_token": 0.000000xxx
    }
  }
}
```

Regenerate with:
```bash
KIT_API_KEY=<key> uv run python scripts/fetch_model_pricing.py
```

The script fetches the live KIT `/models` list (or a curated fallback without the
key), fetches OpenRouter public prices, and maps KIT IDs to OpenRouter slugs via a
hand-verified override table + vendor heuristic.

---

## `AGENT_INFO` / `ORCHESTRATOR_MODES` — `ama_kbqa/frontend/utils/agent_factory.py`

`ORCHESTRATOR_MODES` maps the two Orchestrator picker entries to the `federation`
constructor flag:

```python
ORCHESTRATOR_MODES: dict[str, bool] = {
    "Orchestrator (Router)": False,
    "Orchestrator (Federated)": True,
}
```

`create_agent(name)` calls `Orchestrator(federation=ORCHESTRATOR_MODES[name])` for
either entry; `is_orchestrator(name)` tests membership. `AGENT_INFO` taglines shown in
the popover picker:

| Agent | Tagline |
|-------|---------|
| Orchestrator (Router) | "Picks the single best specialist for your question, as evaluated in the paper." |
| Orchestrator (Federated) | "Experimental: may ask both specialists in parallel and combine their answers. Slower and uses about twice the tokens; the paper's numbers are single-dispatch." |
| KQAPro | "Facts from a general knowledge graph (films, places, people). Use for factual lookups." |
| SciQA | "Scientific research via the ORKG. Use for research papers and contributions." |

`AGENT_SUGGESTIONS` also has per-mode example questions — the Federated entry's
suggestions deliberately include two cross-graph questions so the fan-out/fusion path
has something visible to demonstrate.

---

## Live Graph Panel

Config-gated side panel (`[frontend].live_graph` in `config.toml` /
`config.docker.toml`, default `true` on this demo line; override per
deployment with `AMA_FRONTEND_LIVE_GRAPH=0/1`) that shows the subgraph an
agent has gathered while it answers, and freezes with the answer highlighted
once the run completes. PRD:
[Tasks/active/live-graph-panel.md](../Tasks/active/live-graph-panel.md).
Design rationale for the two-source approach:
[Decisions/live-graph-two-source-subgraph.md](../Decisions/live-graph-two-source-subgraph.md).

### Why two sources

Neither of the agent's existing outputs alone is enough for a useful graph:

- **The journal** (`JournalState`, via `agent.journal_snapshots`) knows which
  entities were visited and which values were found, but the servers write at
  least seven different `verified_facts` shapes, and two of the main
  exploration tools barely reach it: `GetNodeSummary`
  (`kqapro_server.py:2438-2447`) journals attributes into `found_values` and
  drops its `relations` dict entirely; `FindNode` journals nothing at all. A
  journal-only graph would show the visited entity and its literals with no
  neighbours and no search candidates.
- **Tool-call results**, parsed for a fixed allow-list of tools
  (`live_graph_data.GRAPH_TOOLS`), supply exactly what the journal drops:
  search candidates and neighbour relations.

Both are merged by node id through the same normaliser
(`ama_kbqa/frontend/utils/live_graph_data.py`), so the live view (deltas
parsed as the run progresses) and the frozen view (a completed trace) never
disagree. The KQAPro fast path is a sharper version of the same gap: it takes
a single journal snapshot right after `FindNode`, and resolves the answer
entity's label via `GetNodeLabel` alone, so `GetNodeLabel` /
`GetResourceLabel` (and their batch variants) are also in the allow-list,
purely to upgrade an id-as-label node to a real label once the agent looks
one up (a real label always outranks an id on merge, `GraphData._merge_nodes`).

### Data flow

1. **Worker thread** (`lifecycle_runner.make_listener`): on the `close` of
   every `tool_call` span whose tool name is in `GRAPH_TOOLS`, the raw JSON
   result is parsed into a `GraphData` delta right there (`graph_from_tool_result`,
   source left blank) and pushed onto the run queue as a `("__graph__", {...})`
   item, immediately after the regular trace notification for that span.
   Parsing the (potentially megabyte-sized) tool result on the worker thread,
   not the Streamlit thread, keeps the UI responsive.
2. **Main thread** (`lifecycle_runner.drain_into` / `_absorb_graph_delta`):
   resolves the delta's owner via the same `parent_span_id` walk used for
   every other span (`_owner_for`): `None` for a single-agent run, the
   specialist's name for a router/federated run. It then stamps the owner's
   KG source onto the delta (`source_for_agent`), and merges it into
   `LiveGraphState.by_owner[owner]`. A `version` counter increments only when
   the merge actually added a node or edge, so a repeated tool call (the
   agent re-checking something it already visited) does not force a redraw.
3. **`live_graph_snapshot(state, agent)`** merges `state.graph` (the tool-result
   deltas) with the newest journal snapshot(s): for a direct agent, the last
   entry of `agent.journal_snapshots`; for an Orchestrator, one snapshot per
   specialist via `agent.live_journal_snapshots()`, a read-only accessor
   that iterates `self._agents` and copies each specialist's append-only
   `journal_snapshots` list, so it sees specialists that are still running
   (unlike `Orchestrator.journal_snapshots`, which only gains a specialist's
   entries after `_run_specialist` has finished). Returns
   `(capped_graph, version_key)`; the panel redraws only when the version key
   changes.
4. **`live_graph_panel.py`** renders inside a `@st.fragment(run_every=1.0)`
   that writes into an `st.empty()` placeholder created *outside* the
   fragment. Writing into a pre-existing placeholder, rather than having the
   fragment body emit the iframe directly, is what keeps the vis-network
   canvas from being torn down and remounted on every tick. The fragment
   never touches `live_run["queue"]`; `chat.py`'s existing `_live_tick`
   remains the sole consumer (two fragments draining one queue would lose
   items at random).

### Owner attribution for federated runs

The panel colours nodes by source KG (KQAPro blue, SciQA green via the `group`
field in `to_vis_payload`/`graph_html.py`); a federated run shows both
colours because each specialist's tool-call spans are nested under its own
`delegate` span, and the owner walk (`_owner_for` on the live side,
`graph_from_trace`'s `owner_of` on the frozen side) resolves each span back
to its specialist before merging. `source_for_agent` centralises every
spelling the codebase uses for the same specialist (`kqapro_agent`,
`KQAProAgent`, `KQAPro`, `JournalState.kg_name`, and the ORKG/SciQA aliases).

### Frozen view and answer highlighting

Once a run completes, `graph_from_trace(trace)` rebuilds the full subgraph
from the trace's recorded events and journal snapshots (only the *last*
journal snapshot per source is used, since snapshots are cumulative).
`answer_node_ids(graph, answer)` highlights a node when its id appears
verbatim in the answer text, its label matches on a word boundary
(case-insensitive, 3+ chars, so short labels like "US" cannot light up half
the graph), or, for literals, its value appears as a substring; capped at
50 highlighted nodes.

### Limits

Per `live_graph_data.py`: 300 nodes / 600 edges per graph
(`apply_caps`, entities kept before literals before search-only candidates);
8 literal children per `(entity, attribute)`; 10 search candidates per
lookup call; 10 relation targets and 5 attribute values per summary call; 25
entity nodes (no edges) per raw SPARQL result. A SELECT projection does not
reliably say which variable relates to which, so inventing edges from it
would be fiction. Labels are truncated to 40 characters in the graph (full
text in the tooltip).

### Known limitation

Neighbours returned by `GetNodeSummary` (and `GetResourceSummary` on the
SciQA side) are drawn with their raw ids as labels unless the agent later
resolves them, e.g. via `GetNodeLabel`/`BatchGetNodeLabels` or by visiting
the neighbour directly. This is by design, not a gap to close: the frontend
never issues its own KG queries to pre-resolve labels, so the panel costs no
extra tool calls or tokens (see the ADR's "no extra tool calls" consequence).
A neighbour the agent never looks up by name stays an id on the canvas.

### Config

```toml
[frontend]
live_graph = true
```

`ama_kbqa/config.py`: `get_frontend_config()` returns the `[frontend]`
section dict (`{}` if absent); `get_live_graph_enabled()` reads
`live_graph`, overridable by `AMA_FRONTEND_LIVE_GRAPH=0/1` (env always wins,
same truthy parsing as the `AMA_RETRIEVAL_*` overrides). Getter default is
**False** when the section is absent (old configs stay off); both shipped
toml files on this line set it `true`. `app.py` sets `layout="wide"` only
when the flag is on; `chat.py`'s sidebar toggle ("Live graph") only renders
when the flag is on, and the two-column layout (`chat_col`/`graph_col`) only
activates when the flag is on **and** the toggle is on **and** Simplified
view is off **and** there is something to show (a run in flight or at least
one completed trace); the landing view stays single-column.

### Files

| File | Role |
|------|------|
| `ama_kbqa/frontend/utils/live_graph_data.py` | Streamlit-free normaliser: `GraphNode`/`GraphEdge`/`GraphData`, `GRAPH_TOOLS` allow-list, `graph_from_journal_state`, `graph_from_tool_result`, `graph_from_trace`, `answer_node_ids`, `to_vis_payload`, `apply_caps` |
| `ama_kbqa/frontend/utils/lifecycle_runner.py` | `make_listener` emits `"__graph__"` deltas parsed on the worker thread; `drain_into`/`_absorb_graph_delta` merge with owner attribution; `live_graph_snapshot` |
| `ama_kbqa/frontend/utils/live_graph_panel.py` | `render_live_graph_panel`: 1 Hz fragment writing into `st.empty()` placeholders (live), frozen+highlighted view (post-run) |
| `ama_kbqa/frontend/utils/graph_html.py` | Light-theme vis-network template; localStorage position persistence keyed by `view_state_key`; `reset` flag clears it for a new trace; `journal_to_graph` kept as a compatibility wrapper |
| `ama_kbqa/frontend/chat.py` / `app.py` | `panel_active`, `chat_col`/`graph_col` split, sidebar toggle; `app.py` wide layout when the flag is on |
| `ama_kbqa/config.py` | `get_frontend_config`/`get_live_graph_enabled`; `FRONTEND_LIVE_GRAPH_ENV` |
| `ama_kbqa/agents/orchestrator_agent/agent.py` | `live_journal_snapshots()` read-only accessor |

---

## Test Coverage (demo-only tests)

| File | What it tests |
|------|--------------|
| `tests/test_pricing.py` | `estimate_cost_usd`, `format_cost_usd`, `get_model_pricing`, `known_models` |
| `tests/frontend/test_chat_controls.py` | `filter_selectable_models`, `_is_local_chat_model`, `available_models` (mocked fetch), `default_model`, `display_model_name`, `price_caption`, `apply_chat_settings` |

---

## Related Docs

- [System/project_architecture.md](project_architecture.md) — canonical full-build (`dev`) architecture
- [System/orchestrator_routing.md](orchestrator_routing.md) — Router vs. Federated dispatch mechanics the two Orchestrator picker entries drive
- [Decisions/demo-bwcloud-frontend-divergence.md](../Decisions/demo-bwcloud-frontend-divergence.md) — why these divergences exist and what was rejected
- [Decisions/federated-dispatch-and-fusion.md](../Decisions/federated-dispatch-and-fusion.md) — federated dispatch/fusion ADR; 2026-09-14 addendum covers the per-instance mode switch this frontend uses
- [Decisions/live-graph-two-source-subgraph.md](../Decisions/live-graph-two-source-subgraph.md): ADR for the live graph panel's journal + tool-result merge design
- [Decisions/live-trace-and-chat-unification.md](../Decisions/live-trace-and-chat-unification.md) — live trace pipeline (shared with full build)
- [Decisions/multiturn-direct-agent-conversation.md](../Decisions/multiturn-direct-agent-conversation.md) — multiturn session semantics (shared with full build)
- [SOP/hetzner_demo_deployment.md](../SOP/hetzner_demo_deployment.md) — how this frontend is deployed and smoke-tested
