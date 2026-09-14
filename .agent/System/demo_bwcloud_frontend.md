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
- [Decisions/live-trace-and-chat-unification.md](../Decisions/live-trace-and-chat-unification.md) — live trace pipeline (shared with full build)
- [Decisions/multiturn-direct-agent-conversation.md](../Decisions/multiturn-direct-agent-conversation.md) — multiturn session semantics (shared with full build)
- [SOP/hetzner_demo_deployment.md](../SOP/hetzner_demo_deployment.md) — how this frontend is deployed and smoke-tested
