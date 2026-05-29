# Demo Build Frontend (demo-bwcloud branch)

> **Branch-specific doc.** This describes the `demo-bwcloud` branch only.
> For the canonical full-build frontend, see
> [System/project_architecture.md](project_architecture.md) (Section 7).

---

## Overview

The `demo-bwcloud` frontend is a **single-page, chat-only** Streamlit app stripped
of all research-tooling pages. It serves anonymous visitors at the public bwCloud
endpoint for the SEMANTiCS 2026 conference demo.

**Key differences from the full build:**

| Aspect | Full build (`yannic-dev`/`main`) | Demo build (`demo-bwcloud`) |
|--------|----------------------------------|-----------------------------|
| Pages | 5 pages (Chat, Batch, Eval, Settings, Trace Inspector) | 1 page (Chat only) |
| Entry point | `ama_kbqa/frontend/app.py` + `pages/` dir | `ama_kbqa/frontend/app.py` + `chat.py` (no `pages/`) |
| Agent picker | Sidebar | Chat-bar popover pill |
| Model/temp controls | Dedicated Settings page | Sidebar (KIT-only, session-only) |
| Provider selection | User-configurable | Pinned to KIT, never user-selectable |
| Cost display | None | Illustrative `~$X est.` per answer |
| Onboarding | Sidebar landing markdown | First-visit About dialog + "?" button |

---

## File Structure (demo-bwcloud)

```
ama_kbqa/
├── pricing.py                        # Cost estimation (DEMO-ONLY)
├── data/
│   └── model_pricing.json            # 9 KIT models with OpenRouter list prices (DEMO-ONLY)
├── frontend/
│   ├── app.py                        # Single entry point: About dialog + st.navigation
│   ├── chat.py                       # Chat page (moved from pages/1_Chat.py)
│   └── utils/
│       ├── agent_factory.py          # + tagline field in AGENT_INFO (DEMO-ONLY)
│       ├── chat_controls.py          # KIT model/temperature controls (DEMO-ONLY)
│       ├── styling.py                # + .st-key-about_help_btn CSS (DEMO-ONLY)
│       └── (all other utils shared with full build)
scripts/
└── fetch_model_pricing.py            # Re-runnable pricing JSON generator (DEMO-ONLY)
tests/
├── test_pricing.py                   # (DEMO-ONLY)
└── frontend/
    └── test_chat_controls.py         # (DEMO-ONLY)
```

Files marked `(DEMO-ONLY)` do not exist on `yannic-dev`/`main`. Everything else in
`frontend/utils/` (lifecycle runner, trace panel, etc.) is shared with the full build.

---

## Entry Point: `ama_kbqa/frontend/app.py`

```bash
streamlit run ama_kbqa/frontend/app.py
```

Responsibilities:
1. `st.set_page_config(...)` and `inject_css()`.
2. Define and conditionally open the "About this demo" `st.dialog` (first visit only,
   keyed on `st.session_state["_about_seen"]`).
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
- **Model/temperature controls** are in the sidebar via `render_chat_sidebar()` from
  `chat_controls.py`. These are KIT-only and session-only.
- **Cost footer** renders `~$X est.` after each answer using `estimate_cost_usd` /
  `format_cost_usd` from `ama_kbqa/pricing.py`.
- **No Trace Inspector button** (standalone page removed).
- Lifecycle/Trace/Graph tab panel (in-page, after run) is retained unchanged.
- Multiturn conversation for directly-selected sub-agents is retained unchanged.

---

## Module: `ama_kbqa/frontend/utils/chat_controls.py`

Sidebar model and temperature controls for the demo. All purely functional helpers
are unit-tested in `tests/frontend/test_chat_controls.py`.

| Function | Signature | Purpose |
|----------|-----------|---------|
| `filter_selectable_models` | `(models: list[str]) -> list[str]` | Drop embedding models and Azure OpenAI models (except `gpt-oss`) |
| `available_models` | `() -> list[str]` | Fetch KIT `/models` (5-min cache), fall back to `known_models()` from `pricing.py` |
| `price_caption` | `(model: str | None) -> str | None` | One-line `$/M input · $/M output` string for the sidebar |
| `apply_chat_settings` | `(model: str, temperature: float) -> None` | Mutate `cfg_module._config_cache` in-memory; never writes to disk |

`apply_chat_settings` always sets `chat_provider = "kit"`, enforcing the KIT-only
constraint at the config level.

---

## Module: `ama_kbqa/pricing.py`

Illustrative cost estimation. Data loaded from `ama_kbqa/data/model_pricing.json`
via `lru_cache` (parsed once per process).

| Function | Signature | Purpose |
|----------|-----------|---------|
| `known_models` | `() -> list[str]` | Sorted KIT model IDs in the pricing table |
| `get_model_pricing` | `(model: str | None) -> dict | None` | Pricing entry or `None` if unknown |
| `estimate_cost_usd` | `(model, prompt_tokens, completion_tokens) -> float | None` | Illustrative USD estimate |
| `format_cost_usd` | `(cost: float | None) -> str | None` | Magnitude-aware USD string |

Prices are OpenRouter list prices for the equivalent model. KIT does not bill.
The UI labels this "illustrative" explicitly.

---

## Data: `ama_kbqa/data/model_pricing.json`

9 KIT chat models. Schema:

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

## `AGENT_INFO` — `tagline` field (demo addition)

`ama_kbqa/frontend/utils/agent_factory.py` has a new `tagline` key per agent on this
branch, used as the one-liner shown inside the agent popover picker:

| Agent | Tagline |
|-------|---------|
| Orchestrator | "Auto-routes to the best specialist. Use when you're not sure which agent fits." |
| KQAPro | "Facts from a general knowledge graph (films, places, people). Use for factual lookups." |
| SciQA | "Scientific research via the ORKG. Use for research papers and contributions." |

---

## Test Coverage (demo-only tests)

| File | What it tests |
|------|--------------|
| `tests/test_pricing.py` | `estimate_cost_usd`, `format_cost_usd`, `get_model_pricing`, `known_models` |
| `tests/frontend/test_chat_controls.py` | `filter_selectable_models`, `available_models` (mocked fetch), `price_caption`, `apply_chat_settings` |

---

## Related Docs

- [System/project_architecture.md](project_architecture.md) — canonical full-build architecture
- [Decisions/demo-bwcloud-frontend-divergence.md](../Decisions/demo-bwcloud-frontend-divergence.md) — why these divergences exist and what was rejected
- [Decisions/live-trace-and-chat-unification.md](../Decisions/live-trace-and-chat-unification.md) — live trace pipeline (shared with full build)
- [Decisions/multiturn-direct-agent-conversation.md](../Decisions/multiturn-direct-agent-conversation.md) — multiturn session semantics (shared with full build)
