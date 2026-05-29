# ADR: demo-bwcloud Frontend Divergence

> **Branch:** `demo-bwcloud` only. These changes are NOT present on `yannic-dev` or `main`.
> All file paths below are relative to the `demo-bwcloud` branch.

## Status

Active (as of 2026-05-29). Five commits on top of the last `yannic-dev` merge:
`eae0a74`, `3755aec`, `5d475da`, `140eeb7`, `5d30df4`.

## Context

The `demo-bwcloud` branch is a **public demo build** hosted on bwCloud for
conference/research audiences. The full-build dashboard (multi-page Streamlit,
Settings editor, Evaluation/Batch pages, OpenRouter-accessible provider selection,
Trace Inspector standalone page) is inappropriate for a public endpoint:

- Visitors are anonymous; there is no auth layer.
- The KIT endpoint used by the demo is free — billing is irrelevant, but showing
  illustrative pricing gives a sense of real-world cost.
- A dashboard with batch-run controls, settings editors, and raw trace files would
  be confusing and potentially misused.
- The SEMANTiCS 2026 demo should focus attention on the chat interaction, not
  the full research tooling.

## Decisions

### 1. Single-page app (commit `5d475da`)

**Chose:** Collapse the multi-page Streamlit app to a single chat page served at
the root URL via `st.navigation([chat_page], position="hidden")`. Streamlit's
automatic `pages/` discovery is disabled.

**Rejected:** Keeping the multi-page structure and just hiding pages in the sidebar.
That still exposes the pages to direct URL access.

**Why it matters:** `position="hidden"` removes the page switcher entirely from the
UI. There is no way for a visitor to navigate to Evaluation or Batch Processing.

**File changes:**
- `ama_kbqa/frontend/app.py` — rewritten as the single entry point with the About
  dialog and single-page `st.navigation` call.
- `ama_kbqa/frontend/chat.py` — chat page (moved from `pages/1_Chat.py`).
- `pages/` directory — removed entirely on this branch.

### 2. About dialog (commit `5d475da`)

**Chose:** A first-visit `st.dialog` ("About this demo", width="large") that auto-opens
once per browser session via `st.session_state["_about_seen"]`, with a floating "?"
button (CSS in `styling.py` under `.st-key-about_help_btn`) that reopens it at any time.

**Why:** Replaces the old sidebar landing markdown and provider/model branding. Gives
visitors immediate context without cluttering the chat UI. The "?" button is always
reachable without occupying sidebar space.

### 3. In-chat agent picker as a popover pill (commits `140eeb7`, `5d30df4`)

**Chose:** Agent selector (Orchestrator / KQAPro / SciQA) moved out of the sidebar
into a chat-bar popover "pill" picker, rendered next to the chat input in both the
landing and conversation views.

**Why:** Sidebar space is used for model/temperature controls. The agent picker is a
first-class interaction choice — making it a prominent pill next to the input matches
the mental model of "pick your agent, then type your question".

**Details:**
- Each option shows a `tagline` one-liner (new field in `AGENT_INFO` in
  `ama_kbqa/frontend/utils/agent_factory.py`) and a checkmark on the active agent.
- Switching agents drops the persisted multiturn conversation (`persistent_agent`
  cleared from `st.session_state`).
- Emoji icons removed from the picker labels (commit `5d30df4`) for cleaner appearance;
  checkmark selection indicator retained.
- The Orchestrator remains stateless; sub-agents (KQAPro, SciQA) retain multiturn
  history within a session.

### 4. KIT-only model dropdown and temperature in the sidebar (commit `3755aec`)

**Chose:** Move a minimal subset of model/temperature controls into the chat sidebar,
replacing the full Settings page.

**Settings page removed:** `pages/4_Settings.py` and `ama_kbqa/frontend/utils/settings_ui.py`.

**New module:** `ama_kbqa/frontend/utils/chat_controls.py` with:
- `filter_selectable_models(models)` — drops embedding models and Azure OpenAI models
  (except `gpt-oss`); keeps KIT chat models only.
- `available_models()` — fetches the live KIT `/models` endpoint (5-min TTL via
  `@st.cache_data`), falls back to `known_models()` from `pricing.py` when offline.
- `price_caption(model)` — one-line illustrative price string for the sidebar.
- `apply_chat_settings(model, temperature)` — writes the chosen model and temperature
  into the in-memory config cache (`cfg_module._config_cache`). Session-only; never
  writes `config.toml` to disk. Switching models clears the persisted multiturn agent.

**Why KIT-only:** The public demo is KIT-funded; routing traffic through OpenRouter
or other providers must not be possible. The endpoint is pinned to KIT in
`apply_chat_settings`; provider selection is never exposed.

**Why session-only settings:** The demo runs as a shared public service. Writing
`config.toml` would persist one visitor's choices for all subsequent visitors.

### 5. Per-answer cost display (commit `eae0a74`)

**Chose:** Show an illustrative `~$X est.` cost next to time and tokens in the chat
answer footer.

**Why:** Helps visitors understand real-world LLM cost at a glance even though KIT is
free. Explicitly labelled "illustrative" in the About dialog and in code comments.

**New module:** `ama_kbqa/pricing.py` with:
- `estimate_cost_usd(model, prompt_tokens, completion_tokens)` — multiplies token
  counts by per-token OpenRouter list prices for the equivalent model. Returns `None`
  if the model is unknown.
- `format_cost_usd(cost)` — magnitude-aware USD formatting ($X.XXXX / $X.XXX / $X.XX).
- `get_model_pricing(model)` — returns the pricing dict entry or `None`.
- `known_models()` — sorted list of model IDs present in the pricing table.

**Data file:** `ama_kbqa/data/model_pricing.json` — 9 KIT chat models with OpenRouter
list prices. Generated (and re-generatable) by `scripts/fetch_model_pricing.py`:
- Pulls KIT model list live via `/models` when `KIT_API_KEY` is set; else uses a
  curated fallback list.
- Fetches public OpenRouter `/models` pricing.
- Maps each KIT model ID to its OpenRouter slug via a hand-verified override table
  plus a vendor heuristic.
- Safe to re-run at any time (rewrites the JSON file).

**Why illustrative, not actual:** KIT does not charge per token. Using OpenRouter
list prices gives a plausible real-world reference without misrepresenting usage.

## What is NOT on demo-bwcloud

These full-build features exist on `yannic-dev`/`main` but are absent on this branch:

| Feature | Absent because |
|---------|---------------|
| `pages/2_Batch_Processing.py` | Research tooling; removed in earlier demo commits |
| `pages/3_Evaluation.py` | Results dashboard; removed in earlier demo commits |
| `pages/4_Settings.py` + `settings_ui.py` | Replaced by minimal in-sidebar controls |
| `pages/5_Trace_Inspector.py` | Removed in earlier demo commits |
| `pages/1_Chat.py` | Moved to `ama_kbqa/frontend/chat.py` (root-URL single page) |
| OpenRouter / multi-provider selection | KIT-only on the demo |

## Related Docs

- [System/project_architecture.md](../System/project_architecture.md) — full-build architecture (canonical)
- [System/demo_bwcloud_frontend.md](../System/demo_bwcloud_frontend.md) — current-state reference for demo-build frontend
- [Decisions/live-trace-and-chat-unification.md](live-trace-and-chat-unification.md) — live trace pipeline (shared with full build)
- [Decisions/multiturn-direct-agent-conversation.md](multiturn-direct-agent-conversation.md) — multiturn session semantics (shared with full build)
