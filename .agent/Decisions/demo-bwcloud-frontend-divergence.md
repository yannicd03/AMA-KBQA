# ADR: Demo Frontend Divergence

> **Note (2026-09-14):** This ADR was written on the `demo-bwcloud` branch and
> originally scoped to it. The underlying decisions (single-page shell, About
> dialog, in-chat agent picker, KIT-only sidebar controls, illustrative
> pricing) carried forward unchanged onto the `demo-v2`/`demo-hetzner` line
> (this worktree: `demo-v2-int`) when the demo frontend was rebuilt on top of
> `dev`. File paths below are still accurate for that line. **The "KIT-only
> sidebar controls" half no longer holds on `demo-v2-int` as of 2026-09-17 —
> see the 2026-09-17 addendum.** See the dated addenda at the bottom for what
> changed in the port and since, and
> [System/demo_bwcloud_frontend.md](../System/demo_bwcloud_frontend.md) for
> the current-state reference.

## Status

Historical decisions active since 2026-05-29 (originally five commits on
`demo-bwcloud`: `eae0a74`, `3755aec`, `5d475da`, `140eeb7`, `5d30df4`), carried
forward onto `demo-v2`/`demo-hetzner`. See the 2026-09-14 addendum for the
port and what has since changed.

## Context

The demo build is a **public demo** hosted for conference/research audiences
(originally bwCloud, now the shared Hetzner box — see
[SOP/hetzner_demo_deployment.md](../SOP/hetzner_demo_deployment.md)). The
full-build dashboard (multi-page Streamlit, Settings editor,
Evaluation/Batch pages, OpenRouter-accessible provider selection, Trace
Inspector standalone page) is inappropriate for a public endpoint:

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
- `pages/` directory — removed entirely on this line.

### 2. About dialog (commit `5d475da`)

**Chose:** A first-visit `st.dialog` ("About this demo", width="large") that auto-opens
once per browser session via `st.session_state["_about_seen"]`, with a floating "?"
button (CSS in `styling.py` under `.st-key-about_help_btn`) that reopens it at any time.

**Why:** Replaces the old sidebar landing markdown and provider/model branding. Gives
visitors immediate context without cluttering the chat UI. The "?" button is always
reachable without occupying sidebar space.

### 3. In-chat agent picker as a popover pill (commits `140eeb7`, `5d30df4`)

**Chose:** Agent selector moved out of the sidebar into a chat-bar popover "pill"
picker, rendered next to the chat input in both the landing and conversation views.

**Why:** Sidebar space is used for model/temperature controls. The agent picker is a
first-class interaction choice — making it a prominent pill next to the input matches
the mental model of "pick your agent, then type your question".

**Details:**
- Each option shows a `tagline` one-liner (field in `AGENT_INFO` in
  `ama_kbqa/frontend/utils/agent_factory.py`) and a checkmark on the active agent.
- Switching agents drops the persisted multiturn conversation (`persistent_agent`
  cleared from `st.session_state`).
- Emoji icons removed from the picker labels (commit `5d30df4`) for cleaner appearance;
  checkmark selection indicator retained.
- Sub-agents (KQAPro, SciQA) retain multiturn history within a session; the
  Orchestrator (in either dispatch mode, see the addendum below) remains stateless.

> **2026-09-14 update:** the picker originally had 3 entries (Orchestrator, KQAPro,
> SciQA); it now has 4 — "Orchestrator (Router)" and "Orchestrator (Federated)"
> replace the single "Orchestrator" entry, both resolving to the same `Orchestrator`
> class constructed with a different `federation` flag. This is a frontend-only
> extension of decision 3, motivated by the federated dispatch backend feature — see
> the addendum below and
> [Decisions/federated-dispatch-and-fusion.md](federated-dispatch-and-fusion.md).

### 4. KIT-only model dropdown and temperature in the sidebar (commit `3755aec`)

**Chose:** Move a minimal subset of model/temperature controls into the chat sidebar,
replacing the full Settings page.

**Settings page removed:** `pages/4_Settings.py` and `ama_kbqa/frontend/utils/settings_ui.py`.

**New module:** `ama_kbqa/frontend/utils/chat_controls.py` with:
- `filter_selectable_models(entries)` — keeps only locally-hosted KIT chat models
  (drops embedding/reranker/TTS/STT/image models, presets, and routing aliases).
- `available_models()` — fetches the live KIT `/models` endpoint (5-min TTL via
  `@st.cache_data`), falls back to a static list when offline.
- `price_caption(model)` — one-line illustrative price string for the sidebar.
- `apply_chat_settings(model, temperature)` — writes the chosen model and temperature
  into the in-memory config cache (`cfg_module._config_cache`). Session-only; never
  writes `config.toml` to disk. Switching models clears the persisted multiturn agent.

**Why KIT-only:** The public demo is KIT-funded; routing traffic through OpenRouter
or other providers must not be possible. The endpoint is pinned to KIT in
`apply_chat_settings`; provider selection is never exposed.

**Why session-only settings:** The demo runs as a shared public service. Writing
`config.toml` would persist one visitor's choices for all subsequent visitors.

> **2026-09-14 update:** the model list is now populated by **auto-discovery**
> against the live KIT `/models` endpoint (`_is_local_chat_model` metadata filter)
> rather than a hardcoded whitelist, and the config mutation is paired with
> `AMA_KBQA_CHAT_MODEL`/`AMA_KBQA_CHAT_TEMPERATURE` environment variables so the
> Orchestrator/specialist MCP subprocesses pick up the choice too. Neither changes
> the KIT-only / session-only rationale above. See
> `System/demo_bwcloud_frontend.md` and `SOP/hetzner_demo_deployment.md` §4 for the
> "Model not found" incident that motivated the env-var half.

> **2026-09-17 update — superseded on this branch:** the KIT-only rule above no
> longer describes `demo-v2-int`; its picker is now provider-aware. See the
> 2026-09-17 addendum at the bottom and
> [Decisions/demo-picker-provider-routing.md](demo-picker-provider-routing.md).

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

**Data file:** `ama_kbqa/data/model_pricing.json` — KIT chat models with OpenRouter
list prices. Generated (and re-generatable) by `scripts/fetch_model_pricing.py`.

**Why illustrative, not actual:** KIT does not charge per token. Using OpenRouter
list prices gives a plausible real-world reference without misrepresenting usage.

## What is NOT on this line

These full-build features exist on `dev`/`main` but are absent on the demo build:

| Feature | Absent because |
|---------|---------------|
| `pages/2_Batch_Processing.py` | Research tooling; removed in earlier demo commits |
| `pages/3_Evaluation.py` | Results dashboard; removed in earlier demo commits |
| `pages/4_Settings.py` + `settings_ui.py` | Replaced by minimal in-sidebar controls |
| `pages/5_Trace_Inspector.py` | Removed in earlier demo commits |
| `pages/1_Chat.py` | Moved to `ama_kbqa/frontend/chat.py` (root-URL single page) |
| OpenRouter / multi-provider selection | KIT-only on the demo |

## Related Docs

- [System/project_architecture.md](../System/project_architecture.md) — full-build (`dev`) architecture (canonical)
- [System/demo_bwcloud_frontend.md](../System/demo_bwcloud_frontend.md) — current-state reference for the demo-build frontend
- [Decisions/federated-dispatch-and-fusion.md](federated-dispatch-and-fusion.md) — backend federated dispatch decision behind the Router/Federated picker split
- [Decisions/demo-picker-provider-routing.md](demo-picker-provider-routing.md) — **current position on the model picker**, superseding decision 4's KIT-only rule on `demo-v2-int`
- [Decisions/live-trace-and-chat-unification.md](live-trace-and-chat-unification.md) — live trace pipeline (shared with full build)
- [Decisions/multiturn-direct-agent-conversation.md](multiturn-direct-agent-conversation.md) — multiturn session semantics (shared with full build)
- [SOP/hetzner_demo_deployment.md](../SOP/hetzner_demo_deployment.md) — deployment runbook, including the v2 staging/switch procedure

---

## Addendum 2026-09-14: ported onto `demo-v2`/`demo-hetzner`

The demo frontend described above was rebuilt on top of `dev` (bringing in the
`ama_kbqa/retrieval/` hybrid-search package, `framework/adapters/`, the `chatkit`
dependency, and `dev`'s one-round-trip Orchestrator routing — see
[System/orchestrator_routing.md](../System/orchestrator_routing.md)) rather than the
older pre-hybrid `main` snapshot `demo-bwcloud` had branched from. The demo UI landed
on the `dev`-based line via merge commit `4c68a93` ("Merge demo-v2-ui: demo frontend,
KIT-only config and Hetzner packaging"). None of decisions
1–5 above changed in substance during that port; only the two update notes inline
(decision 3's picker entry count, decision 4's model-discovery mechanism) reflect
concurrent, independently-motivated changes.

**One new decision from the port:** the Orchestrator picker entry was split into
"Orchestrator (Router)" and "Orchestrator (Federated)" (commit `7fe31a7`, "Split the
Orchestrator picker into Router and Federated entries") once federated multi-specialist
dispatch was ported from `feature/federated-retrieval` (commit `fc8e365`, documented in
full in [Decisions/federated-dispatch-and-fusion.md](federated-dispatch-and-fusion.md)).
This is additive to decision 3, not a reversal of it: the picker is still a chat-bar
popover pill with taglines and a checkmark; it just now has one more entry, backed by
`ORCHESTRATOR_MODES` (`agent_factory.py`) rather than a single hardcoded agent class.
The lifecycle figure (`orchestrator_svg.py`) was extended in the same window (commit
`a1064db`, "Light the orchestrator figure correctly for federated dispatch") to light
the Answer Combination node and both specialist dispatch edges when the active run is
Federated.

**Rejected alternative (considered, not built):** a single "Orchestrator" entry with a
sidebar toggle for federation, mirroring the model/temperature sidebar controls.
Rejected because the paper explicitly evaluates Router-mode dispatch and Federated
mode is experimental with different cost/latency characteristics (~2x tokens) — making
them two distinct, separately-labelled picker entries keeps that distinction visible
in the UI itself rather than hidden behind a toggle a visitor might not notice, and
keeps `AGENT_SUGGESTIONS` mode-specific (Federated's example questions are chosen to
plausibly span both graphs; Router's are single-domain).

---

## Addendum 2026-09-17: `demo-v2-int` is no longer the KIT-only branch

Decision 4's "KIT-only model dropdown" was accurate when written and remained
accurate through 2026-09-16. It no longer describes this branch. `demo-v2-int`
now carries a provider-aware picker: two DeepSeek presets routed through
OpenRouter (`deepseek/deepseek-v4-pro`, `deepseek/deepseek-v4.1-flash`, both
`[[frontend.chat_models]]` entries in `config.toml`) plus a free-text custom
OpenRouter model entry, with choice keys of the form `provider:model` rather
than bare ids.

[Decisions/demo-picker-provider-routing.md](demo-picker-provider-routing.md) is
the current position on the picker — rationale, custom-id rules, and the
subprocess-routing fix all live there. Decision 4 above is retained as the
superseded KIT-only decision, not as a description of current state.

**What changed is this branch's status, not a cross-branch reversal.** The
other demo lines were never uniformly KIT-only either: `demo-booth` has carried
its own provider-aware picker with 7 `[[frontend.chat_models]]` entries (5
OpenRouter, 2 direct-DeepSeek), and `demo-llamacpp` configures no cloud entries
at all, substituting local endpoints instead. Decision 4's "KIT-only" claim now
holds only for the older `demo-bwcloud` / `demo-hetzner` line it was written
for.

The consequence worth planning around: these branches each carry an
**independent implementation of the same picker symbols**, rather than one
shared implementation varied by per-branch config. Merging forward between them
is a reconciliation, not a mechanical merge — expect to choose between several
live versions of the picker instead of fast-forwarding one.
