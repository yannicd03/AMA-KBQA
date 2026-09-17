---
type: decision
status: accepted
date: 2026-09-16
summary: The demo picker offers DeepSeek through OpenRouter (not the direct API) plus a free-text OpenRouter id validated by shape and never minted into the catalog; choice keys carry their provider, and AMA_KBQA_CHAT_PROVIDER carries the pick across the MCP subprocess boundary so specialists follow the parent.
addresses: [issue/demo-picker-was-kit-only, issue/chat-provider-lost-across-mcp-subprocess, issue/notice-dedupe-not-pushed-to-v2-int]
affects: [branch/demo-v2-int, feature/provider-aware-model-picker, issue/apply-chat-settings-process-global-concurrency]
relates: [react-frontend-second-ui, demo-bwcloud-frontend-divergence]
evidence:
  - "commit 69cd5a9"
  - "ama_kbqa/frontend/utils/chat_controls.py:270"
  - "ama_kbqa/frontend/utils/chat_controls.py:213"
  - "ama_kbqa/api/meta.py:656"
  - "config.toml:131"
---

# ADR: Demo Model Picker — Provider-Carrying Keys, Custom Ids, and Subprocess Routing

**Status:** Accepted (shipped 2026-09-16 on `demo-v2-int`, commit `69cd5a9`).

## Related Docs
- [System/demo_react_frontend.md](../System/demo_react_frontend.md) — the picker contract, `/api/meta` shape, settings-panel rows
- [Decisions/react-frontend-second-ui.md](./react-frontend-second-ui.md) — the additive React frontend and its process-global override trade-off
- [Decisions/cooperative-run-cancellation.md](./cooperative-run-cancellation.md) — shipped in the same commit; shares these files
- [SOP/hetzner_demo_deployment.md](../SOP/hetzner_demo_deployment.md) §4 — the KIT auto-discovery and `AMA_KBQA_CHAT_MODEL` work this builds on

---

## Context

The demo picker was KIT-only. A booth presenter had no way to show the system
running on a frontier model, and no way to try an arbitrary model without a code
change and a rebuild.

Adding non-KIT models raises three questions that each have a non-obvious
answer: which endpoint serves DeepSeek, what a free-text model id is allowed to
do, and whether picking a provider in the parent process actually moves the
sub-agents.

## Decision 1 — Choice keys are `provider:model`, not bare ids

The same model id exists behind more than one endpoint, so a bare id is
ambiguous once more than one provider is offered. Every picker key is now
`"provider:model"` (`chat_controls.CUSTOM_CHOICE_KEY` follows the same rule),
and that key is what `/api/meta` emits as a model's `id` and what
`POST /api/runs` sends back.

This is also what makes conversation continuity break *correctly*: a session's
stored multiturn agent is keyed by `(agent_name, model_key)`
(`runs.py::StoredAgent`), so switching provider while keeping the same model id
starts a fresh agent instead of silently continuing a conversation against a
different endpoint.

## Decision 2 — Custom model ids are validated by shape, and never minted into the catalog

The picker's last entry is a free-text field (`openrouter:__custom__`, flagged
`custom: true` in `/api/meta`) where the presenter types any OpenRouter model
id. `POST /api/runs` carries it in a separate `custom_model` field alongside the
placeholder key.

- **Validated by shape, not against a catalog** (`meta.custom_model_option`,
  `_CUSTOM_MODEL_RE`, 200-character cap). Checking the typed id against the live
  catalog would defeat the point of the entry, which is to reach something the
  catalog does not list.
- **Never minted into the catalog.** The catalog is TTL-cached in-process, so a
  minted id would be dropped on the next refresh and, worse, would leak between
  sessions until it was.
- **No price is shown** rather than a guessed one. `pricing.register_runtime_pricing`
  ignores a half-known price for exactly this reason: the honest footer for a
  model nobody has a price for is no cost at all.
- A `custom_model` sent alongside any *other* pick is ignored rather than
  rejected, so a stale field from a client that switched entries mid-edit cannot
  fail a perfectly valid question.

## Decision 3 — DeepSeek presets route through OpenRouter, not the direct DeepSeek API

Verified against both vendors' live catalogs on 2026-09-16:

- `deepseek/deepseek-v4.1-flash` exists on OpenRouter. The direct DeepSeek API
  serves only `deepseek-flash` and `deepseek-v4-pro` — there is **no v4.1-flash
  direct**, so a direct-API route could not offer the same two presets.
- Independently of catalog coverage: the direct DeepSeek API runs every model in
  thinking mode, and then requires each assistant message to be replayed with
  its `reasoning_content`. This tool loop does not synthesise that field, so the
  *second* turn of any question would fail with a 400.

A `DEEPSEEK_API_KEY` now exists in `.env` and is **deliberately unused**; it is
not a leftover.

The two presets are config, not code: `[[frontend.chat_models]]` entries in
`config.toml` / `config.docker.toml`. An entry is hidden when its provider's key
is unset, when the live OpenRouter catalog no longer knows the id, or when the
catalog says the model cannot call tools (every agent here answers by calling
tools, so a non-tool-calling model would fail on the first question). When the
catalog is unreachable the configured entries are kept unvalidated and unpriced
— a flaky network must not empty the picker.

> **Config hazard, worth repeating here:** these are TOML *array tables*. Every
> plain `[frontend]` key (`live_graph`, `settings_level`) must stay **above** the
> first `[[frontend.chat_models]]` header, or it silently becomes a key of that
> array entry instead of a `[frontend]` key.

## Decision 4 — `AMA_KBQA_CHAT_PROVIDER` carries the pick across the subprocess boundary

**This is the most operationally valuable finding of the change.**

Specialist and orchestrator MCP tool servers are spawned as subprocesses with
`env=os.environ.copy()` and **load their own `config.toml` independently**.
Mutating this process's in-memory config therefore never reaches them. Before
this fix, picking an OpenRouter model left every sub-agent talking to KIT while
the parent talked to OpenRouter.

The failure was invisible: questions still got answered, because KIT still had a
working model. Nothing errored.

`config.get_chat_provider()` now checks `AMA_KBQA_CHAT_PROVIDER` first and falls
back to `config.toml`, and every reader of `[llm].chat_provider` goes through it
— joining the existing `AMA_KBQA_CHAT_MODEL` / `AMA_KBQA_CHAT_TEMPERATURE`
overrides from the Hetzner model work.

**Proven by a control experiment in the container**, not by inspection: with the
variable stripped, the child process carried model
`deepseek/deepseek-v4-pro` but resolved `base_url` to KIT; with it exported, the
child resolved to OpenRouter.

`[llm].embedding_provider` is deliberately **not** routed through this override —
embeddings, reranking and synthesis stay where `config.toml` puts them no matter
which chat model the visitor picks.

## Trade-offs accepted

- **The override is process-global**, exactly like the existing model and
  temperature overrides, now extended to env vars. A server handling concurrent
  demo sessions applies a pick to all of them. This is the same known limitation
  recorded in `react-frontend-second-ui.md` trade-off #2, widened rather than
  fixed.
- **`openrouter-chat`'s settings-panel status `ok` means the key is present, not
  that the endpoint answered.** Probing was rejected deliberately: it would put a
  network round-trip in front of the demo's first screen on every `/api/meta`
  call. An expired or revoked key still renders `ok`.
- A provider without its key stays **visible** with `status: "no_key"` rather
  than disappearing, because hiding it answers "can I use DeepSeek?" with
  silence.

## Known gaps (unverified as of 2026-09-16)

- **No real billed question has been asked** through a DeepSeek preset or a
  custom id. No end-to-end answer from either exists; the wiring is verified by
  tests and by the container control experiment above, not by a completed run.
- **The custom-model field has never been rendered.** No browser check — Chrome
  automation is broken on this machine (pre-existing regression, see
  `System/demo_react_frontend.md`).
