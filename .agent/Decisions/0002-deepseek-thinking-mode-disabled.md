# ADR 0002: Disable DeepSeek thinking mode on chat calls

## Related Docs
- [Decisions/0001-three-branch-demo-split.md](0001-three-branch-demo-split.md) — the `demo-booth` branch this fix ships on
- [System/demo_bwcloud_frontend.md](../System/demo_bwcloud_frontend.md) "Model picker (booth build)" — where the DeepSeek-direct endpoint is exposed to the presenter
- [SOP/hetzner_demo_deployment.md](../SOP/hetzner_demo_deployment.md) §10 — the deploy-time gotcha writeup
- `Tasks/active/booth-demo-providers.md` §8 — the acceptance run where this was found and fixed

## Status
Accepted, 2026-09-14. Implemented same day (commit `f35b2a7`).

## Context

`demo-booth` added a DeepSeek-direct chat endpoint (`https://api.deepseek.com`,
OpenAI-compatible) to the model picker alongside KIT and OpenRouter. The first
local acceptance run of a DeepSeek-direct question (`demo_smoke_ask.py "KQAPro"
deepseek:deepseek-flash`) failed on the **second** turn of the tool-calling
loop with:

```
400 ... The `reasoning_content` in the thinking mode must be passed back to the API
```

Root cause: DeepSeek's direct API runs every model in thinking mode by
default. Once a request carries `tools` (which every call in this system
does), the API requires each assistant message in the conversation history to
be echoed back with its `reasoning_content` field intact. This agent's fast
tool-call path (`base_agent._append_fast_path_tool_messages`) synthesizes
assistant `tool_calls` messages itself, with no reasoning content to return —
and the main loop does not thread `reasoning_content` through at all. The
result: any DeepSeek-direct question that reaches a second turn (i.e. every
question that actually calls a tool and gets a result back) fails.

OpenRouter was unaffected by the same underlying model family
(`deepseek/deepseek-v4-flash` via OpenRouter passed acceptance before this fix
was found) because OpenRouter reconciles the reasoning-content requirement on
its own side before forwarding to DeepSeek.

## Decision

Chat calls routed to the `deepseek` provider send `{"thinking": {"type":
"disabled"}}` in the request's `extra_body`, via a new
`config.get_chat_extra_body()` helper. It returns `{}` for every other
provider, so every existing call site can merge it unconditionally
(`call_params.setdefault("extra_body", {}).update(extra)`) without changing
behavior for KIT or OpenRouter. Wired into every chat-completion call site
that can reach the deepseek provider: `base_agent._llm_call`,
`base_agent._llm_call_text_only`, the classification call in
`base_agent.py`, `orchestrator_agent.agent.Orchestrator._create_with_retry`
(routing + fusion), and `orchestrator_server.extract_semantics`. The
synthesis call path was deliberately left untouched — synthesis stays on KIT
regardless of the chat-provider pick (see ADR 0001 and the System doc), so it
never routes to DeepSeek.

Opt-out: `[deepseek] thinking_enabled = true` in `config.toml` /
`config.docker.toml` (default `false` in both shipped files) restores
thinking mode, for single-turn experiments where the second-turn failure
mode doesn't apply and the reasoning trace might be wanted.

## Alternatives considered

**Thread `reasoning_content` through the tool-call loop instead of disabling
thinking.** Rejected for this deadline: it would require capturing and
replaying reasoning content through the fast path's synthesized tool-call
messages and the main assistant-message history, a change that touches the
shared conversation-building code used by every provider, not just DeepSeek.
Riskier to get right two days before the booth than a provider-scoped
request-body flag. Left as a future option if DeepSeek's reasoning trace is
ever wanted on stage — noted here rather than in a separate deferred-task doc
since it's a small, contained follow-up.

**Only fix single-turn call sites, leave multi-turn broken.** Not viable:
every demo question that calls a tool (which is effectively all of them)
requires the second turn to succeed. There is no useful subset of DeepSeek
usage that avoids it.

## Consequences

- DeepSeek-direct questions on `demo-booth` never show a reasoning trace in
  the UI, even though the model is capable of one. Acceptable trade for a
  live demo: reliability over a trace nobody was planning to show on stage
  anyway (see the helper's docstring, `ama_kbqa/config.py`
  `get_chat_extra_body`).
- The `extra_body` merge pattern (`setdefault(...).update(...)`, never
  overwrite) is now required at every future chat-call site that might touch
  a provider needing special request-body fields, so it doesn't clobber
  OpenRouter's `provider` routing-preferences key, which lives under the same
  `extra_body` dict. `orchestrator_server.extract_semantics` merges once and
  reuses the same `call_params` across its own internal retries, rather than
  re-deriving it per attempt.
- This fix is scoped to `demo-booth` only by virtue of ADR 0001 (only
  `demo-booth` exposes the DeepSeek-direct picker entry at all) — but the
  underlying `config.py` / `base_agent.py` / `orchestrator_agent/agent.py` /
  `orchestrator_server.py` changes are provider-gated (`if provider !=
  "deepseek": return {}`), so they would be inert, not harmful, if ever
  merged forward onto `demo-v2-int` or `demo-public` without the picker that
  exposes DeepSeek in the first place.
