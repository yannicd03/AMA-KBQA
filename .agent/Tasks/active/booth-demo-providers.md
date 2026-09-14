# PRD: Booth demo branch with OpenRouter and DeepSeek chat endpoints

**Status:** 🛠 In implementation (branch `demo-booth`, off `demo-v2-int` @ `ca90fe4`)
**Owner:** Yannic · **Written:** 2026-09-14 · **Target:** booth demo on Hetzner, Wednesday 2026-09-16
**Scope:** demo frontend, config plumbing for one new provider, deployment config. No agent-loop, prompt, or MCP tool changes.

## 1. Branch model (decided 2026-09-14)

| Branch | Role | Deployed where |
|---|---|---|
| `demo-v2-int` | Shared base: demo v2 with the live graph panel. Shared fixes land here first and are merged forward into both deliverable branches. | nowhere directly |
| `demo-public` | Public-facing demo (`amakbqa.yanlab.de`). KIT endpoint only, demo rate limits on. Identical to `demo-v2-int` today. | Not on Hetzner for the near future (user decision 2026-09-14). The v1 container currently serving the public hostname on Hetzner is untouched by this PRD. |
| `demo-booth` | Booth demo shown by the team (SEMANTiCS 2026). Adds OpenRouter and DeepSeek chat endpoints next to KIT, relaxed rate limits, booth About text. | Hetzner, compose project `amakbqa-next` (port 8504, hostname `amakbqa-next.yanlab.de`), from Wednesday 2026-09-16; API keys supplied by the user. |

Rule: never commit booth-only changes to `demo-v2-int` or `demo-public`; never commit public-only changes to `demo-booth`.

## 2. Goal

The booth presenter can pick, in the sidebar model picker, a chat model from three endpoints:

1. **KIT** (as today: auto-discovered local chat models, free, default),
2. **OpenRouter** (`https://openrouter.ai/api/v1`, key `OPENROUTER_API_KEY`), a curated list of tool-capable models,
3. **DeepSeek direct** (`https://api.deepseek.com`, key `DEEPSEEK_API_KEY`).

Switching the endpoint must reach the MCP tool-server subprocesses (they build their own chat client from their own config), or the specialists silently keep calling KIT with a foreign model id ("Model not found", the bug class fixed on 2026-09-14 for the model name). Embeddings, the reranker, and synthesis stay on KIT regardless of the chat pick; `KIT_API_KEY` therefore remains required.

Verified facts (2026-09-14):
- DeepSeek docs (`https://api-docs.deepseek.com/`, `/quick_start/pricing`): base URL `https://api.deepseek.com`, OpenAI-compatible chat completions with tool calls; current model ids `deepseek-flash` (DeepSeek-V4.1-Flash, 1M context, $0.30 in / $1.20 out per 1M at peak) and `deepseek-v4-pro` (DeepSeek-V4-Pro-0813, $1.32 / $3.96 peak; API service continues after 2026-09-14 per the docs). Legacy `deepseek-v4-flash` is still accepted as an alias.
- OpenRouter `GET /api/v1/models` (public, 447 entries today): entries carry `id`, `name`, `pricing.prompt` / `pricing.completion` (USD per token, strings) and `supported_parameters` (contains `"tools"` for tool-capable models). Verified present and tool-capable: `deepseek/deepseek-v4-flash`, `deepseek/deepseek-v4-pro`, `anthropic/claude-sonnet-4.5`, `openai/gpt-5`, `google/gemini-2.5-pro`.

## 3. Config

### 3.1 `config.toml` and `config.docker.toml` (both, identical additions)

```toml
[deepseek]
# Direct DeepSeek endpoint (OpenAI-compatible). Key: DEEPSEEK_API_KEY.
# Chat only; embeddings stay on KIT.
base_url = "https://api.deepseek.com"
chat_model = "deepseek-flash"
```

Under the existing `[frontend]` section, after `live_graph`:

```toml
# Booth build: chat endpoints offered next to the auto-discovered KIT models.
# An entry is shown only when its provider's API key env var is set. Prices
# are USD per 1M tokens and are what the booth account is actually billed
# (OpenRouter entries fall back to the live catalog price when omitted).
# `default = true` on one entry pre-selects it; otherwise the KIT default
# preference applies.
[[frontend.chat_models]]
provider = "openrouter"
id = "deepseek/deepseek-v4-flash"
name = "DeepSeek V4 Flash"

[[frontend.chat_models]]
provider = "openrouter"
id = "deepseek/deepseek-v4-pro"
name = "DeepSeek V4 Pro"

[[frontend.chat_models]]
provider = "openrouter"
id = "anthropic/claude-sonnet-4.5"
name = "Claude Sonnet 4.5"

[[frontend.chat_models]]
provider = "openrouter"
id = "openai/gpt-5"
name = "GPT-5"

[[frontend.chat_models]]
provider = "openrouter"
id = "google/gemini-2.5-pro"
name = "Gemini 2.5 Pro"

[[frontend.chat_models]]
provider = "deepseek"
id = "deepseek-flash"
name = "DeepSeek Flash (direct)"
prompt_usd_per_m = 0.30
completion_usd_per_m = 1.20

[[frontend.chat_models]]
provider = "deepseek"
id = "deepseek-v4-pro"
name = "DeepSeek V4 Pro (direct)"
prompt_usd_per_m = 1.32
completion_usd_per_m = 3.96
```

TOML note: `[[frontend.chat_models]]` tables must follow the `[frontend]` key/value lines and precede the next top-level section.

### 3.2 `ama_kbqa/config.py`

- `CHAT_PROVIDER_OVERRIDE_ENV_VAR = "AMA_KBQA_CHAT_PROVIDER"` next to the model/temperature override constants, same rationale (MCP subprocesses inherit `os.environ`).
- New `get_chat_provider() -> str`: env override if set, else `[llm].chat_provider`. Every current reader of `config["llm"]["chat_provider"]` (lines 119, 153, 249, 274, 528, 549, 675 at `ca90fe4`) goes through it. With the override set and `AMA_KBQA_CHAT_MODEL` unset, `get_chat_model_name()` returns the override provider's `chat_model`.
- `_get_api_key`: env map gains `"deepseek": "DEEPSEEK_API_KEY"`; `_PROVIDER_API_KEY_ENV_VARS` gains `DEEPSEEK_API_KEY`. `_create_client` needs no change (generic OpenAI client; no extra headers for DeepSeek).
- `get_frontend_chat_models() -> list[dict]`: the `[frontend].chat_models` list, validated (provider in `{"openrouter", "deepseek", "kit"}`, non-empty `id`; invalid entries dropped with a logged warning), never raises.
- `frontend/utils/config_editor.py::_PROVIDER_API_KEY_ENV` gains `deepseek`.
- `.env_example` gains `DEEPSEEK_API_KEY`.

## 4. Model picker (`ama_kbqa/frontend/utils/chat_controls.py`)

```python
@dataclass(frozen=True)
class ChatModelChoice:
    provider: str            # "kit" | "openrouter" | "deepseek"
    model: str               # provider-specific id, e.g. "anthropic/claude-sonnet-4.5"
    name: str                # display label
    prompt_usd_per_token: float | None = None
    completion_usd_per_token: float | None = None
    default: bool = False

    @property
    def key(self) -> str: return f"{self.provider}:{self.model}"

def available_choices() -> tuple[list[ChatModelChoice], list[str]]
    # (choices, notices). KIT choices exactly as `available_models()` today
    # (auto-discovery + offline fallback), then config entries in config order.
    # A config entry whose provider key env var is unset is dropped and one
    # notice per provider is produced ("OpenRouter models hidden: OPENROUTER_API_KEY not set").
    # OpenRouter entries are checked against the live catalog (`fetch_provider_models_meta("openrouter")`
    # or a raw fetch that keeps `pricing`; cached 5 min like the KIT fetch):
    # unknown ids are dropped with a notice; catalog `name` fills a missing name;
    # catalog pricing fills missing prices. Catalog unreachable => keep the
    # config entries as they are (no validation, no prices).
def default_choice(choices) -> ChatModelChoice   # first `default = true`, else DEFAULT_MODEL_PREFERENCE on KIT, else first
def display_choice(choice) -> str                # e.g. "Claude Sonnet 4.5 · OpenRouter", KIT entries keep today's label ("Mistral Small 4 · KIT" is acceptable)
def price_caption_for(choice) -> str | None      # config/catalog prices for non-KIT; today's illustrative table for KIT
def apply_chat_settings(model: str, temperature: float, provider: str = "kit") -> None
    # as today plus: cfg["llm"]["chat_provider"] = provider; cfg[provider]["chat_model"] = model;
    # os.environ[AMA_KBQA_CHAT_PROVIDER] = provider. Keeps the two-argument call working.
```

Keep `available_models()`, `default_model()`, `display_model_name()`, `price_caption()` working for existing callers and tests; the new functions are additive.

### 4.1 `chat.py` sidebar

- Selectbox over `choices` keyed by `choice.key`, `format_func=display_choice`, default from `default_choice`. Notices render as one `st.caption` line each under the picker.
- `st.session_state["_active_chat_model"]` stores the choice key (a provider switch drops the persistent agent exactly like a model switch does today).
- `apply_chat_settings(choice.model, temperature, provider=choice.provider)`.
- Price caption via `price_caption_for(choice)`. For non-KIT models the caption ends with "(billed)" so the presenter knows it is real money; the message footer keeps `~$x est.`.
- The message footer's `estimate_cost_usd(message["model"], ...)` must price non-KIT models: register the choice's per-token prices in a runtime registry (`pricing.register_runtime_pricing(model_id, prompt_per_token, completion_per_token)`; `get_model_pricing` consults it before the static table). Persist the provider on the message dict (`"provider": choice.provider`) for later display.

### 4.2 About dialog (`app.py`, booth only)

Replace the sentence "The cost is for intuition only: this demo runs on a free KIT-hosted endpoint, so nobody is actually billed." with: "KIT-hosted models are free to us, so their cost is illustrative. OpenRouter and DeepSeek models are billed at the list price shown next to the picker." Keep everything else.

### 4.3 Smoke script (`scripts/demo_smoke_ask.py`)

Model argument accepts `provider:model` (e.g. `openrouter:deepseek/deepseek-v4-flash`, `deepseek:deepseek-flash`); a bare id keeps meaning KIT. Print the provider in the result line.

## 5. Deployment config (booth stack on Hetzner)

- `docker-compose.hetzner-next.yml`: header comment updated (this is the booth stack built from `demo-booth`); `environment` gains `DEMO_MAX_QUERIES_PER_SESSION=500` and `DEMO_MIN_SECONDS_BETWEEN_QUERIES=1` (presenter usage); `env_file: ./.env` already passes the keys (the user adds `OPENROUTER_API_KEY` and `DEEPSEEK_API_KEY` to `~/amakbqa-next/.env` on the box).
- `.agent/SOP/hetzner_demo_deployment.md` §9: branch to stage is `demo-booth`; key list; smoke commands per provider; note that `assert_provider_api_key_present` only needs one key but the picker hides providers whose key is missing.

## 6. Non-goals

- No OpenRouter or DeepSeek embeddings; no provider switch for synthesis/judge.
- No per-session isolation of the provider override (same process-global caveat as the model override today).
- No changes to `demo-public` beyond the branch creation.

## 7. Tests

1. `tests/test_config_chat_provider.py`: `get_chat_provider()` env override wins; `get_chat_model_name()` follows the override provider's `chat_model`; `get_chat_client()` builds a DeepSeek client with `base_url` from `[deepseek]` and key from `DEEPSEEK_API_KEY` (monkeypatched env, no network); missing key raises `KeyError` naming the var; `get_frontend_chat_models()` drops invalid entries and never raises; both shipped tomls parse and contain the `[deepseek]` section and at least one `chat_models` entry per provider.
2. `tests/frontend/test_chat_controls.py` (extend): `available_choices` with no keys yields KIT only plus two notices; with `OPENROUTER_API_KEY` set and a fake catalog, unknown ids are dropped with a notice, names/prices are filled from the catalog; catalog unreachable keeps config entries; `DEEPSEEK_API_KEY` set yields the DeepSeek entries with config prices; `default_choice` honours `default = true`; `apply_chat_settings(..., provider="deepseek")` sets `AMA_KBQA_CHAT_PROVIDER`, `AMA_KBQA_CHAT_MODEL`, and the in-memory config; the two-argument call still pins KIT.
3. `tests/test_pricing.py` (extend): runtime registry beats the static table; unregistered model still returns None.
4. AppTest (`tests/frontend/test_agent_picker_apptest.py` style): page loads with the shipped config and no provider keys: no exception, the picker lists only KIT fallback models, and the two notices are present.
5. Smoke-script arg parsing unit test (factor the parsing into a small function).
6. Existing suites stay green: `uv run pytest -q`, `uv run ruff check .`.

## 8. Acceptance

- Local, with `OPENROUTER_API_KEY` from `.env` and `AMA_RETRIEVAL_RERANKER_ENABLED=false`: `scripts/demo_smoke_ask.py "KQAPro" openrouter:deepseek/deepseek-v4-flash` answers "Who is the director of Inception?" end to end, and the trace shows the specialist and its MCP server used OpenRouter (log line "Created chat client for provider 'openrouter'").
- Browser: picker shows KIT models plus the OpenRouter entries, DeepSeek entries hidden with the notice until `DEEPSEEK_API_KEY` is set; selecting an OpenRouter model and asking a question works; the footer shows a billed cost estimate.
- DeepSeek direct: **PASSED 2026-09-14** (key supplied, local, `AMA_RETRIEVAL_RERANKER_ENABLED=false`):
  - `demo_smoke_ask.py "KQAPro" deepseek:deepseek-flash` -> `[OK] KQAPro (15s) deepseek:deepseek-flash`, answer names Christopher Nolan.
  - `demo_smoke_ask.py "Orchestrator (Router)" deepseek:deepseek-flash` -> `[OK] Orchestrator (Router) (14s) deepseek:deepseek-flash`, answer names Ulm.
  - `Created chat client for provider 'deepseek' at https://api.deepseek.com` appears in both the agent process and the MCP server subprocess; embeddings stayed on KIT.
  - OpenRouter regression re-run after the fix: `[OK] KQAPro (17s) openrouter:deepseek/deepseek-v4-flash`.

  Blocker found and fixed on the way (2026-09-14). The first DeepSeek run failed on the
  second turn of the tool loop with `400 ... The 'reasoning_content' in the thinking mode
  must be passed back to the API`. DeepSeek's direct API runs thinking mode by default and,
  for any request carrying `tools`, requires every assistant message to be returned with its
  `reasoning_content`. Our fast path synthesises assistant `tool_calls` messages
  (`base_agent._append_fast_path_tool_messages`) that have no reasoning to return, and the
  main loop does not preserve it either. Rather than thread `reasoning_content` through the
  loop, chat calls now send `thinking.type = "disabled"` for the DeepSeek provider via
  `config.get_chat_extra_body()`, opt-out-able with `[deepseek] thinking_enabled = true`
  (default false in both shipped tomls). OpenRouter is unaffected: it reconciles this itself,
  which is why the OpenRouter path passed before the fix.

## 9. Implementation plan

| Phase | Agent | Deliverables | Done when |
|---|---|---|---|
| 1 | Opus "booth-providers" | §3, §4, §4.2, §4.3, §5 compose change, tests §7 | full suite green, ruff clean, local OpenRouter smoke passes |
| 2 | orchestrator | review, acceptance §8 (OpenRouter), commit, push `demo-booth` | acceptance recorded here |
| 3 | `docs-agent` / `wiki` | SOP §9 booth section, System doc picker section, ADR "three-branch demo split"; wiki active-branches + TODO | links here |
