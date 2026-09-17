"""The free-text ("OpenRouter (custom)") model entry.

The picker's custom entry carries no model id of its own: the client sends its
key plus the id the visitor typed, and the server validates that id by *shape*
rather than against the catalog — the whole point is to reach a model this
build has never heard of. These tests pin both halves: the id reaches
``apply_chat_settings`` verbatim, and nothing unusable gets that far.
"""

from __future__ import annotations

import pytest

from ama_kbqa.api import meta

from .conftest import CUSTOM_KEY, MODEL, read_sse

TYPED = "anthropic/claude-sonnet-4.5"


# ---------------------------------------------------------------------------
# /api/meta
# ---------------------------------------------------------------------------

def test_meta_offers_the_custom_entry_last(custom_harness):
    models = custom_harness.client.get("/api/meta").json()["models"]
    entry = models[-1]
    assert entry["id"] == CUSTOM_KEY
    assert entry["provider"] == "openrouter"
    # The marker the React app switches the text field on.
    assert entry["custom"] is True
    # No price is invented for an entry that has no model yet.
    assert entry["price"] is None


def test_real_models_carry_no_custom_marker(custom_harness):
    models = custom_harness.client.get("/api/meta").json()["models"]
    for entry in models[:-1]:
        assert "custom" not in entry


def test_the_custom_entry_is_never_the_default(custom_harness):
    # It cannot answer anything on its own, so a build must not open on it.
    assert custom_harness.client.get("/api/meta").json()["default_model"] != CUSTOM_KEY


# ---------------------------------------------------------------------------
# The happy path
# ---------------------------------------------------------------------------

def test_typed_id_reaches_apply_chat_settings_verbatim(custom_harness):
    run_id, frames = custom_harness.run(model=CUSTOM_KEY, custom_model=TYPED)

    # Verbatim, on the entry's provider — not the sentinel, not a preset.
    assert custom_harness.applied == [(TYPED, 1.0, "openrouter")]

    done = frames[-1]["data"]
    assert done["status"] == "done"
    assert done["model"] == TYPED
    assert done["model_key"] == f"openrouter:{TYPED}"
    assert done["provider"] == "openrouter"

    record = custom_harness.client.get(f"/api/runs/{run_id}").json()
    assert (record["model"], record["provider"]) == (TYPED, "openrouter")


def test_surrounding_whitespace_is_trimmed(custom_harness):
    custom_harness.run(model=CUSTOM_KEY, custom_model=f"  {TYPED}\n")
    assert custom_harness.applied == [(TYPED, 1.0, "openrouter")]


def test_no_cost_is_invented_for_an_unknown_id(custom_harness):
    _run_id, frames = custom_harness.run(model=CUSTOM_KEY, custom_model=TYPED)
    # Nobody knows what an arbitrary id costs. A blank footer is right;
    # a guessed number would not be.
    assert frames[-1]["data"]["cost"] is None


def test_followups_are_keyed_by_the_typed_id(custom_harness):
    h = custom_harness

    def ask(typed):
        resp = h.post(model=CUSTOM_KEY, custom_model=typed)
        assert resp.status_code == 201, resp.text
        assert read_sse(h.client, resp.json()["run_id"])[-1]["event"] == "done"
        return resp.json()["continuation"]

    assert ask(TYPED) is False
    assert ask(TYPED) is True          # same id: a follow-up turn
    assert ask("openai/gpt-5") is False  # different id: a fresh agent
    assert ask("openai/gpt-5") is True
    assert len(h.created) == 2


# ---------------------------------------------------------------------------
# Rejection
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("typed", ["", "   ", "\t\n"])
def test_empty_or_whitespace_id_is_rejected(custom_harness, typed):
    resp = custom_harness.post(model=CUSTOM_KEY, custom_model=typed)
    assert resp.status_code == 400
    assert "model id" in resp.json()["detail"]
    # Nothing was applied and no agent was built.
    assert custom_harness.applied == []
    assert custom_harness.created == []


def test_a_missing_custom_model_field_is_rejected(custom_harness):
    resp = custom_harness.post(model=CUSTOM_KEY)
    assert resp.status_code == 400
    assert "model id" in resp.json()["detail"]
    assert custom_harness.applied == []


@pytest.mark.parametrize("typed", [
    "has space/model",
    'quote"/model',
    "semi;colon",
    "new\nline",
    "__custom__",            # the sentinel itself is not a model
    "/leading-slash",
    "-leading-dash",
    "x" * 201,               # over the length cap
])
def test_malformed_ids_are_rejected(custom_harness, typed):
    resp = custom_harness.post(model=CUSTOM_KEY, custom_model=typed)
    assert resp.status_code == 400
    assert custom_harness.applied == []
    assert custom_harness.created == []


def test_the_sentinel_never_reaches_a_provider(custom_harness):
    """The placeholder's own key must not be usable as a model id: posting it
    without an id is a 400, never a request for a model called __custom__."""
    resp = custom_harness.post(model=CUSTOM_KEY, custom_model="__custom__")
    assert resp.status_code == 400
    assert all("__custom__" != applied[0] for applied in custom_harness.applied)


def test_custom_model_is_ignored_for_a_normal_pick(custom_harness):
    """A stale field from a client that switched entries mid-edit must not
    fail an otherwise valid question."""
    kit_key = f"kit:{MODEL}"
    resp = custom_harness.post(model=kit_key, custom_model="junk not a model")
    assert resp.status_code == 201
    assert custom_harness.applied == [(MODEL, 1.0, "kit")]


def test_an_over_long_field_is_refused_by_the_request_schema(custom_harness):
    resp = custom_harness.post(model=CUSTOM_KEY, custom_model="a" * 5000)
    assert resp.status_code == 400
    assert custom_harness.created == []


# ---------------------------------------------------------------------------
# The resolver, directly
# ---------------------------------------------------------------------------

PLACEHOLDER = meta.ModelOption(
    key=CUSTOM_KEY, provider="openrouter", model="__custom__",
    name="OpenRouter (custom)", price=None, custom=True,
)


def test_custom_model_option_builds_an_ordinary_option():
    option = meta.custom_model_option(PLACEHOLDER, f" {TYPED} ")
    assert (option.key, option.provider, option.model) == (
        f"openrouter:{TYPED}", "openrouter", TYPED,
    )
    assert option.name == TYPED
    assert option.price is None
    # The result is a real pick, not another placeholder.
    assert option.custom is False


@pytest.mark.parametrize("typed", [None, "", "  ", "bad id", "x" * 201])
def test_custom_model_option_rejects_unusable_ids(typed):
    with pytest.raises(meta.InvalidCustomModel):
        meta.custom_model_option(PLACEHOLDER, typed)


def test_custom_model_option_accepts_the_shapes_openrouter_uses():
    for typed in (
        "deepseek/deepseek-v4-pro",
        "deepseek/deepseek-v4.1-flash",
        "anthropic/claude-sonnet-4.5",
        "openai/gpt-5",
        "qwen/qwen3-235b-a22b:free",
        "meta-llama/llama-3.3-70b-instruct",
    ):
        assert meta.custom_model_option(PLACEHOLDER, typed).model == typed


def test_the_error_message_tells_the_user_what_to_type():
    with pytest.raises(meta.InvalidCustomModel) as excinfo:
        meta.custom_model_option(PLACEHOLDER, "")
    # An example, not a stack trace: this string is shown in the UI.
    assert "deepseek/deepseek-v4-pro" in str(excinfo.value)
