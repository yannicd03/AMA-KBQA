"""Session-wide test isolation.

The three MCP server modules configure a loguru file sink at *import* time
(`ama_kbqa/server/{kqapro,sciqa,orchestrator}_server.py`). Any test that
imports one of them therefore appends to the repo's real `logs/*.log`,
interleaving fixture strings such as "boom", "bad" and "localhost:59999"
with production log lines. That noise was mistaken for evidence of failed
benchmark runs during a 2026-07-25 investigation.

Redirect the sink before those modules can be imported. This must run at
conftest import time, not inside a fixture: the servers read the variable
while their module body executes, which happens on first import from a test
module, i.e. after conftest is loaded but before any fixture runs.
"""

import os
import tempfile

import pytest

os.environ.setdefault(
    "AMA_KBQA_LOG_DIR",
    tempfile.mkdtemp(prefix="ama-kbqa-test-logs-"),
)

# The demo's model picker (chat_controls.apply_chat_settings) writes these
# straight into os.environ on purpose: MCP tool-server subprocesses inherit
# the environment and that is the only channel that reaches them. Any test
# that exercises the picker, directly or by running the Streamlit page under
# AppTest, therefore leaves them set for the rest of the session. Since
# config.get_chat_provider() / get_chat_model_name() read them first, a leaked
# value silently redirects a *later* test away from the config it monkeypatched
# (observed: AMA_KBQA_CHAT_PROVIDER=kit from an AppTest run breaking the
# llamacpp stub in test_config_max_retries.py). Snapshot and restore around
# every test so ordering cannot matter.
_CHAT_OVERRIDE_ENV_VARS = (
    "AMA_KBQA_CHAT_PROVIDER",
    "AMA_KBQA_CHAT_MODEL",
    "AMA_KBQA_CHAT_TEMPERATURE",
)


@pytest.fixture(autouse=True)
def _isolate_chat_override_env():
    saved = {name: os.environ.get(name) for name in _CHAT_OVERRIDE_ENV_VARS}
    yield
    for name, value in saved.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value
