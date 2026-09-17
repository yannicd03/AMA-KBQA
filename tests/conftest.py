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


@pytest.fixture(autouse=True)
def _restore_chat_overrides():
    """Undo the process-global env writes ``apply_chat_settings`` makes.

    ``chat_controls.apply_chat_settings`` points the MCP tool-server
    subprocesses at the picked endpoint by writing AMA_KBQA_CHAT_PROVIDER /
    _MODEL / _TEMPERATURE with a plain ``os.environ[...] = ...``. Any test that
    calls it therefore leaks those into every test that runs afterwards.

    ``monkeypatch.delenv(name, raising=False)`` does NOT protect against this,
    which is the trap worth recording: pytest registers an undo only when the
    variable *already existed*, so a var that starts unset is simply deleted
    with no restore recorded, and a later plain assignment survives the test.

    That leak is exactly what made a stale ``AMA_KBQA_CHAT_PROVIDER=kit``
    silently win over the monkeypatched ``load_config`` in
    ``tests/test_config_max_retries.py`` — the test passed alone and failed in
    the full suite. Restoring the real values here kills the whole class.
    """
    # Imported late: this module runs before AMA_KBQA_LOG_DIR is honoured by
    # the server modules, so nothing heavy may be imported at its top level.
    import ama_kbqa.config as cfg

    names = (
        cfg.CHAT_PROVIDER_OVERRIDE_ENV_VAR,
        cfg.CHAT_MODEL_OVERRIDE_ENV_VAR,
        cfg.CHAT_TEMPERATURE_OVERRIDE_ENV_VAR,
    )
    saved = {name: os.environ.get(name) for name in names}
    yield
    for name, value in saved.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value
