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

os.environ.setdefault(
    "AMA_KBQA_LOG_DIR",
    tempfile.mkdtemp(prefix="ama-kbqa-test-logs-"),
)
