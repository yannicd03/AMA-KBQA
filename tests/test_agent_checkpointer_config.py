"""Tests for the Phase 4 opt-in graph-engine checkpointer config keys:
``[agent].checkpointer`` / ``AMA_AGENT_CHECKPOINTER`` and
``[agent].checkpointer_path`` / ``AMA_AGENT_CHECKPOINTER_PATH``.

Mirrors the env-first/config-fallback/default pattern already established
for ``get_agent_engine`` (``ama_kbqa/config.py``).
"""

from __future__ import annotations

from ama_kbqa import config as config_module
from ama_kbqa.config import (
    DEFAULT_AGENT_CHECKPOINTER_PATH,
    get_agent_checkpointer,
    get_agent_checkpointer_path,
)


def _clear_env(monkeypatch):
    monkeypatch.delenv("AMA_AGENT_CHECKPOINTER", raising=False)
    monkeypatch.delenv("AMA_AGENT_CHECKPOINTER_PATH", raising=False)


# ---------------------------------------------------------------------------
# get_agent_checkpointer
# ---------------------------------------------------------------------------


def test_checkpointer_defaults_to_none_when_unset(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setattr(config_module, "load_config", lambda: {"agent": {}})

    assert get_agent_checkpointer() == "none"


def test_checkpointer_reads_from_config_toml(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setattr(config_module, "load_config", lambda: {"agent": {"checkpointer": "memory"}})

    assert get_agent_checkpointer() == "memory"


def test_checkpointer_env_var_overrides_config(monkeypatch):
    monkeypatch.setenv("AMA_AGENT_CHECKPOINTER", "sqlite")
    monkeypatch.setattr(config_module, "load_config", lambda: {"agent": {"checkpointer": "memory"}})

    assert get_agent_checkpointer() == "sqlite"


def test_checkpointer_unrecognized_value_falls_back_to_none(monkeypatch):
    monkeypatch.setenv("AMA_AGENT_CHECKPOINTER", "postgres")

    assert get_agent_checkpointer() == "none"


def test_checkpointer_env_var_case_insensitive_and_trimmed(monkeypatch):
    monkeypatch.setenv("AMA_AGENT_CHECKPOINTER", "  Memory  ")

    assert get_agent_checkpointer() == "memory"


def test_checkpointer_blank_env_var_falls_through_to_config(monkeypatch):
    monkeypatch.setenv("AMA_AGENT_CHECKPOINTER", "   ")
    monkeypatch.setattr(config_module, "load_config", lambda: {"agent": {"checkpointer": "sqlite"}})

    assert get_agent_checkpointer() == "sqlite"


# ---------------------------------------------------------------------------
# get_agent_checkpointer_path
# ---------------------------------------------------------------------------


def test_checkpointer_path_defaults_when_unset(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setattr(config_module, "load_config", lambda: {"agent": {}})

    assert get_agent_checkpointer_path() == DEFAULT_AGENT_CHECKPOINTER_PATH


def test_checkpointer_path_reads_from_config_toml(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setattr(
        config_module,
        "load_config",
        lambda: {"agent": {"checkpointer_path": "custom/checkpoints.sqlite"}},
    )

    assert get_agent_checkpointer_path() == "custom/checkpoints.sqlite"


def test_checkpointer_path_env_var_overrides_config(monkeypatch):
    monkeypatch.setenv("AMA_AGENT_CHECKPOINTER_PATH", "/tmp/override.sqlite")
    monkeypatch.setattr(
        config_module,
        "load_config",
        lambda: {"agent": {"checkpointer_path": "custom/checkpoints.sqlite"}},
    )

    assert get_agent_checkpointer_path() == "/tmp/override.sqlite"


# ---------------------------------------------------------------------------
# Real config.toml: sanity check the checked-in defaults are what phase 4
# documents (engine stays "legacy", checkpointer stays "none" — no behaviour
# change for anyone not opting in).
# ---------------------------------------------------------------------------


def test_real_config_toml_defaults_checkpointer_to_none(monkeypatch):
    _clear_env(monkeypatch)
    assert get_agent_checkpointer() == "none"
