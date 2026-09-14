"""Tests for the [frontend].live_graph flag and its env override.

The flag gates a layout change (the chat page goes from `centered` to a
two-column `wide` page), so getting the default wrong is not cosmetic: a
deployment that never opted in, or an older config.toml with no [frontend]
section at all, must keep the pre-feature single-column page. Hence default
False in the getter even though both shipped toml files set it true.

The env override exists for the same reason the AMA_RETRIEVAL_* ones do: a
container mounts its config.toml read-only, so turning the panel off in a
deployment has to be possible without editing that file.
"""

from __future__ import annotations

import pytest

import ama_kbqa.config as cfg


@pytest.fixture(autouse=True)
def _no_env_override(monkeypatch):
    monkeypatch.delenv(cfg.FRONTEND_LIVE_GRAPH_ENV, raising=False)


def _with_config(monkeypatch, config: dict) -> None:
    monkeypatch.setattr(cfg, "_config_cache", config)


class TestGetFrontendConfig:
    def test_missing_section_is_an_empty_dict(self, monkeypatch):
        _with_config(monkeypatch, {"llm": {}})
        assert cfg.get_frontend_config() == {}

    def test_section_is_returned_as_a_copy(self, monkeypatch):
        _with_config(monkeypatch, {"frontend": {"live_graph": True}})
        section = cfg.get_frontend_config()
        assert section == {"live_graph": True}
        section["live_graph"] = False
        assert cfg.get_frontend_config() == {"live_graph": True}


class TestGetLiveGraphEnabled:
    def test_defaults_to_false_without_the_section(self, monkeypatch):
        _with_config(monkeypatch, {"llm": {}})
        assert cfg.get_live_graph_enabled() is False

    def test_defaults_to_false_without_the_key(self, monkeypatch):
        _with_config(monkeypatch, {"frontend": {}})
        assert cfg.get_live_graph_enabled() is False

    def test_reads_true_from_the_toml(self, monkeypatch):
        _with_config(monkeypatch, {"frontend": {"live_graph": True}})
        assert cfg.get_live_graph_enabled() is True

    @pytest.mark.parametrize("raw", ["0", "false", "no", "off", "", "nonsense"])
    def test_env_override_can_turn_it_off(self, monkeypatch, raw):
        _with_config(monkeypatch, {"frontend": {"live_graph": True}})
        monkeypatch.setenv(cfg.FRONTEND_LIVE_GRAPH_ENV, raw)
        assert cfg.get_live_graph_enabled() is False

    @pytest.mark.parametrize("raw", ["1", "true", "TRUE", "yes", "on"])
    def test_env_override_can_turn_it_on(self, monkeypatch, raw):
        _with_config(monkeypatch, {"frontend": {"live_graph": False}})
        monkeypatch.setenv(cfg.FRONTEND_LIVE_GRAPH_ENV, raw)
        assert cfg.get_live_graph_enabled() is True


class TestShippedConfigs:
    def test_both_shipped_toml_files_enable_the_panel(self):
        """The demo line ships with the panel on; this guards against the
        [frontend] section being dropped by a merge."""
        import tomllib
        from pathlib import Path

        for name in ("config.toml", "config.docker.toml"):
            path = Path(cfg.REPO_ROOT) / name
            data = tomllib.loads(path.read_text())
            assert data.get("frontend", {}).get("live_graph") is True, name
