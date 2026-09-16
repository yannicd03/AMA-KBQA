"""Tests for the [frontend].settings_level flag and its env override.

The flag decides how much the React demo's settings panel shows: "minimal"
(model, temperature, view toggles) or "full" (additionally the endpoint list
and the read-only build facts). It is what makes the public demo differ from
the booth/local builds without forking the React code.

Default "minimal" for the same reason live_graph defaults False: an old
config.toml with no key, or a deployment that never opted in, must not start
advertising which endpoint it talks to. Both shipped toml files opt in.
"""

from __future__ import annotations

import pytest

import ama_kbqa.config as cfg


@pytest.fixture(autouse=True)
def _no_env_override(monkeypatch):
    monkeypatch.delenv(cfg.FRONTEND_SETTINGS_LEVEL_ENV, raising=False)


def _with_config(monkeypatch, config: dict) -> None:
    monkeypatch.setattr(cfg, "_config_cache", config)


class TestGetFrontendSettingsLevel:
    def test_defaults_to_minimal_without_the_section(self, monkeypatch):
        _with_config(monkeypatch, {"llm": {}})
        assert cfg.get_frontend_settings_level() == "minimal"

    def test_defaults_to_minimal_without_the_key(self, monkeypatch):
        _with_config(monkeypatch, {"frontend": {"live_graph": True}})
        assert cfg.get_frontend_settings_level() == "minimal"

    @pytest.mark.parametrize("raw,expected", [
        ("full", "full"), ("minimal", "minimal"), ("FULL", "full"), (" full ", "full"),
    ])
    def test_reads_the_toml(self, monkeypatch, raw, expected):
        _with_config(monkeypatch, {"frontend": {"settings_level": raw}})
        assert cfg.get_frontend_settings_level() == expected

    @pytest.mark.parametrize("raw", ["", "verbose", "1", "none"])
    def test_unknown_values_fall_back_to_minimal(self, monkeypatch, raw):
        _with_config(monkeypatch, {"frontend": {"settings_level": raw}})
        assert cfg.get_frontend_settings_level() == "minimal"

    def test_env_override_wins_over_the_toml(self, monkeypatch):
        _with_config(monkeypatch, {"frontend": {"settings_level": "minimal"}})
        monkeypatch.setenv(cfg.FRONTEND_SETTINGS_LEVEL_ENV, "full")
        assert cfg.get_frontend_settings_level() == "full"

    def test_env_override_can_cut_it_back(self, monkeypatch):
        _with_config(monkeypatch, {"frontend": {"settings_level": "full"}})
        monkeypatch.setenv(cfg.FRONTEND_SETTINGS_LEVEL_ENV, "minimal")
        assert cfg.get_frontend_settings_level() == "minimal"

    def test_invalid_env_override_falls_back_to_minimal(self, monkeypatch):
        # Not "keep the toml value": an operator who typed the override meant
        # to change something, and minimal is the safe reading.
        _with_config(monkeypatch, {"frontend": {"settings_level": "full"}})
        monkeypatch.setenv(cfg.FRONTEND_SETTINGS_LEVEL_ENV, "everything")
        assert cfg.get_frontend_settings_level() == "minimal"


class TestShippedConfigs:
    def test_both_shipped_toml_files_declare_a_valid_level(self):
        """Guards against the key being dropped by a merge, or misspelled.

        Deliberately not "== full": demo-public ships the same test with
        settings_level = "minimal".
        """
        import tomllib
        from pathlib import Path

        for name in ("config.toml", "config.docker.toml"):
            path = Path(cfg.REPO_ROOT) / name
            data = tomllib.loads(path.read_text())
            level = data.get("frontend", {}).get("settings_level")
            assert level in cfg.FRONTEND_SETTINGS_LEVELS, f"{name}: {level!r}"
