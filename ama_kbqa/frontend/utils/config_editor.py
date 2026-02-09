"""Read/write config.toml with backup support for the Settings page."""

from __future__ import annotations

import shutil
import tomllib
from pathlib import Path

import toml

CONFIG_PATH = Path(__file__).resolve().parents[3] / "config.toml"


def load_config_raw() -> dict:
    """Load config.toml directly (bypasses the cached loader)."""
    with open(CONFIG_PATH, "rb") as f:
        return tomllib.load(f)


def save_config(config: dict, backup: bool = True) -> None:
    """Write config dict to config.toml.

    Args:
        config: Full configuration dictionary to write.
        backup: If True, create a .bak copy before overwriting.
    """
    if backup and CONFIG_PATH.exists():
        shutil.copy2(CONFIG_PATH, CONFIG_PATH.with_suffix(".toml.bak"))

    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        toml.dump(config, f)

    # Invalidate the cached config so the app picks up the new values
    _invalidate_config_cache()


def apply_to_session(config: dict) -> None:
    """Apply config dict to the in-memory cache without writing to disk.

    This modifies the running process's config but does not persist.
    """
    import ama_kbqa.config as cfg_module
    cfg_module._config_cache = config


def _invalidate_config_cache() -> None:
    """Reset the config module's cache so the next load reads from disk."""
    try:
        import ama_kbqa.config as cfg_module
        cfg_module._config_cache = None
    except Exception:
        pass
