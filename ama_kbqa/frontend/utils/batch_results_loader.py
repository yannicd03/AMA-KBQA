"""Utilities for loading and parsing benchmark_results/ directories."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


BENCHMARK_RESULTS_DIR = Path(__file__).resolve().parents[3] / "benchmark_results"


def list_batches() -> list[dict[str, Any]]:
    """Return batch run entries sorted newest-first.

    Scans ``benchmark_results/<YYYY-MM-DD-N>/<agent>/<model>/`` directories.
    Each entry that contains a ``summary.json`` is returned as a dict with
    keys: ``label``, ``timestamp``, ``agent``, ``model``, ``path``.

    Returns:
        List of run descriptor dicts, newest first.
    """
    if not BENCHMARK_RESULTS_DIR.exists():
        return []

    runs: list[dict[str, Any]] = []

    for ts_dir in BENCHMARK_RESULTS_DIR.iterdir():
        if not ts_dir.is_dir():
            continue
        for agent_dir in ts_dir.iterdir():
            if not agent_dir.is_dir():
                continue
            for model_dir in agent_dir.iterdir():
                if not model_dir.is_dir():
                    continue
                if (model_dir / "summary.json").exists():
                    runs.append({
                        "label": f"{ts_dir.name} / {agent_dir.name} / {model_dir.name}",
                        "timestamp": ts_dir.name,
                        "agent": agent_dir.name,
                        "model": model_dir.name,
                        "path": str(model_dir),
                    })

    # Sort by timestamp descending (newest first)
    runs.sort(key=lambda r: r["timestamp"], reverse=True)
    return runs


def _resolve_path(run: dict[str, Any] | str) -> Path:
    """Resolve a run descriptor or legacy batch name to a directory path."""
    if isinstance(run, dict):
        return Path(run["path"])
    # Legacy fallback: treat as direct folder name under benchmark_results
    return BENCHMARK_RESULTS_DIR / run


def load_summary(run: dict[str, Any] | str) -> dict[str, Any]:
    """Load summary.json for a run."""
    path = _resolve_path(run) / "summary.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_results(run: dict[str, Any] | str) -> list[dict[str, Any]]:
    """Load results.json (per-question details) for a run."""
    path = _resolve_path(run) / "results.json"
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_judgments(run: dict[str, Any] | str) -> dict[str, Any] | None:
    """Load judgments.json for a run, if it exists."""
    path = _resolve_path(run) / "judgments.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_detailed_log(run: dict[str, Any] | str) -> str | None:
    """Load detailed_log.txt for a run, if it exists."""
    path = _resolve_path(run) / "detailed_log.txt"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return f.read()
