"""Utilities for loading and parsing batch_results/ directories."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


BATCH_RESULTS_DIR = Path(__file__).resolve().parents[3] / "batch_results"


def list_batches() -> list[str]:
    """Return batch folder names sorted newest-first (by folder name descending).

    Only includes folders that contain a summary.json.
    """
    if not BATCH_RESULTS_DIR.exists():
        return []

    batches = []
    for p in BATCH_RESULTS_DIR.iterdir():
        if p.is_dir() and (p / "summary.json").exists():
            batches.append(p.name)

    # Sort descending so newest batches (highest number) come first
    batches.sort(key=lambda n: n.lower(), reverse=True)
    return batches


def load_summary(batch_name: str) -> dict[str, Any]:
    """Load summary.json for a batch."""
    path = BATCH_RESULTS_DIR / batch_name / "summary.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_results(batch_name: str) -> list[dict[str, Any]]:
    """Load results.json (per-question details) for a batch."""
    path = BATCH_RESULTS_DIR / batch_name / "results.json"
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_judgments(batch_name: str) -> dict[str, Any] | None:
    """Load judgments.json for a batch, if it exists."""
    path = BATCH_RESULTS_DIR / batch_name / "judgments.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
