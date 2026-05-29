#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["httpx>=0.27"]
# ///
"""Build the illustrative model-pricing table for the chat UI.

The KIT endpoint does not bill us, so we have no real per-token prices. To still
show an "estimated cost" under each chat answer, we map every KIT chat model to
the equivalent model on OpenRouter and borrow OpenRouter's public list price.
The numbers are therefore *illustrative*, not amounts anyone is charged.

What this script does:
  1. Get the list of KIT chat models. Tries the live ``/models`` endpoint
     (needs ``KIT_API_KEY``); falls back to the curated ``KNOWN_KIT_MODELS``
     list baked in below so the script still works without a KIT key.
  2. Fetch the public OpenRouter model catalogue (no key required).
  3. Map each KIT model id to an OpenRouter slug via ``KIT_TO_OPENROUTER``
     overrides, with a vendor heuristic as a fallback for new models.
  4. Write ``ama_kbqa/data/model_pricing.json``.

Run from the repo root:  ``uv run scripts/fetch_model_pricing.py``
Refresh on a host that has a real KIT key to pick up newly added models.
"""

from __future__ import annotations

import json
import os
import re
import sys
import tomllib
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "config.toml"
OUT_PATH = REPO_ROOT / "ama_kbqa" / "data" / "model_pricing.json"
OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"

# Curated KIT chat models, used when the live /models endpoint is unreachable
# (e.g. no KIT_API_KEY locally). Mirrors the KIT entries in benchmark_agents.py.
KNOWN_KIT_MODELS = [
    "kit.gpt-oss-120b",
    "kit.qwen3-vl-235b-a22b-instruct",
    "azure.gpt-4.1-mini",
    "azure.o4-mini",
    "kit.mixtral-8x22b-instruct",
    "kit.minimax-m2.1-229b",
    "kit.minimax-m2.7-229b",
    "kit.gemma4-31b-it",
    "kit.qwen3.5-397b-A17b",
]

# Hand-verified KIT-id -> OpenRouter-slug mapping. The KIT side annotates
# parameter counts (``-229b``, ``-397b``) and uses ``gemma4`` where OpenRouter
# writes ``gemma-4``, so an exact string match is not possible. Keep this in
# sync when KIT adds models; anything not listed falls back to the heuristic.
KIT_TO_OPENROUTER = {
    "kit.gpt-oss-120b": "openai/gpt-oss-120b",
    "kit.qwen3-vl-235b-a22b-instruct": "qwen/qwen3-vl-235b-a22b-instruct",
    "azure.gpt-4.1-mini": "openai/gpt-4.1-mini",
    "azure.o4-mini": "openai/o4-mini",
    "kit.mixtral-8x22b-instruct": "mistralai/mixtral-8x22b-instruct",
    "kit.minimax-m2.1-229b": "minimax/minimax-m2.1",
    "kit.minimax-m2.7-229b": "minimax/minimax-m2.7",
    "kit.gemma4-31b-it": "google/gemma-4-31b-it",
    "kit.qwen3.5-397b-A17b": "qwen/qwen3.5-397b-a17b",
}

# Vendor prefixes used by the heuristic fallback (longest hint wins).
_VENDOR_HINTS = [
    ("gpt-oss", "openai/"),
    ("gpt-", "openai/"),
    ("o4-", "openai/"),
    ("o3-", "openai/"),
    ("gemma", "google/"),
    ("gemini", "google/"),
    ("qwen", "qwen/"),
    ("mixtral", "mistralai/"),
    ("mistral", "mistralai/"),
    ("minimax", "minimax/"),
    ("llama", "meta-llama/"),
    ("deepseek", "deepseek/"),
]


def _norm(s: str) -> str:
    """Lowercase, drop the provider prefix and all non-alphanumerics."""
    s = s.split(".", 1)[-1] if "." in s else s
    return re.sub(r"[^a-z0-9]", "", s.lower())


def fetch_kit_models() -> tuple[list[str], str]:
    """Return ``(model_ids, source)`` from the live KIT endpoint or the fallback."""
    try:
        with open(CONFIG_PATH, "rb") as f:
            cfg = tomllib.load(f)
        base_url = cfg.get("kit", {}).get("base_url")
    except (OSError, tomllib.TOMLDecodeError):
        base_url = None

    api_key = os.getenv("KIT_API_KEY")
    if base_url and api_key and api_key != "your_kit_api_key":
        try:
            resp = httpx.get(
                f"{base_url.rstrip('/')}/models",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=15.0,
            )
            resp.raise_for_status()
            data = resp.json().get("data", [])
            ids = sorted({m["id"] for m in data if isinstance(m, dict) and m.get("id")})
            if ids:
                return ids, "live KIT /models endpoint"
        except (httpx.HTTPError, ValueError) as e:
            print(f"  live KIT fetch failed ({e}); using KNOWN_KIT_MODELS", file=sys.stderr)
    else:
        print("  no usable KIT_API_KEY; using KNOWN_KIT_MODELS", file=sys.stderr)
    return list(KNOWN_KIT_MODELS), "curated KNOWN_KIT_MODELS fallback"


def fetch_openrouter_pricing() -> dict[str, dict]:
    """Map OpenRouter slug -> ``{prompt, completion}`` USD-per-token floats."""
    resp = httpx.get(OPENROUTER_MODELS_URL, timeout=30.0)
    resp.raise_for_status()
    out: dict[str, dict] = {}
    for m in resp.json().get("data", []):
        slug = m.get("id")
        pricing = m.get("pricing") or {}
        if not slug:
            continue
        try:
            out[slug] = {
                "prompt": float(pricing.get("prompt", 0) or 0),
                "completion": float(pricing.get("completion", 0) or 0),
            }
        except (TypeError, ValueError):
            continue
    return out


def match_slug(kit_id: str, or_pricing: dict[str, dict]) -> str | None:
    """Resolve a KIT id to an OpenRouter slug: override first, then heuristic."""
    if kit_id in KIT_TO_OPENROUTER:
        slug = KIT_TO_OPENROUTER[kit_id]
        return slug if slug in or_pricing else None

    target = _norm(kit_id)
    bare = kit_id.split(".", 1)[-1].lower()
    vendor = next((v for hint, v in _VENDOR_HINTS if hint in bare), None)

    candidates = []
    for slug in or_pricing:
        if slug.endswith(":free") or "/" not in slug:
            continue
        if vendor and not slug.startswith(vendor):
            continue
        body = _norm(slug.split("/", 1)[1])
        if body == target or body.startswith(target) or target.startswith(body):
            candidates.append(slug)
    # Prefer the shortest (most exact) slug.
    return min(candidates, key=len) if candidates else None


def main() -> int:
    print("Fetching KIT models...", file=sys.stderr)
    kit_models, kit_source = fetch_kit_models()
    print(f"  {len(kit_models)} models from {kit_source}", file=sys.stderr)

    print("Fetching OpenRouter pricing...", file=sys.stderr)
    or_pricing = fetch_openrouter_pricing()
    print(f"  {len(or_pricing)} OpenRouter models", file=sys.stderr)

    models: dict[str, dict] = {}
    matched = 0
    for kit_id in kit_models:
        slug = match_slug(kit_id, or_pricing)
        if slug:
            price = or_pricing[slug]
            models[kit_id] = {
                "openrouter_id": slug,
                "prompt_usd_per_token": price["prompt"],
                "completion_usd_per_token": price["completion"],
                "matched": True,
            }
            matched += 1
            print(f"  ✓ {kit_id} -> {slug}", file=sys.stderr)
        else:
            models[kit_id] = {
                "openrouter_id": None,
                "prompt_usd_per_token": None,
                "completion_usd_per_token": None,
                "matched": False,
            }
            print(f"  ✗ {kit_id} -> (no OpenRouter match)", file=sys.stderr)

    out = {
        "note": (
            "Illustrative prices only. The KIT endpoint does not charge us; these "
            "USD-per-token figures are OpenRouter list prices for the equivalent "
            "model, used to show an estimated cost per answer. Regenerate with "
            "scripts/fetch_model_pricing.py."
        ),
        "source": "openrouter.ai/api/v1/models",
        "kit_model_source": kit_source,
        "currency": "USD",
        "unit": "per_token",
        "models": dict(sorted(models.items())),
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(
        f"\nWrote {OUT_PATH.relative_to(REPO_ROOT)} "
        f"({matched}/{len(kit_models)} models priced).",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
