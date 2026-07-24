"""
Deployment-time overrides for adapter-declared ``GraphConfig`` / ``VectorConfig``.

Design: the adapter is the single place a KG *declares* its endpoint, named
graph, and vector-collection defaults (see ``KQAProAdapter._create_config``,
``SciQAAdapter._create_config``). Deployment environments (local dev vs.
Docker Compose, per-host overrides) still need to be able to change those
values without editing code, exactly like the legacy ``config.toml``-driven
getters in ``ama_kbqa/config.py`` used to. This module is that override layer,
factored out so a NEW KG never has to add a getter function to
``ama_kbqa/config.py`` — it only has to declare defaults in its adapter and,
optionally, add a ``[kg.<code>]`` section to ``config.toml``.

Resolution precedence for each field (highest wins):

1. ``AMA_KBQA_<CODE>_<FIELD>`` environment variable (e.g.
   ``AMA_KBQA_SCIQA_ENDPOINT``). Uppercased adapter code + uppercased field
   name. This is the generic override channel available to every KG,
   including ones added after this module was written.
2. ``[kg.<code>]`` section in ``config.toml`` (e.g. ``[kg.sciqa]
   endpoint = "..."``). Also generic — the extension point new KGs use
   instead of a bespoke getter.
3. Legacy KG-specific ``config.toml`` keys, kept ONLY for the two KGs that
   predate the generic ``[kg.*]`` section (KQAPro, SciQA) so existing
   ``config.toml`` / ``config.docker.toml`` deployments (including the
   Docker Compose demo) keep working unmodified. This table does not grow
   when a new KG is onboarded; new KGs use tier 2 or tier 1 instead.
4. The adapter's own declared default.
"""

from __future__ import annotations

import os
from dataclasses import replace
from typing import Any, Callable, Dict, Optional, Tuple, TypeVar

from ama_kbqa.config import load_config
from ama_kbqa.framework.config import GraphConfig, VectorConfig

T = TypeVar("T")

ENV_PREFIX = "AMA_KBQA_"

# code -> field -> (config.toml section, key). Closed set: KQAPro and SciQA
# only. See module docstring — new KGs must not be added here.
_LEGACY_GRAPH_KEYS: Dict[str, Dict[str, Tuple[str, str]]] = {
    "kqapro": {
        "endpoint": ("database", "virtuoso_endpoint"),
    },
    "sciqa": {
        "endpoint": ("database", "virtuoso_endpoint"),
        "graph_uri": ("search", "sciqa_virtuoso_graph"),
    },
}

_LEGACY_VECTOR_KEYS: Dict[str, Dict[str, Tuple[str, str]]] = {
    "kqapro": {
        "entity_collection": ("database", "collection_entities"),
        "relation_collection": ("database", "collection_relations"),
        "host": ("database", "qdrant_host"),
        "port": ("database", "qdrant_port"),
    },
    "sciqa": {
        "entity_collection": ("search", "sciqa_collection_entities"),
        "relation_collection": ("search", "sciqa_collection_relations"),
        "host": ("database", "qdrant_host"),
        "port": ("database", "qdrant_port"),
    },
}


def _env_var_name(code: str, field: str) -> str:
    return f"{ENV_PREFIX}{code.upper()}_{field.upper()}"


def _resolve_field(
    config: Dict[str, Any],
    code: str,
    field: str,
    default: T,
    legacy_keys: Dict[str, Dict[str, Tuple[str, str]]],
    parse: Callable[[str], T] = str,  # type: ignore[assignment]
) -> T:
    """Resolve a single field through the precedence chain described above."""
    env_raw = os.environ.get(_env_var_name(code, field))
    if env_raw is not None:
        try:
            return parse(env_raw)
        except (TypeError, ValueError):
            pass  # fall through to config.toml / default on a bad env value

    kg_section = config.get("kg", {}).get(code, {})
    if field in kg_section and kg_section[field] is not None:
        return kg_section[field]

    legacy = legacy_keys.get(code, {}).get(field)
    if legacy is not None:
        section, key = legacy
        value = config.get(section, {}).get(key)
        if value is not None:
            return value

    return default


def resolve_graph_config(code: str, default: GraphConfig) -> GraphConfig:
    """Resolve a ``GraphConfig`` for adapter ``code`` against config.toml/env overrides."""
    config = load_config()
    endpoint: str = _resolve_field(
        config, code, "endpoint", default.endpoint, _LEGACY_GRAPH_KEYS, str
    )
    graph_uri: Optional[str] = _resolve_field(
        config, code, "graph_uri", default.graph_uri, _LEGACY_GRAPH_KEYS, str
    )
    timeout_ms: int = _resolve_field(
        config, code, "timeout_ms", default.timeout_ms, _LEGACY_GRAPH_KEYS, int
    )
    return replace(default, endpoint=endpoint, graph_uri=graph_uri, timeout_ms=timeout_ms)


def resolve_vector_config(code: str, default: VectorConfig) -> VectorConfig:
    """Resolve a ``VectorConfig`` for adapter ``code`` against config.toml/env overrides.

    Note: ``entity_threshold``, ``relation_threshold``, and ``top_k_default``
    are intentionally NOT overridden here — no deployment-level override for
    those exists today (the KQAPro server reads a shared ``[search]
    score_threshold``; SciQA reads its own ``[sciqa]`` thresholds directly).
    Only the fields that previously had per-KG getters in
    ``ama_kbqa/config.py`` (collections, endpoint, graph) are in scope.
    """
    config = load_config()
    entity_collection: str = _resolve_field(
        config, code, "entity_collection", default.entity_collection, _LEGACY_VECTOR_KEYS, str
    )
    relation_collection: str = _resolve_field(
        config, code, "relation_collection", default.relation_collection, _LEGACY_VECTOR_KEYS, str
    )
    host: str = _resolve_field(
        config, code, "host", default.host, _LEGACY_VECTOR_KEYS, str
    )
    port: int = _resolve_field(
        config, code, "port", default.port, _LEGACY_VECTOR_KEYS, int
    )
    return replace(
        default,
        entity_collection=entity_collection,
        relation_collection=relation_collection,
        host=host,
        port=port,
    )
