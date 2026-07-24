"""Tests for deployment-time GraphConfig/VectorConfig resolution.

Covers the fix for the SEMANTiCS-2026-paper audit finding that
``GraphConfig``/``VectorConfig`` were populated by adapters but never read
at runtime — servers instead read separate, KG-specific getters in
``ama_kbqa/config.py``. This suite verifies:

1. With no overrides, resolution returns the adapter's own declared
   defaults (``resolve_graph_config``/``resolve_vector_config`` are true
   pass-throughs absent config.toml/env).
2. Override precedence: env var > generic ``[kg.<code>]`` config.toml
   section > legacy KG-specific config.toml keys > adapter default.
3. Regression guard: under both the repo's real ``config.toml`` and
   ``config.docker.toml``, ``KQAProAdapter``/``SciQAAdapter``
   ``.resolved_config`` produces exactly the values the old
   ``get_virtuoso_endpoint`` / ``get_collection_entities`` /
   ``get_collection_relations`` / ``get_sciqa_collection_entities`` /
   ``get_sciqa_collection_relations`` / ``get_sciqa_virtuoso_graph``
   getters produced from the same file.
"""
from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

import ama_kbqa.config as cfg_module
from ama_kbqa.config import (
    get_collection_entities,
    get_collection_relations,
    get_sciqa_collection_entities,
    get_sciqa_collection_relations,
    get_sciqa_virtuoso_graph,
    get_virtuoso_endpoint,
)
from ama_kbqa.framework.adapters.kqapro_adapter import KQAProAdapter
from ama_kbqa.framework.adapters.sciqa_adapter import SciQAAdapter
from ama_kbqa.framework.adapters.resolve import (
    resolve_graph_config,
    resolve_vector_config,
)
from ama_kbqa.framework.config import GraphConfig, VectorConfig

REPO_ROOT = Path(__file__).resolve().parents[2]

ALL_OVERRIDE_ENV_VARS = [
    "AMA_KBQA_KQAPRO_ENDPOINT",
    "AMA_KBQA_KQAPRO_GRAPH_URI",
    "AMA_KBQA_KQAPRO_ENTITY_COLLECTION",
    "AMA_KBQA_KQAPRO_RELATION_COLLECTION",
    "AMA_KBQA_SCIQA_ENDPOINT",
    "AMA_KBQA_SCIQA_GRAPH_URI",
    "AMA_KBQA_SCIQA_ENTITY_COLLECTION",
    "AMA_KBQA_SCIQA_RELATION_COLLECTION",
    "AMA_KBQA_NEWKG_ENDPOINT",
    "AMA_KBQA_NEWKG_ENTITY_COLLECTION",
]


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Clear any override env vars so tests are hermetic regardless of order."""
    for var in ALL_OVERRIDE_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    yield


@pytest.fixture
def empty_config(monkeypatch):
    """Pin config.toml to an empty dict (no [database]/[search]/[kg] sections)."""
    monkeypatch.setattr(cfg_module, "_config_cache", {})
    return {}


class TestAdapterDeclaredDefaults:
    """With no config.toml or env overrides, resolution is a pass-through."""

    def test_kqapro_graph_defaults_unchanged(self, empty_config):
        default = KQAProAdapter().config.graph
        resolved = resolve_graph_config("kqapro", default)
        assert resolved.endpoint == default.endpoint
        assert resolved.graph_uri is None
        assert resolved.timeout_ms == default.timeout_ms

    def test_kqapro_vector_defaults_unchanged(self, empty_config):
        default = KQAProAdapter().config.vectors
        resolved = resolve_vector_config("kqapro", default)
        assert resolved.entity_collection == default.entity_collection
        assert resolved.relation_collection == default.relation_collection
        assert resolved.host == default.host
        assert resolved.port == default.port

    def test_sciqa_graph_defaults_unchanged(self, empty_config):
        default = SciQAAdapter().config.graph
        resolved = resolve_graph_config("sciqa", default)
        assert resolved.endpoint == default.endpoint
        assert resolved.graph_uri == "http://sciqa.org/kg"

    def test_unknown_kg_code_has_no_legacy_mapping(self, empty_config):
        """A brand-new KG with no legacy entry still resolves to its own default."""
        default = GraphConfig(endpoint="http://newkg.example/sparql")
        resolved = resolve_graph_config("newkg", default)
        assert resolved.endpoint == "http://newkg.example/sparql"


class TestOverridePrecedence:
    """env > [kg.<code>] > legacy config.toml keys > adapter default."""

    def test_legacy_toml_key_overrides_default(self, monkeypatch):
        monkeypatch.setattr(
            cfg_module,
            "_config_cache",
            {"database": {"virtuoso_endpoint": "http://legacy:8890/sparql"}},
        )
        default = GraphConfig(endpoint="http://adapter-default:8890/sparql")
        resolved = resolve_graph_config("kqapro", default)
        assert resolved.endpoint == "http://legacy:8890/sparql"

    def test_generic_kg_section_overrides_legacy(self, monkeypatch):
        monkeypatch.setattr(
            cfg_module,
            "_config_cache",
            {
                "database": {"virtuoso_endpoint": "http://legacy:8890/sparql"},
                "kg": {"kqapro": {"endpoint": "http://generic:8890/sparql"}},
            },
        )
        default = GraphConfig(endpoint="http://adapter-default:8890/sparql")
        resolved = resolve_graph_config("kqapro", default)
        assert resolved.endpoint == "http://generic:8890/sparql"

    def test_env_overrides_everything(self, monkeypatch):
        monkeypatch.setattr(
            cfg_module,
            "_config_cache",
            {
                "database": {"virtuoso_endpoint": "http://legacy:8890/sparql"},
                "kg": {"kqapro": {"endpoint": "http://generic:8890/sparql"}},
            },
        )
        monkeypatch.setenv("AMA_KBQA_KQAPRO_ENDPOINT", "http://env-wins:8890/sparql")
        default = GraphConfig(endpoint="http://adapter-default:8890/sparql")
        resolved = resolve_graph_config("kqapro", default)
        assert resolved.endpoint == "http://env-wins:8890/sparql"

    def test_new_kg_onboards_via_generic_section_only(self, monkeypatch):
        """A new KG never touches ama_kbqa/config.py; [kg.<code>] is enough."""
        monkeypatch.setattr(
            cfg_module,
            "_config_cache",
            {
                "kg": {
                    "newkg": {
                        "entity_collection": "newkg-entities",
                        "relation_collection": "newkg-relations",
                    }
                }
            },
        )
        default = VectorConfig(
            entity_collection="newkg_entities_default",
            relation_collection="newkg_relations_default",
        )
        resolved = resolve_vector_config("newkg", default)
        assert resolved.entity_collection == "newkg-entities"
        assert resolved.relation_collection == "newkg-relations"

    def test_new_kg_env_override(self, monkeypatch):
        monkeypatch.setattr(cfg_module, "_config_cache", {})
        monkeypatch.setenv("AMA_KBQA_NEWKG_ENTITY_COLLECTION", "env-newkg-entities")
        default = VectorConfig(
            entity_collection="newkg_entities_default",
            relation_collection="newkg_relations_default",
        )
        resolved = resolve_vector_config("newkg", default)
        assert resolved.entity_collection == "env-newkg-entities"

    def test_sciqa_graph_uri_overridable_via_generic_section(self, monkeypatch):
        monkeypatch.setattr(
            cfg_module,
            "_config_cache",
            {"kg": {"sciqa": {"graph_uri": "http://staging.sciqa.org/kg"}}},
        )
        default = SciQAAdapter().config.graph
        resolved = resolve_graph_config("sciqa", default)
        assert resolved.graph_uri == "http://staging.sciqa.org/kg"

    def test_invalid_int_env_falls_back(self, monkeypatch):
        """A malformed env override doesn't crash resolution; it's ignored."""
        monkeypatch.setattr(cfg_module, "_config_cache", {})
        monkeypatch.setenv("AMA_KBQA_NEWKG_TIMEOUT_MS", "not-a-number")
        default = GraphConfig(endpoint="http://x/sparql", timeout_ms=12345)
        resolved = resolve_graph_config("newkg", default)
        assert resolved.timeout_ms == 12345


class TestKQAProSciQANamedGraphIsolation:
    """Guard the specific behaviors called out in the task: SciQA's named
    graph for FROM-clause isolation, and KQAPro's default-graph behavior."""

    def test_kqapro_resolved_graph_uri_stays_none_by_default(self, empty_config):
        resolved = KQAProAdapter().resolved_config
        assert resolved.graph.graph_uri is None

    def test_sciqa_resolved_graph_uri_stays_named_graph_by_default(self, empty_config):
        resolved = SciQAAdapter().resolved_config
        assert resolved.graph.graph_uri == "http://sciqa.org/kg"


class TestRegressionAgainstOldGetters:
    """Both adapters must resolve to exactly what the deprecated per-KG
    getters produced, under both the real dev config.toml and the real
    Docker config.docker.toml — a regression guard for the getter -> adapter
    migration."""

    @pytest.fixture(autouse=True)
    def _fresh_cache(self, monkeypatch):
        # Force load_config() to re-parse from whatever _config_cache we set,
        # rather than reusing a value left over by another test module.
        monkeypatch.setattr(cfg_module, "_config_cache", None)
        yield

    def _assert_resolution_matches_old_getters(self):
        kqapro_resolved = KQAProAdapter().resolved_config
        assert kqapro_resolved.graph.endpoint == get_virtuoso_endpoint()
        assert kqapro_resolved.vectors.entity_collection == get_collection_entities()
        assert kqapro_resolved.vectors.relation_collection == get_collection_relations()

        sciqa_resolved = SciQAAdapter().resolved_config
        assert sciqa_resolved.graph.endpoint == get_virtuoso_endpoint()
        assert sciqa_resolved.graph.graph_uri == get_sciqa_virtuoso_graph()
        assert sciqa_resolved.vectors.entity_collection == get_sciqa_collection_entities()
        assert sciqa_resolved.vectors.relation_collection == get_sciqa_collection_relations()

    def test_matches_old_getters_under_dev_config(self, monkeypatch):
        with open(REPO_ROOT / "config.toml", "rb") as f:
            dev_config = tomllib.load(f)
        monkeypatch.setattr(cfg_module, "_config_cache", dev_config)
        self._assert_resolution_matches_old_getters()

    def test_matches_old_getters_under_docker_config(self, monkeypatch):
        with open(REPO_ROOT / "config.docker.toml", "rb") as f:
            docker_config = tomllib.load(f)
        monkeypatch.setattr(cfg_module, "_config_cache", docker_config)
        self._assert_resolution_matches_old_getters()

    def test_docker_endpoint_differs_from_dev_endpoint(self, monkeypatch):
        """Sanity check that the two fixture files actually differ, so the
        two tests above aren't accidentally comparing identical values."""
        with open(REPO_ROOT / "config.toml", "rb") as f:
            dev_config = tomllib.load(f)
        with open(REPO_ROOT / "config.docker.toml", "rb") as f:
            docker_config = tomllib.load(f)
        assert (
            dev_config["database"]["virtuoso_endpoint"]
            != docker_config["database"]["virtuoso_endpoint"]
        )
