"""Tests for framework configuration classes."""

import pytest
import json
from ama_kbqa.framework.config import (
    NamespaceConfig,
    VectorConfig,
    GraphConfig,
    PromptConfig,
    KnowledgeGraphConfig,
    create_default_kqapro_config,
    create_default_sciqa_config,
)


class TestNamespaceConfig:
    """Tests for NamespaceConfig dataclass."""

    def test_create_namespace_config(self):
        """Test creating a NamespaceConfig."""
        ns = NamespaceConfig(
            entity_prefix="http://example.org/entity/",
            property_prefix="http://example.org/property/",
            attribute_prefix="http://example.org/attribute/",
            qualifier_prefix="http://example.org/qualifier/",
            sparql_prefixes="PREFIX ex: <http://example.org/entity/>",
            prefix_map={"ex": "http://example.org/entity/"},
        )

        assert ns.entity_prefix == "http://example.org/entity/"
        assert ns.property_prefix == "http://example.org/property/"
        assert ns.qualifier_prefix == "http://example.org/qualifier/"
        assert "PREFIX" in ns.sparql_prefixes

    def test_get_full_entity_uri(self):
        """Test getting full entity URI."""
        ns = NamespaceConfig(
            entity_prefix="http://example.org/entity/",
            property_prefix="http://example.org/property/",
            attribute_prefix="http://example.org/attribute/",
        )

        # From ID
        uri = ns.get_full_entity_uri("Q123")
        assert uri == "http://example.org/entity/Q123"

        # Already full URI
        uri = ns.get_full_entity_uri("http://other.org/entity/Q456")
        assert uri == "http://other.org/entity/Q456"

    def test_get_full_property_uri(self):
        """Test getting full property URI."""
        ns = NamespaceConfig(
            entity_prefix="http://example.org/entity/",
            property_prefix="http://example.org/property/",
            attribute_prefix="http://example.org/attribute/",
        )

        uri = ns.get_full_property_uri("P123")
        assert uri == "http://example.org/property/P123"

    def test_extract_id_from_uri(self):
        """Test extracting ID from URI."""
        ns = NamespaceConfig(
            entity_prefix="http://example.org/entity/",
            property_prefix="http://example.org/property/",
            attribute_prefix="http://example.org/attribute/",
        )

        # Entity URI
        entity_id = ns.extract_id_from_uri("http://example.org/entity/Q123")
        assert entity_id == "Q123"

        # Property URI
        prop_id = ns.extract_id_from_uri("http://example.org/property/P456")
        assert prop_id == "P456"

        # Unknown URI - extract from last segment
        unknown_id = ns.extract_id_from_uri("http://other.org/resource/X789")
        assert unknown_id == "X789"

        # Hash URI
        hash_id = ns.extract_id_from_uri("http://example.org/schema#Entity")
        assert hash_id == "Entity"


class TestVectorConfig:
    """Tests for VectorConfig dataclass."""

    def test_create_vector_config(self):
        """Test creating a VectorConfig."""
        vc = VectorConfig(
            entity_collection="entities",
            relation_collection="relations",
            entity_threshold=0.75,
            relation_threshold=0.70,
            top_k_default=10,
            host="localhost",
            port=6333,
        )

        assert vc.entity_collection == "entities"
        assert vc.relation_collection == "relations"
        assert vc.entity_threshold == 0.75
        assert vc.top_k_default == 10

    def test_vector_config_defaults(self):
        """Test VectorConfig default values."""
        vc = VectorConfig(
            entity_collection="entities",
            relation_collection="relations",
        )

        assert vc.entity_threshold == 0.70
        assert vc.relation_threshold == 0.65
        assert vc.top_k_default == 5
        assert vc.host == "localhost"
        assert vc.port == 6333


class TestGraphConfig:
    """Tests for GraphConfig dataclass."""

    def test_create_graph_config(self):
        """Test creating a GraphConfig."""
        gc = GraphConfig(
            endpoint="http://localhost:8890/sparql",
            graph_uri="http://example.org/graph",
            supports_reification=True,
            has_temporal_data=True,
            timeout_ms=60000,
        )

        assert gc.endpoint == "http://localhost:8890/sparql"
        assert gc.graph_uri == "http://example.org/graph"
        assert gc.supports_reification is True
        assert gc.has_temporal_data is True
        assert gc.timeout_ms == 60000

    def test_graph_config_defaults(self):
        """Test GraphConfig default values."""
        gc = GraphConfig(endpoint="http://localhost:8890/sparql")

        assert gc.graph_uri is None
        assert gc.supports_reification is False
        assert gc.has_temporal_data is False
        assert gc.timeout_ms == 30000


class TestPromptConfig:
    """Tests for PromptConfig dataclass."""

    def test_create_prompt_config(self):
        """Test creating a PromptConfig."""
        pc = PromptConfig(
            system_prompt="You are a KBQA agent.",
            classification_prompt="Classify this question: {question}",
            entity_extraction_prompt="Extract entities.",
            synthesis_prompt="Synthesize answer.",
            qtype_strategies={"Query": "Default strategy"},
            tool_loop_guidance={"RunSPARQL": "Check syntax"},
            generic_loop_guidance="Try different approach",
        )

        assert "KBQA" in pc.system_prompt
        assert "{question}" in pc.classification_prompt
        assert "Query" in pc.qtype_strategies

    def test_get_strategy_for_qtype(self):
        """Test getting strategy for question type."""
        pc = PromptConfig(
            system_prompt="",
            classification_prompt="",
            entity_extraction_prompt="",
            synthesis_prompt="",
            qtype_strategies={
                "Count": "Count strategy",
                "Query": "Default strategy",
            },
        )

        assert pc.get_strategy_for_qtype("Count") == "Count strategy"
        assert pc.get_strategy_for_qtype("Unknown") == "Default strategy"

    def test_get_loop_guidance_for_tool(self):
        """Test getting loop guidance for tool."""
        pc = PromptConfig(
            system_prompt="",
            classification_prompt="",
            entity_extraction_prompt="",
            synthesis_prompt="",
            tool_loop_guidance={"FindNode": "Try different terms"},
            generic_loop_guidance="Generic guidance",
        )

        assert pc.get_loop_guidance_for_tool("FindNode") == "Try different terms"
        assert pc.get_loop_guidance_for_tool("Unknown") == "Generic guidance"


class TestKnowledgeGraphConfig:
    """Tests for KnowledgeGraphConfig dataclass."""

    def test_create_kg_config(self):
        """Test creating a KnowledgeGraphConfig."""
        ns = NamespaceConfig(
            entity_prefix="http://example.org/entity/",
            property_prefix="http://example.org/property/",
            attribute_prefix="http://example.org/attribute/",
        )
        vc = VectorConfig(
            entity_collection="entities",
            relation_collection="relations",
        )
        gc = GraphConfig(endpoint="http://localhost:8890/sparql")

        config = KnowledgeGraphConfig(
            name="TestKG",
            code="test",
            namespaces=ns,
            vectors=vc,
            graph=gc,
            node_types={"entity": "Entity"},
            domain_settings={"max_iterations": 50},
        )

        assert config.name == "TestKG"
        assert config.code == "test"
        assert config.namespaces.entity_prefix == "http://example.org/entity/"
        assert config.vectors.entity_collection == "entities"
        assert config.node_types["entity"] == "Entity"

    def test_kg_config_to_dict(self):
        """Test serializing KnowledgeGraphConfig to dict."""
        ns = NamespaceConfig(
            entity_prefix="http://example.org/entity/",
            property_prefix="http://example.org/property/",
            attribute_prefix="http://example.org/attribute/",
        )
        vc = VectorConfig(
            entity_collection="entities",
            relation_collection="relations",
        )
        gc = GraphConfig(endpoint="http://localhost:8890/sparql")

        config = KnowledgeGraphConfig(
            name="TestKG",
            code="test",
            namespaces=ns,
            vectors=vc,
            graph=gc,
        )

        d = config.to_dict()
        assert d["name"] == "TestKG"
        assert d["namespaces"]["entity_prefix"] == "http://example.org/entity/"

    def test_kg_config_to_json(self):
        """Test serializing KnowledgeGraphConfig to JSON."""
        config = create_default_kqapro_config()
        json_str = config.to_json()
        parsed = json.loads(json_str)

        assert parsed["name"] == "KQAPro"
        assert "entity_prefix" in parsed["namespaces"]


class TestDefaultConfigs:
    """Tests for default configuration factories."""

    def test_create_default_kqapro_config(self):
        """Test creating default KQAPro config."""
        config = create_default_kqapro_config()

        assert config.name == "KQAPro"
        assert config.code == "kqapro"
        assert config.namespaces.entity_prefix == "http://kqapro.org/entity/"
        assert config.namespaces.qualifier_prefix == "http://kqapro.org/qualifier/"
        assert config.graph.supports_reification is True
        assert config.graph.has_temporal_data is True
        assert "kqapro_entities" in config.vectors.entity_collection

    def test_create_default_sciqa_config(self):
        """Test creating default SciQA config."""
        config = create_default_sciqa_config()

        assert config.name == "SciQA/ORKG"
        assert config.code == "sciqa"
        assert config.namespaces.entity_prefix == "http://orkg.org/orkg/resource/"
        assert config.namespaces.qualifier_prefix is None
        assert config.graph.supports_reification is False
        assert config.graph.graph_uri == "http://sciqa.org/kg"
        assert "sciqa" in config.vectors.entity_collection

    def test_kqapro_config_has_all_prefixes(self):
        """Test KQAPro config has all required SPARQL prefixes."""
        config = create_default_kqapro_config()
        prefixes = config.namespaces.sparql_prefixes

        assert "PREFIX ex:" in prefixes
        assert "PREFIX prop:" in prefixes
        assert "PREFIX attr:" in prefixes
        assert "PREFIX qual:" in prefixes
        assert "PREFIX rdfs:" in prefixes

    def test_sciqa_config_has_all_prefixes(self):
        """Test SciQA config has all required SPARQL prefixes."""
        config = create_default_sciqa_config()
        prefixes = config.namespaces.sparql_prefixes

        assert "PREFIX orkgr:" in prefixes
        assert "PREFIX orkgp:" in prefixes
        assert "PREFIX orkgc:" in prefixes
        assert "PREFIX rdfs:" in prefixes
