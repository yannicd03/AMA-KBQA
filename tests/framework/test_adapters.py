"""Tests for framework adapters."""

import pytest
from ama_kbqa.framework.adapters.base_adapter import BaseKGAdapter
from ama_kbqa.framework.adapters.kqapro_adapter import KQAProAdapter
from ama_kbqa.framework.adapters.sciqa_adapter import SciQAAdapter


class TestKQAProAdapter:
    """Tests for KQAProAdapter."""

    def test_create_adapter(self):
        """Test creating a KQAProAdapter."""
        adapter = KQAProAdapter()
        config = adapter.config

        assert config.name == "KQAPro"
        assert config.code == "kqapro"

    def test_config_has_correct_prefixes(self):
        """Test KQAPro config has correct namespace prefixes."""
        adapter = KQAProAdapter()
        ns = adapter.config.namespaces

        assert ns.entity_prefix == "http://kqapro.org/entity/"
        assert ns.property_prefix == "http://kqapro.org/property/"
        assert ns.attribute_prefix == "http://kqapro.org/attribute/"
        assert ns.qualifier_prefix == "http://kqapro.org/qualifier/"

    def test_format_entity_uri(self):
        """Test formatting entity URIs."""
        adapter = KQAProAdapter()

        # Simple ID
        uri = adapter.format_entity_uri("Q123")
        assert uri == "http://kqapro.org/entity/Q123"

        # Already full URI
        uri = adapter.format_entity_uri("http://other.org/Q456")
        assert uri == "http://other.org/Q456"

    def test_format_property_uri(self):
        """Test formatting property URIs."""
        adapter = KQAProAdapter()

        uri = adapter.format_property_uri("P789")
        assert uri == "http://kqapro.org/property/P789"

    def test_format_qualifier_uri(self):
        """Test formatting qualifier URIs."""
        adapter = KQAProAdapter()

        uri = adapter.format_qualifier_uri("point_in_time")
        assert uri == "http://kqapro.org/qualifier/point_in_time"

    def test_extract_id_from_uri(self):
        """Test extracting ID from URI."""
        adapter = KQAProAdapter()

        entity_id = adapter.extract_id_from_uri("http://kqapro.org/entity/Q12345")
        assert entity_id == "Q12345"

        prop_id = adapter.extract_id_from_uri("http://kqapro.org/property/P999")
        assert prop_id == "P999"

    def test_extract_prefix_and_id(self):
        """Test extracting prefix and ID from URI."""
        adapter = KQAProAdapter()

        prefix, local_id = adapter.extract_prefix_and_id("http://kqapro.org/entity/Q123")
        assert prefix == "ex"
        assert local_id == "Q123"

        prefix, local_id = adapter.extract_prefix_and_id("http://kqapro.org/property/P456")
        assert prefix == "prop"
        assert local_id == "P456"

    def test_format_prefixed_uri(self):
        """Test formatting URI with prefix."""
        adapter = KQAProAdapter()

        prefixed = adapter.format_prefixed_uri("http://kqapro.org/entity/Q123")
        assert prefixed == "ex:Q123"

        prefixed = adapter.format_prefixed_uri("http://kqapro.org/property/P456")
        assert prefixed == "prop:P456"

    def test_is_valid_entity_id(self):
        """Test validating KQAPro entity IDs."""
        adapter = KQAProAdapter()

        assert adapter.is_valid_entity_id("Q123") is True
        assert adapter.is_valid_entity_id("Q1234567") is True
        assert adapter.is_valid_entity_id("http://kqapro.org/entity/Q123") is True

        assert adapter.is_valid_entity_id("P123") is False
        assert adapter.is_valid_entity_id("123") is False
        assert adapter.is_valid_entity_id("") is False
        assert adapter.is_valid_entity_id("Qxyz") is False

    def test_is_valid_property_id(self):
        """Test validating KQAPro property IDs."""
        adapter = KQAProAdapter()

        assert adapter.is_valid_property_id("P123") is True
        assert adapter.is_valid_property_id("P9999") is True
        assert adapter.is_valid_property_id("http://kqapro.org/property/P123") is True

        assert adapter.is_valid_property_id("Q123") is False
        assert adapter.is_valid_property_id("") is False

    def test_normalize_entity_id(self):
        """Test normalizing KQAPro entity IDs."""
        adapter = KQAProAdapter()

        assert adapter.normalize_entity_id("Q123") == "Q123"
        assert adapter.normalize_entity_id("q123") == "Q123"
        assert adapter.normalize_entity_id("http://kqapro.org/entity/Q123") == "Q123"

    def test_normalize_property_name(self):
        """Test normalizing KQAPro property names."""
        adapter = KQAProAdapter()

        assert adapter.normalize_property_name("P123") == "P123"
        assert adapter.normalize_property_name("p123") == "P123"

    def test_inject_sparql_prefixes(self):
        """Test injecting SPARQL prefixes."""
        adapter = KQAProAdapter()

        query = "SELECT ?x WHERE { ?x a ex:Entity }"
        result = adapter.inject_sparql_prefixes(query)

        assert "PREFIX ex:" in result
        assert "PREFIX prop:" in result
        assert query in result

        # Already has prefixes
        query_with_prefix = "PREFIX ex: <http://test/> SELECT ?x WHERE { ?x a ex:Entity }"
        result = adapter.inject_sparql_prefixes(query_with_prefix)
        assert result == query_with_prefix

    def test_supports_reification(self):
        """Test reification support check."""
        adapter = KQAProAdapter()
        assert adapter.supports_reification() is True

    def test_has_temporal_data(self):
        """Test temporal data support check."""
        adapter = KQAProAdapter()
        assert adapter.has_temporal_data() is True

    def test_get_reification_predicates(self):
        """Test getting reification predicates."""
        adapter = KQAProAdapter()
        preds = adapter.get_reification_predicates()

        assert "fact_subject" in preds
        assert "fact_relation" in preds
        assert "fact_object" in preds

    def test_build_qualified_fact_query(self):
        """Test building qualified fact query."""
        adapter = KQAProAdapter()

        query = adapter.build_qualified_fact_query(
            subject_id="Q123",
            predicate="population",
            target_value="1000000"
        )

        assert "ex:Q123" in query
        assert "prop:population" in query
        assert '"1000000"' in query
        assert "qualifier_pred" in query


class TestSciQAAdapter:
    """Tests for SciQAAdapter."""

    def test_create_adapter(self):
        """Test creating a SciQAAdapter."""
        adapter = SciQAAdapter()
        config = adapter.config

        assert config.name == "SciQA/ORKG"
        assert config.code == "sciqa"

    def test_config_has_correct_prefixes(self):
        """Test SciQA config has correct namespace prefixes."""
        adapter = SciQAAdapter()
        ns = adapter.config.namespaces

        assert ns.entity_prefix == "http://orkg.org/orkg/resource/"
        assert ns.property_prefix == "http://orkg.org/orkg/predicate/"
        assert ns.class_prefix == "http://orkg.org/orkg/class/"
        assert ns.qualifier_prefix is None

    def test_format_entity_uri(self):
        """Test formatting ORKG resource URIs."""
        adapter = SciQAAdapter()

        uri = adapter.format_entity_uri("R12345")
        assert uri == "http://orkg.org/orkg/resource/R12345"

    def test_format_property_uri(self):
        """Test formatting ORKG predicate URIs."""
        adapter = SciQAAdapter()

        uri = adapter.format_property_uri("P30")
        assert uri == "http://orkg.org/orkg/predicate/P30"

    def test_is_valid_entity_id(self):
        """Test validating ORKG resource IDs."""
        adapter = SciQAAdapter()

        assert adapter.is_valid_entity_id("R123") is True
        assert adapter.is_valid_entity_id("R1234567") is True
        assert adapter.is_valid_entity_id("http://orkg.org/orkg/resource/R123") is True

        assert adapter.is_valid_entity_id("P123") is False
        assert adapter.is_valid_entity_id("Q123") is False
        assert adapter.is_valid_entity_id("") is False

    def test_is_valid_property_id(self):
        """Test validating ORKG predicate IDs."""
        adapter = SciQAAdapter()

        assert adapter.is_valid_property_id("P30") is True
        assert adapter.is_valid_property_id("P0") is True

        assert adapter.is_valid_property_id("R123") is False

    def test_normalize_entity_id(self):
        """Test normalizing ORKG resource IDs."""
        adapter = SciQAAdapter()

        assert adapter.normalize_entity_id("R123") == "R123"
        assert adapter.normalize_entity_id("r123") == "R123"

    def test_supports_reification(self):
        """Test reification support check."""
        adapter = SciQAAdapter()
        assert adapter.supports_reification() is False

    def test_has_temporal_data(self):
        """Test temporal data support check."""
        adapter = SciQAAdapter()
        assert adapter.has_temporal_data() is False

    def test_inject_graph_clause(self):
        """Test injecting named graph clause."""
        adapter = SciQAAdapter()

        query = "SELECT ?x WHERE { ?x a orkgc:Paper }"
        result = adapter.inject_graph_clause(query)

        assert "FROM <http://sciqa.org/kg>" in result

        # Already has FROM clause
        query_with_from = "SELECT ?x FROM <http://other.org/> WHERE { ?x a orkgc:Paper }"
        result = adapter.inject_graph_clause(query_with_from)
        assert result == query_with_from

    def test_prepare_sparql_query(self):
        """Test preparing full SPARQL query."""
        adapter = SciQAAdapter()

        query = "SELECT ?x WHERE { ?x a orkgc:Paper }"
        result = adapter.prepare_sparql_query(query)

        # Should have both prefixes and FROM clause
        assert "PREFIX orkgr:" in result
        assert "PREFIX orkgp:" in result
        assert "FROM <http://sciqa.org/kg>" in result
        assert "SELECT ?x WHERE" in result

    def test_get_common_predicates(self):
        """Test getting common ORKG predicates."""
        adapter = SciQAAdapter()
        preds = adapter.get_common_predicates()

        assert preds["addresses"] == "P0"
        assert preds["yields"] == "P1"
        assert preds["employs"] == "P2"
        assert preds["author"] == "P27"
        assert preds["research_field"] == "P30"

    def test_get_common_classes(self):
        """Test getting common ORKG classes."""
        adapter = SciQAAdapter()
        classes = adapter.get_common_classes()

        assert classes["paper"] == "Paper"
        assert classes["author"] == "Author"
        assert classes["contribution"] == "Contribution"

    def test_build_paper_query(self):
        """Test building paper details query."""
        adapter = SciQAAdapter()

        query = adapter.build_paper_query("R12345")

        assert "orkgr:R12345" in query
        assert "?predicate ?object" in query

    def test_build_papers_by_field_query(self):
        """Test building papers by field query."""
        adapter = SciQAAdapter()

        # With field name
        query = adapter.build_papers_by_field_query("Natural Language Processing")
        assert "Natural Language Processing" in query
        assert "P30" in query

        # With resource ID
        query = adapter.build_papers_by_field_query("R100")
        assert "orkgr:R100" in query

    def test_build_paper_authors_query(self):
        """Test building paper authors query."""
        adapter = SciQAAdapter()

        query = adapter.build_paper_authors_query("R12345")

        assert "orkgr:R12345" in query
        assert "P27" in query or "P6" in query

    def test_build_contribution_methods_query(self):
        """Test building contribution methods query."""
        adapter = SciQAAdapter()

        query = adapter.build_contribution_methods_query("R99999")

        assert "orkgr:R99999" in query
        assert "P2" in query


class TestAdapterInheritance:
    """Tests for adapter inheritance behavior."""

    def test_adapters_share_base_methods(self):
        """Test that all adapters inherit common methods."""
        kqapro = KQAProAdapter()
        sciqa = SciQAAdapter()

        # Both should have these methods
        assert hasattr(kqapro, "format_entity_uri")
        assert hasattr(kqapro, "format_property_uri")
        assert hasattr(kqapro, "extract_id_from_uri")
        assert hasattr(kqapro, "inject_sparql_prefixes")
        assert hasattr(kqapro, "get_kg_name")
        assert hasattr(kqapro, "get_kg_code")

        assert hasattr(sciqa, "format_entity_uri")
        assert hasattr(sciqa, "format_property_uri")
        assert hasattr(sciqa, "extract_id_from_uri")
        assert hasattr(sciqa, "inject_sparql_prefixes")
        assert hasattr(sciqa, "get_kg_name")
        assert hasattr(sciqa, "get_kg_code")

    def test_adapters_have_different_configs(self):
        """Test that adapters produce different configurations."""
        kqapro = KQAProAdapter()
        sciqa = SciQAAdapter()

        assert kqapro.config.name != sciqa.config.name
        assert kqapro.config.code != sciqa.config.code
        assert kqapro.config.namespaces.entity_prefix != sciqa.config.namespaces.entity_prefix
