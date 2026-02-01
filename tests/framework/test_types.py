"""Tests for framework response types."""

import pytest
import json
from ama_kbqa.framework.types import (
    EntityMatch,
    AttributeValue,
    NodeDetails,
    NavigationResult,
    QualifierResult,
    SPARQLResult,
    ToolResult,
)


class TestEntityMatch:
    """Tests for EntityMatch dataclass."""

    def test_create_entity_match(self):
        """Test creating an EntityMatch."""
        match = EntityMatch(
            id="Q123",
            label="Test Entity",
            node_type="Entity",
            relevance_score=0.95,
            available_attributes=["population", "country"],
            available_relations=["located_in", "founded_by"],
            metadata={"source": "vector_search"},
        )

        assert match.id == "Q123"
        assert match.label == "Test Entity"
        assert match.node_type == "Entity"
        assert match.relevance_score == 0.95
        assert "population" in match.available_attributes
        assert "located_in" in match.available_relations
        assert match.metadata["source"] == "vector_search"

    def test_entity_match_to_dict(self):
        """Test converting EntityMatch to dictionary."""
        match = EntityMatch(
            id="Q123",
            label="Test Entity",
            node_type="Entity",
            relevance_score=0.95,
        )

        d = match.to_dict()
        assert d["id"] == "Q123"
        assert d["label"] == "Test Entity"
        assert isinstance(d["available_attributes"], list)

    def test_entity_match_to_json(self):
        """Test converting EntityMatch to JSON."""
        match = EntityMatch(
            id="Q123",
            label="Test Entity",
            node_type="Entity",
            relevance_score=0.95,
        )

        json_str = match.to_json()
        parsed = json.loads(json_str)
        assert parsed["id"] == "Q123"

    def test_entity_match_from_dict(self):
        """Test creating EntityMatch from dictionary."""
        data = {
            "id": "Q456",
            "label": "Another Entity",
            "node_type": "Class",
            "relevance_score": 0.8,
            "available_attributes": ["attr1"],
            "available_relations": ["rel1"],
            "metadata": {"key": "value"},
        }

        match = EntityMatch.from_dict(data)
        assert match.id == "Q456"
        assert match.label == "Another Entity"
        assert match.available_attributes == ["attr1"]


class TestAttributeValue:
    """Tests for AttributeValue dataclass."""

    def test_create_attribute_value(self):
        """Test creating an AttributeValue."""
        attr = AttributeValue(
            value=12345,
            unit="km",
            qualifiers={"time": "2020"},
            datatype="integer",
        )

        assert attr.value == 12345
        assert attr.unit == "km"
        assert attr.qualifiers["time"] == "2020"
        assert attr.datatype == "integer"

    def test_attribute_value_defaults(self):
        """Test AttributeValue default values."""
        attr = AttributeValue(value="test")

        assert attr.value == "test"
        assert attr.unit is None
        assert attr.qualifiers == {}
        assert attr.datatype == "string"

    def test_attribute_value_roundtrip(self):
        """Test AttributeValue serialization roundtrip."""
        original = AttributeValue(
            value="Paris",
            unit=None,
            qualifiers={"lang": "en"},
            datatype="string",
        )

        d = original.to_dict()
        restored = AttributeValue.from_dict(d)

        assert restored.value == original.value
        assert restored.unit == original.unit
        assert restored.qualifiers == original.qualifiers


class TestNodeDetails:
    """Tests for NodeDetails dataclass."""

    def test_create_node_details(self):
        """Test creating NodeDetails."""
        attr_val = AttributeValue(value="Paris", datatype="string")
        details = NodeDetails(
            id="Q123",
            label="France",
            node_type="Country",
            attributes={"capital": [attr_val]},
            relations={"located_in": ["Q789"]},
            stats={"attribute_count": 1, "relation_count": 1},
            status="success",
        )

        assert details.id == "Q123"
        assert details.label == "France"
        assert "capital" in details.attributes
        assert details.attributes["capital"][0].value == "Paris"
        assert "located_in" in details.relations

    def test_node_details_with_error(self):
        """Test NodeDetails with error."""
        details = NodeDetails(
            id="Q123",
            label="Unknown",
            node_type="Unknown",
            status="error",
            error="Entity not found",
        )

        assert details.status == "error"
        assert details.error == "Entity not found"

    def test_node_details_to_dict(self):
        """Test NodeDetails serialization."""
        attr_val = AttributeValue(value=100, unit="km", datatype="integer")
        details = NodeDetails(
            id="Q123",
            label="Test",
            node_type="Entity",
            attributes={"distance": [attr_val]},
            relations={},
            stats={},
        )

        d = details.to_dict()
        assert d["id"] == "Q123"
        assert d["attributes"]["distance"][0]["value"] == 100


class TestNavigationResult:
    """Tests for NavigationResult dataclass."""

    def test_create_navigation_result(self):
        """Test creating NavigationResult."""
        result = NavigationResult(
            start_node="Q123",
            relation="located_in",
            targets=["Q456", "Q789"],
            target_labels={"Q456": "Paris", "Q789": "Lyon"},
            direction="outgoing",
            status="success",
        )

        assert result.start_node == "Q123"
        assert result.relation == "located_in"
        assert len(result.targets) == 2
        assert result.target_labels["Q456"] == "Paris"
        assert result.direction == "outgoing"

    def test_navigation_result_defaults(self):
        """Test NavigationResult default values."""
        result = NavigationResult(
            start_node="Q123",
            relation="test",
        )

        assert result.targets == []
        assert result.target_labels == {}
        assert result.direction == "outgoing"
        assert result.status == "success"


class TestQualifierResult:
    """Tests for QualifierResult dataclass."""

    def test_create_qualifier_result(self):
        """Test creating QualifierResult."""
        result = QualifierResult(
            subject_id="Q123",
            predicate="population",
            object_value="1000000",
            qualifiers={"point_in_time": "2020", "determination_method": "census"},
            status="success",
        )

        assert result.subject_id == "Q123"
        assert result.predicate == "population"
        assert result.object_value == "1000000"
        assert result.qualifiers["point_in_time"] == "2020"

    def test_qualifier_result_with_object_id(self):
        """Test QualifierResult with object_id instead of value."""
        result = QualifierResult(
            subject_id="Q123",
            predicate="located_in",
            object_id="Q456",
            qualifiers={"start_time": "1990"},
        )

        assert result.object_id == "Q456"
        assert result.object_value is None


class TestSPARQLResult:
    """Tests for SPARQLResult dataclass."""

    def test_create_sparql_result(self):
        """Test creating SPARQLResult."""
        result = SPARQLResult(
            query="SELECT ?x WHERE { ?x a ex:Entity }",
            variables=["x"],
            bindings=[{"x": "Q123"}, {"x": "Q456"}],
            row_count=2,
            status="success",
            execution_time_ms=150.5,
        )

        assert "SELECT" in result.query
        assert result.variables == ["x"]
        assert len(result.bindings) == 2
        assert result.row_count == 2
        assert result.execution_time_ms == 150.5

    def test_sparql_result_with_error(self):
        """Test SPARQLResult with error."""
        result = SPARQLResult(
            query="INVALID QUERY",
            status="error",
            error="Syntax error at position 0",
        )

        assert result.status == "error"
        assert "Syntax error" in result.error


class TestToolResult:
    """Tests for ToolResult wrapper."""

    def test_create_tool_result_success(self):
        """Test creating successful ToolResult."""
        entity = EntityMatch(
            id="Q123",
            label="Test",
            node_type="Entity",
            relevance_score=0.9,
        )
        result = ToolResult(
            tool_name="FindNode",
            success=True,
            data=entity,
            execution_time_seconds=0.5,
        )

        assert result.tool_name == "FindNode"
        assert result.success is True
        assert result.data.id == "Q123"
        assert result.execution_time_seconds == 0.5

    def test_create_tool_result_failure(self):
        """Test creating failed ToolResult."""
        result = ToolResult(
            tool_name="RunSPARQL",
            success=False,
            error="Query timeout",
            execution_time_seconds=30.0,
        )

        assert result.success is False
        assert result.error == "Query timeout"

    def test_tool_result_to_dict(self):
        """Test ToolResult serialization."""
        entity = EntityMatch(
            id="Q123",
            label="Test",
            node_type="Entity",
            relevance_score=0.9,
        )
        result = ToolResult(
            tool_name="FindNode",
            success=True,
            data=entity,
            execution_time_seconds=0.5,
        )

        d = result.to_dict()
        assert d["tool_name"] == "FindNode"
        assert d["success"] is True
        assert d["data"]["id"] == "Q123"
