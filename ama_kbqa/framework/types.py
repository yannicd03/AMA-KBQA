"""
Standard response types for KBQA tools.

These dataclasses provide a consistent interface for tool responses
across different knowledge graph implementations.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional
import json


@dataclass
class EntityMatch:
    """
    Represents an entity match from semantic search.

    Returned by FindNode/FindResource tools after vector similarity search.
    """
    id: str
    label: str
    node_type: str
    relevance_score: float
    available_attributes: List[str] = field(default_factory=list)
    available_relations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EntityMatch":
        """Create from dictionary."""
        return cls(
            id=data.get("id", ""),
            label=data.get("label", ""),
            node_type=data.get("node_type", ""),
            relevance_score=data.get("relevance_score", 0.0),
            available_attributes=data.get("available_attributes", []),
            available_relations=data.get("available_relations", []),
            metadata=data.get("metadata", {}),
        )


@dataclass
class AttributeValue:
    """
    Represents an attribute value with optional metadata.

    Supports units (for numeric values), qualifiers (for contextual facts),
    and datatype information.
    """
    value: Any
    unit: Optional[str] = None
    qualifiers: Dict[str, Any] = field(default_factory=dict)
    datatype: str = "string"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AttributeValue":
        """Create from dictionary."""
        return cls(
            value=data.get("value"),
            unit=data.get("unit"),
            qualifiers=data.get("qualifiers", {}),
            datatype=data.get("datatype", "string"),
        )


@dataclass
class NodeDetails:
    """
    Comprehensive details about a knowledge graph node.

    Returned by GetNodeSummary/GetResourceDetails tools.
    Contains all attributes, relations, and statistics.
    """
    id: str
    label: str
    node_type: str
    attributes: Dict[str, List[AttributeValue]] = field(default_factory=dict)
    relations: Dict[str, List[str]] = field(default_factory=dict)
    stats: Dict[str, int] = field(default_factory=dict)
    status: str = "success"
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "id": self.id,
            "label": self.label,
            "node_type": self.node_type,
            "attributes": {
                k: [v.to_dict() for v in vals]
                for k, vals in self.attributes.items()
            },
            "relations": self.relations,
            "stats": self.stats,
            "status": self.status,
        }
        if self.error:
            result["error"] = self.error
        return result

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NodeDetails":
        """Create from dictionary."""
        attributes = {}
        for k, vals in data.get("attributes", {}).items():
            attributes[k] = [AttributeValue.from_dict(v) for v in vals]

        return cls(
            id=data.get("id", ""),
            label=data.get("label", ""),
            node_type=data.get("node_type", ""),
            attributes=attributes,
            relations=data.get("relations", {}),
            stats=data.get("stats", {}),
            status=data.get("status", "success"),
            error=data.get("error"),
        )


@dataclass
class NavigationResult:
    """
    Result of navigating a relation from a node.

    Returned by GetRelationDetails/GetRelationTargets tools.
    Contains the target nodes reached via the specified relation.
    """
    start_node: str
    relation: str
    targets: List[str] = field(default_factory=list)
    target_labels: Dict[str, str] = field(default_factory=dict)
    direction: str = "outgoing"  # "outgoing" or "incoming"
    status: str = "success"
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "start_node": self.start_node,
            "relation": self.relation,
            "targets": self.targets,
            "target_labels": self.target_labels,
            "direction": self.direction,
            "status": self.status,
        }
        if self.error:
            result["error"] = self.error
        return result

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NavigationResult":
        """Create from dictionary."""
        return cls(
            start_node=data.get("start_node", ""),
            relation=data.get("relation", ""),
            targets=data.get("targets", []),
            target_labels=data.get("target_labels", {}),
            direction=data.get("direction", "outgoing"),
            status=data.get("status", "success"),
            error=data.get("error"),
        )


@dataclass
class QualifierResult:
    """
    Result of querying edge qualifiers (reification).

    Used for KGs that support "facts about facts" - e.g., temporal
    qualifiers on relationships like "population in 2020".
    """
    subject_id: str
    predicate: str
    object_value: Optional[str] = None
    object_id: Optional[str] = None
    qualifiers: Dict[str, Any] = field(default_factory=dict)
    status: str = "success"
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "subject_id": self.subject_id,
            "predicate": self.predicate,
            "qualifiers": self.qualifiers,
            "status": self.status,
        }
        if self.object_value is not None:
            result["object_value"] = self.object_value
        if self.object_id is not None:
            result["object_id"] = self.object_id
        if self.error:
            result["error"] = self.error
        return result

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QualifierResult":
        """Create from dictionary."""
        return cls(
            subject_id=data.get("subject_id", ""),
            predicate=data.get("predicate", ""),
            object_value=data.get("object_value"),
            object_id=data.get("object_id"),
            qualifiers=data.get("qualifiers", {}),
            status=data.get("status", "success"),
            error=data.get("error"),
        )


@dataclass
class SPARQLResult:
    """
    Result of a SPARQL query execution.

    Returned by RunSPARQL/RunORKGSPARQL tools.
    Contains the query, variable bindings, and row count.
    """
    query: str
    variables: List[str] = field(default_factory=list)
    bindings: List[Dict[str, Any]] = field(default_factory=list)
    row_count: int = 0
    status: str = "success"
    error: Optional[str] = None
    execution_time_ms: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "query": self.query,
            "variables": self.variables,
            "bindings": self.bindings,
            "row_count": self.row_count,
            "status": self.status,
        }
        if self.error:
            result["error"] = self.error
        if self.execution_time_ms is not None:
            result["execution_time_ms"] = self.execution_time_ms
        return result

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SPARQLResult":
        """Create from dictionary."""
        return cls(
            query=data.get("query", ""),
            variables=data.get("variables", []),
            bindings=data.get("bindings", []),
            row_count=data.get("row_count", 0),
            status=data.get("status", "success"),
            error=data.get("error"),
            execution_time_ms=data.get("execution_time_ms"),
        )


@dataclass
class ToolResult:
    """
    Generic wrapper for tool execution results.

    Provides a consistent interface for all tool responses.
    """
    tool_name: str
    success: bool
    data: Any = None
    error: Optional[str] = None
    execution_time_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "tool_name": self.tool_name,
            "success": self.success,
            "execution_time_seconds": self.execution_time_seconds,
        }
        if self.data is not None:
            if hasattr(self.data, "to_dict"):
                result["data"] = self.data.to_dict()
            else:
                result["data"] = self.data
        if self.error:
            result["error"] = self.error
        return result

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)
