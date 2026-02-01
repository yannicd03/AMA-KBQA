"""
Configuration dataclasses for KBQA framework.

These classes define the configuration structure for knowledge graphs,
including namespaces, vector databases, and prompts.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional
import json


@dataclass
class NamespaceConfig:
    """
    Configuration for knowledge graph namespaces and prefixes.

    Defines the URI patterns and SPARQL prefixes for a specific KG.
    """
    entity_prefix: str
    property_prefix: str
    attribute_prefix: str
    qualifier_prefix: Optional[str] = None
    unit_prefix: Optional[str] = None
    class_prefix: Optional[str] = None
    sparql_prefixes: str = ""
    prefix_map: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    def get_full_entity_uri(self, entity_id: str) -> str:
        """Get the full URI for an entity ID."""
        if entity_id.startswith("http://") or entity_id.startswith("https://"):
            return entity_id
        return f"{self.entity_prefix}{entity_id}"

    def get_full_property_uri(self, property_name: str) -> str:
        """Get the full URI for a property name."""
        if property_name.startswith("http://") or property_name.startswith("https://"):
            return property_name
        return f"{self.property_prefix}{property_name}"

    def get_full_attribute_uri(self, attribute_name: str) -> str:
        """Get the full URI for an attribute name."""
        if attribute_name.startswith("http://") or attribute_name.startswith("https://"):
            return attribute_name
        return f"{self.attribute_prefix}{attribute_name}"

    def extract_id_from_uri(self, uri: str) -> str:
        """Extract the local ID from a full URI."""
        for prefix_uri in [
            self.entity_prefix,
            self.property_prefix,
            self.attribute_prefix,
            self.qualifier_prefix,
            self.unit_prefix,
            self.class_prefix,
        ]:
            if prefix_uri and uri.startswith(prefix_uri):
                return uri[len(prefix_uri):]
        # Fallback: return everything after the last / or #
        if "#" in uri:
            return uri.split("#")[-1]
        return uri.split("/")[-1]


@dataclass
class VectorConfig:
    """
    Configuration for vector database collections.

    Defines collection names and search thresholds for semantic search.
    """
    entity_collection: str
    relation_collection: str
    entity_threshold: float = 0.70
    relation_threshold: float = 0.65
    top_k_default: int = 5
    host: str = "localhost"
    port: int = 6333

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


@dataclass
class GraphConfig:
    """
    Configuration for the graph database endpoint.

    Defines connection settings and KG-specific features.
    """
    endpoint: str
    graph_uri: Optional[str] = None
    supports_reification: bool = False
    has_temporal_data: bool = False
    timeout_ms: int = 30000

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


@dataclass
class PromptConfig:
    """
    Configuration for agent prompts and strategies.

    Contains the system prompt, classification prompt, and question-type
    specific strategies.
    """
    system_prompt: str
    classification_prompt: str
    entity_extraction_prompt: str
    synthesis_prompt: str
    qtype_strategies: Dict[str, str] = field(default_factory=dict)
    tool_loop_guidance: Dict[str, str] = field(default_factory=dict)
    generic_loop_guidance: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    def get_strategy_for_qtype(self, qtype: str) -> str:
        """Get the reasoning strategy for a question type."""
        return self.qtype_strategies.get(qtype, self.qtype_strategies.get("Query", ""))

    def get_loop_guidance_for_tool(self, tool_name: str) -> str:
        """Get loop recovery guidance for a specific tool."""
        return self.tool_loop_guidance.get(tool_name, self.generic_loop_guidance)


@dataclass
class KnowledgeGraphConfig:
    """
    Complete configuration for a knowledge graph.

    Aggregates all sub-configurations into a single object.
    """
    name: str
    code: str
    namespaces: NamespaceConfig
    vectors: VectorConfig
    graph: GraphConfig
    prompts: Optional[PromptConfig] = None
    node_types: Dict[str, str] = field(default_factory=dict)
    domain_settings: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "code": self.code,
            "namespaces": self.namespaces.to_dict(),
            "vectors": self.vectors.to_dict(),
            "graph": self.graph.to_dict(),
            "prompts": self.prompts.to_dict() if self.prompts else None,
            "node_types": self.node_types,
            "domain_settings": self.domain_settings,
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KnowledgeGraphConfig":
        """Create from dictionary."""
        namespaces = NamespaceConfig(**data.get("namespaces", {}))
        vectors = VectorConfig(**data.get("vectors", {}))
        graph = GraphConfig(**data.get("graph", {}))
        prompts = None
        if data.get("prompts"):
            prompts = PromptConfig(**data["prompts"])

        return cls(
            name=data.get("name", ""),
            code=data.get("code", ""),
            namespaces=namespaces,
            vectors=vectors,
            graph=graph,
            prompts=prompts,
            node_types=data.get("node_types", {}),
            domain_settings=data.get("domain_settings", {}),
        )


def create_default_kqapro_config() -> KnowledgeGraphConfig:
    """Create default configuration for KQAPro knowledge graph."""
    return KnowledgeGraphConfig(
        name="KQAPro",
        code="kqapro",
        namespaces=NamespaceConfig(
            entity_prefix="http://kqapro.org/entity/",
            property_prefix="http://kqapro.org/property/",
            attribute_prefix="http://kqapro.org/attribute/",
            qualifier_prefix="http://kqapro.org/qualifier/",
            unit_prefix="http://kqapro.org/unit/",
            class_prefix="http://kqapro.org/class/",
            sparql_prefixes="""PREFIX ex:   <http://kqapro.org/entity/>
PREFIX prop: <http://kqapro.org/property/>
PREFIX attr: <http://kqapro.org/attribute/>
PREFIX qual: <http://kqapro.org/qualifier/>
PREFIX unit: <http://kqapro.org/unit/>
PREFIX pred: <http://kqapro.org/predicate/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX xsd:  <http://www.w3.org/2001/XMLSchema#>
""",
            prefix_map={
                "ex": "http://kqapro.org/entity/",
                "prop": "http://kqapro.org/property/",
                "attr": "http://kqapro.org/attribute/",
                "qual": "http://kqapro.org/qualifier/",
                "unit": "http://kqapro.org/unit/",
                "pred": "http://kqapro.org/predicate/",
            },
        ),
        vectors=VectorConfig(
            entity_collection="kqapro_entities",
            relation_collection="kqapro_relations",
            entity_threshold=0.70,
            relation_threshold=0.65,
            top_k_default=5,
        ),
        graph=GraphConfig(
            endpoint="http://localhost:8890/sparql",
            graph_uri=None,
            supports_reification=True,
            has_temporal_data=True,
            timeout_ms=30000,
        ),
        node_types={
            "entity": "Entity",
            "class": "Class",
            "property": "Property",
            "attribute": "Attribute",
        },
        domain_settings={
            "max_iterations": 50,
            "journal_refresh_interval": 5,
        },
    )


def create_default_sciqa_config() -> KnowledgeGraphConfig:
    """Create default configuration for SciQA/ORKG knowledge graph."""
    return KnowledgeGraphConfig(
        name="SciQA/ORKG",
        code="sciqa",
        namespaces=NamespaceConfig(
            entity_prefix="http://orkg.org/orkg/resource/",
            property_prefix="http://orkg.org/orkg/predicate/",
            attribute_prefix="http://orkg.org/orkg/predicate/",
            qualifier_prefix=None,  # ORKG doesn't use reification
            unit_prefix=None,
            class_prefix="http://orkg.org/orkg/class/",
            sparql_prefixes="""PREFIX orkgr: <http://orkg.org/orkg/resource/>
PREFIX orkgp: <http://orkg.org/orkg/predicate/>
PREFIX orkgc: <http://orkg.org/orkg/class/>
PREFIX rdfs:  <http://www.w3.org/2000/01/rdf-schema#>
PREFIX rdf:   <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX xsd:   <http://www.w3.org/2001/XMLSchema#>
""",
            prefix_map={
                "orkgr": "http://orkg.org/orkg/resource/",
                "orkgp": "http://orkg.org/orkg/predicate/",
                "orkgc": "http://orkg.org/orkg/class/",
            },
        ),
        vectors=VectorConfig(
            entity_collection="sciqa_resources",
            relation_collection="sciqa_predicates",
            entity_threshold=0.65,
            relation_threshold=0.60,
            top_k_default=5,
        ),
        graph=GraphConfig(
            endpoint="http://localhost:8890/sparql",
            graph_uri="http://sciqa.org/kg",
            supports_reification=False,
            has_temporal_data=False,
            timeout_ms=30000,
        ),
        node_types={
            "resource": "Resource",
            "paper": "Paper",
            "author": "Author",
            "contribution": "Contribution",
            "venue": "Venue",
            "research_field": "ResearchField",
        },
        domain_settings={
            "max_iterations": 50,
            "journal_refresh_interval": 5,
        },
    )
