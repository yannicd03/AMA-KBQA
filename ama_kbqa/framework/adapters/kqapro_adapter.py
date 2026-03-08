"""
KQAPro knowledge graph adapter.

Provides configuration and utilities specific to the KQAPro dataset.
"""

from __future__ import annotations
import re

from ama_kbqa.framework.config import (
    KnowledgeGraphConfig,
    NamespaceConfig,
    VectorConfig,
    GraphConfig,
)
from ama_kbqa.framework.adapters.base_adapter import BaseKGAdapter


class KQAProAdapter(BaseKGAdapter):
    """
    Adapter for the KQAPro knowledge graph.

    KQAPro is a large-scale knowledge graph QA dataset with:
    - Reification support (facts about facts)
    - Temporal data (dates, time periods)
    - Multiple entity types (entities, classes, properties)
    - Qualifier predicates for contextual information
    """

    # KQAPro entity ID pattern (e.g., Q123, Q1234567)
    ENTITY_ID_PATTERN = re.compile(r'^Q\d+$')

    # KQAPro property ID pattern (e.g., P123)
    PROPERTY_ID_PATTERN = re.compile(r'^P\d+$')

    def _create_config(self) -> KnowledgeGraphConfig:
        """Create KQAPro-specific configuration."""
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
                    "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
                    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
                    "xsd": "http://www.w3.org/2001/XMLSchema#",
                },
            ),
            vectors=VectorConfig(
                entity_collection="kqapro_entities",
                relation_collection="kqapro_relations",
                entity_threshold=0.70,
                relation_threshold=0.65,
                top_k_default=5,
                host="localhost",
                port=6333,
            ),
            graph=GraphConfig(
                endpoint="http://localhost:8890/sparql",
                graph_uri=None,  # KQAPro uses default graph
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
                "max_iterations": 25,
                "journal_refresh_interval": 7,
                "enable_fast_path": True,
                "fewshot_examples_dir": "db/datasets/kqapro/fewshot-examples",
            },
        )

    # =========================================================================
    # KQAPro-Specific Validation
    # =========================================================================

    def is_valid_entity_id(self, entity_id: str) -> bool:
        """
        Check if an entity ID is valid for KQAPro.

        Valid KQAPro entity IDs start with Q followed by digits.

        Args:
            entity_id: Entity ID to validate

        Returns:
            True if valid KQAPro entity ID
        """
        if not entity_id:
            return False

        # Handle full URIs
        if entity_id.startswith("http://"):
            entity_id = self.extract_id_from_uri(entity_id)

        return bool(self.ENTITY_ID_PATTERN.match(entity_id))

    def is_valid_property_id(self, prop_id: str) -> bool:
        """
        Check if a property ID is valid for KQAPro.

        Valid KQAPro property IDs start with P followed by digits.

        Args:
            prop_id: Property ID to validate

        Returns:
            True if valid KQAPro property ID
        """
        if not prop_id:
            return False

        # Handle full URIs
        if prop_id.startswith("http://"):
            prop_id = self.extract_id_from_uri(prop_id)

        return bool(self.PROPERTY_ID_PATTERN.match(prop_id))

    def normalize_entity_id(self, entity_id: str) -> str:
        """
        Normalize a KQAPro entity ID.

        Args:
            entity_id: Entity ID to normalize

        Returns:
            Normalized entity ID (e.g., "Q123")
        """
        if not entity_id:
            return ""

        # Handle full URIs
        if entity_id.startswith("http://"):
            entity_id = self.extract_id_from_uri(entity_id)

        # Ensure uppercase Q prefix
        if entity_id.lower().startswith("q") and entity_id[1:].isdigit():
            return "Q" + entity_id[1:]

        return entity_id

    def normalize_property_name(self, prop_name: str) -> str:
        """
        Normalize a KQAPro property name.

        Args:
            prop_name: Property name to normalize

        Returns:
            Normalized property name (e.g., "P123")
        """
        if not prop_name:
            return ""

        # Handle full URIs
        if prop_name.startswith("http://"):
            prop_name = self.extract_id_from_uri(prop_name)

        # Ensure uppercase P prefix
        if prop_name.lower().startswith("p") and prop_name[1:].isdigit():
            return "P" + prop_name[1:]

        return prop_name

    # =========================================================================
    # KQAPro-Specific Utilities
    # =========================================================================

    def format_qualifier_uri(self, qualifier_name: str) -> str:
        """
        Format a qualifier name as a full URI.

        Args:
            qualifier_name: The qualifier name

        Returns:
            Full URI string
        """
        if qualifier_name.startswith("http://") or qualifier_name.startswith("https://"):
            return qualifier_name
        qualifier_prefix = self.config.namespaces.qualifier_prefix
        if qualifier_prefix:
            return f"{qualifier_prefix}{qualifier_name}"
        return qualifier_name

    def format_unit_uri(self, unit_name: str) -> str:
        """
        Format a unit name as a full URI.

        Args:
            unit_name: The unit name

        Returns:
            Full URI string
        """
        if unit_name.startswith("http://") or unit_name.startswith("https://"):
            return unit_name
        unit_prefix = self.config.namespaces.unit_prefix
        if unit_prefix:
            return f"{unit_prefix}{unit_name}"
        return unit_name

    def get_reification_predicates(self) -> dict:
        """
        Get the predicate URIs used for reification in KQAPro.

        Returns:
            Dictionary with reification predicate URIs
        """
        return {
            "fact_subject": "http://kqapro.org/predicate/fact_h",
            "fact_relation": "http://kqapro.org/predicate/fact_r",
            "fact_object": "http://kqapro.org/predicate/fact_t",
        }

    def build_qualified_fact_query(
        self,
        subject_id: str,
        predicate: str,
        target_value: str = None,
        target_id: str = None
    ) -> str:
        """
        Build a SPARQL query to retrieve qualifiers for a fact.

        Args:
            subject_id: Subject entity ID
            predicate: Predicate name
            target_value: Target literal value (optional)
            target_id: Target entity ID (optional)

        Returns:
            SPARQL query string
        """
        reif = self.get_reification_predicates()

        if target_value:
            target_clause = f'"{target_value}"'
        elif target_id:
            target_clause = f"ex:{target_id}"
        else:
            target_clause = "?target"

        query = f"""
SELECT ?qualifier_pred ?qualifier_value WHERE {{
    ?fact_node <{reif['fact_subject']}> ex:{subject_id} ;
               <{reif['fact_relation']}> prop:{predicate} ;
               <{reif['fact_object']}> {target_clause} .
    ?fact_node ?qualifier_pred ?qualifier_value .
    FILTER(STRSTARTS(STR(?qualifier_pred), "{self.config.namespaces.qualifier_prefix}"))
}}
"""
        return query.strip()
