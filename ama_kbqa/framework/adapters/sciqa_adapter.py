"""
SciQA/ORKG knowledge graph adapter.

Provides configuration and utilities specific to the Open Research
Knowledge Graph (ORKG) used in the SciQA dataset.
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


class SciQAAdapter(BaseKGAdapter):
    """
    Adapter for the SciQA/ORKG knowledge graph.

    ORKG (Open Research Knowledge Graph) is a scientific knowledge graph with:
    - No reification (simple triples)
    - Research-focused entity types (papers, authors, contributions)
    - Standardized predicates (P0, P1, P2, etc.)
    - Named graph storage
    """

    # ORKG resource ID pattern (e.g., R123)
    RESOURCE_ID_PATTERN = re.compile(r'^R\d+$')

    # ORKG predicate ID pattern (e.g., P123)
    PREDICATE_ID_PATTERN = re.compile(r'^P\d+$')

    # ORKG class ID pattern (e.g., C123 or class names)
    CLASS_ID_PATTERN = re.compile(r'^C\d+$')

    def _create_config(self) -> KnowledgeGraphConfig:
        """Create SciQA/ORKG-specific configuration."""
        return KnowledgeGraphConfig(
            name="SciQA/ORKG",
            code="sciqa",
            namespaces=NamespaceConfig(
                entity_prefix="http://orkg.org/orkg/resource/",
                property_prefix="http://orkg.org/orkg/predicate/",
                attribute_prefix="http://orkg.org/orkg/predicate/",  # Same as property
                qualifier_prefix=None,  # ORKG doesn't use reification
                unit_prefix=None,
                class_prefix="http://orkg.org/orkg/class/",
                sparql_prefixes="""PREFIX orkgr: <http://orkg.org/orkg/resource/>
PREFIX orkgp: <http://orkg.org/orkg/predicate/>
PREFIX orkgc: <http://orkg.org/orkg/class/>
PREFIX rdfs:  <http://www.w3.org/2000/01/rdf-schema#>
PREFIX rdf:   <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX xsd:   <http://www.w3.org/2001/XMLSchema#>
PREFIX owl:   <http://www.w3.org/2002/07/owl#>
""",
                prefix_map={
                    "orkgr": "http://orkg.org/orkg/resource/",
                    "orkgp": "http://orkg.org/orkg/predicate/",
                    "orkgc": "http://orkg.org/orkg/class/",
                    "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
                    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
                    "xsd": "http://www.w3.org/2001/XMLSchema#",
                    "owl": "http://www.w3.org/2002/07/owl#",
                },
            ),
            vectors=VectorConfig(
                entity_collection="sciqa_resources",
                relation_collection="sciqa_predicates",
                entity_threshold=0.65,
                relation_threshold=0.60,
                top_k_default=5,
                host="localhost",
                port=6333,
            ),
            graph=GraphConfig(
                endpoint="http://localhost:8890/sparql",
                graph_uri="http://sciqa.org/kg",  # Named graph
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
                "problem": "Problem",
            },
            domain_settings={
                "max_iterations": 40,
                "journal_refresh_interval": 5,
                "find_resource_cap": 8,
                "sparql_cap": 10,
                "context_limit": 100000,
                "max_tool_calls": 25,
            },
        )

    def get_operation_bindings(self) -> dict:
        """Map abstract atomic operations to SciQA/ORKG MCP tool names."""
        return {
            "find_entity": "FindResource",
            "get_label": "GetResourceLabel",
            "get_labels": "BatchGetResourceLabels",
            "get_summary": "GetResourceSummary",
            "get_relation_targets": "GetRelationTargets",
            "reverse_lookup": "FindByPredicateValue",
            "follow_path": "FollowRelationPath",
            "compare": "CompareResources",
            # Counting in ORKG goes through the aggregator's agg="count" mode
            "count": "AggregateComparisonValues",
            "verify_numeric": "VerifyNumericCondition",
            "run_sparql": "RunORKGSPARQL",
            # Optional tier (ORKG's comparison tables support group-by aggregation)
            "aggregate": "AggregateComparisonValues",
            "frequent_values": "FindFrequentValues",
        }

    # =========================================================================
    # ORKG-Specific Validation
    # =========================================================================

    def is_valid_entity_id(self, entity_id: str) -> bool:
        """
        Check if an entity ID is valid for ORKG.

        Valid ORKG resource IDs start with R followed by digits.

        Args:
            entity_id: Entity ID to validate

        Returns:
            True if valid ORKG resource ID
        """
        if not entity_id:
            return False

        # Handle full URIs
        if entity_id.startswith("http://"):
            entity_id = self.extract_id_from_uri(entity_id)

        return bool(self.RESOURCE_ID_PATTERN.match(entity_id))

    def is_valid_property_id(self, prop_id: str) -> bool:
        """
        Check if a property ID is valid for ORKG.

        Valid ORKG predicate IDs start with P followed by digits.

        Args:
            prop_id: Property ID to validate

        Returns:
            True if valid ORKG predicate ID
        """
        if not prop_id:
            return False

        # Handle full URIs
        if prop_id.startswith("http://"):
            prop_id = self.extract_id_from_uri(prop_id)

        return bool(self.PREDICATE_ID_PATTERN.match(prop_id))

    def is_valid_class_id(self, class_id: str) -> bool:
        """
        Check if a class ID is valid for ORKG.

        Args:
            class_id: Class ID to validate

        Returns:
            True if valid ORKG class ID
        """
        if not class_id:
            return False

        # Handle full URIs
        if class_id.startswith("http://"):
            class_id = self.extract_id_from_uri(class_id)

        # ORKG classes can be C123 format or named classes like "Paper"
        return bool(self.CLASS_ID_PATTERN.match(class_id)) or class_id in self.config.node_types.values()

    def normalize_entity_id(self, entity_id: str) -> str:
        """
        Normalize an ORKG resource ID.

        Args:
            entity_id: Entity ID to normalize

        Returns:
            Normalized resource ID (e.g., "R123")
        """
        if not entity_id:
            return ""

        # Handle full URIs
        if entity_id.startswith("http://"):
            entity_id = self.extract_id_from_uri(entity_id)

        # Ensure uppercase R prefix
        if entity_id.lower().startswith("r") and entity_id[1:].isdigit():
            return "R" + entity_id[1:]

        return entity_id

    def normalize_property_name(self, prop_name: str) -> str:
        """
        Normalize an ORKG predicate name.

        Args:
            prop_name: Property name to normalize

        Returns:
            Normalized predicate name (e.g., "P123")
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
    # ORKG-Specific Utilities
    # =========================================================================

    def get_common_predicates(self) -> dict:
        """
        Get common ORKG predicate mappings.

        Returns:
            Dictionary mapping semantic names to predicate IDs
        """
        return {
            "addresses": "P0",       # Problem addressed
            "yields": "P1",          # Result/contribution
            "employs": "P2",         # Method used
            "author": "P27",         # Paper author (also P6)
            "author_alt": "P6",      # Alternative author predicate
            "affiliation": "P7",     # Author affiliation
            "doi": "P26",            # DOI (also P10)
            "doi_alt": "P10",        # Alternative DOI predicate
            "year": "P29",           # Publication year
            "research_field": "P30", # Research field
            "has_contribution": "P31", # Paper contributions
            "research_problem": "P32", # Research problem
            "compareContribution": "compareContribution",  # Comparison -> Contribution
            "has_value": "HAS_VALUE",  # Generic value predicate
            "efficiency": "P43156",    # Energy efficiency
            "vegetable_source": "P35148",  # Vegetable source
            "integrity_constraints": "P41333",  # Integrity constraints
            "population_sample_size": "P23161",  # Population/sample size
            "electricity_generation": "P43134",  # Electricity generation
        }

    def get_common_classes(self) -> dict:
        """
        Get common ORKG class mappings.

        Returns:
            Dictionary mapping semantic names to class IDs/names
        """
        return {
            "paper": "Paper",
            "author": "Author",
            "contribution": "Contribution",
            "venue": "Venue",
            "research_field": "ResearchField",
            "problem": "Problem",
            "method": "Method",
            "result": "Result",
        }

    def build_paper_query(self, paper_id: str) -> str:
        """
        Build a SPARQL query to get paper details.

        Args:
            paper_id: Paper resource ID

        Returns:
            SPARQL query string
        """
        query = f"""
SELECT ?predicate ?object ?objectLabel WHERE {{
    orkgr:{paper_id} ?predicate ?object .
    OPTIONAL {{ ?object rdfs:label ?objectLabel }}
}}
"""
        return query.strip()

    def build_papers_by_field_query(self, field_name: str) -> str:
        """
        Build a SPARQL query to find papers in a research field.

        Args:
            field_name: Research field name or ID

        Returns:
            SPARQL query string
        """
        predicates = self.get_common_predicates()

        # If field_name looks like an ID, use it directly
        if self.is_valid_entity_id(field_name):
            field_clause = f"orkgr:{field_name}"
        else:
            # Search by label
            field_clause = f'?field . ?field rdfs:label "{field_name}"'

        query = f"""
SELECT DISTINCT ?paper ?paperLabel WHERE {{
    ?paper orkgp:{predicates['research_field']} {field_clause} .
    ?paper rdfs:label ?paperLabel .
}}
LIMIT 100
"""
        return query.strip()

    def build_paper_authors_query(self, paper_id: str) -> str:
        """
        Build a SPARQL query to get paper authors.

        Args:
            paper_id: Paper resource ID

        Returns:
            SPARQL query string
        """
        predicates = self.get_common_predicates()

        query = f"""
SELECT ?author ?authorLabel WHERE {{
    {{
        orkgr:{paper_id} orkgp:{predicates['author']} ?author .
    }}
    UNION
    {{
        orkgr:{paper_id} orkgp:{predicates['author_alt']} ?author .
    }}
    OPTIONAL {{ ?author rdfs:label ?authorLabel }}
}}
"""
        return query.strip()

    def build_contribution_methods_query(self, contribution_id: str) -> str:
        """
        Build a SPARQL query to get methods used in a contribution.

        Args:
            contribution_id: Contribution resource ID

        Returns:
            SPARQL query string
        """
        predicates = self.get_common_predicates()

        query = f"""
SELECT ?method ?methodLabel WHERE {{
    orkgr:{contribution_id} orkgp:{predicates['employs']} ?method .
    OPTIONAL {{ ?method rdfs:label ?methodLabel }}
}}
"""
        return query.strip()
