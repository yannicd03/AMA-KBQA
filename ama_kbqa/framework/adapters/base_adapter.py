"""
Abstract base class for knowledge graph adapters.

Provides common URI formatting and configuration utilities
that all KG adapters share.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Optional
import re

from ama_kbqa.framework.config import KnowledgeGraphConfig
from ama_kbqa.framework.operations import CoverageReport, validate_bindings


class BaseKGAdapter(ABC):
    """
    Abstract base class for knowledge graph adapters.

    Provides common utilities for URI formatting, prefix injection,
    and configuration management.
    """

    def __init__(self):
        """Initialize the adapter with its configuration."""
        self._config: Optional[KnowledgeGraphConfig] = None

    @property
    def config(self) -> KnowledgeGraphConfig:
        """Get the KG configuration, creating it if needed."""
        if self._config is None:
            self._config = self._create_config()
        return self._config

    @abstractmethod
    def _create_config(self) -> KnowledgeGraphConfig:
        """
        Create the configuration for this knowledge graph.

        Must be implemented by subclasses.

        Returns:
            KnowledgeGraphConfig for this KG
        """
        pass

    @abstractmethod
    def get_operation_bindings(self) -> Dict[str, str]:
        """
        Map abstract atomic operations to this KG's concrete MCP tool names.

        Keys are operation names from ``framework.operations.ATOMIC_OPERATIONS``
        (e.g. "find_entity", "get_label"); values are the tool names the KG's
        MCP server registers (e.g. "FindNode", "GetResourceLabel"). Concrete
        tool names stay KG-flavored because they appear in prompts, fewshots,
        and recorded benchmark traces; this binding is what makes the tool
        surfaces comparable across KGs.

        Returns:
            Dict mapping abstract operation name to concrete tool name
        """
        pass

    def validate_operation_coverage(self) -> CoverageReport:
        """
        Check this adapter's bindings against the abstract operation contract.

        Returns:
            CoverageReport with any missing required operations or unknown bindings
        """
        return validate_bindings(self.get_operation_bindings())

    # =========================================================================
    # URI Formatting Utilities
    # =========================================================================

    def format_entity_uri(self, entity_id: str) -> str:
        """
        Format an entity ID as a full URI.

        Args:
            entity_id: The entity ID (e.g., "Q123" or full URI)

        Returns:
            Full URI string
        """
        if entity_id.startswith("http://") or entity_id.startswith("https://"):
            return entity_id
        return f"{self.config.namespaces.entity_prefix}{entity_id}"

    def format_property_uri(self, prop_name: str) -> str:
        """
        Format a property name as a full URI.

        Args:
            prop_name: The property name (e.g., "P123" or full URI)

        Returns:
            Full URI string
        """
        if prop_name.startswith("http://") or prop_name.startswith("https://"):
            return prop_name
        return f"{self.config.namespaces.property_prefix}{prop_name}"

    def format_attribute_uri(self, attr_name: str) -> str:
        """
        Format an attribute name as a full URI.

        Args:
            attr_name: The attribute name (e.g., "population" or full URI)

        Returns:
            Full URI string
        """
        if attr_name.startswith("http://") or attr_name.startswith("https://"):
            return attr_name
        return f"{self.config.namespaces.attribute_prefix}{attr_name}"

    def format_class_uri(self, class_name: str) -> str:
        """
        Format a class name as a full URI.

        Args:
            class_name: The class name (e.g., "Paper" or full URI)

        Returns:
            Full URI string
        """
        if class_name.startswith("http://") or class_name.startswith("https://"):
            return class_name
        if self.config.namespaces.class_prefix:
            return f"{self.config.namespaces.class_prefix}{class_name}"
        return class_name

    def extract_id_from_uri(self, uri: str) -> str:
        """
        Extract the local ID from a full URI.

        Args:
            uri: Full URI string

        Returns:
            Local ID string
        """
        return self.config.namespaces.extract_id_from_uri(uri)

    def extract_prefix_and_id(self, uri: str) -> tuple[str, str]:
        """
        Extract both the prefix short name and local ID from a URI.

        Args:
            uri: Full URI string

        Returns:
            Tuple of (prefix_short_name, local_id)
        """
        for short, long_uri in self.config.namespaces.prefix_map.items():
            if uri.startswith(long_uri):
                return short, uri[len(long_uri):]

        # Fallback
        if "#" in uri:
            return "", uri.split("#")[-1]
        return "", uri.split("/")[-1]

    def format_prefixed_uri(self, uri: str) -> str:
        """
        Convert a full URI to prefixed format (e.g., ex:Q123).

        Args:
            uri: Full URI string

        Returns:
            Prefixed URI string (e.g., "ex:Q123")
        """
        prefix, local_id = self.extract_prefix_and_id(uri)
        if prefix:
            return f"{prefix}:{local_id}"
        return uri

    # =========================================================================
    # SPARQL Utilities
    # =========================================================================

    def inject_sparql_prefixes(self, query: str) -> str:
        """
        Inject the standard SPARQL prefixes into a query.

        Args:
            query: SPARQL query string (may or may not have prefixes)

        Returns:
            Query with prefixes prepended if not already present
        """
        prefixes = self.config.namespaces.sparql_prefixes

        # Check if query already has PREFIX declarations
        if query.strip().upper().startswith("PREFIX"):
            return query

        return f"{prefixes}\n{query}"

    def inject_graph_clause(self, query: str) -> str:
        """
        Inject a FROM clause for the named graph if configured.

        Args:
            query: SPARQL query string

        Returns:
            Query with FROM clause added if graph_uri is set
        """
        graph_uri = self.config.graph.graph_uri
        if not graph_uri:
            return query

        # Check if query already has a FROM clause
        if "FROM" in query.upper():
            return query

        # Find the position to insert FROM clause
        # It should go after PREFIX declarations and before SELECT/ASK/CONSTRUCT
        lines = query.split('\n')
        result_lines = []
        inserted = False

        for line in lines:
            stripped = line.strip().upper()
            # Insert FROM before SELECT, ASK, CONSTRUCT, or DESCRIBE
            if (not inserted and
                (stripped.startswith("SELECT") or
                 stripped.startswith("ASK") or
                 stripped.startswith("CONSTRUCT") or
                 stripped.startswith("DESCRIBE"))):
                result_lines.append(f"FROM <{graph_uri}>")
                inserted = True
            result_lines.append(line)

        return '\n'.join(result_lines)

    def prepare_sparql_query(self, query: str) -> str:
        """
        Prepare a SPARQL query by injecting prefixes and graph clause.

        Args:
            query: Raw SPARQL query

        Returns:
            Prepared query ready for execution
        """
        query = self.inject_sparql_prefixes(query)
        query = self.inject_graph_clause(query)
        return query

    # =========================================================================
    # Validation Utilities
    # =========================================================================

    def is_valid_entity_id(self, entity_id: str) -> bool:
        """
        Check if an entity ID is valid for this KG.

        Override in subclass for KG-specific validation.

        Args:
            entity_id: Entity ID to validate

        Returns:
            True if valid
        """
        return bool(entity_id and len(entity_id) > 0)

    def is_valid_property_id(self, prop_id: str) -> bool:
        """
        Check if a property ID is valid for this KG.

        Override in subclass for KG-specific validation.

        Args:
            prop_id: Property ID to validate

        Returns:
            True if valid
        """
        return bool(prop_id and len(prop_id) > 0)

    def normalize_entity_id(self, entity_id: str) -> str:
        """
        Normalize an entity ID to canonical form.

        Override in subclass for KG-specific normalization.

        Args:
            entity_id: Entity ID to normalize

        Returns:
            Normalized entity ID
        """
        # If it's a full URI, extract the ID
        if entity_id.startswith("http://") or entity_id.startswith("https://"):
            return self.extract_id_from_uri(entity_id)
        return entity_id

    def normalize_property_name(self, prop_name: str) -> str:
        """
        Normalize a property name to canonical form.

        Override in subclass for KG-specific normalization.

        Args:
            prop_name: Property name to normalize

        Returns:
            Normalized property name
        """
        # If it's a full URI, extract the ID
        if prop_name.startswith("http://") or prop_name.startswith("https://"):
            return self.extract_id_from_uri(prop_name)
        return prop_name

    # =========================================================================
    # Feature Detection
    # =========================================================================

    def supports_reification(self) -> bool:
        """Check if this KG supports reification (facts about facts)."""
        return self.config.graph.supports_reification

    def has_temporal_data(self) -> bool:
        """Check if this KG has temporal (time-based) data."""
        return self.config.graph.has_temporal_data

    def get_kg_name(self) -> str:
        """Get the human-readable name of this KG."""
        return self.config.name

    def get_kg_code(self) -> str:
        """Get the short code for this KG."""
        return self.config.code
