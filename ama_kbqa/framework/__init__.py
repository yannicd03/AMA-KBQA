"""
Generic KBQA Framework.

This package provides abstract base classes for KBQA agents and their tools,
enabling code reuse across different knowledge graph implementations (KQAPro, SciQA/ORKG, etc.).
"""

from ama_kbqa.framework.config import (
    NamespaceConfig,
    VectorConfig,
    GraphConfig,
    PromptConfig,
    KnowledgeGraphConfig,
)
from ama_kbqa.framework.state import JournalState, JournalManager
from ama_kbqa.framework.mcp_client import MCPClient
from ama_kbqa.framework.base_agent import BaseKBQAAgent
from ama_kbqa.framework.operations import (
    AtomicOperation,
    ATOMIC_OPERATIONS,
    CoverageReport,
    validate_bindings,
)
from ama_kbqa.framework.deterministic import (
    NumericComparison,
    compare_numeric,
    parse_numeric,
)

__all__ = [
    # Config
    "NamespaceConfig",
    "VectorConfig",
    "GraphConfig",
    "PromptConfig",
    "KnowledgeGraphConfig",
    # State
    "JournalState",
    "JournalManager",
    # MCP
    "MCPClient",
    # Base Agent
    "BaseKBQAAgent",
    # Abstract operation contract
    "AtomicOperation",
    "ATOMIC_OPERATIONS",
    "CoverageReport",
    "validate_bindings",
    # Deterministic cores
    "NumericComparison",
    "compare_numeric",
    "parse_numeric",
]
