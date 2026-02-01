"""
Generic KBQA Framework.

This package provides abstract base classes for KBQA agents and their tools,
enabling code reuse across different knowledge graph implementations (KQAPro, SciQA/ORKG, etc.).
"""

from ama_kbqa.framework.types import (
    EntityMatch,
    AttributeValue,
    NodeDetails,
    NavigationResult,
    QualifierResult,
    SPARQLResult,
)
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

__all__ = [
    # Types
    "EntityMatch",
    "AttributeValue",
    "NodeDetails",
    "NavigationResult",
    "QualifierResult",
    "SPARQLResult",
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
]
