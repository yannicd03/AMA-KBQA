"""
Knowledge Graph Adapters.

This package contains adapter implementations for specific knowledge graphs.
Each adapter provides configuration and URI handling for its respective KG.
"""

from ama_kbqa.framework.adapters.base_adapter import BaseKGAdapter
from ama_kbqa.framework.adapters.kqapro_adapter import KQAProAdapter
from ama_kbqa.framework.adapters.sciqa_adapter import SciQAAdapter

__all__ = [
    "BaseKGAdapter",
    "KQAProAdapter",
    "SciQAAdapter",
]
