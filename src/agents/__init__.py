"""
Agents package for RAG system Phase 2.

Contains advanced reasoning and optimization agents:
- QueryRewriter: Optimizes user queries for better retrieval
- MultiQueryGenerator: Generates query variations for improved recall
- MultiQueryRetriever: Retrieves using multiple query variations
- Self-critique: Validates generated answers (coming soon)
- ReAct Agent: Multi-step reasoning loops (coming soon)
"""

from src.agents.query_rewriter import QueryRewriter
from src.agents.multi_query import (
    MultiQueryGenerator,
    MultiQueryRetriever,
    generate_query_variations,
)

__all__ = [
    'QueryRewriter',
    'MultiQueryGenerator',
    'MultiQueryRetriever',
    'generate_query_variations',
]

__version__ = "2.0.0"
