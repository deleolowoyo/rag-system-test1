"""
Agents package for RAG system Phase 2.

Contains advanced reasoning and optimization agents:
- QueryRewriter: Optimizes user queries for better retrieval
- Self-critique: Validates generated answers (coming soon)
- ReAct Agent: Multi-step reasoning loops (coming soon)
"""

from src.agents.query_rewriter import QueryRewriter

__all__ = [
    'QueryRewriter',
]

__version__ = "2.0.0"
