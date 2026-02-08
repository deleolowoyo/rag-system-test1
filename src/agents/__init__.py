"""
Agents package for RAG system Phase 2.

Contains advanced reasoning and optimization agents:
- QueryRewriter: Optimizes user queries for better retrieval
- MultiQueryGenerator: Generates query variations for improved recall
- MultiQueryRetriever: Retrieves using multiple query variations
- ReActAgent: Multi-step reasoning loops with iterative refinement
- SelfCritiqueAgent: Validates generated answers for quality and accuracy
"""

from src.agents.query_rewriter import QueryRewriter
from src.agents.multi_query import (
    MultiQueryGenerator,
    MultiQueryRetriever,
    generate_query_variations,
)
from src.agents.react_agent import (
    ReActAgent,
    AgentAction,
    AgentState,
    create_react_agent,
)
from src.agents.self_critique import (
    SelfCritiqueAgent,
    create_self_critique_agent,
)

__all__ = [
    'QueryRewriter',
    'MultiQueryGenerator',
    'MultiQueryRetriever',
    'generate_query_variations',
    'ReActAgent',
    'AgentAction',
    'AgentState',
    'create_react_agent',
    'SelfCritiqueAgent',
    'create_self_critique_agent',
]

__version__ = "2.0.0"
