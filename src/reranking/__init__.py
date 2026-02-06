"""
Document re-ranking module for improved retrieval precision.

This module provides re-ranking strategies to improve the ordering of
retrieved documents by relevance.

Classes:
    LLMReranker: LLM-based relevance scoring and re-ranking
    HybridReranker: Combines vector similarity and LLM scores

Functions:
    create_llm_reranker: Create an LLM re-ranker instance
    create_hybrid_reranker: Create a hybrid re-ranker instance

Example:
    >>> from src.reranking import LLMReranker, HybridReranker
    >>>
    >>> # LLM-based re-ranking
    >>> reranker = LLMReranker()
    >>> docs = retriever.retrieve("What is RAG?")
    >>> ranked = reranker.rerank("What is RAG?", docs, top_k=5)
    >>>
    >>> # Hybrid re-ranking (combines similarity + LLM scores)
    >>> hybrid = HybridReranker(reranker, llm_weight=0.7)
    >>> docs_with_scores = retriever.retrieve(query, return_scores=True)
    >>> ranked = hybrid.rerank(query, docs_with_scores)
"""

from src.reranking.reranker import (
    LLMReranker,
    HybridReranker,
    create_llm_reranker,
    create_hybrid_reranker,
    RELEVANCE_PROMPT,
)

__all__ = [
    'LLMReranker',
    'HybridReranker',
    'create_llm_reranker',
    'create_hybrid_reranker',
    'RELEVANCE_PROMPT',
]
