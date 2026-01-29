"""
Query rewriting for improved retrieval performance.

Uses LLM to optimize user queries by:
- Expanding abbreviations and acronyms
- Adding specificity and context
- Removing conversational filler
- Reformulating for better semantic search
"""
import logging
from typing import Optional
from langchain_core.messages import HumanMessage

from src.generation.llm import LLMGenerator
from src.config.settings import settings

logger = logging.getLogger(__name__)


# Prompt template for query rewriting
REWRITE_PROMPT = """You are an expert at optimizing search queries for semantic retrieval systems.

Your task is to rewrite the user's question to make it more effective for finding relevant information in a document database.

OPTIMIZATION RULES:
1. Expand abbreviations and acronyms to full terms
2. Add specificity where the intent is clear
3. Remove conversational filler words (like, um, just, etc.)
4. Preserve the core intent and meaning
5. Keep it concise (1-2 sentences max)
6. Use clear, direct language
7. If the query is already well-formed, return it with minimal changes

EXAMPLES:
- "What's RAG?" → "What is Retrieval Augmented Generation (RAG)?"
- "How do I setup the thing?" → "How do I set up and configure the system?"
- "Tell me about ML models" → "What are machine learning models and their applications?"
- "API docs" → "Where can I find the API documentation?"

Original query: {query}

Rewritten query (only output the rewritten query, nothing else):"""


class QueryRewriter:
    """
    Rewrites user queries to optimize retrieval performance.

    Uses an LLM with carefully tuned temperature to balance creativity
    and determinism when reformulating queries for better semantic search.

    Key Features:
    - Expands abbreviations for clarity
    - Adds context where appropriate
    - Removes conversational noise
    - Falls back gracefully on errors
    - Skips rewriting for very short queries

    Example:
        >>> rewriter = QueryRewriter()
        >>> optimized = rewriter.rewrite("What's LLM?")
        >>> print(optimized)
        "What is a Large Language Model (LLM)?"
    """

    def __init__(
        self,
        temperature: float = 0.3,
        max_tokens: int = 100,
    ):
        """
        Initialize query rewriter.

        Args:
            temperature: LLM temperature (0.3 = mostly deterministic with some creativity)
            max_tokens: Maximum tokens in rewritten query (default: 100)
        """
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Initialize LLM with specific settings for query rewriting
        self.llm = LLMGenerator(
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            streaming=False,
        )

        logger.info(
            f"Initialized QueryRewriter: temperature={temperature}, "
            f"max_tokens={max_tokens}"
        )

    def rewrite(self, query: str) -> str:
        """
        Rewrite a query to optimize it for retrieval.

        The rewriting process:
        1. Validates query length (skips very short queries)
        2. Sends query to LLM with optimization instructions
        3. Extracts and cleans the rewritten query
        4. Falls back to original query on any errors

        Args:
            query: Original user query

        Returns:
            Optimized query string (or original if rewriting fails/unnecessary)

        Example:
            >>> rewriter = QueryRewriter()
            >>> rewriter.rewrite("What's RAG?")
            "What is Retrieval Augmented Generation (RAG)?"
        """
        # Validate input
        if not query or not query.strip():
            logger.warning("Empty query provided, returning as-is")
            return query

        query = query.strip()

        # Skip rewriting for very short queries
        if len(query) < 10:
            logger.debug(f"Query too short ({len(query)} chars), skipping rewrite: '{query}'")
            return query

        logger.info(f"Rewriting query: '{query}'")

        try:
            # Format prompt with user query
            prompt = REWRITE_PROMPT.format(query=query)

            # Generate rewritten query
            messages = [HumanMessage(content=prompt)]
            rewritten = self.llm.generate(messages)

            # Clean up the response
            rewritten = rewritten.strip()

            # Remove common prefix artifacts if present
            if rewritten.lower().startswith("rewritten query:"):
                rewritten = rewritten[len("rewritten query:"):].strip()
            if rewritten.startswith('"') and rewritten.endswith('"'):
                rewritten = rewritten[1:-1].strip()
            if rewritten.startswith("'") and rewritten.endswith("'"):
                rewritten = rewritten[1:-1].strip()

            # Validate rewritten query is reasonable
            if not rewritten or len(rewritten) < 3:
                logger.warning("Rewritten query too short, using original")
                return query

            # Log the transformation
            if rewritten.lower() != query.lower():
                logger.info(f"Query rewritten: '{query}' → '{rewritten}'")
            else:
                logger.debug("Query unchanged after rewriting")

            return rewritten

        except Exception as e:
            logger.error(f"Error rewriting query: {str(e)}, falling back to original")
            logger.debug(f"Error details: ", exc_info=True)
            return query

    def rewrite_batch(self, queries: list[str]) -> list[str]:
        """
        Rewrite multiple queries in batch.

        Note: Currently processes queries sequentially. Future optimization
        could use LLM batch generation for better performance.

        Args:
            queries: List of query strings

        Returns:
            List of rewritten queries in same order

        Example:
            >>> rewriter = QueryRewriter()
            >>> queries = ["What's ML?", "API docs"]
            >>> rewriter.rewrite_batch(queries)
            ["What is Machine Learning (ML)?", "Where can I find API documentation?"]
        """
        if not queries:
            logger.warning("Empty query list provided")
            return []

        logger.info(f"Batch rewriting {len(queries)} queries")

        rewritten_queries = []
        for query in queries:
            rewritten = self.rewrite(query)
            rewritten_queries.append(rewritten)

        logger.info(f"Completed batch rewriting of {len(queries)} queries")
        return rewritten_queries

    def get_rewriter_config(self) -> dict:
        """
        Get current rewriter configuration.

        Returns:
            Dictionary with temperature, max_tokens, and model info
        """
        return {
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'llm_config': self.llm.get_model_info(),
        }


# Convenience function
def rewrite_query(query: str, temperature: float = 0.3) -> str:
    """
    Convenience function to rewrite a single query.

    Creates a QueryRewriter instance and rewrites the query.
    For multiple queries, prefer creating a QueryRewriter instance
    and reusing it.

    Args:
        query: Query string to rewrite
        temperature: LLM temperature (default: 0.3)

    Returns:
        Rewritten query string

    Example:
        >>> from src.agents.query_rewriter import rewrite_query
        >>> rewrite_query("What's NLP?")
        "What is Natural Language Processing (NLP)?"
    """
    rewriter = QueryRewriter(temperature=temperature)
    return rewriter.rewrite(query)
