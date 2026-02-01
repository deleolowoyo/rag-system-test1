"""
Multi-query generation for improved retrieval recall.

Generates multiple query variations to explore different angles
of the same question, improving the chances of finding all relevant documents.
"""
import logging
import hashlib
from typing import List, Optional, Set
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document

from src.generation.llm import LLMGenerator
from src.retrieval.retriever import AdvancedRetriever
from src.config.settings import settings

logger = logging.getLogger(__name__)


# Prompt for generating query variations
MULTI_QUERY_PROMPT = """You are an expert at generating query variations for information retrieval.

Your task is to generate {num_queries} different versions of the user's question. Each variation should:
1. Explore the question from a different angle or perspective
2. Use different keywords and phrasing
3. Maintain the same core intent and information need
4. Be specific and searchable

IMPORTANT: Generate EXACTLY {num_queries} variations. Output ONLY the questions, one per line, numbered.

Example:
Original: "What is RAG?"
Variations:
1. What is Retrieval Augmented Generation and how does it work?
2. How does RAG combine retrieval and language models?
3. What are the key components of a RAG system?

Original question: {query}

Generate {num_queries} variations (numbered 1-{num_queries}):"""


class MultiQueryGenerator:
    """
    Generates multiple variations of a user query for improved retrieval.

    Uses an LLM with higher temperature (0.7) to create diverse query
    variations that explore different angles of the same question.

    Example:
        >>> generator = MultiQueryGenerator()
        >>> queries = generator.generate_queries("What is RAG?", num_queries=3)
        >>> print(queries)
        [
            "What is RAG?",  # Original query always included
            "What is Retrieval Augmented Generation and how does it work?",
            "How does RAG combine retrieval and language models?",
            "What are the key components of a RAG system?"
        ]
    """

    def __init__(
        self,
        temperature: float = 0.7,
        max_tokens: int = 300,
    ):
        """
        Initialize multi-query generator.

        Args:
            temperature: LLM temperature (0.7 = more diverse variations)
            max_tokens: Maximum tokens for generated queries
        """
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Initialize LLM with higher temperature for diversity
        self.llm = LLMGenerator(
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            streaming=False,
        )

        logger.info(
            f"Initialized MultiQueryGenerator: temperature={temperature}, "
            f"max_tokens={max_tokens}"
        )

    def generate_queries(
        self,
        query: str,
        num_queries: int = 3,
    ) -> List[str]:
        """
        Generate multiple variations of a query.

        Args:
            query: Original user query
            num_queries: Number of variations to generate (default: 3)

        Returns:
            List of queries including original and variations

        Example:
            >>> generator = MultiQueryGenerator()
            >>> queries = generator.generate_queries("What is machine learning?", num_queries=3)
            >>> len(queries)
            4  # Original + 3 variations
        """
        if not query or not query.strip():
            logger.warning("Empty query provided")
            return [query] if query is not None else []

        query = query.strip()

        logger.info(f"Generating {num_queries} query variations for: '{query}'")

        try:
            # Format prompt
            prompt = MULTI_QUERY_PROMPT.format(
                num_queries=num_queries,
                query=query,
            )

            # Generate variations
            messages = [HumanMessage(content=prompt)]
            response = self.llm.generate(messages)

            # Parse variations from response
            variations = self._parse_variations(response, num_queries)

            # Always include original query
            all_queries = [query] + variations

            # Remove duplicates while preserving order
            unique_queries = []
            seen = set()
            for q in all_queries:
                q_lower = q.lower().strip()
                if q_lower not in seen:
                    seen.add(q_lower)
                    unique_queries.append(q)

            logger.info(f"Generated {len(unique_queries)} unique queries:")
            for i, q in enumerate(unique_queries, 1):
                logger.info(f"  {i}. {q}")

            return unique_queries

        except Exception as e:
            logger.error(f"Error generating query variations: {str(e)}, using original only")
            logger.debug("Error details: ", exc_info=True)
            return [query]

    def _parse_variations(self, response: str, expected_count: int) -> List[str]:
        """
        Parse query variations from LLM response.

        Handles various formats:
        - Numbered lists (1., 2., 3.)
        - Bullet points (-, *, •)
        - Plain lines

        Args:
            response: LLM response text
            expected_count: Expected number of variations

        Returns:
            List of parsed query variations
        """
        if not response or not response.strip():
            logger.warning("Empty response from LLM")
            return []

        lines = response.strip().split('\n')
        variations = []

        for line in lines:
            line = line.strip()

            # Skip empty lines
            if not line:
                continue

            # Remove common prefixes
            # Numbered: "1.", "1)", "1:"
            if line and line[0].isdigit():
                # Find where the actual query starts
                for i, char in enumerate(line):
                    if char in '.):':
                        line = line[i+1:].strip()
                        break

            # Bullet points: "- ", "* ", "• "
            for prefix in ['- ', '* ', '• ', '→ ', '> ']:
                if line.startswith(prefix):
                    line = line[len(prefix):].strip()
                    break

            # Remove quotes if present
            if line.startswith('"') and line.endswith('"'):
                line = line[1:-1].strip()
            elif line.startswith("'") and line.endswith("'"):
                line = line[1:-1].strip()

            # Add if non-empty and looks like a question
            if line and len(line) > 5:
                variations.append(line)

        # Limit to expected count
        variations = variations[:expected_count]

        logger.debug(f"Parsed {len(variations)} variations from response")
        return variations

    def get_generator_config(self) -> dict:
        """Get current generator configuration."""
        return {
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'llm_config': self.llm.get_model_info(),
        }


class MultiQueryRetriever:
    """
    Retrieves documents using multiple query variations.

    Combines MultiQueryGenerator with a retriever to fetch documents
    for multiple query variations, then deduplicates and merges results.

    Example:
        >>> from src.retrieval.retriever import AdvancedRetriever
        >>> from src.storage.vector_store import VectorStoreManager
        >>>
        >>> vector_store = VectorStoreManager()
        >>> base_retriever = AdvancedRetriever(vector_store)
        >>> multi_retriever = MultiQueryRetriever(base_retriever)
        >>>
        >>> docs = multi_retriever.retrieve("What is RAG?")
        >>> # Returns deduplicated documents from multiple query variations
    """

    def __init__(
        self,
        retriever: AdvancedRetriever,
        generator: Optional[MultiQueryGenerator] = None,
        num_queries: int = 3,
        top_k_per_query: int = 3,
    ):
        """
        Initialize multi-query retriever.

        Args:
            retriever: Base retriever to use for document retrieval
            generator: Query generator (creates new one if not provided)
            num_queries: Number of query variations to generate
            top_k_per_query: Documents to retrieve per query variation
        """
        self.retriever = retriever
        self.generator = generator or MultiQueryGenerator()
        self.num_queries = num_queries
        self.top_k_per_query = top_k_per_query

        logger.info(
            f"Initialized MultiQueryRetriever: "
            f"num_queries={num_queries}, top_k_per_query={top_k_per_query}"
        )

    def retrieve(
        self,
        query: str,
        num_queries: Optional[int] = None,
        top_k_per_query: Optional[int] = None,
        return_scores: bool = False,
    ) -> List[Document] | List[tuple[Document, float]]:
        """
        Retrieve documents using multiple query variations.

        Process:
        1. Generate query variations
        2. Retrieve documents for each variation
        3. Deduplicate by content hash
        4. Merge and return results

        Args:
            query: Original user query
            num_queries: Override default number of query variations
            top_k_per_query: Override default documents per query
            return_scores: Whether to return similarity scores

        Returns:
            List of deduplicated documents (or doc/score tuples)
        """
        num_queries = num_queries or self.num_queries
        top_k = top_k_per_query or self.top_k_per_query

        logger.info(f"Multi-query retrieval for: '{query}'")
        logger.info(f"Generating {num_queries} variations, retrieving {top_k} docs per query")

        # Generate query variations
        queries = self.generator.generate_queries(query, num_queries=num_queries)

        logger.info(f"Generated {len(queries)} total queries (including original)")

        # Retrieve documents for each query
        all_docs = []
        seen_hashes: Set[str] = set()

        for i, q in enumerate(queries, 1):
            logger.info(f"Retrieving for query {i}/{len(queries)}: '{q}'")

            try:
                # Retrieve documents
                docs = self.retriever.retrieve(
                    query=q,
                    top_k=top_k,
                    return_scores=return_scores,
                )

                # Process documents
                if return_scores:
                    # Filter duplicates for scored results
                    for doc, score in docs:
                        doc_hash = self._hash_document(doc)
                        if doc_hash not in seen_hashes:
                            seen_hashes.add(doc_hash)
                            all_docs.append((doc, score))
                else:
                    # Filter duplicates for regular results
                    for doc in docs:
                        doc_hash = self._hash_document(doc)
                        if doc_hash not in seen_hashes:
                            seen_hashes.add(doc_hash)
                            all_docs.append(doc)

                logger.debug(f"Retrieved {len(docs)} docs, {len(all_docs)} unique so far")

            except Exception as e:
                logger.warning(f"Error retrieving for query '{q}': {str(e)}")
                continue

        logger.info(
            f"Multi-query retrieval complete: {len(all_docs)} unique documents "
            f"from {len(queries)} queries"
        )

        return all_docs

    def _hash_document(self, doc: Document) -> str:
        """
        Generate hash for document deduplication.

        Uses content and source metadata to create a unique identifier.

        Args:
            doc: Document to hash

        Returns:
            Hash string
        """
        # Combine content and source for hashing
        content = doc.page_content
        source = doc.metadata.get('source', '')

        # Create hash
        hash_input = f"{content}|{source}".encode('utf-8')
        return hashlib.md5(hash_input).hexdigest()

    def get_retriever_config(self) -> dict:
        """Get current multi-query retriever configuration."""
        return {
            'num_queries': self.num_queries,
            'top_k_per_query': self.top_k_per_query,
            'generator_config': self.generator.get_generator_config(),
            'base_retriever_config': self.retriever.get_retriever_config(),
        }


# Convenience functions
def generate_query_variations(
    query: str,
    num_queries: int = 3,
    temperature: float = 0.7,
) -> List[str]:
    """
    Convenience function to generate query variations.

    Args:
        query: Original query
        num_queries: Number of variations
        temperature: LLM temperature

    Returns:
        List of query variations including original
    """
    generator = MultiQueryGenerator(temperature=temperature)
    return generator.generate_queries(query, num_queries=num_queries)
