"""
Document re-ranking for improved retrieval precision.

Provides LLM-based and hybrid re-ranking strategies to improve the ordering
of retrieved documents by relevance to the user's query.
"""
import logging
import re
from typing import List, Tuple, Optional
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

from src.generation.llm import LLMGenerator
from src.config.settings import settings

logger = logging.getLogger(__name__)


# Prompt template for LLM-based relevance scoring
RELEVANCE_PROMPT = """You are an expert at evaluating document relevance for search queries.

Your task is to rate how relevant a document is to a given query on a scale of 0-10.

RATING SCALE:
0 - Completely irrelevant, no connection to query
1-3 - Minimally relevant, tangential mention
4-6 - Somewhat relevant, contains related information
7-8 - Highly relevant, directly addresses query
9-10 - Perfectly relevant, comprehensive answer

INSTRUCTIONS:
1. Consider semantic relevance, not just keyword matching
2. Evaluate how well the content answers or relates to the query
3. Account for specificity and completeness
4. Be objective and consistent in your ratings
5. Return ONLY a single number from 0-10, nothing else

Query: {query}

Document:
{document}

Relevance score (0-10):"""


class LLMReranker:
    """
    LLM-based document re-ranker using relevance scoring.

    Uses an LLM with deterministic settings (temperature=0.0) to score
    each document's relevance to a query on a 0-10 scale, then re-ranks
    documents by score.

    Key Features:
    - Deterministic scoring (temperature=0.0)
    - Graceful error handling (defaults to 5.0)
    - Document truncation to avoid context limits
    - Detailed logging of scores

    Example:
        >>> reranker = LLMReranker()
        >>> docs = retriever.retrieve("What is RAG?")
        >>> ranked = reranker.rerank("What is RAG?", docs, top_k=5)
        >>> for doc, score in ranked:
        ...     print(f"Score: {score:.1f} - {doc.page_content[:50]}")
    """

    def __init__(
        self,
        temperature: float = 0.0,
        max_tokens: int = 10,
        max_doc_length: int = 1000,
    ):
        """
        Initialize LLM re-ranker.

        Args:
            temperature: LLM temperature (0.0 = deterministic)
            max_tokens: Maximum tokens in response (should be small, just a number)
            max_doc_length: Maximum document chars to send to LLM (default: 1000)
        """
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_doc_length = max_doc_length

        # Initialize LLM with deterministic settings
        self.llm = LLMGenerator(
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            streaming=False,
        )

        logger.info(
            f"Initialized LLMReranker: temperature={temperature}, "
            f"max_doc_length={max_doc_length}"
        )

    def score_document(
        self,
        query: str,
        document: Document,
    ) -> float:
        """
        Score a single document's relevance to a query.

        Args:
            query: User query
            document: Document to score

        Returns:
            Relevance score from 0.0 to 10.0

        Example:
            >>> reranker = LLMReranker()
            >>> doc = Document(page_content="RAG is Retrieval Augmented Generation")
            >>> score = reranker.score_document("What is RAG?", doc)
            >>> print(f"Relevance: {score}/10")
        """
        try:
            # Truncate document content if needed
            content = document.page_content
            if len(content) > self.max_doc_length:
                content = content[:self.max_doc_length] + "..."
                logger.debug(
                    f"Truncated document from {len(document.page_content)} "
                    f"to {self.max_doc_length} chars"
                )

            # Format prompt
            prompt = RELEVANCE_PROMPT.format(
                query=query,
                document=content,
            )

            # Get score from LLM
            messages = [HumanMessage(content=prompt)]
            response = self.llm.generate(messages).strip()

            # Parse score from response
            score = self._parse_score(response)

            logger.debug(
                f"Scored document: {score:.1f}/10 "
                f"(content: '{content[:50]}...')"
            )

            return score

        except Exception as e:
            logger.warning(
                f"Error scoring document: {str(e)}, using default score 5.0"
            )
            logger.debug("Error details: ", exc_info=True)
            return 5.0  # Default middle-range score on error

    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: Optional[int] = None,
    ) -> List[Tuple[Document, float]]:
        """
        Re-rank documents by LLM-scored relevance.

        Scores each document independently, then sorts by score descending.
        Returns the top_k highest-scoring documents.

        Args:
            query: User query
            documents: List of documents to re-rank
            top_k: Number of top documents to return (default: all)

        Returns:
            List of (document, score) tuples sorted by score descending

        Example:
            >>> reranker = LLMReranker()
            >>> docs = retriever.retrieve("What is RAG?")
            >>> ranked = reranker.rerank("What is RAG?", docs, top_k=3)
            >>> print(f"Top document score: {ranked[0][1]:.1f}/10")
        """
        if not documents:
            logger.warning("No documents provided for re-ranking")
            return []

        logger.info(
            f"Re-ranking {len(documents)} documents for query: '{query[:50]}...'"
        )

        # Score all documents
        scored_docs = []
        for i, doc in enumerate(documents):
            score = self.score_document(query, doc)
            scored_docs.append((doc, score))
            logger.info(
                f"Document {i+1}/{len(documents)}: score={score:.1f}/10 "
                f"(source: {doc.metadata.get('source', 'unknown')})"
            )

        # Sort by score descending
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        # Return top_k if specified
        if top_k:
            scored_docs = scored_docs[:top_k]
            logger.info(f"Returning top {top_k} documents after re-ranking")
        else:
            logger.info(f"Returning all {len(scored_docs)} re-ranked documents")

        # Log summary
        if scored_docs:
            avg_score = sum(score for _, score in scored_docs) / len(scored_docs)
            logger.info(
                f"Re-ranking complete: avg_score={avg_score:.2f}, "
                f"top_score={scored_docs[0][1]:.1f}, "
                f"bottom_score={scored_docs[-1][1]:.1f}"
            )

        return scored_docs

    def _parse_score(self, response: str) -> float:
        """
        Parse score from LLM response.

        Extracts a number from the response and validates it's in range 0-10.

        Args:
            response: LLM response string

        Returns:
            Parsed score (0.0-10.0)

        Raises:
            ValueError: If no valid score can be parsed
        """
        # Try to extract a number from the response
        # Look for patterns like "8", "8.5", "8/10", "Score: 8"
        patterns = [
            r'(\d+\.?\d*)\s*/?\s*10',  # "8/10" or "8 / 10"
            r'(\d+\.?\d*)',             # Just a number
        ]

        for pattern in patterns:
            match = re.search(pattern, response)
            if match:
                try:
                    score = float(match.group(1))

                    # Validate range
                    if 0 <= score <= 10:
                        return score
                    elif score > 10:
                        logger.warning(
                            f"Score {score} > 10, clamping to 10.0"
                        )
                        return 10.0
                    else:
                        logger.warning(
                            f"Score {score} < 0, clamping to 0.0"
                        )
                        return 0.0

                except ValueError:
                    continue

        # If no valid number found, raise error
        raise ValueError(
            f"Could not parse score from response: '{response}'"
        )

    def get_reranker_config(self) -> dict:
        """
        Get current re-ranker configuration.

        Returns:
            Dictionary with temperature, max_tokens, max_doc_length, and model info
        """
        return {
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'max_doc_length': self.max_doc_length,
            'llm_config': self.llm.get_model_info(),
        }


class HybridReranker:
    """
    Hybrid re-ranker combining vector similarity and LLM scores.

    Normalizes and combines scores from both vector similarity search and
    LLM-based relevance scoring using configurable weights.

    Key Features:
    - Combines vector similarity (fast) with LLM scoring (accurate)
    - Configurable weighting between methods
    - Score normalization for fair combination
    - Best of both worlds: speed and precision

    Example:
        >>> llm_reranker = LLMReranker()
        >>> hybrid = HybridReranker(llm_reranker, llm_weight=0.7)
        >>> # Get docs with similarity scores from retriever
        >>> docs_with_scores = retriever.retrieve(query, return_scores=True)
        >>> ranked = hybrid.rerank(query, docs_with_scores)
    """

    def __init__(
        self,
        llm_reranker: LLMReranker,
        llm_weight: float = 0.7,
        similarity_weight: float = 0.3,
    ):
        """
        Initialize hybrid re-ranker.

        Args:
            llm_reranker: LLMReranker instance for LLM scoring
            llm_weight: Weight for LLM scores (default: 0.7)
            similarity_weight: Weight for similarity scores (default: 0.3)

        Raises:
            ValueError: If weights don't sum to 1.0
        """
        if abs(llm_weight + similarity_weight - 1.0) > 1e-6:
            raise ValueError(
                f"Weights must sum to 1.0, got {llm_weight + similarity_weight}"
            )

        self.llm_reranker = llm_reranker
        self.llm_weight = llm_weight
        self.similarity_weight = similarity_weight

        logger.info(
            f"Initialized HybridReranker: llm_weight={llm_weight}, "
            f"similarity_weight={similarity_weight}"
        )

    def rerank(
        self,
        query: str,
        documents_with_scores: List[Tuple[Document, float]],
        top_k: Optional[int] = None,
    ) -> List[Tuple[Document, float]]:
        """
        Re-rank documents using hybrid scoring.

        Combines normalized vector similarity scores with LLM relevance scores
        using configured weights.

        Args:
            query: User query
            documents_with_scores: List of (document, similarity_score) tuples
            top_k: Number of top documents to return (default: all)

        Returns:
            List of (document, combined_score) tuples sorted descending

        Example:
            >>> hybrid = HybridReranker(llm_reranker, llm_weight=0.7)
            >>> docs_with_scores = retriever.retrieve(query, return_scores=True)
            >>> ranked = hybrid.rerank(query, docs_with_scores, top_k=5)
            >>> for doc, score in ranked:
            ...     print(f"Combined score: {score:.2f}")
        """
        if not documents_with_scores:
            logger.warning("No documents provided for hybrid re-ranking")
            return []

        logger.info(
            f"Hybrid re-ranking {len(documents_with_scores)} documents "
            f"for query: '{query[:50]}...'"
        )

        # Extract documents and similarity scores
        documents = [doc for doc, _ in documents_with_scores]
        similarity_scores = [score for _, score in documents_with_scores]

        # Normalize similarity scores to 0-10 scale
        normalized_sim_scores = self._normalize_scores(similarity_scores)

        # Get LLM scores for all documents
        llm_scores = []
        for doc in documents:
            llm_score = self.llm_reranker.score_document(query, doc)
            llm_scores.append(llm_score)

        # Combine scores with weights
        combined_scored_docs = []
        for i, doc in enumerate(documents):
            combined_score = (
                self.llm_weight * llm_scores[i] +
                self.similarity_weight * normalized_sim_scores[i]
            )
            combined_scored_docs.append((doc, combined_score))

            logger.info(
                f"Document {i+1}: sim={normalized_sim_scores[i]:.1f}/10, "
                f"llm={llm_scores[i]:.1f}/10, "
                f"combined={combined_score:.2f}/10 "
                f"(source: {doc.metadata.get('source', 'unknown')})"
            )

        # Sort by combined score descending
        combined_scored_docs.sort(key=lambda x: x[1], reverse=True)

        # Return top_k if specified
        if top_k:
            combined_scored_docs = combined_scored_docs[:top_k]
            logger.info(f"Returning top {top_k} documents after hybrid re-ranking")
        else:
            logger.info(
                f"Returning all {len(combined_scored_docs)} hybrid re-ranked documents"
            )

        # Log summary
        if combined_scored_docs:
            avg_score = sum(s for _, s in combined_scored_docs) / len(combined_scored_docs)
            logger.info(
                f"Hybrid re-ranking complete: avg_score={avg_score:.2f}, "
                f"top_score={combined_scored_docs[0][1]:.2f}, "
                f"bottom_score={combined_scored_docs[-1][1]:.2f}"
            )

        return combined_scored_docs

    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """
        Normalize scores to 0-10 scale using min-max normalization.

        Args:
            scores: List of raw scores

        Returns:
            List of normalized scores (0.0-10.0)
        """
        if not scores:
            return []

        min_score = min(scores)
        max_score = max(scores)

        # If all scores are the same, return middle value
        if max_score == min_score:
            logger.debug(
                f"All similarity scores identical ({min_score}), "
                "returning normalized score of 5.0"
            )
            return [5.0] * len(scores)

        # Min-max normalization to 0-10 scale
        normalized = [
            10.0 * (score - min_score) / (max_score - min_score)
            for score in scores
        ]

        logger.debug(
            f"Normalized {len(scores)} scores: "
            f"range [{min_score:.3f}, {max_score:.3f}] → [0.0, 10.0]"
        )

        return normalized

    def get_reranker_config(self) -> dict:
        """
        Get current hybrid re-ranker configuration.

        Returns:
            Dictionary with weights and LLM re-ranker config
        """
        return {
            'llm_weight': self.llm_weight,
            'similarity_weight': self.similarity_weight,
            'llm_reranker_config': self.llm_reranker.get_reranker_config(),
        }


# Convenience functions
def create_llm_reranker(**kwargs) -> LLMReranker:
    """
    Create an LLM re-ranker instance.

    Args:
        **kwargs: LLMReranker configuration

    Returns:
        LLMReranker instance

    Example:
        >>> reranker = create_llm_reranker(max_doc_length=500)
    """
    return LLMReranker(**kwargs)


def create_hybrid_reranker(
    llm_weight: float = 0.7,
    similarity_weight: float = 0.3,
    **llm_reranker_kwargs,
) -> HybridReranker:
    """
    Create a hybrid re-ranker instance.

    Args:
        llm_weight: Weight for LLM scores (default: 0.7)
        similarity_weight: Weight for similarity scores (default: 0.3)
        **llm_reranker_kwargs: Additional LLMReranker configuration

    Returns:
        HybridReranker instance

    Example:
        >>> reranker = create_hybrid_reranker(llm_weight=0.6, similarity_weight=0.4)
    """
    llm_reranker = LLMReranker(**llm_reranker_kwargs)
    return HybridReranker(
        llm_reranker=llm_reranker,
        llm_weight=llm_weight,
        similarity_weight=similarity_weight,
    )
