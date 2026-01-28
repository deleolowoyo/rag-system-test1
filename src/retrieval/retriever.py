"""
Advanced retrieval strategies for finding relevant documents.
Supports similarity search, MMR, and metadata filtering.
"""
import logging
from typing import List, Optional, Dict, Any
from langchain_core.documents import Document
from src.storage.vector_store import VectorStoreManager
from src.config.settings import settings

logger = logging.getLogger(__name__)


class AdvancedRetriever:
    """
    Advanced retriever with multiple search strategies.
    
    Search Types:
    - Similarity: Pure cosine similarity
    - MMR: Maximal Marginal Relevance (diversity)
    - Filtered: Metadata-based filtering
    """
    
    def __init__(
        self,
        vector_store: VectorStoreManager,
        search_type: str = None,
        top_k: int = None,
        score_threshold: float = None,
    ):
        """
        Initialize advanced retriever.
        
        Args:
            vector_store: Vector store to search
            search_type: Type of search ('similarity' or 'mmr')
            top_k: Number of results to return
            score_threshold: Minimum similarity score
        """
        self.vector_store = vector_store
        self.search_type = search_type or settings.search_type
        self.top_k = top_k or settings.retrieval_top_k
        self.score_threshold = score_threshold or settings.retrieval_score_threshold
        
        logger.info(
            f"Initialized retriever: type={self.search_type}, "
            f"top_k={self.top_k}, threshold={self.score_threshold}"
        )
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter: Optional[Dict[str, Any]] = None,
        return_scores: bool = False,
    ) -> List[Document] | List[tuple[Document, float]]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Search query
            top_k: Override default top_k
            filter: Metadata filter (e.g., {'source': 'file.pdf'})
            return_scores: Whether to return similarity scores
            
        Returns:
            List of documents or (document, score) tuples
        """
        k = top_k or self.top_k
        
        logger.info(f"Retrieving documents for query: '{query[:50]}...'")
        
        try:
            if return_scores:
                results = self.vector_store.similarity_search_with_score(
                    query=query,
                    k=k,
                    filter=filter,
                )
                
                # Filter by score threshold
                filtered_results = [
                    (doc, score) for doc, score in results
                    if score >= self.score_threshold
                ]
                
                logger.info(
                    f"Retrieved {len(filtered_results)} documents "
                    f"(scores: {[f'{s:.3f}' for _, s in filtered_results]})"
                )
                
                return filtered_results
            else:
                results = self.vector_store.similarity_search(
                    query=query,
                    k=k,
                    filter=filter,
                    score_threshold=self.score_threshold,
                )
                
                logger.info(f"Retrieved {len(results)} documents")
                return results
                
        except Exception as e:
            logger.error(f"Error during retrieval: {str(e)}")
            raise
    
    def retrieve_with_context(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Retrieve documents with rich context information.
        
        Args:
            query: Search query
            top_k: Number of results
            filter: Metadata filter
            
        Returns:
            Dictionary with documents, scores, and metadata
        """
        results = self.retrieve(
            query=query,
            top_k=top_k,
            filter=filter,
            return_scores=True,
        )
        
        # Build context dictionary
        context = {
            'query': query,
            'num_results': len(results),
            'documents': [],
        }
        
        for doc, score in results:
            context['documents'].append({
                'content': doc.page_content,
                'score': score,
                'metadata': doc.metadata,
                'source': doc.metadata.get('source', 'unknown'),
                'page': doc.metadata.get('page', None),
            })
        
        return context
    
    def retrieve_by_source(
        self,
        query: str,
        source: str,
        top_k: Optional[int] = None,
    ) -> List[Document]:
        """
        Retrieve documents filtered by source.
        
        Args:
            query: Search query
            source: Source file or identifier
            top_k: Number of results
            
        Returns:
            List of documents from specified source
        """
        return self.retrieve(
            query=query,
            top_k=top_k,
            filter={'source': source},
        )
    
    def get_retriever_config(self) -> Dict[str, Any]:
        """Get current retriever configuration."""
        return {
            'search_type': self.search_type,
            'top_k': self.top_k,
            'score_threshold': self.score_threshold,
        }


class MultiQueryRetriever:
    """
    Advanced retriever that generates multiple query variations
    for better recall. (Preview of Phase 2 capabilities)
    """
    
    def __init__(self, base_retriever: AdvancedRetriever):
        """
        Initialize multi-query retriever.
        
        Args:
            base_retriever: Base retriever to use
        """
        self.base_retriever = base_retriever
        logger.info("Initialized MultiQueryRetriever")
    
    def retrieve(
        self,
        query: str,
        num_variations: int = 3,
    ) -> List[Document]:
        """
        Retrieve using multiple query variations.
        
        Note: This is a simplified version. Full implementation
        in Phase 2 will use LLM to generate query variations.
        
        Args:
            query: Original query
            num_variations: Number of query variations to generate
            
        Returns:
            Deduplicated list of retrieved documents
        """
        # For now, just use the original query
        # Phase 2 will add LLM-based query generation
        logger.info(f"Multi-query retrieval for: '{query}'")
        
        return self.base_retriever.retrieve(query)


# Convenience function
def create_retriever(
    vector_store: VectorStoreManager,
    **kwargs,
) -> AdvancedRetriever:
    """
    Create a retriever instance.
    
    Args:
        vector_store: Vector store to search
        **kwargs: Additional retriever configuration
        
    Returns:
        AdvancedRetriever instance
    """
    return AdvancedRetriever(vector_store=vector_store, **kwargs)
