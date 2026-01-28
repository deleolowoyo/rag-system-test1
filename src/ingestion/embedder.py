"""
Embedding generation for documents and queries.
Handles batch processing and caching for efficiency.
"""
import logging
from typing import List
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from src.config.settings import settings

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """
    Generates embeddings for documents and queries.
    
    Key Features:
    - Batch processing for efficiency
    - Uses OpenAI's latest embedding models
    - Consistent embedding for documents and queries (critical!)
    """
    
    def __init__(
        self,
        model: str = None,
        dimensions: int = None,
    ):
        """
        Initialize embedding generator.
        
        Args:
            model: Embedding model name (default from settings)
            dimensions: Embedding dimensions (default from settings)
        """
        self.model = model or settings.embedding_model
        self.dimensions = dimensions or settings.embedding_dimensions
        
        # Initialize OpenAI embeddings
        self.embeddings = OpenAIEmbeddings(
            model=self.model,
            dimensions=self.dimensions if self.model == "text-embedding-3-large" else None,
            api_key=settings.openai_api_key,
        )
        
        logger.info(
            f"Initialized embeddings with model={self.model}, "
            f"dimensions={self.dimensions}"
        )
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple documents.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            logger.warning("No texts provided for embedding")
            return []
        
        logger.info(f"Generating embeddings for {len(texts)} documents...")
        
        try:
            embeddings = self.embeddings.embed_documents(texts)
            logger.info(f"Successfully generated {len(embeddings)} embeddings")
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise
    
    def embed_query(self, text: str) -> List[float]:
        """
        Generate embedding for a single query.
        
        Args:
            text: Query text to embed
            
        Returns:
            Embedding vector
        """
        try:
            embedding = self.embeddings.embed_query(text)
            return embedding
            
        except Exception as e:
            logger.error(f"Error embedding query: {str(e)}")
            raise
    
    def get_embedding_model(self):
        """Return the underlying embedding model for use in vector stores."""
        return self.embeddings


# Convenience functions
def get_embeddings(model: str = None) -> OpenAIEmbeddings:
    """
    Get configured embeddings instance.
    
    Args:
        model: Optional model override
        
    Returns:
        OpenAIEmbeddings instance
    """
    generator = EmbeddingGenerator(model=model)
    return generator.get_embedding_model()
