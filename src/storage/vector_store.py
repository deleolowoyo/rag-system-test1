"""
Vector store management using FAISS.
Handles document storage, retrieval, and persistence.
"""
import logging
import os
from pathlib import Path
from typing import List, Optional, Dict, Any
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from src.config.settings import settings
from src.ingestion.embedder import get_embeddings

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """
    Manages vector store operations with FAISS.

    Features:
    - Persistent storage on disk
    - Efficient similarity search
    - Metadata filtering
    - Fast approximate nearest neighbor search
    """

    def __init__(
        self,
        collection_name: str = None,
        persist_directory: str = None,
        embedding_function = None,
    ):
        """
        Initialize vector store manager.

        Args:
            collection_name: Name of the collection (default from settings)
            persist_directory: Path for persistence (default from settings)
            embedding_function: Optional custom embedding function
        """
        self.collection_name = collection_name or settings.collection_name
        self.persist_directory = persist_directory or settings.vector_db_path

        # Create persistence directory if it doesn't exist
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)

        # Get embedding function
        self.embedding_function = embedding_function or get_embeddings()

        # Initialize or load vector store
        self.vector_store: Optional[FAISS] = None
        self._initialize_store()

        logger.info(
            f"Initialized vector store: collection='{self.collection_name}', "
            f"persist_dir='{self.persist_directory}'"
        )

    def _initialize_store(self):
        """Initialize or load existing vector store."""
        try:
            # Try to load existing collection
            index_path = os.path.join(self.persist_directory, self.collection_name)
            if os.path.exists(index_path):
                self.vector_store = FAISS.load_local(
                    index_path,
                    self.embedding_function,
                    allow_dangerous_deserialization=True
                )
                doc_count = self.vector_store.index.ntotal
                logger.info(f"Loaded existing collection with {doc_count} documents")
            else:
                # Create new empty vector store with a dummy document
                dummy_doc = Document(page_content="Initialization document", metadata={"dummy": True})
                self.vector_store = FAISS.from_documents(
                    [dummy_doc],
                    self.embedding_function
                )
                # Remove the dummy document
                self.vector_store.delete([self.vector_store.index_to_docstore_id[0]])
                logger.info("Initialized new empty collection")

        except Exception as e:
            logger.error(f"Error initializing vector store: {str(e)}")
            raise
    
    def add_documents(
        self,
        documents: List[Document],
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Add documents to the vector store.

        Args:
            documents: List of documents to add
            ids: Optional list of document IDs

        Returns:
            List of document IDs
        """
        if not documents:
            logger.warning("No documents provided to add")
            return []

        logger.info(f"Adding {len(documents)} documents to vector store...")

        try:
            # Add documents and get IDs
            doc_ids = self.vector_store.add_documents(
                documents=documents,
                ids=ids,
            )

            # Save to disk
            index_path = os.path.join(self.persist_directory, self.collection_name)
            self.vector_store.save_local(index_path)

            logger.info(f"Successfully added {len(doc_ids)} documents and saved to disk")
            return doc_ids

        except Exception as e:
            logger.error(f"Error adding documents: {str(e)}")
            raise
    
    def similarity_search(
        self,
        query: str,
        k: int = None,
        filter: Optional[Dict[str, Any]] = None,
        score_threshold: Optional[float] = None,
    ) -> List[Document]:
        """
        Search for similar documents.
        
        Args:
            query: Search query
            k: Number of results to return (default from settings)
            filter: Optional metadata filter
            score_threshold: Minimum similarity score (default from settings)
            
        Returns:
            List of most similar documents
        """
        k = k or settings.retrieval_top_k
        score_threshold = score_threshold or settings.retrieval_score_threshold
        
        logger.info(f"Searching for top {k} similar documents...")
        
        try:
            if score_threshold:
                # Search with score threshold
                results = self.vector_store.similarity_search_with_score(
                    query=query,
                    k=k,
                    filter=filter,
                )
                
                # Filter by score threshold
                filtered_results = [
                    doc for doc, score in results
                    if score >= score_threshold
                ]
                
                logger.info(
                    f"Found {len(filtered_results)} documents above "
                    f"threshold {score_threshold}"
                )
                return filtered_results
            else:
                # Standard similarity search
                results = self.vector_store.similarity_search(
                    query=query,
                    k=k,
                    filter=filter,
                )
                
                logger.info(f"Found {len(results)} similar documents")
                return results
                
        except Exception as e:
            logger.error(f"Error during similarity search: {str(e)}")
            raise
    
    def similarity_search_with_score(
        self,
        query: str,
        k: int = None,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[tuple[Document, float]]:
        """
        Search with similarity scores.
        
        Args:
            query: Search query
            k: Number of results
            filter: Optional metadata filter
            
        Returns:
            List of (document, score) tuples
        """
        k = k or settings.retrieval_top_k
        
        try:
            results = self.vector_store.similarity_search_with_score(
                query=query,
                k=k,
                filter=filter,
            )
            
            logger.info(
                f"Found {len(results)} documents with scores: "
                f"{[f'{score:.3f}' for _, score in results]}"
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Error during scored search: {str(e)}")
            raise
    
    def delete_collection(self):
        """Delete the entire collection."""
        try:
            import shutil
            index_path = os.path.join(self.persist_directory, self.collection_name)
            if os.path.exists(index_path):
                shutil.rmtree(index_path)
                logger.info(f"Deleted collection '{self.collection_name}'")

                # Reinitialize empty vector store
                dummy_doc = Document(page_content="Initialization document", metadata={"dummy": True})
                self.vector_store = FAISS.from_documents(
                    [dummy_doc],
                    self.embedding_function
                )
                self.vector_store.delete([self.vector_store.index_to_docstore_id[0]])
            else:
                logger.info(f"Collection '{self.collection_name}' does not exist")

        except Exception as e:
            logger.error(f"Error deleting collection: {str(e)}")
            raise

    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the collection.

        Returns:
            Dictionary with collection stats
        """
        try:
            count = self.vector_store.index.ntotal if self.vector_store else 0

            return {
                'collection_name': self.collection_name,
                'document_count': count,
                'persist_directory': self.persist_directory,
            }

        except Exception as e:
            logger.error(f"Error getting collection stats: {str(e)}")
            raise
    
    def as_retriever(self, **kwargs):
        """
        Get a retriever interface for the vector store.
        
        Args:
            **kwargs: Additional retriever configuration
            
        Returns:
            VectorStoreRetriever instance
        """
        return self.vector_store.as_retriever(**kwargs)


# Convenience function
def create_vector_store(
    documents: List[Document] = None,
    collection_name: str = None,
    persist_directory: str = None,
) -> VectorStoreManager:
    """
    Create and optionally populate a vector store.
    
    Args:
        documents: Optional documents to add
        collection_name: Collection name
        persist_directory: Persistence directory
        
    Returns:
        VectorStoreManager instance
    """
    store = VectorStoreManager(
        collection_name=collection_name,
        persist_directory=persist_directory,
    )
    
    if documents:
        store.add_documents(documents)
    
    return store
