"""
End-to-end RAG pipeline orchestrating all components.
This is the main interface for the RAG system.
"""
import logging
from typing import List, Optional, Dict, Any
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

from src.ingestion.loaders import DocumentLoader
from src.ingestion.splitters import DocumentSplitter
from src.storage.vector_store import VectorStoreManager
from src.retrieval.retriever import AdvancedRetriever
from src.generation.prompts import format_documents_for_context, RAG_SYSTEM_PROMPT
from src.generation.llm import LLMGenerator
from src.config.settings import settings

logging.basicConfig(
    level=settings.log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    Complete RAG pipeline from document ingestion to answer generation.
    
    Architecture:
    1. Document Loading → 2. Splitting → 3. Embedding → 4. Storage
    5. Query → 6. Retrieval → 7. Generation → 8. Response
    """
    
    def __init__(
        self,
        collection_name: str = None,
        persist_directory: str = None,
        reinitialize: bool = False,
    ):
        """
        Initialize RAG pipeline.
        
        Args:
            collection_name: Vector store collection name
            persist_directory: Path for vector store persistence
            reinitialize: If True, delete existing collection and start fresh
        """
        logger.info("Initializing RAG Pipeline...")
        
        # Initialize components
        self.loader = DocumentLoader()
        self.splitter = DocumentSplitter()
        
        # Initialize vector store
        self.vector_store = VectorStoreManager(
            collection_name=collection_name,
            persist_directory=persist_directory,
        )
        
        # Reinitialize if requested
        if reinitialize:
            logger.warning("Reinitializing vector store (deleting existing data)...")
            self.vector_store.delete_collection()
            self.vector_store = VectorStoreManager(
                collection_name=collection_name,
                persist_directory=persist_directory,
            )
        
        # Initialize retriever
        self.retriever = AdvancedRetriever(vector_store=self.vector_store)
        
        # Initialize LLM
        self.llm = LLMGenerator()
        
        logger.info("RAG Pipeline initialized successfully")
    
    def ingest_documents(
        self,
        file_paths: List[str] = None,
        directory_path: str = None,
    ) -> Dict[str, Any]:
        """
        Ingest documents into the RAG system.
        
        Args:
            file_paths: List of individual file paths
            directory_path: Path to directory containing documents
            
        Returns:
            Dictionary with ingestion statistics
        """
        logger.info("Starting document ingestion...")
        
        documents = []
        
        # Load from files
        if file_paths:
            for file_path in file_paths:
                try:
                    docs = self.loader.load_file(file_path)
                    documents.extend(docs)
                except Exception as e:
                    logger.error(f"Failed to load {file_path}: {str(e)}")
        
        # Load from directory
        if directory_path:
            try:
                docs = self.loader.load_directory(directory_path)
                documents.extend(docs)
            except Exception as e:
                logger.error(f"Failed to load directory {directory_path}: {str(e)}")
        
        if not documents:
            logger.warning("No documents loaded")
            return {
                'success': False,
                'documents_loaded': 0,
                'chunks_created': 0,
            }
        
        logger.info(f"Loaded {len(documents)} documents")
        
        # Split documents
        chunks = self.splitter.split_documents(documents)
        logger.info(f"Created {len(chunks)} chunks")
        
        # Add to vector store
        doc_ids = self.vector_store.add_documents(chunks)
        logger.info(f"Added {len(doc_ids)} chunks to vector store")
        
        stats = {
            'success': True,
            'documents_loaded': len(documents),
            'chunks_created': len(chunks),
            'chunk_stats': self.splitter.get_chunk_stats(chunks),
            'files_processed': self.loader.get_loaded_files(),
        }
        
        logger.info("Document ingestion completed successfully")
        return stats
    
    def query(
        self,
        question: str,
        top_k: int = None,
        return_sources: bool = True,
        return_context: bool = False,
    ) -> Dict[str, Any]:
        """
        Query the RAG system.
        
        Args:
            question: User question
            top_k: Number of documents to retrieve
            return_sources: Include source documents in response
            return_context: Include formatted context in response
            
        Returns:
            Dictionary with answer and optional sources/context
        """
        logger.info(f"Processing query: '{question}'")
        
        # Retrieve relevant documents
        retrieved_docs = self.retriever.retrieve(
            query=question,
            top_k=top_k,
            return_scores=True,
        )
        
        if not retrieved_docs:
            logger.warning("No relevant documents found")
            return {
                'answer': "I don't have enough information to answer that question.",
                'sources': [],
                'num_sources': 0,
            }
        
        # Separate documents and scores
        documents = [doc for doc, _ in retrieved_docs]
        scores = [score for _, score in retrieved_docs]
        
        logger.info(f"Retrieved {len(documents)} relevant documents")
        
        # Format context
        context = format_documents_for_context(documents)
        
        # Create prompt
        prompt = f"{context}\n\nQuestion: {question}\n\nAnswer:"
        
        # Generate response
        messages = [
            HumanMessage(content=RAG_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]
        
        answer = self.llm.generate(messages)
        
        # Build response
        response = {
            'answer': answer,
            'num_sources': len(documents),
        }
        
        if return_sources:
            response['sources'] = [
                {
                    'content': doc.page_content[:200] + '...',  # Truncate for brevity
                    'metadata': doc.metadata,
                    'score': score,
                }
                for doc, score in zip(documents, scores)
            ]
        
        if return_context:
            response['context'] = context
        
        logger.info("Query processed successfully")
        return response
    
    def query_stream(
        self,
        question: str,
        top_k: int = None,
    ):
        """
        Query with streaming response.
        
        Args:
            question: User question
            top_k: Number of documents to retrieve
            
        Yields:
            Response tokens
        """
        logger.info(f"Processing streaming query: '{question}'")
        
        # Retrieve documents
        retrieved_docs = self.retriever.retrieve(
            query=question,
            top_k=top_k,
        )
        
        if not retrieved_docs:
            yield "I don't have enough information to answer that question."
            return
        
        # Format context
        context = format_documents_for_context(retrieved_docs)
        
        # Create prompt
        prompt = f"{context}\n\nQuestion: {question}\n\nAnswer:"
        
        messages = [
            HumanMessage(content=RAG_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]
        
        # Stream response
        streaming_llm = LLMGenerator(streaming=True)
        for token in streaming_llm.stream_generate(messages):
            yield token
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get pipeline statistics.
        
        Returns:
            Dictionary with system statistics
        """
        return {
            'vector_store': self.vector_store.get_collection_stats(),
            'retriever': self.retriever.get_retriever_config(),
            'llm': self.llm.get_model_info(),
        }
    
    def reset(self):
        """Reset the pipeline by clearing the vector store."""
        logger.warning("Resetting pipeline...")
        self.vector_store.delete_collection()
        self.vector_store = VectorStoreManager()
        self.retriever = AdvancedRetriever(vector_store=self.vector_store)
        logger.info("Pipeline reset complete")


# Convenience function
def create_pipeline(
    collection_name: str = None,
    reinitialize: bool = False,
) -> RAGPipeline:
    """
    Create a RAG pipeline instance.
    
    Args:
        collection_name: Optional collection name
        reinitialize: Whether to start fresh
        
    Returns:
        RAGPipeline instance
    """
    return RAGPipeline(
        collection_name=collection_name,
        reinitialize=reinitialize,
    )
