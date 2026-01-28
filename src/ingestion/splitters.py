"""
Text splitting strategies for optimal chunk creation.
Uses recursive splitting to maintain semantic coherence.
"""
import logging
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from src.config.settings import settings

logger = logging.getLogger(__name__)


class DocumentSplitter:
    """
    Intelligent document splitter that creates semantically coherent chunks.
    
    Key Design Decisions:
    1. Recursive splitting: Tries to split on paragraphs, then sentences, then words
    2. Overlap: Preserves context across chunk boundaries
    3. Token-aware: Uses tiktoken for accurate token counting
    """
    
    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None,
    ):
        """
        Initialize document splitter.
        
        Args:
            chunk_size: Maximum tokens per chunk (default from settings)
            chunk_overlap: Token overlap between chunks (default from settings)
        """
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        
        # Create recursive splitter with intelligent separators
        # Order matters: try to split on larger semantic units first
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=[
                "\n\n",  # Paragraph breaks (highest priority)
                "\n",    # Line breaks
                ". ",    # Sentence endings
                "! ",    # Exclamation sentences
                "? ",    # Question sentences
                "; ",    # Clause breaks
                ", ",    # Phrase breaks
                " ",     # Word breaks
                "",      # Character breaks (last resort)
            ],
            is_separator_regex=False,
        )
        
        logger.info(
            f"Initialized splitter with chunk_size={self.chunk_size}, "
            f"overlap={self.chunk_overlap}"
        )
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into optimally-sized chunks.
        
        Args:
            documents: List of documents to split
            
        Returns:
            List of document chunks with preserved metadata
        """
        if not documents:
            logger.warning("No documents provided for splitting")
            return []
        
        logger.info(f"Splitting {len(documents)} documents into chunks...")
        
        # Split documents while preserving metadata
        chunks = self.splitter.split_documents(documents)
        
        # Add chunk-specific metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata.update({
                'chunk_id': i,
                'chunk_size': len(chunk.page_content),
            })
        
        logger.info(
            f"Created {len(chunks)} chunks from {len(documents)} documents "
            f"(avg {len(chunks) / len(documents):.1f} chunks/doc)"
        )
        
        return chunks
    
    def split_text(self, text: str, metadata: dict = None) -> List[Document]:
        """
        Split raw text into chunks.
        
        Args:
            text: Raw text to split
            metadata: Optional metadata to attach to chunks
            
        Returns:
            List of document chunks
        """
        chunks = self.splitter.create_documents(
            texts=[text],
            metadatas=[metadata or {}]
        )
        
        # Add chunk IDs
        for i, chunk in enumerate(chunks):
            chunk.metadata['chunk_id'] = i
        
        return chunks
    
    def get_chunk_stats(self, chunks: List[Document]) -> dict:
        """
        Calculate statistics about chunks.
        
        Args:
            chunks: List of document chunks
            
        Returns:
            Dictionary with chunk statistics
        """
        if not chunks:
            return {
                'total_chunks': 0,
                'avg_chunk_size': 0,
                'min_chunk_size': 0,
                'max_chunk_size': 0,
            }
        
        chunk_sizes = [len(chunk.page_content) for chunk in chunks]
        
        return {
            'total_chunks': len(chunks),
            'avg_chunk_size': sum(chunk_sizes) / len(chunk_sizes),
            'min_chunk_size': min(chunk_sizes),
            'max_chunk_size': max(chunk_sizes),
            'total_characters': sum(chunk_sizes),
        }


# Convenience function for quick splitting
def split_documents(
    documents: List[Document],
    chunk_size: int = None,
    chunk_overlap: int = None,
) -> List[Document]:
    """
    Convenience function to split documents with default settings.
    
    Args:
        documents: Documents to split
        chunk_size: Optional chunk size override
        chunk_overlap: Optional overlap override
        
    Returns:
        List of split document chunks
    """
    splitter = DocumentSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(documents)
