"""
Document loaders for various file formats.
Handles PDF, DOCX, TXT, and MD files with proper error handling.
"""
import logging
from pathlib import Path
from typing import List, Optional
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class DocumentLoader:
    """Unified document loader supporting multiple file formats."""
    
    SUPPORTED_EXTENSIONS = {
        '.pdf': PyPDFLoader,
        '.docx': Docx2txtLoader,
        '.txt': TextLoader,
        '.md': UnstructuredMarkdownLoader,
    }
    
    def __init__(self):
        """Initialize document loader."""
        self.loaded_files: List[str] = []
    
    def load_file(self, file_path: str) -> List[Document]:
        """
        Load a single document file.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            List of Document objects with content and metadata
            
        Raises:
            ValueError: If file type is not supported
            FileNotFoundError: If file doesn't exist
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        extension = path.suffix.lower()
        if extension not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type: {extension}. "
                f"Supported types: {list(self.SUPPORTED_EXTENSIONS.keys())}"
            )
        
        loader_class = self.SUPPORTED_EXTENSIONS[extension]
        
        try:
            logger.info(f"Loading document: {file_path}")
            loader = loader_class(str(path))
            documents = loader.load()
            
            # Enrich metadata
            for doc in documents:
                doc.metadata.update({
                    'source': str(path),
                    'file_name': path.name,
                    'file_type': extension,
                })
            
            self.loaded_files.append(str(path))
            logger.info(f"Successfully loaded {len(documents)} pages from {path.name}")
            
            return documents
            
        except Exception as e:
            logger.error(f"Error loading {file_path}: {str(e)}")
            raise
    
    def load_directory(
        self, 
        directory_path: str,
        recursive: bool = True,
        file_pattern: Optional[str] = None
    ) -> List[Document]:
        """
        Load all supported documents from a directory.
        
        Args:
            directory_path: Path to directory containing documents
            recursive: Whether to search subdirectories
            file_pattern: Optional glob pattern to filter files (e.g., "*.pdf")
            
        Returns:
            List of all loaded documents
        """
        directory = Path(directory_path)
        
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        if not directory.is_dir():
            raise ValueError(f"Not a directory: {directory_path}")
        
        all_documents = []
        
        # Determine search pattern
        if recursive:
            pattern = "**/*" if file_pattern is None else f"**/{file_pattern}"
        else:
            pattern = "*" if file_pattern is None else file_pattern
        
        # Find all matching files
        files = [
            f for f in directory.glob(pattern)
            if f.is_file() and f.suffix.lower() in self.SUPPORTED_EXTENSIONS
        ]
        
        logger.info(f"Found {len(files)} supported documents in {directory_path}")
        
        # Load each file
        for file_path in files:
            try:
                documents = self.load_file(str(file_path))
                all_documents.extend(documents)
            except Exception as e:
                logger.warning(f"Skipping {file_path.name}: {str(e)}")
                continue
        
        logger.info(
            f"Successfully loaded {len(all_documents)} total pages "
            f"from {len(self.loaded_files)} files"
        )
        
        return all_documents
    
    def get_loaded_files(self) -> List[str]:
        """Return list of successfully loaded files."""
        return self.loaded_files.copy()
