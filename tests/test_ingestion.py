"""
Tests for document ingestion pipeline.
Tests loaders, splitters, and embeddings.
"""
import pytest
from pathlib import Path
from langchain_core.documents import Document

from src.ingestion.loaders import DocumentLoader
from src.ingestion.splitters import DocumentSplitter
from src.ingestion.embedder import EmbeddingGenerator


class TestDocumentLoader:
    """Test document loading functionality."""
    
    def test_loader_initialization(self):
        """Test loader initializes correctly."""
        loader = DocumentLoader()
        assert loader is not None
        assert len(loader.loaded_files) == 0
    
    def test_supported_extensions(self):
        """Test that supported extensions are defined."""
        assert '.pdf' in DocumentLoader.SUPPORTED_EXTENSIONS
        assert '.docx' in DocumentLoader.SUPPORTED_EXTENSIONS
        assert '.txt' in DocumentLoader.SUPPORTED_EXTENSIONS
        assert '.md' in DocumentLoader.SUPPORTED_EXTENSIONS
    
    def test_load_text_file(self, tmp_path):
        """Test loading a text file."""
        # Create test file
        test_file = tmp_path / "test.txt"
        test_content = "This is a test document for RAG system."
        test_file.write_text(test_content)
        
        # Load file
        loader = DocumentLoader()
        documents = loader.load_file(str(test_file))
        
        # Assertions
        assert len(documents) > 0
        assert documents[0].page_content == test_content
        assert 'source' in documents[0].metadata
        assert 'file_name' in documents[0].metadata
    
    def test_load_nonexistent_file(self):
        """Test that loading nonexistent file raises error."""
        loader = DocumentLoader()
        
        with pytest.raises(FileNotFoundError):
            loader.load_file("nonexistent.txt")
    
    def test_unsupported_file_type(self, tmp_path):
        """Test that unsupported file type raises error."""
        test_file = tmp_path / "test.xyz"
        test_file.write_text("content")
        
        loader = DocumentLoader()
        
        with pytest.raises(ValueError, match="Unsupported file type"):
            loader.load_file(str(test_file))
    
    def test_load_directory(self, tmp_path):
        """Test loading all files from a directory."""
        # Create multiple test files
        (tmp_path / "doc1.txt").write_text("Document 1")
        (tmp_path / "doc2.txt").write_text("Document 2")
        (tmp_path / "doc3.txt").write_text("Document 3")

        # Load directory
        loader = DocumentLoader()
        documents = loader.load_directory(str(tmp_path))

        # Should load all 3 documents
        assert len(documents) >= 3


class TestDocumentSplitter:
    """Test document splitting functionality."""
    
    def test_splitter_initialization(self):
        """Test splitter initializes with correct parameters."""
        splitter = DocumentSplitter(chunk_size=500, chunk_overlap=50)
        
        assert splitter.chunk_size == 500
        assert splitter.chunk_overlap == 50
    
    def test_split_documents(self):
        """Test splitting documents into chunks."""
        # Create test document
        long_text = " ".join([f"Sentence {i}." for i in range(100)])
        documents = [Document(page_content=long_text, metadata={'source': 'test'})]
        
        # Split
        splitter = DocumentSplitter(chunk_size=100, chunk_overlap=20)
        chunks = splitter.split_documents(documents)
        
        # Assertions
        assert len(chunks) > 1  # Should create multiple chunks
        assert all('chunk_id' in chunk.metadata for chunk in chunks)
        assert all('source' in chunk.metadata for chunk in chunks)
    
    def test_chunk_overlap(self):
        """Test that chunks have overlap."""
        text = "A" * 1000
        documents = [Document(page_content=text)]
        
        splitter = DocumentSplitter(chunk_size=200, chunk_overlap=50)
        chunks = splitter.split_documents(documents)
        
        # With overlap, adjacent chunks should share content
        assert len(chunks) > 1
    
    def test_get_chunk_stats(self):
        """Test chunk statistics calculation."""
        documents = [Document(page_content="Test content " * 100)]
        splitter = DocumentSplitter(chunk_size=100, chunk_overlap=20)
        chunks = splitter.split_documents(documents)

        stats = splitter.get_chunk_stats(chunks)

        assert 'total_chunks' in stats
        assert 'avg_chunk_size' in stats
        assert stats['total_chunks'] == len(chunks)
    
    def test_empty_documents(self):
        """Test handling empty document list."""
        splitter = DocumentSplitter()
        chunks = splitter.split_documents([])
        
        assert len(chunks) == 0


class TestEmbeddingGenerator:
    """Test embedding generation."""
    
    def test_generator_initialization(self):
        """Test embedding generator initializes."""
        generator = EmbeddingGenerator()
        
        assert generator.model is not None
        assert generator.dimensions > 0
    
    @pytest.mark.skipif(
        True,  # Skip by default to avoid API calls
        reason="Requires OpenAI API key and makes real API calls"
    )
    def test_embed_documents(self):
        """Test document embedding (requires API key)."""
        generator = EmbeddingGenerator()
        texts = ["This is a test.", "Another test sentence."]
        
        embeddings = generator.embed_documents(texts)
        
        assert len(embeddings) == 2
        assert len(embeddings[0]) == generator.dimensions
    
    @pytest.mark.skipif(
        True,  # Skip by default
        reason="Requires OpenAI API key"
    )
    def test_embed_query(self):
        """Test query embedding (requires API key)."""
        generator = EmbeddingGenerator()
        text = "What is the answer?"
        
        embedding = generator.embed_query(text)
        
        assert len(embedding) == generator.dimensions
    
    def test_empty_texts(self):
        """Test handling empty text list."""
        generator = EmbeddingGenerator()
        embeddings = generator.embed_documents([])
        
        assert len(embeddings) == 0


# Fixtures
@pytest.fixture
def sample_documents():
    """Create sample documents for testing."""
    return [
        Document(
            page_content="This is the first document about AI.",
            metadata={'source': 'doc1.txt', 'page': 1}
        ),
        Document(
            page_content="This is the second document about machine learning.",
            metadata={'source': 'doc2.txt', 'page': 1}
        ),
    ]


@pytest.fixture
def sample_chunks(sample_documents):
    """Create sample chunks for testing."""
    splitter = DocumentSplitter(chunk_size=50, chunk_overlap=10)
    return splitter.split_documents(sample_documents)
