"""
Tests for vector store management with FAISS.
"""
import pytest
import tempfile
import shutil
from pathlib import Path
from typing import List
from unittest.mock import Mock, patch, MagicMock
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from src.storage.vector_store import VectorStoreManager, create_vector_store


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path)


class FakeEmbeddings(Embeddings):
    """Fake embeddings for testing that FAISS can work with."""

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Return fake embeddings for documents."""
        import random
        return [[random.random() for _ in range(1536)] for _ in texts]

    def embed_query(self, text: str) -> List[float]:
        """Return fake embedding for a query."""
        import random
        return [random.random() for _ in range(1536)]


@pytest.fixture
def mock_embeddings():
    """Create mock embeddings function."""
    return FakeEmbeddings()


@pytest.fixture
def sample_documents():
    """Create sample documents for testing."""
    return [
        Document(
            page_content="This is the first test document.",
            metadata={'source': 'doc1.txt', 'page': 1}
        ),
        Document(
            page_content="This is the second test document.",
            metadata={'source': 'doc2.txt', 'page': 1}
        ),
    ]


class TestVectorStoreManager:
    """Test VectorStoreManager functionality."""

    @patch('src.storage.vector_store.get_embeddings')
    def test_initialization(self, mock_get_embeddings, temp_dir, mock_embeddings):
        """Test vector store initializes correctly."""
        mock_get_embeddings.return_value = mock_embeddings

        store = VectorStoreManager(
            collection_name="test_collection",
            persist_directory=temp_dir,
            embedding_function=mock_embeddings
        )

        assert store.collection_name == "test_collection"
        assert store.persist_directory == temp_dir
        assert store.vector_store is not None

    @patch('src.storage.vector_store.get_embeddings')
    def test_add_documents(self, mock_get_embeddings, temp_dir, mock_embeddings, sample_documents):
        """Test adding documents to vector store."""
        mock_get_embeddings.return_value = mock_embeddings

        store = VectorStoreManager(
            collection_name="test_collection",
            persist_directory=temp_dir,
            embedding_function=mock_embeddings
        )

        doc_ids = store.add_documents(sample_documents)

        assert len(doc_ids) == len(sample_documents)
        assert all(isinstance(doc_id, str) for doc_id in doc_ids)

    @patch('src.storage.vector_store.get_embeddings')
    def test_add_empty_documents(self, mock_get_embeddings, temp_dir, mock_embeddings):
        """Test adding empty document list."""
        mock_get_embeddings.return_value = mock_embeddings

        store = VectorStoreManager(
            collection_name="test_collection",
            persist_directory=temp_dir,
            embedding_function=mock_embeddings
        )

        doc_ids = store.add_documents([])

        assert len(doc_ids) == 0

    @patch('src.storage.vector_store.get_embeddings')
    def test_similarity_search(self, mock_get_embeddings, temp_dir, mock_embeddings, sample_documents):
        """Test similarity search."""
        mock_get_embeddings.return_value = mock_embeddings

        store = VectorStoreManager(
            collection_name="test_collection",
            persist_directory=temp_dir,
            embedding_function=mock_embeddings
        )

        # Add documents
        store.add_documents(sample_documents)

        # Search
        results = store.similarity_search("test query", k=2)

        assert isinstance(results, list)
        assert len(results) <= 2

    @patch('src.storage.vector_store.get_embeddings')
    def test_similarity_search_with_score(self, mock_get_embeddings, temp_dir, mock_embeddings, sample_documents):
        """Test similarity search with scores."""
        mock_get_embeddings.return_value = mock_embeddings

        store = VectorStoreManager(
            collection_name="test_collection",
            persist_directory=temp_dir,
            embedding_function=mock_embeddings
        )

        # Add documents
        store.add_documents(sample_documents)

        # Search with scores
        results = store.similarity_search_with_score("test query", k=2)

        assert isinstance(results, list)
        assert all(isinstance(item, tuple) and len(item) == 2 for item in results)

    @patch('src.storage.vector_store.get_embeddings')
    def test_get_collection_stats(self, mock_get_embeddings, temp_dir, mock_embeddings, sample_documents):
        """Test getting collection statistics."""
        mock_get_embeddings.return_value = mock_embeddings

        store = VectorStoreManager(
            collection_name="test_collection",
            persist_directory=temp_dir,
            embedding_function=mock_embeddings
        )

        # Add documents
        store.add_documents(sample_documents)

        # Get stats
        stats = store.get_collection_stats()

        assert 'collection_name' in stats
        assert 'document_count' in stats
        assert 'persist_directory' in stats
        assert stats['collection_name'] == "test_collection"
        assert stats['document_count'] >= 0

    @patch('src.storage.vector_store.get_embeddings')
    def test_delete_collection(self, mock_get_embeddings, temp_dir, mock_embeddings, sample_documents):
        """Test deleting a collection."""
        mock_get_embeddings.return_value = mock_embeddings

        store = VectorStoreManager(
            collection_name="test_collection",
            persist_directory=temp_dir,
            embedding_function=mock_embeddings
        )

        # Add documents
        store.add_documents(sample_documents)

        # Delete collection
        store.delete_collection()

        # Verify collection is empty
        stats = store.get_collection_stats()
        assert stats['document_count'] == 0

    @patch('src.storage.vector_store.get_embeddings')
    def test_persistence(self, mock_get_embeddings, temp_dir, mock_embeddings, sample_documents):
        """Test that vector store persists to disk."""
        mock_get_embeddings.return_value = mock_embeddings

        # Create store and add documents
        store1 = VectorStoreManager(
            collection_name="test_collection",
            persist_directory=temp_dir,
            embedding_function=mock_embeddings
        )
        store1.add_documents(sample_documents)

        # Create new store instance with same path
        store2 = VectorStoreManager(
            collection_name="test_collection",
            persist_directory=temp_dir,
            embedding_function=mock_embeddings
        )

        # Should load existing data
        stats = store2.get_collection_stats()
        assert stats['document_count'] >= len(sample_documents)

    @patch('src.storage.vector_store.get_embeddings')
    def test_as_retriever(self, mock_get_embeddings, temp_dir, mock_embeddings):
        """Test getting retriever interface."""
        mock_get_embeddings.return_value = mock_embeddings

        store = VectorStoreManager(
            collection_name="test_collection",
            persist_directory=temp_dir,
            embedding_function=mock_embeddings
        )

        retriever = store.as_retriever(search_kwargs={'k': 3})

        assert retriever is not None


class TestCreateVectorStore:
    """Test create_vector_store convenience function."""

    @patch('src.storage.vector_store.get_embeddings')
    def test_create_empty_store(self, mock_get_embeddings, temp_dir, mock_embeddings):
        """Test creating an empty vector store."""
        mock_get_embeddings.return_value = mock_embeddings

        store = create_vector_store(
            collection_name="test_collection",
            persist_directory=temp_dir
        )

        assert store is not None
        assert isinstance(store, VectorStoreManager)

    @patch('src.storage.vector_store.get_embeddings')
    def test_create_with_documents(self, mock_get_embeddings, temp_dir, mock_embeddings, sample_documents):
        """Test creating vector store with initial documents."""
        mock_get_embeddings.return_value = mock_embeddings

        store = create_vector_store(
            documents=sample_documents,
            collection_name="test_collection",
            persist_directory=temp_dir
        )

        assert store is not None
        stats = store.get_collection_stats()
        assert stats['document_count'] >= len(sample_documents)
