"""
Tests for retrieval functionality.
"""
import pytest
from unittest.mock import Mock, patch
from langchain_core.documents import Document

from src.retrieval.retriever import AdvancedRetriever


@pytest.fixture
def mock_vector_store():
    """Create mock vector store."""
    mock = Mock()
    mock.similarity_search = Mock(return_value=[
        Document(page_content="Test document 1", metadata={'source': 'doc1.txt'}),
        Document(page_content="Test document 2", metadata={'source': 'doc2.txt'}),
    ])
    mock.similarity_search_with_score = Mock(return_value=[
        (Document(page_content="Test document 1", metadata={'source': 'doc1.txt'}), 0.95),
        (Document(page_content="Test document 2", metadata={'source': 'doc2.txt'}), 0.87),
    ])
    return mock


@pytest.fixture
def mock_store_manager(mock_vector_store):
    """Create mock VectorStoreManager."""
    mock = Mock()
    mock.vector_store = mock_vector_store
    mock.similarity_search = mock_vector_store.similarity_search
    mock.similarity_search_with_score = mock_vector_store.similarity_search_with_score
    return mock


class TestAdvancedRetriever:
    """Test AdvancedRetriever functionality."""

    def test_initialization(self, mock_store_manager):
        """Test retriever initializes correctly."""
        retriever = AdvancedRetriever(
            vector_store=mock_store_manager,
            search_type="similarity",
            top_k=5
        )

        assert retriever.vector_store == mock_store_manager
        assert retriever.search_type == "similarity"
        assert retriever.top_k == 5

    def test_initialization_with_defaults(self, mock_store_manager):
        """Test retriever initializes with default settings."""
        retriever = AdvancedRetriever(vector_store=mock_store_manager)

        assert retriever.vector_store == mock_store_manager
        assert retriever.search_type is not None
        assert retriever.top_k > 0

    def test_retrieve_similarity(self, mock_store_manager):
        """Test similarity-based retrieval."""
        retriever = AdvancedRetriever(
            vector_store=mock_store_manager,
            search_type="similarity"
        )

        results = retriever.retrieve("test query")

        assert isinstance(results, list)
        assert len(results) > 0
        assert all(isinstance(doc, Document) for doc in results)
        mock_store_manager.similarity_search.assert_called_once()

    def test_retrieve_with_metadata_filter(self, mock_store_manager):
        """Test retrieval with metadata filtering."""
        retriever = AdvancedRetriever(vector_store=mock_store_manager)

        metadata_filter = {'source': 'doc1.txt'}
        results = retriever.retrieve("test query", filter=metadata_filter)

        assert isinstance(results, list)
        mock_store_manager.similarity_search.assert_called()

    def test_retrieve_with_custom_k(self, mock_store_manager):
        """Test retrieval with custom k value."""
        retriever = AdvancedRetriever(
            vector_store=mock_store_manager,
            top_k=3
        )

        results = retriever.retrieve("test query", top_k=2)

        assert isinstance(results, list)
        mock_store_manager.similarity_search.assert_called()

    def test_retrieve_with_score_threshold(self, mock_store_manager):
        """Test retrieval with score threshold."""
        retriever = AdvancedRetriever(
            vector_store=mock_store_manager,
            score_threshold=0.9
        )

        results = retriever.retrieve("test query")

        assert isinstance(results, list)

    def test_retrieve_with_scores(self, mock_store_manager):
        """Test retrieval returning scores."""
        retriever = AdvancedRetriever(vector_store=mock_store_manager)

        results = retriever.retrieve("test query", return_scores=True)

        assert isinstance(results, list)
        assert all(isinstance(item, tuple) and len(item) == 2 for item in results)
        assert all(isinstance(item[0], Document) for item in results)
        assert all(isinstance(item[1], (int, float)) for item in results)
        mock_store_manager.similarity_search_with_score.assert_called_once()

    def test_get_retriever_config(self, mock_store_manager):
        """Test getting retriever configuration."""
        retriever = AdvancedRetriever(
            vector_store=mock_store_manager,
            search_type="similarity",
            top_k=4,
            score_threshold=0.8
        )

        config = retriever.get_retriever_config()

        assert isinstance(config, dict)
        assert 'search_type' in config
        assert 'top_k' in config
        assert 'score_threshold' in config
        assert config['search_type'] == "similarity"
        assert config['top_k'] == 4
        assert config['score_threshold'] == 0.8

    def test_empty_query(self, mock_store_manager):
        """Test handling of empty query."""
        retriever = AdvancedRetriever(vector_store=mock_store_manager)

        # Should handle empty query gracefully
        results = retriever.retrieve("")

        assert isinstance(results, list)

    def test_rerank_documents(self, mock_store_manager):
        """Test document reranking if implemented."""
        retriever = AdvancedRetriever(vector_store=mock_store_manager)

        results = retriever.retrieve("test query", return_scores=True)

        # Documents should be sorted by score (descending)
        if len(results) > 1:
            scores = [score for _, score in results]
            assert scores == sorted(scores, reverse=True)


class TestRetrieverSearchTypes:
    """Test different search type configurations."""

    def test_similarity_search_type(self, mock_store_manager):
        """Test similarity search type."""
        retriever = AdvancedRetriever(
            vector_store=mock_store_manager,
            search_type="similarity"
        )

        assert retriever.search_type == "similarity"

    def test_mmr_search_type(self, mock_store_manager):
        """Test MMR search type."""
        retriever = AdvancedRetriever(
            vector_store=mock_store_manager,
            search_type="mmr"
        )

        assert retriever.search_type == "mmr"

    @patch('src.config.settings.settings')
    def test_default_from_settings(self, mock_settings, mock_store_manager):
        """Test that defaults come from settings."""
        mock_settings.search_type = "similarity"
        mock_settings.retrieval_top_k = 5
        mock_settings.retrieval_score_threshold = 0.75

        retriever = AdvancedRetriever(vector_store=mock_store_manager)

        # Should use defaults from settings
        assert retriever.top_k > 0
        assert retriever.score_threshold >= 0
