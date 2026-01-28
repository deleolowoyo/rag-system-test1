"""
Tests for RAG pipeline functionality.
"""
import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from langchain_core.documents import Document

from src.pipeline import RAGPipeline


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def mock_embeddings():
    """Create mock embeddings function."""
    mock = Mock()
    # Return embeddings matching the number of input texts
    mock.embed_documents = Mock(side_effect=lambda texts: [[0.1] * 1536 for _ in texts])
    mock.embed_query = Mock(return_value=[0.15] * 1536)
    return mock


@pytest.fixture
def sample_text_file(temp_dir):
    """Create a sample text file for testing."""
    file_path = Path(temp_dir) / "test.txt"
    file_path.write_text("This is a test document for the RAG system. It contains multiple sentences.")
    return str(file_path)


class TestRAGPipeline:
    """Test RAGPipeline functionality."""

    @patch('src.pipeline.LLMGenerator')
    @patch('src.pipeline.VectorStoreManager')
    def test_initialization(self, mock_vector_store, mock_llm_generator, temp_dir, mock_embeddings):
        """Test pipeline initializes correctly."""
        mock_store_instance = Mock()
        mock_vector_store.return_value = mock_store_instance
        mock_llm_instance = Mock()
        mock_llm_generator.return_value = mock_llm_instance

        pipeline = RAGPipeline(
            collection_name="test_collection",
            persist_directory=temp_dir,
            reinitialize=False
        )

        assert pipeline.loader is not None
        assert pipeline.splitter is not None
        assert pipeline.vector_store is not None
        assert pipeline.retriever is not None
        assert pipeline.llm is not None

    @patch('src.pipeline.LLMGenerator')
    @patch('src.pipeline.VectorStoreManager')
    def test_initialization_with_reinitialize(self, mock_vector_store, mock_llm_generator, temp_dir, mock_embeddings):
        """Test pipeline with reinitialize flag."""
        mock_store_instance = Mock()
        mock_store_instance.delete_collection = Mock()
        mock_vector_store.return_value = mock_store_instance
        mock_llm_instance = Mock()
        mock_llm_generator.return_value = mock_llm_instance

        pipeline = RAGPipeline(
            collection_name="test_collection",
            persist_directory=temp_dir,
            reinitialize=True
        )

        # Should have called delete_collection
        mock_store_instance.delete_collection.assert_called_once()

    @patch('src.pipeline.LLMGenerator')
    @patch('src.pipeline.VectorStoreManager')
    def test_ingest_documents(self, mock_vector_store, mock_llm_generator, temp_dir, mock_embeddings, sample_text_file):
        """Test document ingestion."""
        mock_store_instance = Mock()
        mock_store_instance.add_documents = Mock(return_value=['doc1'])
        mock_vector_store.return_value = mock_store_instance
        mock_llm_instance = Mock()
        mock_llm_generator.return_value = mock_llm_instance

        pipeline = RAGPipeline(
            collection_name="test_collection",
            persist_directory=temp_dir
        )

        # Create directory with sample file
        data_dir = Path(temp_dir) / "data"
        data_dir.mkdir()
        (data_dir / "test.txt").write_text("Test content")

        result = pipeline.ingest_documents(directory_path=str(data_dir))

        assert 'files_processed' in result
        assert 'chunks_created' in result
        assert len(result['files_processed']) > 0
        mock_store_instance.add_documents.assert_called()

    @patch('src.pipeline.LLMGenerator')
    @patch('src.pipeline.VectorStoreManager')
    def test_ingest_single_file(self, mock_vector_store, mock_llm_generator, temp_dir, mock_embeddings, sample_text_file):
        """Test ingesting a single file."""
        mock_store_instance = Mock()
        mock_store_instance.add_documents = Mock(return_value=['doc1'])
        mock_vector_store.return_value = mock_store_instance
        mock_llm_instance = Mock()
        mock_llm_generator.return_value = mock_llm_instance

        pipeline = RAGPipeline(
            collection_name="test_collection",
            persist_directory=temp_dir
        )

        result = pipeline.ingest_documents(file_paths=[sample_text_file])

        assert len(result['files_processed']) == 1
        assert result['chunks_created'] > 0

    @patch('src.generation.llm.ChatAnthropic')
    @patch('src.pipeline.VectorStoreManager')
    def test_query(self, mock_vector_store, mock_chat_anthropic, temp_dir, mock_embeddings):
        """Test querying the pipeline."""
        # Mock vector store with similarity_search_with_score method
        mock_store_instance = Mock()
        mock_store_instance.similarity_search_with_score = Mock(return_value=[
            (Document(page_content="Test content", metadata={'source': 'test.txt'}), 0.95)
        ])
        mock_vector_store.return_value = mock_store_instance

        # Mock LLM
        mock_llm = Mock()
        mock_llm.invoke = Mock(return_value=Mock(content="This is a test answer."))
        mock_chat_anthropic.return_value = mock_llm

        pipeline = RAGPipeline(
            collection_name="test_collection",
            persist_directory=temp_dir
        )

        result = pipeline.query("What is this about?")

        assert 'answer' in result
        assert 'sources' in result
        assert isinstance(result['answer'], str)
        assert len(result['answer']) > 0

    @patch('src.generation.llm.ChatAnthropic')
    @patch('src.pipeline.VectorStoreManager')
    def test_query_with_no_results(self, mock_vector_store, mock_chat_anthropic, temp_dir, mock_embeddings):
        """Test querying with no relevant documents."""
        # Mock vector store with no results
        mock_store_instance = Mock()
        mock_store_instance.similarity_search_with_score = Mock(return_value=[])
        mock_vector_store.return_value = mock_store_instance

        # Mock LLM
        mock_llm = Mock()
        mock_llm.invoke = Mock(return_value=Mock(content="I don't have enough information."))
        mock_chat_anthropic.return_value = mock_llm

        pipeline = RAGPipeline(
            collection_name="test_collection",
            persist_directory=temp_dir
        )

        result = pipeline.query("What is this about?")

        assert 'answer' in result
        assert 'sources' in result
        assert len(result['sources']) == 0

    @patch('src.generation.llm.ChatAnthropic')
    @patch('src.pipeline.VectorStoreManager')
    def test_stream_query(self, mock_vector_store, mock_chat_anthropic, temp_dir, mock_embeddings):
        """Test streaming query."""
        # Mock vector store with similarity_search method
        mock_store_instance = Mock()
        mock_store_instance.similarity_search = Mock(return_value=[
            Document(page_content="Test content", metadata={'source': 'test.txt'})
        ])
        mock_vector_store.return_value = mock_store_instance

        # Mock streaming LLM
        mock_llm = Mock()
        mock_llm.stream = Mock(return_value=[
            Mock(content="Test "),
            Mock(content="streaming "),
            Mock(content="response"),
        ])
        mock_chat_anthropic.return_value = mock_llm

        pipeline = RAGPipeline(
            collection_name="test_collection",
            persist_directory=temp_dir
        )

        chunks = list(pipeline.query_stream("What is this about?"))

        assert len(chunks) > 0
        assert all(isinstance(chunk, str) for chunk in chunks)

    @patch('src.pipeline.LLMGenerator')
    @patch('src.pipeline.VectorStoreManager')
    def test_get_stats(self, mock_vector_store, mock_llm_generator, temp_dir, mock_embeddings):
        """Test getting pipeline statistics."""
        mock_store_instance = Mock()
        mock_store_instance.get_collection_stats = Mock(return_value={
            'collection_name': 'test',
            'document_count': 5,
            'persist_directory': temp_dir
        })
        mock_vector_store.return_value = mock_store_instance
        mock_llm_instance = Mock()
        mock_llm_generator.return_value = mock_llm_instance

        pipeline = RAGPipeline(
            collection_name="test_collection",
            persist_directory=temp_dir
        )

        stats = pipeline.get_stats()

        assert 'vector_store' in stats
        assert 'retriever' in stats
        assert 'llm' in stats
        assert isinstance(stats['vector_store'], dict)

    @patch('src.pipeline.LLMGenerator')
    @patch('src.pipeline.VectorStoreManager')
    def test_reset(self, mock_vector_store, mock_llm_generator, temp_dir, mock_embeddings):
        """Test resetting the pipeline."""
        mock_store_instance = Mock()
        mock_store_instance.delete_collection = Mock()
        mock_vector_store.return_value = mock_store_instance
        mock_llm_instance = Mock()
        mock_llm_generator.return_value = mock_llm_instance

        pipeline = RAGPipeline(
            collection_name="test_collection",
            persist_directory=temp_dir
        )

        pipeline.reset()

        mock_store_instance.delete_collection.assert_called()


class TestPipelineIntegration:
    """Integration tests for pipeline components."""

    @patch('src.generation.llm.ChatAnthropic')
    @patch('src.pipeline.VectorStoreManager')
    def test_full_workflow(self, mock_vector_store, mock_chat_anthropic, temp_dir, mock_embeddings):
        """Test complete ingestion and query workflow."""
        # Mock vector store
        mock_store_instance = Mock()
        mock_store_instance.add_documents = Mock(return_value=['doc1'])
        mock_store_instance.similarity_search_with_score = Mock(return_value=[
            (Document(page_content="Test content about AI", metadata={'source': 'test.txt'}), 0.95)
        ])
        mock_vector_store.return_value = mock_store_instance

        # Mock LLM
        mock_llm = Mock()
        mock_llm.invoke = Mock(return_value=Mock(content="AI is artificial intelligence."))
        mock_chat_anthropic.return_value = mock_llm

        # Create pipeline
        pipeline = RAGPipeline(
            collection_name="test_collection",
            persist_directory=temp_dir
        )

        # Create test file
        data_dir = Path(temp_dir) / "data"
        data_dir.mkdir()
        (data_dir / "test.txt").write_text("AI stands for Artificial Intelligence.")

        # Ingest
        ingest_result = pipeline.ingest_documents(directory_path=str(data_dir))
        assert len(ingest_result['files_processed']) > 0

        # Query
        query_result = pipeline.query("What is AI?")
        assert 'answer' in query_result
        assert len(query_result['answer']) > 0


class TestPipelineErrorHandling:
    """Test error handling in pipeline."""

    @patch('src.pipeline.LLMGenerator')
    @patch('src.pipeline.VectorStoreManager')
    def test_ingest_invalid_directory(self, mock_vector_store, mock_llm_generator, temp_dir, mock_embeddings):
        """Test ingesting from invalid directory."""
        mock_vector_store.return_value = Mock()
        mock_llm_instance = Mock()
        mock_llm_generator.return_value = mock_llm_instance

        pipeline = RAGPipeline(
            collection_name="test_collection",
            persist_directory=temp_dir
        )

        # Pipeline catches exceptions and returns error status
        result = pipeline.ingest_documents(directory_path="/nonexistent/directory")
        assert result['success'] == False
        assert result['documents_loaded'] == 0

    @patch('src.generation.llm.ChatAnthropic')
    @patch('src.pipeline.VectorStoreManager')
    def test_query_empty_string(self, mock_vector_store, mock_chat_anthropic, temp_dir, mock_embeddings):
        """Test querying with empty string."""
        mock_store_instance = Mock()
        mock_store_instance.similarity_search_with_score = Mock(return_value=[])
        mock_vector_store.return_value = mock_store_instance

        mock_llm = Mock()
        mock_llm.invoke = Mock(return_value=Mock(content="Please provide a question."))
        mock_chat_anthropic.return_value = mock_llm

        pipeline = RAGPipeline(
            collection_name="test_collection",
            persist_directory=temp_dir
        )

        # Should handle gracefully
        result = pipeline.query("")
        assert 'answer' in result
