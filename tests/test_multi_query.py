"""
Tests for multi-query generation and retrieval.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from langchain_core.documents import Document

from src.agents.multi_query import (
    MultiQueryGenerator,
    MultiQueryRetriever,
    generate_query_variations,
    MULTI_QUERY_PROMPT,
)


class TestMultiQueryGenerator:
    """Test MultiQueryGenerator functionality."""

    def test_initialization(self):
        """Test MultiQueryGenerator initializes correctly."""
        generator = MultiQueryGenerator()

        assert generator.temperature == 0.7
        assert generator.max_tokens == 300
        assert generator.llm is not None

    def test_initialization_custom_params(self):
        """Test MultiQueryGenerator with custom parameters."""
        generator = MultiQueryGenerator(temperature=0.5, max_tokens=200)

        assert generator.temperature == 0.5
        assert generator.max_tokens == 200

    @patch('src.agents.multi_query.LLMGenerator')
    def test_generate_queries_basic(self, mock_llm_class):
        """Test basic query generation."""
        # Mock LLM response
        mock_llm = Mock()
        mock_llm.generate = Mock(return_value="""1. What is Retrieval Augmented Generation?
2. How does RAG work with language models?
3. What are the components of a RAG system?""")
        mock_llm.get_model_info = Mock(return_value={'model': 'claude-sonnet-4'})
        mock_llm_class.return_value = mock_llm

        generator = MultiQueryGenerator()
        queries = generator.generate_queries("What is RAG?", num_queries=3)

        # Should include original + 3 variations
        assert len(queries) >= 4
        assert queries[0] == "What is RAG?"  # Original is first
        assert "Retrieval Augmented Generation" in queries[1]

    @patch('src.agents.multi_query.LLMGenerator')
    def test_generate_queries_deduplicates(self, mock_llm_class):
        """Test that duplicate queries are removed."""
        mock_llm = Mock()
        # LLM returns a duplicate of the original
        mock_llm.generate = Mock(return_value="""1. What is RAG?
2. How does RAG work?
3. What are RAG components?""")
        mock_llm.get_model_info = Mock(return_value={'model': 'claude-sonnet-4'})
        mock_llm_class.return_value = mock_llm

        generator = MultiQueryGenerator()
        queries = generator.generate_queries("What is RAG?", num_queries=3)

        # Should not have duplicates
        assert len(queries) == len(set(q.lower().strip() for q in queries))

    def test_parse_variations_numbered(self):
        """Test parsing numbered list variations."""
        generator = MultiQueryGenerator()

        response = """1. First variation
2. Second variation
3. Third variation"""

        variations = generator._parse_variations(response, 3)

        assert len(variations) == 3
        assert variations[0] == "First variation"
        assert variations[1] == "Second variation"
        assert variations[2] == "Third variation"

    def test_parse_variations_bullets(self):
        """Test parsing bullet point variations."""
        generator = MultiQueryGenerator()

        response = """- First variation
* Second variation
• Third variation"""

        variations = generator._parse_variations(response, 3)

        assert len(variations) == 3
        assert variations[0] == "First variation"
        assert variations[1] == "Second variation"
        assert variations[2] == "Third variation"

    def test_parse_variations_mixed_formats(self):
        """Test parsing mixed format variations."""
        generator = MultiQueryGenerator()

        response = """1. First variation
- Second variation
3) Third variation
→ Fourth variation"""

        variations = generator._parse_variations(response, 4)

        assert len(variations) == 4
        assert "First variation" in variations
        assert "Second variation" in variations

    def test_parse_variations_with_quotes(self):
        """Test parsing variations with quotes."""
        generator = MultiQueryGenerator()

        response = """1. "First variation"
2. 'Second variation'
3. Third variation"""

        variations = generator._parse_variations(response, 3)

        assert variations[0] == "First variation"
        assert variations[1] == "Second variation"
        assert variations[2] == "Third variation"

    def test_parse_variations_empty_response(self):
        """Test parsing empty or invalid responses."""
        generator = MultiQueryGenerator()

        variations = generator._parse_variations("", 3)
        assert variations == []

        variations = generator._parse_variations("   \n\n   ", 3)
        assert variations == []

    @patch('src.agents.multi_query.LLMGenerator')
    def test_generate_queries_error_fallback(self, mock_llm_class):
        """Test that errors fall back to original query only."""
        mock_llm = Mock()
        mock_llm.generate = Mock(side_effect=Exception("API Error"))
        mock_llm.get_model_info = Mock(return_value={'model': 'claude-sonnet-4'})
        mock_llm_class.return_value = mock_llm

        generator = MultiQueryGenerator()
        queries = generator.generate_queries("What is machine learning?", num_queries=3)

        # Should return only original query on error
        assert len(queries) == 1
        assert queries[0] == "What is machine learning?"

    def test_generate_queries_empty_input(self):
        """Test handling of empty queries."""
        generator = MultiQueryGenerator()

        # Empty string
        queries = generator.generate_queries("", num_queries=3)
        assert queries == [""]

        # None
        queries = generator.generate_queries(None, num_queries=3)
        assert queries == []

    def test_get_generator_config(self):
        """Test getting generator configuration."""
        generator = MultiQueryGenerator(temperature=0.5, max_tokens=200)

        config = generator.get_generator_config()

        assert config['temperature'] == 0.5
        assert config['max_tokens'] == 200
        assert 'llm_config' in config


class TestMultiQueryRetriever:
    """Test MultiQueryRetriever functionality."""

    @pytest.fixture
    def mock_retriever(self):
        """Create a mock retriever."""
        retriever = Mock()
        retriever.get_retriever_config = Mock(return_value={
            'search_type': 'similarity',
            'top_k': 4,
            'score_threshold': 0.7,
        })
        return retriever

    @pytest.fixture
    def mock_generator(self):
        """Create a mock query generator."""
        generator = Mock()
        generator.get_generator_config = Mock(return_value={
            'temperature': 0.7,
            'max_tokens': 300,
        })
        return generator

    def test_initialization(self, mock_retriever):
        """Test MultiQueryRetriever initializes correctly."""
        multi_retriever = MultiQueryRetriever(mock_retriever)

        assert multi_retriever.retriever is not None
        assert multi_retriever.generator is not None
        assert multi_retriever.num_queries == 3
        assert multi_retriever.top_k_per_query == 3

    def test_initialization_custom_params(self, mock_retriever, mock_generator):
        """Test MultiQueryRetriever with custom parameters."""
        multi_retriever = MultiQueryRetriever(
            retriever=mock_retriever,
            generator=mock_generator,
            num_queries=5,
            top_k_per_query=4,
        )

        assert multi_retriever.num_queries == 5
        assert multi_retriever.top_k_per_query == 4

    def test_retrieve_basic(self, mock_retriever, mock_generator):
        """Test basic multi-query retrieval."""
        # Mock generator to return query variations
        mock_generator.generate_queries = Mock(return_value=[
            "What is RAG?",
            "What is Retrieval Augmented Generation?",
            "How does RAG work?",
        ])

        # Mock retriever to return documents
        mock_retriever.retrieve = Mock(return_value=[
            Document(page_content="RAG is a technique...", metadata={'source': 'doc1.txt'}),
            Document(page_content="RAG combines retrieval...", metadata={'source': 'doc2.txt'}),
        ])

        multi_retriever = MultiQueryRetriever(
            retriever=mock_retriever,
            generator=mock_generator,
        )

        docs = multi_retriever.retrieve("What is RAG?")

        # Should have retrieved documents
        assert isinstance(docs, list)
        assert len(docs) > 0

        # Verify generator was called
        mock_generator.generate_queries.assert_called_once()

        # Verify retriever was called for each query variation
        assert mock_retriever.retrieve.call_count == 3

    def test_retrieve_deduplicates(self, mock_retriever, mock_generator):
        """Test that duplicate documents are removed."""
        mock_generator.generate_queries = Mock(return_value=[
            "Query 1",
            "Query 2",
        ])

        # Return same document for both queries (duplicate)
        duplicate_doc = Document(
            page_content="Same content",
            metadata={'source': 'doc1.txt'}
        )

        mock_retriever.retrieve = Mock(return_value=[duplicate_doc])

        multi_retriever = MultiQueryRetriever(
            retriever=mock_retriever,
            generator=mock_generator,
        )

        docs = multi_retriever.retrieve("Test query")

        # Should deduplicate - only 1 doc despite 2 queries returning same doc
        assert len(docs) == 1

    def test_retrieve_with_scores(self, mock_retriever, mock_generator):
        """Test retrieval with similarity scores."""
        mock_generator.generate_queries = Mock(return_value=[
            "Query 1",
        ])

        mock_retriever.retrieve = Mock(return_value=[
            (Document(page_content="Content 1", metadata={'source': 'doc1.txt'}), 0.95),
            (Document(page_content="Content 2", metadata={'source': 'doc2.txt'}), 0.85),
        ])

        multi_retriever = MultiQueryRetriever(
            retriever=mock_retriever,
            generator=mock_generator,
        )

        docs = multi_retriever.retrieve("Test query", return_scores=True)

        # Should return tuples with scores
        assert isinstance(docs, list)
        assert len(docs) > 0
        assert isinstance(docs[0], tuple)
        assert len(docs[0]) == 2
        assert isinstance(docs[0][0], Document)
        assert isinstance(docs[0][1], float)

    def test_hash_document(self, mock_retriever):
        """Test document hashing for deduplication."""
        multi_retriever = MultiQueryRetriever(mock_retriever)

        doc1 = Document(page_content="Test content", metadata={'source': 'doc1.txt'})
        doc2 = Document(page_content="Test content", metadata={'source': 'doc1.txt'})
        doc3 = Document(page_content="Different content", metadata={'source': 'doc1.txt'})

        hash1 = multi_retriever._hash_document(doc1)
        hash2 = multi_retriever._hash_document(doc2)
        hash3 = multi_retriever._hash_document(doc3)

        # Same content should have same hash
        assert hash1 == hash2

        # Different content should have different hash
        assert hash1 != hash3

    def test_retrieve_error_handling(self, mock_retriever, mock_generator):
        """Test that retrieval errors are handled gracefully."""
        mock_generator.generate_queries = Mock(return_value=[
            "Query 1",
            "Query 2",
        ])

        # First query succeeds, second fails
        mock_retriever.retrieve = Mock(side_effect=[
            [Document(page_content="Success", metadata={'source': 'doc1.txt'})],
            Exception("Retrieval error"),
        ])

        multi_retriever = MultiQueryRetriever(
            retriever=mock_retriever,
            generator=mock_generator,
        )

        # Should not raise, should return partial results
        docs = multi_retriever.retrieve("Test query")

        assert len(docs) == 1  # Only from successful query

    def test_get_retriever_config(self, mock_retriever, mock_generator):
        """Test getting multi-query retriever configuration."""
        multi_retriever = MultiQueryRetriever(
            retriever=mock_retriever,
            generator=mock_generator,
            num_queries=5,
            top_k_per_query=4,
        )

        config = multi_retriever.get_retriever_config()

        assert config['num_queries'] == 5
        assert config['top_k_per_query'] == 4
        assert 'generator_config' in config
        assert 'base_retriever_config' in config


class TestMultiQueryPrompt:
    """Test the multi-query prompt."""

    def test_prompt_exists(self):
        """Test that MULTI_QUERY_PROMPT is defined."""
        assert MULTI_QUERY_PROMPT is not None
        assert len(MULTI_QUERY_PROMPT) > 0

    def test_prompt_has_placeholders(self):
        """Test that prompt has required placeholders."""
        assert "{num_queries}" in MULTI_QUERY_PROMPT
        assert "{query}" in MULTI_QUERY_PROMPT

    def test_prompt_formatting(self):
        """Test that prompt can be formatted."""
        formatted = MULTI_QUERY_PROMPT.format(
            num_queries=3,
            query="What is machine learning?"
        )

        assert "What is machine learning?" in formatted
        assert "3" in formatted
        assert "{query}" not in formatted
        assert "{num_queries}" not in formatted


class TestConvenienceFunction:
    """Test the convenience function."""

    @patch('src.agents.multi_query.MultiQueryGenerator')
    def test_generate_query_variations(self, mock_generator_class):
        """Test the convenience function."""
        mock_generator = Mock()
        mock_generator.generate_queries = Mock(return_value=[
            "Original query",
            "Variation 1",
            "Variation 2",
        ])
        mock_generator_class.return_value = mock_generator

        result = generate_query_variations("Original query", num_queries=2, temperature=0.8)

        mock_generator_class.assert_called_once_with(temperature=0.8)
        mock_generator.generate_queries.assert_called_once_with("Original query", num_queries=2)
        assert result == ["Original query", "Variation 1", "Variation 2"]
