"""
Tests for query rewriting functionality.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from src.agents.query_rewriter import QueryRewriter, rewrite_query, REWRITE_PROMPT


class TestQueryRewriter:
    """Test QueryRewriter functionality."""

    def test_initialization(self):
        """Test QueryRewriter initializes correctly."""
        rewriter = QueryRewriter()

        assert rewriter.temperature == 0.3
        assert rewriter.max_tokens == 100
        assert rewriter.llm is not None

    def test_initialization_custom_params(self):
        """Test QueryRewriter with custom parameters."""
        rewriter = QueryRewriter(temperature=0.5, max_tokens=150)

        assert rewriter.temperature == 0.5
        assert rewriter.max_tokens == 150

    def test_rewrite_short_query(self):
        """Test that short queries are returned unchanged."""
        rewriter = QueryRewriter()

        short_queries = ["API", "help", "test", "a b c"]

        for query in short_queries:
            result = rewriter.rewrite(query)
            assert result == query, f"Short query '{query}' should be unchanged"

    def test_rewrite_empty_query(self):
        """Test handling of empty queries."""
        rewriter = QueryRewriter()

        assert rewriter.rewrite("") == ""
        assert rewriter.rewrite("   ") == "   "
        assert rewriter.rewrite(None) is None

    @patch('src.agents.query_rewriter.LLMGenerator')
    def test_rewrite_with_llm(self, mock_llm_class):
        """Test query rewriting with mocked LLM."""
        # Mock LLM response
        mock_llm = Mock()
        mock_llm.generate = Mock(return_value="What is Retrieval Augmented Generation (RAG)?")
        mock_llm.get_model_info = Mock(return_value={'model': 'claude-sonnet-4'})
        mock_llm_class.return_value = mock_llm

        rewriter = QueryRewriter()
        result = rewriter.rewrite("What's RAG?")

        assert result == "What is Retrieval Augmented Generation (RAG)?"
        mock_llm.generate.assert_called_once()

    @patch('src.agents.query_rewriter.LLMGenerator')
    def test_rewrite_removes_quotes(self, mock_llm_class):
        """Test that surrounding quotes are removed from rewritten queries."""
        mock_llm = Mock()
        mock_llm.generate = Mock(return_value='"What is machine learning?"')
        mock_llm.get_model_info = Mock(return_value={'model': 'claude-sonnet-4'})
        mock_llm_class.return_value = mock_llm

        rewriter = QueryRewriter()
        result = rewriter.rewrite("What's ML?")

        assert result == "What is machine learning?"
        assert not result.startswith('"')
        assert not result.endswith('"')

    @patch('src.agents.query_rewriter.LLMGenerator')
    def test_rewrite_removes_prefix(self, mock_llm_class):
        """Test that 'Rewritten query:' prefix is removed."""
        mock_llm = Mock()
        mock_llm.generate = Mock(return_value="Rewritten query: What is Natural Language Processing?")
        mock_llm.get_model_info = Mock(return_value={'model': 'claude-sonnet-4'})
        mock_llm_class.return_value = mock_llm

        rewriter = QueryRewriter()
        result = rewriter.rewrite("What's NLP?")

        assert result == "What is Natural Language Processing?"
        assert not result.lower().startswith("rewritten")

    @patch('src.agents.query_rewriter.LLMGenerator')
    def test_rewrite_error_fallback(self, mock_llm_class):
        """Test that errors fall back to original query."""
        mock_llm = Mock()
        mock_llm.generate = Mock(side_effect=Exception("API Error"))
        mock_llm.get_model_info = Mock(return_value={'model': 'claude-sonnet-4'})
        mock_llm_class.return_value = mock_llm

        rewriter = QueryRewriter()
        original_query = "What is machine learning?"
        result = rewriter.rewrite(original_query)

        # Should fall back to original query
        assert result == original_query

    @patch('src.agents.query_rewriter.LLMGenerator')
    def test_rewrite_invalid_response_fallback(self, mock_llm_class):
        """Test that invalid LLM responses fall back to original."""
        mock_llm = Mock()
        mock_llm.generate = Mock(return_value="  ")  # Empty response
        mock_llm.get_model_info = Mock(return_value={'model': 'claude-sonnet-4'})
        mock_llm_class.return_value = mock_llm

        rewriter = QueryRewriter()
        original_query = "What is machine learning?"
        result = rewriter.rewrite(original_query)

        # Should fall back to original query
        assert result == original_query

    @patch('src.agents.query_rewriter.LLMGenerator')
    def test_rewrite_batch(self, mock_llm_class):
        """Test batch query rewriting."""
        mock_llm = Mock()
        # Mock different responses for different queries
        mock_llm.generate = Mock(side_effect=[
            "What is Retrieval Augmented Generation?",
            "Where can I find the API documentation?",
            "What is Natural Language Processing?",
        ])
        mock_llm.get_model_info = Mock(return_value={'model': 'claude-sonnet-4'})
        mock_llm_class.return_value = mock_llm

        rewriter = QueryRewriter()
        queries = [
            "What's RAG?",
            "API docs location",
            "Tell me about NLP",
        ]

        results = rewriter.rewrite_batch(queries)

        assert len(results) == 3
        assert results[0] == "What is Retrieval Augmented Generation?"
        assert results[1] == "Where can I find the API documentation?"
        assert results[2] == "What is Natural Language Processing?"

    def test_rewrite_batch_empty_list(self):
        """Test batch rewriting with empty list."""
        rewriter = QueryRewriter()
        results = rewriter.rewrite_batch([])

        assert results == []

    def test_get_rewriter_config(self):
        """Test getting rewriter configuration."""
        rewriter = QueryRewriter(temperature=0.5, max_tokens=150)

        config = rewriter.get_rewriter_config()

        assert config['temperature'] == 0.5
        assert config['max_tokens'] == 150
        assert 'llm_config' in config

    @patch('src.agents.query_rewriter.QueryRewriter')
    def test_convenience_function(self, mock_rewriter_class):
        """Test the convenience rewrite_query function."""
        mock_rewriter = Mock()
        mock_rewriter.rewrite = Mock(return_value="Rewritten query")
        mock_rewriter_class.return_value = mock_rewriter

        result = rewrite_query("Original query", temperature=0.5)

        mock_rewriter_class.assert_called_once_with(temperature=0.5)
        mock_rewriter.rewrite.assert_called_once_with("Original query")
        assert result == "Rewritten query"


class TestRewritePrompt:
    """Test the query rewrite prompt."""

    def test_prompt_exists(self):
        """Test that REWRITE_PROMPT is defined."""
        assert REWRITE_PROMPT is not None
        assert len(REWRITE_PROMPT) > 0

    def test_prompt_has_placeholder(self):
        """Test that prompt has query placeholder."""
        assert "{query}" in REWRITE_PROMPT

    def test_prompt_formatting(self):
        """Test that prompt can be formatted."""
        formatted = REWRITE_PROMPT.format(query="What's ML?")

        assert "What's ML?" in formatted
        assert "{query}" not in formatted


class TestQueryRewriterIntegration:
    """Integration tests requiring API keys."""

    @pytest.mark.integration
    def test_real_query_rewrite(self):
        """
        Test actual query rewriting with real LLM.

        Note: Requires valid API keys. Skipped in normal test runs.
        """
        rewriter = QueryRewriter()

        # Test with a query that needs optimization
        original = "What's RAG?"
        rewritten = rewriter.rewrite(original)

        # Verify rewriting occurred
        assert rewritten != original
        assert len(rewritten) > len(original)
        # Should expand "RAG"
        assert "retrieval" in rewritten.lower() or "augmented" in rewritten.lower()
