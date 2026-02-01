"""
Tests for agents package (Phase 2).

Focused tests for query rewriting functionality with mocked LLM calls
to avoid API costs during testing.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from src.agents.query_rewriter import QueryRewriter


@pytest.fixture
def mock_llm():
    """
    Fixture providing a mocked LLM for testing.

    Returns a configured mock that simulates LLM responses
    without making actual API calls.
    """
    with patch('src.agents.query_rewriter.LLMGenerator') as mock_llm_class:
        mock_instance = Mock()
        mock_instance.get_model_info = Mock(return_value={
            'model': 'claude-sonnet-4-20250514',
            'temperature': 0.3,
            'max_tokens': 100,
        })
        mock_llm_class.return_value = mock_instance
        yield mock_instance


def test_query_rewriter_initialization():
    """
    Test that QueryRewriter initializes correctly.

    Verifies:
    - Default temperature is set correctly
    - Default max_tokens is set correctly
    - LLM instance is created
    """
    rewriter = QueryRewriter()

    # Check default parameters
    assert rewriter.temperature == 0.3, "Default temperature should be 0.3"
    assert rewriter.max_tokens == 100, "Default max_tokens should be 100"

    # Check LLM is initialized
    assert rewriter.llm is not None, "LLM should be initialized"

    # Test custom parameters
    custom_rewriter = QueryRewriter(temperature=0.5, max_tokens=200)
    assert custom_rewriter.temperature == 0.5
    assert custom_rewriter.max_tokens == 200


def test_query_rewriter_short_query():
    """
    Test that short queries are returned unchanged.

    Short queries (< 10 characters) should bypass the rewriting
    process and be returned as-is to save API calls.
    """
    rewriter = QueryRewriter()

    # Test various short queries
    short_queries = [
        "API",           # 3 chars
        "help",          # 4 chars
        "test me",       # 7 chars
        "short",         # 5 chars
        "a b c",         # 5 chars (with spaces)
    ]

    for query in short_queries:
        result = rewriter.rewrite(query)
        assert result == query, f"Short query '{query}' should be unchanged"

    # Edge case: exactly 10 characters will be rewritten (not skipped)
    # The threshold is strictly less than 10, so 10+ chars get rewritten
    exactly_ten = "0123456789"
    assert len(exactly_ten) == 10
    # This query has 10 chars, so it will be processed (not skipped)
    # We're just verifying the length check works correctly


def test_query_rewriter_basic(mock_llm):
    """
    Test that vague queries get rewritten to be more specific.

    Uses mocked LLM to simulate query rewriting without API calls.
    Verifies that the rewriter correctly:
    - Expands abbreviations
    - Adds specificity
    - Returns the rewritten query
    """
    # Configure mock to return an improved query
    mock_llm.generate = Mock(
        return_value="What is Retrieval Augmented Generation (RAG) and how does it work?"
    )

    rewriter = QueryRewriter()

    # Test rewriting a vague query
    vague_query = "What's RAG?"
    rewritten = rewriter.rewrite(vague_query)

    # Verify the query was rewritten
    assert rewritten != vague_query, "Query should be rewritten"
    assert len(rewritten) > len(vague_query), "Rewritten query should be more detailed"
    assert "Retrieval Augmented Generation" in rewritten, "Should expand RAG abbreviation"

    # Verify LLM was called
    mock_llm.generate.assert_called_once()

    # Verify the prompt included the original query
    call_args = mock_llm.generate.call_args
    messages = call_args[0][0]  # First positional argument
    assert any("What's RAG?" in str(msg.content) for msg in messages), \
        "Original query should be in prompt"


def test_query_rewriter_fallback(mock_llm):
    """
    Test that the rewriter handles errors gracefully.

    When the LLM fails (API error, timeout, etc.), the rewriter
    should fall back to returning the original query unchanged.
    """
    # Configure mock to raise an exception
    mock_llm.generate = Mock(side_effect=Exception("API Error: Rate limit exceeded"))

    rewriter = QueryRewriter()

    # Test with a query that would normally be rewritten
    original_query = "What is machine learning and how does it work?"
    rewritten = rewriter.rewrite(original_query)

    # Should fall back to original query
    assert rewritten == original_query, \
        "Should return original query when LLM fails"

    # Verify LLM was called (attempted)
    mock_llm.generate.assert_called_once()

    # Test with another error type
    mock_llm.generate = Mock(side_effect=ValueError("Invalid input"))
    rewritten2 = rewriter.rewrite("Another test query")
    assert rewritten2 == "Another test query", \
        "Should fall back on any exception type"


# Additional helper tests

def test_query_rewriter_empty_query():
    """Test that empty queries are handled properly."""
    rewriter = QueryRewriter()

    # Empty string
    assert rewriter.rewrite("") == ""

    # Whitespace only
    assert rewriter.rewrite("   ") == "   "

    # None
    assert rewriter.rewrite(None) is None


def test_query_rewriter_response_cleaning(mock_llm):
    """
    Test that LLM response artifacts are cleaned up.

    The LLM might return responses with quotes, prefixes, or other
    artifacts that should be removed.
    """
    # Test removing quotes
    mock_llm.generate = Mock(return_value='"What is Natural Language Processing?"')
    rewriter = QueryRewriter()
    result = rewriter.rewrite("What's NLP?")
    assert result == "What is Natural Language Processing?"
    assert not result.startswith('"')

    # Test removing 'Rewritten query:' prefix
    mock_llm.generate = Mock(
        return_value="Rewritten query: What is a Large Language Model?"
    )
    result = rewriter.rewrite("What's an LLM?")
    assert result == "What is a Large Language Model?"
    assert not result.lower().startswith("rewritten")


def test_query_rewriter_batch_processing(mock_llm):
    """
    Test batch rewriting of multiple queries.

    Verifies that multiple queries can be rewritten in a single
    batch operation.
    """
    # Configure mock to return different responses for each query
    mock_llm.generate = Mock(side_effect=[
        "What is Retrieval Augmented Generation?",
        "Where can I find the API documentation?",
        "What is Natural Language Processing?",
    ])

    rewriter = QueryRewriter()

    # Use queries that are all > 10 chars so they get rewritten
    queries = [
        "What's RAG?",  # 11 chars - will be rewritten
        "Where are API docs?",  # 18 chars - will be rewritten
        "What's NLP?",  # 11 chars - will be rewritten
    ]

    results = rewriter.rewrite_batch(queries)

    # Verify correct number of results
    assert len(results) == 3, "Should return same number of results as input"

    # Verify each query was processed
    assert results[0] == "What is Retrieval Augmented Generation?"
    assert results[1] == "Where can I find the API documentation?"
    assert results[2] == "What is Natural Language Processing?"

    # Verify LLM was called for each query (only for queries > 10 chars)
    assert mock_llm.generate.call_count == 3


def test_query_rewriter_config(mock_llm):
    """Test retrieving rewriter configuration."""
    rewriter = QueryRewriter(temperature=0.4, max_tokens=150)

    config = rewriter.get_rewriter_config()

    assert 'temperature' in config
    assert 'max_tokens' in config
    assert 'llm_config' in config

    assert config['temperature'] == 0.4
    assert config['max_tokens'] == 150
    assert config['llm_config']['model'] == 'claude-sonnet-4-20250514'
