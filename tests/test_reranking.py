"""
Tests for re-ranking module (Phase 2).

Focused tests for LLM-based and hybrid re-ranking with mocked LLM calls
to avoid API costs during testing.
"""
import pytest
from unittest.mock import Mock, patch
from langchain_core.documents import Document

from src.reranking.reranker import LLMReranker, HybridReranker


@pytest.fixture
def sample_documents():
    """Create sample documents for testing."""
    return [
        Document(
            page_content="RAG combines retrieval with generation for better answers.",
            metadata={'source': 'doc1.txt', 'page': 1}
        ),
        Document(
            page_content="Machine learning models learn from data patterns.",
            metadata={'source': 'doc2.txt', 'page': 2}
        ),
        Document(
            page_content="Vector databases store embeddings for semantic search.",
            metadata={'source': 'doc3.txt', 'page': 3}
        ),
    ]


@pytest.fixture
def sample_documents_with_scores():
    """Create sample documents with similarity scores for testing."""
    return [
        (
            Document(
                page_content="RAG combines retrieval with generation.",
                metadata={'source': 'doc1.txt'}
            ),
            0.85  # High similarity
        ),
        (
            Document(
                page_content="Machine learning basics.",
                metadata={'source': 'doc2.txt'}
            ),
            0.45  # Low similarity
        ),
        (
            Document(
                page_content="Vector search uses embeddings.",
                metadata={'source': 'doc3.txt'}
            ),
            0.62  # Medium similarity
        ),
    ]


# LLMReranker Tests

@patch('src.reranking.reranker.LLMGenerator')
def test_llm_reranker_initialization(mock_llm_class):
    """
    Test that LLMReranker initializes correctly.

    Verifies:
    - Default temperature is 0.0 (deterministic)
    - Default max_tokens is 10
    - Default max_doc_length is 1000
    - LLM instance is created
    """
    # Configure mock
    mock_llm = Mock()
    mock_llm.get_model_info = Mock(return_value={
        'model': 'claude-sonnet-4',
        'temperature': 0.0,
        'max_tokens': 10,
    })
    mock_llm_class.return_value = mock_llm

    reranker = LLMReranker()

    # Check default parameters
    assert reranker.temperature == 0.0, "Default temperature should be 0.0"
    assert reranker.max_tokens == 10, "Default max_tokens should be 10"
    assert reranker.max_doc_length == 1000, "Default max_doc_length should be 1000"

    # Check LLM is initialized
    assert reranker.llm is not None, "LLM should be initialized"

    # Test custom parameters
    custom_reranker = LLMReranker(
        temperature=0.1,
        max_tokens=20,
        max_doc_length=500
    )
    assert custom_reranker.temperature == 0.1
    assert custom_reranker.max_tokens == 20
    assert custom_reranker.max_doc_length == 500


@patch('src.reranking.reranker.LLMGenerator')
def test_llm_reranker_scoring(mock_llm_class, sample_documents):
    """
    Test that LLMReranker scores documents correctly.

    Verifies:
    - Documents receive relevance scores 0-10
    - Scores are parsed from LLM responses
    - Different response formats are handled
    - Scores are returned as floats
    """
    # Configure mock to return different scores
    mock_llm = Mock()
    mock_llm.get_model_info = Mock(return_value={'model': 'claude-sonnet-4'})

    # Test different score formats (match number of sample_documents = 3)
    score_responses = ["9", "7.5", "8/10"]
    mock_llm.generate = Mock(side_effect=score_responses)
    mock_llm_class.return_value = mock_llm

    reranker = LLMReranker()

    # Score documents
    query = "What is RAG?"
    scores = []
    for doc in sample_documents:
        score = reranker.score_document(query, doc)
        scores.append(score)

    # Verify scores
    assert len(scores) == len(sample_documents), "Should score all documents"
    assert scores[0] == 9.0, "Should parse '9' as 9.0"
    assert scores[1] == 7.5, "Should parse '7.5' as 7.5"
    assert scores[2] == 8.0, "Should parse '8/10' as 8.0"

    # Verify all scores are in valid range
    for score in scores:
        assert 0.0 <= score <= 10.0, f"Score {score} should be in range 0-10"

    # Verify LLM was called for each document
    assert mock_llm.generate.call_count == len(sample_documents)


@patch('src.reranking.reranker.LLMGenerator')
def test_llm_reranker_sorting(mock_llm_class, sample_documents):
    """
    Test that LLMReranker sorts documents by relevance.

    Verifies:
    - Documents are sorted by score descending
    - Top_k parameter works correctly
    - Returns (document, score) tuples
    - Highest scoring document is first
    """
    # Configure mock to return scores in specific order
    mock_llm = Mock()
    mock_llm.get_model_info = Mock(return_value={'model': 'claude-sonnet-4'})

    # Return scores: 5.0, 9.0, 7.0 (middle, high, medium)
    # After sorting: 9.0, 7.0, 5.0
    mock_llm.generate = Mock(side_effect=["5", "9", "7"])
    mock_llm_class.return_value = mock_llm

    reranker = LLMReranker()

    # Re-rank documents
    query = "What is RAG?"
    ranked_docs = reranker.rerank(query, sample_documents)

    # Verify sorting
    assert len(ranked_docs) == 3, "Should return all 3 documents"
    assert ranked_docs[0][1] == 9.0, "Highest score should be first"
    assert ranked_docs[1][1] == 7.0, "Middle score should be second"
    assert ranked_docs[2][1] == 5.0, "Lowest score should be last"

    # Verify documents are in correct order
    assert ranked_docs[0][0] == sample_documents[1], "Doc with score 9 should be first"
    assert ranked_docs[1][0] == sample_documents[2], "Doc with score 7 should be second"
    assert ranked_docs[2][0] == sample_documents[0], "Doc with score 5 should be third"

    # Verify scores are descending
    for i in range(len(ranked_docs) - 1):
        assert ranked_docs[i][1] >= ranked_docs[i + 1][1], \
            "Scores should be in descending order"


@patch('src.reranking.reranker.LLMGenerator')
def test_llm_reranker_top_k(mock_llm_class, sample_documents):
    """
    Test that top_k parameter limits results correctly.

    Verifies:
    - Only top_k documents are returned
    - Highest scoring documents are selected
    - Sorting is maintained
    """
    # Configure mock
    mock_llm = Mock()
    mock_llm.get_model_info = Mock(return_value={'model': 'claude-sonnet-4'})
    mock_llm.generate = Mock(side_effect=["5", "9", "7"])
    mock_llm_class.return_value = mock_llm

    reranker = LLMReranker()

    # Re-rank with top_k=2
    query = "What is RAG?"
    ranked_docs = reranker.rerank(query, sample_documents, top_k=2)

    # Verify only top 2 are returned
    assert len(ranked_docs) == 2, "Should return only top 2 documents"
    assert ranked_docs[0][1] == 9.0, "Highest score should be first"
    assert ranked_docs[1][1] == 7.0, "Second highest score should be second"


@patch('src.reranking.reranker.LLMGenerator')
def test_llm_reranker_score_parsing(mock_llm_class):
    """
    Test score parsing from various response formats.

    Verifies:
    - Parses simple numbers (8, 7.5)
    - Parses fractions (8/10)
    - Parses with prefixes (Score: 8)
    - Clamps scores to 0-10 range
    - Handles invalid scores gracefully
    """
    mock_llm = Mock()
    mock_llm.get_model_info = Mock(return_value={'model': 'claude-sonnet-4'})
    mock_llm_class.return_value = mock_llm

    reranker = LLMReranker()

    # Test valid formats
    test_cases = [
        ("8", 8.0),
        ("7.5", 7.5),
        ("9/10", 9.0),
        ("Score: 6", 6.0),
        ("The score is 8.5", 8.5),
        ("10", 10.0),
        ("0", 0.0),
    ]

    for response, expected_score in test_cases:
        score = reranker._parse_score(response)
        assert score == expected_score, \
            f"Failed to parse '{response}', expected {expected_score}, got {score}"

    # Test clamping
    assert reranker._parse_score("15") == 10.0, "Should clamp scores > 10 to 10.0"

    # Test invalid format raises ValueError
    with pytest.raises(ValueError):
        reranker._parse_score("invalid")


@patch('src.reranking.reranker.LLMGenerator')
def test_llm_reranker_error_handling(mock_llm_class, sample_documents):
    """
    Test that LLMReranker handles errors gracefully.

    Verifies:
    - Falls back to default score 5.0 on LLM error
    - Falls back on parsing error
    - Continues processing other documents
    """
    # Configure mock to raise error
    mock_llm = Mock()
    mock_llm.get_model_info = Mock(return_value={'model': 'claude-sonnet-4'})
    mock_llm.generate = Mock(side_effect=Exception("API Error"))
    mock_llm_class.return_value = mock_llm

    reranker = LLMReranker()

    # Score document with error
    query = "What is RAG?"
    score = reranker.score_document(query, sample_documents[0])

    # Should fall back to 5.0
    assert score == 5.0, "Should return default score 5.0 on error"

    # Test parsing error fallback
    mock_llm.generate = Mock(return_value="invalid response")
    score = reranker.score_document(query, sample_documents[0])
    assert score == 5.0, "Should return default score 5.0 on parsing error"


@patch('src.reranking.reranker.LLMGenerator')
def test_llm_reranker_document_truncation(mock_llm_class):
    """
    Test that long documents are truncated correctly.

    Verifies:
    - Documents longer than max_doc_length are truncated
    - Truncation adds "..." marker
    - Scoring still works on truncated content
    """
    mock_llm = Mock()
    mock_llm.get_model_info = Mock(return_value={'model': 'claude-sonnet-4'})
    mock_llm.generate = Mock(return_value="8")
    mock_llm_class.return_value = mock_llm

    # Create reranker with small max_doc_length
    reranker = LLMReranker(max_doc_length=50)

    # Create long document
    long_content = "A" * 200
    doc = Document(page_content=long_content, metadata={'source': 'test.txt'})

    # Score document
    query = "Test query"
    score = reranker.score_document(query, doc)

    # Verify scoring worked
    assert score == 8.0, "Should score truncated document"

    # Verify LLM received truncated content
    call_args = mock_llm.generate.call_args
    messages = call_args[0][0]
    prompt_content = messages[0].content

    # Check that content was truncated
    assert "..." in prompt_content, "Truncated content should have ... marker"
    assert len(doc.page_content) > reranker.max_doc_length, \
        "Original doc should be longer than max_doc_length"


# HybridReranker Tests

@patch('src.reranking.reranker.LLMGenerator')
def test_hybrid_reranker_initialization(mock_llm_class):
    """
    Test that HybridReranker initializes correctly.

    Verifies:
    - Default weights sum to 1.0
    - Custom weights are accepted
    - Weight validation works
    - LLMReranker is initialized
    """
    mock_llm = Mock()
    mock_llm.get_model_info = Mock(return_value={'model': 'claude-sonnet-4'})
    mock_llm_class.return_value = mock_llm

    # Test default weights
    llm_reranker = LLMReranker()
    hybrid = HybridReranker(llm_reranker)

    assert hybrid.llm_weight == 0.7, "Default LLM weight should be 0.7"
    assert hybrid.similarity_weight == 0.3, "Default similarity weight should be 0.3"
    assert hybrid.llm_weight + hybrid.similarity_weight == 1.0, "Weights should sum to 1.0"

    # Test custom weights
    hybrid2 = HybridReranker(llm_reranker, llm_weight=0.6, similarity_weight=0.4)
    assert hybrid2.llm_weight == 0.6
    assert hybrid2.similarity_weight == 0.4

    # Test weight validation (should raise error if not sum to 1.0)
    with pytest.raises(ValueError, match="Weights must sum to 1.0"):
        HybridReranker(llm_reranker, llm_weight=0.6, similarity_weight=0.5)


@patch('src.reranking.reranker.LLMGenerator')
def test_hybrid_reranker(mock_llm_class, sample_documents_with_scores):
    """
    Test that HybridReranker combines scores correctly.

    Verifies:
    - Similarity scores are normalized to 0-10 scale
    - LLM scores are obtained for each document
    - Scores are combined with correct weights
    - Documents are sorted by combined score
    """
    # Configure mock LLM to return specific scores
    mock_llm = Mock()
    mock_llm.get_model_info = Mock(return_value={'model': 'claude-sonnet-4'})

    # LLM scores: 8, 9, 6 (high for doc1, very high for doc2, medium for doc3)
    mock_llm.generate = Mock(side_effect=["8", "9", "6"])
    mock_llm_class.return_value = mock_llm

    # Create hybrid reranker with equal weights for easier math
    llm_reranker = LLMReranker()
    hybrid = HybridReranker(llm_reranker, llm_weight=0.5, similarity_weight=0.5)

    # Re-rank documents
    query = "What is RAG?"
    ranked_docs = hybrid.rerank(query, sample_documents_with_scores)

    # Verify all documents returned
    assert len(ranked_docs) == 3, "Should return all 3 documents"

    # Verify scores are combined (normalized similarity + LLM scores)
    # Original similarity scores: [0.85, 0.45, 0.62]
    # Normalized (min-max to 0-10): [10.0, 0.0, 4.25]
    # LLM scores: [8, 9, 6]
    # Combined (50/50): [9.0, 4.5, 5.125]
    # So order should be: doc1 (9.0), doc3 (5.125), doc2 (4.5)

    # First doc should have highest combined score
    assert ranked_docs[0][0].metadata['source'] == 'doc1.txt', \
        "Doc1 should be ranked first (high similarity + high LLM score)"

    # Verify scores are in descending order
    for i in range(len(ranked_docs) - 1):
        assert ranked_docs[i][1] >= ranked_docs[i + 1][1], \
            "Combined scores should be in descending order"


@patch('src.reranking.reranker.LLMGenerator')
def test_hybrid_reranker_score_normalization(mock_llm_class):
    """
    Test that similarity scores are normalized correctly.

    Verifies:
    - Min-max normalization to 0-10 scale
    - Handles edge cases (all same scores)
    - Normalization is applied correctly
    """
    mock_llm = Mock()
    mock_llm.get_model_info = Mock(return_value={'model': 'claude-sonnet-4'})
    mock_llm_class.return_value = mock_llm

    llm_reranker = LLMReranker()
    hybrid = HybridReranker(llm_reranker)

    # Test normalization
    scores = [0.5, 0.7, 0.9, 0.3]
    normalized = hybrid._normalize_scores(scores)

    # Min = 0.3, Max = 0.9, range = 0.6
    # Normalized: [(s - 0.3) / 0.6 * 10]
    # Expected: [3.33, 6.67, 10.0, 0.0]
    assert len(normalized) == 4, "Should normalize all scores"
    assert abs(normalized[0] - 3.33) < 0.1, "Should normalize first score correctly"
    assert abs(normalized[1] - 6.67) < 0.1, "Should normalize second score correctly"
    assert normalized[2] == 10.0, "Max score should be 10.0"
    assert normalized[3] == 0.0, "Min score should be 0.0"

    # Test edge case: all same scores
    same_scores = [0.5, 0.5, 0.5]
    normalized_same = hybrid._normalize_scores(same_scores)
    assert all(s == 5.0 for s in normalized_same), \
        "All identical scores should normalize to 5.0"


@patch('src.reranking.reranker.LLMGenerator')
def test_hybrid_reranker_top_k(mock_llm_class, sample_documents_with_scores):
    """
    Test that top_k parameter works with hybrid re-ranking.

    Verifies:
    - Only top_k documents are returned
    - Highest combined scores are selected
    """
    mock_llm = Mock()
    mock_llm.get_model_info = Mock(return_value={'model': 'claude-sonnet-4'})
    mock_llm.generate = Mock(side_effect=["8", "9", "6"])
    mock_llm_class.return_value = mock_llm

    llm_reranker = LLMReranker()
    hybrid = HybridReranker(llm_reranker)

    # Re-rank with top_k=2
    query = "What is RAG?"
    ranked_docs = hybrid.rerank(query, sample_documents_with_scores, top_k=2)

    # Verify only top 2 returned
    assert len(ranked_docs) == 2, "Should return only top 2 documents"

    # Verify they are highest scoring
    assert ranked_docs[0][1] >= ranked_docs[1][1], \
        "Top 2 should be in descending order"


# Edge Cases and Error Handling

@patch('src.reranking.reranker.LLMGenerator')
def test_reranker_empty_docs(mock_llm_class):
    """
    Test that re-rankers handle empty document lists correctly.

    Verifies:
    - LLMReranker returns empty list for empty input
    - HybridReranker returns empty list for empty input
    - No errors are raised
    """
    mock_llm = Mock()
    mock_llm.get_model_info = Mock(return_value={'model': 'claude-sonnet-4'})
    mock_llm_class.return_value = mock_llm

    # Test LLMReranker
    llm_reranker = LLMReranker()
    result = llm_reranker.rerank("What is RAG?", [])
    assert result == [], "LLMReranker should return empty list for empty input"

    # Test HybridReranker
    hybrid = HybridReranker(llm_reranker)
    result = hybrid.rerank("What is RAG?", [])
    assert result == [], "HybridReranker should return empty list for empty input"


@patch('src.reranking.reranker.LLMGenerator')
def test_reranker_config(mock_llm_class):
    """
    Test retrieving reranker configurations.

    Verifies:
    - LLMReranker config includes all parameters
    - HybridReranker config includes weights and LLM config
    """
    mock_llm = Mock()
    mock_llm.get_model_info = Mock(return_value={
        'model': 'claude-sonnet-4',
        'temperature': 0.0,
        'max_tokens': 10,
    })
    mock_llm_class.return_value = mock_llm

    # Test LLMReranker config
    llm_reranker = LLMReranker(temperature=0.0, max_tokens=15, max_doc_length=800)
    config = llm_reranker.get_reranker_config()

    assert 'temperature' in config
    assert 'max_tokens' in config
    assert 'max_doc_length' in config
    assert 'llm_config' in config

    assert config['temperature'] == 0.0
    assert config['max_tokens'] == 15
    assert config['max_doc_length'] == 800

    # Test HybridReranker config
    hybrid = HybridReranker(llm_reranker, llm_weight=0.6, similarity_weight=0.4)
    hybrid_config = hybrid.get_reranker_config()

    assert 'llm_weight' in hybrid_config
    assert 'similarity_weight' in hybrid_config
    assert 'llm_reranker_config' in hybrid_config

    assert hybrid_config['llm_weight'] == 0.6
    assert hybrid_config['similarity_weight'] == 0.4
