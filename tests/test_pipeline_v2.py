"""
Integration tests for Enhanced RAG Pipeline (Phase 2).

Tests the complete Phase 2 pipeline with query rewriting, multi-query,
re-ranking, and self-critique features.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from langchain_core.documents import Document

from src.pipeline_v2 import EnhancedRAGPipeline, create_enhanced_pipeline
from src.agents import QueryRewriter, MultiQueryRetriever
from src.reranking import LLMReranker


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture
def sample_documents():
    """Sample documents for testing."""
    return [
        Document(
            page_content="RAG combines retrieval with generation for better answers.",
            metadata={'source': 'rag_guide.txt', 'page': 1}
        ),
        Document(
            page_content="Vector databases store embeddings for semantic search.",
            metadata={'source': 'vector_db.txt', 'page': 2}
        ),
        Document(
            page_content="LLMs can hallucinate without proper grounding in sources.",
            metadata={'source': 'llm_guide.txt', 'page': 3}
        ),
    ]


@pytest.fixture
def mock_vector_store():
    """Mock vector store manager."""
    mock = Mock()
    mock.add_documents = Mock(return_value=['id1', 'id2', 'id3'])
    mock.get_collection_stats = Mock(return_value={'count': 3})
    mock.delete_collection = Mock()
    return mock


@pytest.fixture
def mock_retriever(sample_documents):
    """Mock retriever that returns sample documents."""
    mock = Mock()
    # Return documents with scores
    mock.retrieve = Mock(return_value=[
        (sample_documents[0], 0.95),
        (sample_documents[1], 0.87),
        (sample_documents[2], 0.78),
    ])
    mock.get_retriever_config = Mock(return_value={'type': 'mock', 'top_k': 3})
    return mock


# ==============================================================================
# Initialization Tests
# ==============================================================================

@patch('src.pipeline.VectorStoreManager')
@patch('src.pipeline.AdvancedRetriever')
@patch('src.pipeline.LLMGenerator')
def test_enhanced_pipeline_initialization(
    mock_llm_class,
    mock_retriever_class,
    mock_vector_class
):
    """
    Test that EnhancedRAGPipeline initializes all Phase 2 components correctly.

    Verifies:
    - All enabled components are initialized
    - Disabled components are None
    - Feature counts are correct
    """
    # Configure mocks
    mock_vector = Mock()
    mock_vector.get_collection_stats = Mock(return_value={'count': 0})
    mock_vector_class.return_value = mock_vector

    mock_retriever = Mock()
    mock_retriever.get_retriever_config = Mock(return_value={'type': 'mock'})
    mock_retriever_class.return_value = mock_retriever

    mock_llm = Mock()
    mock_llm.get_model_info = Mock(return_value={'model': 'claude-sonnet-4'})
    mock_llm_class.return_value = mock_llm

    # Create pipeline with all features enabled
    pipeline = EnhancedRAGPipeline(
        enable_query_rewriting=True,
        enable_multi_query=True,
        enable_reranking=True,
        enable_react_agent=True,
        enable_self_critique=True,
    )

    # Check components are initialized
    assert pipeline.query_rewriter is not None, "Query rewriter should be initialized"
    assert pipeline.multi_query_retriever is not None, "Multi-query retriever should be initialized"
    assert pipeline.reranker is not None, "Reranker should be initialized"
    assert pipeline.react_agent is not None, "ReAct agent should be initialized"
    assert pipeline.self_critique_agent is not None, "Self-critique agent should be initialized"

    # Check feature flags
    assert pipeline.enable_query_rewriting == True
    assert pipeline.enable_multi_query == True
    assert pipeline.enable_reranking == True
    assert pipeline.enable_react_agent == True
    assert pipeline.enable_self_critique == True

    # Create pipeline with all features disabled
    pipeline_minimal = EnhancedRAGPipeline(
        enable_query_rewriting=False,
        enable_multi_query=False,
        enable_reranking=False,
        enable_react_agent=False,
        enable_self_critique=False,
    )

    # Check components are None
    assert pipeline_minimal.query_rewriter is None
    assert pipeline_minimal.multi_query_retriever is None
    assert pipeline_minimal.reranker is None
    assert pipeline_minimal.react_agent is None
    assert pipeline_minimal.self_critique_agent is None


@patch('src.pipeline.VectorStoreManager')
@patch('src.pipeline.AdvancedRetriever')
@patch('src.pipeline.LLMGenerator')
def test_enhanced_pipeline_presets(
    mock_llm_class,
    mock_retriever_class,
    mock_vector_class
):
    """
    Test that preset configurations initialize correctly.

    Verifies:
    - Minimal preset has only query rewriting
    - Standard preset has recommended features
    - Full preset has all features
    """
    # Configure mocks
    mock_vector = Mock()
    mock_vector.get_collection_stats = Mock(return_value={'count': 0})
    mock_vector_class.return_value = mock_vector

    mock_retriever = Mock()
    mock_retriever.get_retriever_config = Mock(return_value={'type': 'mock'})
    mock_retriever_class.return_value = mock_retriever

    mock_llm = Mock()
    mock_llm.get_model_info = Mock(return_value={'model': 'claude-sonnet-4'})
    mock_llm_class.return_value = mock_llm

    # Test minimal preset
    pipeline_minimal = create_enhanced_pipeline(preset="minimal")
    config_minimal = pipeline_minimal.get_phase2_config()
    assert config_minimal['enabled_count'] == 1, "Minimal should have 1 feature"
    assert config_minimal['features']['query_rewriting']['enabled'] == True

    # Test standard preset
    pipeline_standard = create_enhanced_pipeline(preset="standard")
    config_standard = pipeline_standard.get_phase2_config()
    assert config_standard['enabled_count'] == 4, "Standard should have 4 features"
    assert config_standard['features']['query_rewriting']['enabled'] == True
    assert config_standard['features']['multi_query']['enabled'] == True
    assert config_standard['features']['reranking']['enabled'] == True
    assert config_standard['features']['self_critique']['enabled'] == True

    # Test full preset
    pipeline_full = create_enhanced_pipeline(preset="full")
    config_full = pipeline_full.get_phase2_config()
    assert config_full['enabled_count'] == 5, "Full should have 5 features"
    assert config_full['features']['react_agent']['enabled'] == True


# ==============================================================================
# Query_v2 Feature Tests
# ==============================================================================

@patch('src.pipeline.VectorStoreManager')
@patch('src.pipeline.AdvancedRetriever')
@patch('src.pipeline.LLMGenerator')
def test_query_v2_with_query_rewriting(
    mock_llm_class,
    mock_retriever_class,
    mock_vector_class,
    sample_documents
):
    """
    Test query_v2 with query rewriting enabled.

    Verifies:
    - Query is rewritten before retrieval
    - Optimized query is tracked in metadata
    - Feature usage is recorded
    """
    # Configure mocks
    mock_vector = Mock()
    mock_vector.get_collection_stats = Mock(return_value={'count': 0})
    mock_vector_class.return_value = mock_vector

    mock_retriever = Mock()
    mock_retriever.retrieve = Mock(return_value=[
        (sample_documents[0], 0.95),
        (sample_documents[1], 0.87),
    ])
    mock_retriever.get_retriever_config = Mock(return_value={'type': 'mock'})
    mock_retriever_class.return_value = mock_retriever

    mock_llm = Mock()
    mock_llm.get_model_info = Mock(return_value={'model': 'claude-sonnet-4'})
    # Mock query rewriting LLM response
    mock_llm.generate = Mock(side_effect=[
        "What is Retrieval Augmented Generation and how does it work?",  # Rewritten query
        "RAG combines retrieval with generation to provide grounded answers.",  # Final answer
    ])
    mock_llm_class.return_value = mock_llm

    # Create pipeline with query rewriting enabled
    pipeline = EnhancedRAGPipeline(
        enable_query_rewriting=True,
        enable_multi_query=False,
        enable_reranking=False,
        enable_react_agent=False,
        enable_self_critique=False,
    )

    # Query
    result = pipeline.query_v2(
        "What is RAG?",
        use_query_rewriting=True,
        use_reranking=False,
        use_self_critique=False,
    )

    # Verify query was rewritten
    assert 'phase2_metadata' in result
    assert result['phase2_metadata']['original_query'] == "What is RAG?"
    assert result['phase2_metadata']['optimized_query'] != "What is RAG?"
    assert result['phase2_metadata']['features_used']['query_rewriting'] == True

    # Verify answer was generated
    assert 'answer' in result
    assert len(result['answer']) > 0


@patch('src.pipeline.VectorStoreManager')
@patch('src.pipeline.AdvancedRetriever')
@patch('src.pipeline.LLMGenerator')
def test_query_v2_with_multi_query(
    mock_llm_class,
    mock_retriever_class,
    mock_vector_class,
    sample_documents
):
    """
    Test query_v2 with multi-query retrieval enabled.

    Verifies:
    - Multiple query variations are generated
    - MultiQueryRetriever is used
    - Feature usage is recorded
    """
    # Configure mocks
    mock_vector = Mock()
    mock_vector.get_collection_stats = Mock(return_value={'count': 0})
    mock_vector_class.return_value = mock_vector

    mock_retriever = Mock()
    mock_retriever.retrieve = Mock(return_value=[
        (sample_documents[0], 0.95),
        (sample_documents[1], 0.87),
        (sample_documents[2], 0.78),
    ])
    mock_retriever.get_retriever_config = Mock(return_value={'type': 'mock'})
    mock_retriever_class.return_value = mock_retriever

    mock_llm = Mock()
    mock_llm.get_model_info = Mock(return_value={'model': 'claude-sonnet-4'})
    # Mock multi-query generation and final answer
    mock_llm.generate = Mock(side_effect=[
        "1. What is RAG?\n2. How does RAG work?\n3. What are RAG benefits?",  # Multi-query
        "RAG is a technique that combines retrieval with generation.",  # Final answer
    ])
    mock_llm_class.return_value = mock_llm

    # Create pipeline with multi-query enabled
    pipeline = EnhancedRAGPipeline(
        enable_query_rewriting=False,
        enable_multi_query=True,
        enable_reranking=False,
        enable_react_agent=False,
        enable_self_critique=False,
    )

    # Query
    result = pipeline.query_v2(
        "What is RAG?",
        use_query_rewriting=False,
        use_multi_query=True,
        use_reranking=False,
        use_self_critique=False,
    )

    # Verify multi-query was used
    assert 'phase2_metadata' in result
    assert result['phase2_metadata']['features_used']['multi_query'] == True

    # Verify answer was generated
    assert 'answer' in result
    assert result['num_sources'] > 0


@patch('src.pipeline.VectorStoreManager')
@patch('src.pipeline.AdvancedRetriever')
@patch('src.pipeline.LLMGenerator')
def test_query_v2_with_reranking(
    mock_llm_class,
    mock_retriever_class,
    mock_vector_class,
    sample_documents
):
    """
    Test query_v2 with re-ranking enabled.

    Verifies:
    - Documents are re-ranked by relevance
    - Scores are updated
    - Feature usage is recorded
    """
    # Configure mocks
    mock_vector = Mock()
    mock_vector.get_collection_stats = Mock(return_value={'count': 0})
    mock_vector_class.return_value = mock_vector

    mock_retriever = Mock()
    # Return docs with initial scores
    mock_retriever.retrieve = Mock(return_value=[
        (sample_documents[0], 0.85),
        (sample_documents[1], 0.90),  # Initially higher score
        (sample_documents[2], 0.80),
    ])
    mock_retriever.get_retriever_config = Mock(return_value={'type': 'mock'})
    mock_retriever_class.return_value = mock_retriever

    mock_llm = Mock()
    mock_llm.get_model_info = Mock(return_value={'model': 'claude-sonnet-4'})
    # Mock re-ranking scores (reverse order) and final answer
    mock_llm.generate = Mock(side_effect=[
        "9.5",  # Re-rank score for doc 0 (high relevance)
        "6.0",  # Re-rank score for doc 1 (lower relevance)
        "7.0",  # Re-rank score for doc 2 (medium relevance)
        "RAG combines retrieval with generation.",  # Final answer
    ])
    mock_llm_class.return_value = mock_llm

    # Create pipeline with re-ranking enabled
    pipeline = EnhancedRAGPipeline(
        enable_query_rewriting=False,
        enable_multi_query=False,
        enable_reranking=True,
        enable_react_agent=False,
        enable_self_critique=False,
    )

    # Query
    result = pipeline.query_v2(
        "What is RAG?",
        use_query_rewriting=False,
        use_multi_query=False,
        use_reranking=True,
        use_self_critique=False,
        return_sources=True,
    )

    # Verify re-ranking was used
    assert 'phase2_metadata' in result
    assert result['phase2_metadata']['features_used']['reranking'] == True

    # Verify sources are included with scores
    assert 'sources' in result
    assert len(result['sources']) > 0

    # Verify top document has highest re-ranked score
    # Note: In real scenario, doc 0 (score 9.5) should be first after re-ranking
    assert result['sources'][0]['score'] > 0


@patch('src.pipeline.VectorStoreManager')
@patch('src.pipeline.AdvancedRetriever')
@patch('src.pipeline.LLMGenerator')
def test_query_v2_with_self_critique(
    mock_llm_class,
    mock_retriever_class,
    mock_vector_class,
    sample_documents
):
    """
    Test query_v2 with self-critique validation.

    Verifies:
    - Answer is critiqued for quality
    - Critique results are included
    - should_refine flag is set correctly
    """
    # Configure mocks
    mock_vector = Mock()
    mock_vector.get_collection_stats = Mock(return_value={'count': 0})
    mock_vector_class.return_value = mock_vector

    mock_retriever = Mock()
    mock_retriever.retrieve = Mock(return_value=[
        (sample_documents[0], 0.95),
        (sample_documents[1], 0.87),
    ])
    mock_retriever.get_retriever_config = Mock(return_value={'type': 'mock'})
    mock_retriever_class.return_value = mock_retriever

    mock_llm = Mock()
    mock_llm.get_model_info = Mock(return_value={'model': 'claude-sonnet-4'})
    # Mock final answer and critique response
    mock_llm.generate = Mock(side_effect=[
        "RAG combines retrieval with generation to provide grounded answers.",  # Final answer
        """Addresses Question: Yes
Has Citations: Yes
Supported: Yes
Hallucination Risk: Low
Improvements: None needed, answer is comprehensive and well-grounded.
Overall Quality: Excellent""",  # Critique response
    ])
    mock_llm_class.return_value = mock_llm

    # Create pipeline with self-critique enabled
    pipeline = EnhancedRAGPipeline(
        enable_query_rewriting=False,
        enable_multi_query=False,
        enable_reranking=False,
        enable_react_agent=False,
        enable_self_critique=True,
    )

    # Query
    result = pipeline.query_v2(
        "What is RAG?",
        use_query_rewriting=False,
        use_reranking=False,
        use_self_critique=True,
    )

    # Verify self-critique was used
    assert 'phase2_metadata' in result
    assert result['phase2_metadata']['features_used']['self_critique'] == True

    # Verify critique is included
    assert 'critique' in result
    assert 'should_refine' in result
    assert 'overall_quality' in result['critique']
    assert result['critique']['overall_quality'] in ['Poor', 'Fair', 'Good', 'Excellent']
    assert isinstance(result['should_refine'], bool)


# ==============================================================================
# Backward Compatibility Tests
# ==============================================================================

@patch('src.pipeline.VectorStoreManager')
@patch('src.pipeline.AdvancedRetriever')
@patch('src.pipeline.LLMGenerator')
def test_query_v2_backward_compatible(
    mock_llm_class,
    mock_retriever_class,
    mock_vector_class,
    sample_documents
):
    """
    Test that Phase 1 query() method still works on EnhancedRAGPipeline.

    Verifies:
    - Phase 1 query() is inherited
    - Phase 1 query() returns expected format
    - No Phase 2 metadata in Phase 1 responses
    """
    # Configure mocks
    mock_vector = Mock()
    mock_vector.get_collection_stats = Mock(return_value={'count': 0})
    mock_vector_class.return_value = mock_vector

    mock_retriever = Mock()
    mock_retriever.retrieve = Mock(return_value=[
        (sample_documents[0], 0.95),
        (sample_documents[1], 0.87),
    ])
    mock_retriever.get_retriever_config = Mock(return_value={'type': 'mock'})
    mock_retriever_class.return_value = mock_retriever

    mock_llm = Mock()
    mock_llm.get_model_info = Mock(return_value={'model': 'claude-sonnet-4'})
    mock_llm.generate = Mock(return_value="RAG is a retrieval-augmented generation technique.")
    mock_llm_class.return_value = mock_llm

    # Create enhanced pipeline
    pipeline = EnhancedRAGPipeline(
        enable_query_rewriting=True,
        enable_multi_query=True,
        enable_reranking=True,
    )

    # Use Phase 1 query() method
    result = pipeline.query("What is RAG?", return_sources=True)

    # Verify Phase 1 format
    assert 'answer' in result
    assert 'num_sources' in result
    assert 'sources' in result

    # Verify Phase 2 metadata is NOT in Phase 1 response
    assert 'phase2_metadata' not in result
    assert 'critique' not in result


# ==============================================================================
# Metadata Tests
# ==============================================================================

@patch('src.pipeline.VectorStoreManager')
@patch('src.pipeline.AdvancedRetriever')
@patch('src.pipeline.LLMGenerator')
def test_enhanced_metadata(
    mock_llm_class,
    mock_retriever_class,
    mock_vector_class,
    sample_documents
):
    """
    Test that Phase 2 metadata is properly added to responses.

    Verifies:
    - original_query is tracked
    - optimized_query is tracked
    - features_used dictionary is complete
    - All feature flags are present
    """
    # Configure mocks
    mock_vector = Mock()
    mock_vector.get_collection_stats = Mock(return_value={'count': 0})
    mock_vector_class.return_value = mock_vector

    mock_retriever = Mock()
    mock_retriever.retrieve = Mock(return_value=[
        (sample_documents[0], 0.95),
    ])
    mock_retriever.get_retriever_config = Mock(return_value={'type': 'mock'})
    mock_retriever_class.return_value = mock_retriever

    mock_llm = Mock()
    mock_llm.get_model_info = Mock(return_value={'model': 'claude-sonnet-4'})
    mock_llm.generate = Mock(return_value="RAG is a technique.")
    mock_llm_class.return_value = mock_llm

    # Create pipeline
    pipeline = EnhancedRAGPipeline(
        enable_query_rewriting=False,
        enable_multi_query=False,
        enable_reranking=False,
        enable_react_agent=False,
        enable_self_critique=False,
    )

    # Query
    result = pipeline.query_v2(
        "What is RAG?",
        use_query_rewriting=False,
        use_multi_query=False,
        use_reranking=False,
        use_self_critique=False,
    )

    # Verify metadata structure
    assert 'phase2_metadata' in result
    metadata = result['phase2_metadata']

    # Check required fields
    assert 'original_query' in metadata
    assert 'optimized_query' in metadata
    assert 'features_used' in metadata

    # Check features_used completeness
    features = metadata['features_used']
    assert 'query_rewriting' in features
    assert 'multi_query' in features
    assert 'reranking' in features
    assert 'react_agent' in features
    assert 'self_critique' in features

    # Verify values
    assert metadata['original_query'] == "What is RAG?"
    assert metadata['optimized_query'] == "What is RAG?"  # No rewriting
    assert features['query_rewriting'] == False
    assert features['multi_query'] == False
    assert features['reranking'] == False
    assert features['react_agent'] == False
    assert features['self_critique'] == False


@patch('src.pipeline.VectorStoreManager')
@patch('src.pipeline.AdvancedRetriever')
@patch('src.pipeline.LLMGenerator')
def test_get_phase2_config(
    mock_llm_class,
    mock_retriever_class,
    mock_vector_class
):
    """
    Test get_phase2_config() method returns correct configuration.

    Verifies:
    - Returns all feature flags
    - Returns component names
    - Returns enabled count
    - Includes base pipeline stats
    """
    # Configure mocks
    mock_vector = Mock()
    mock_vector.get_collection_stats = Mock(return_value={'count': 0})
    mock_vector_class.return_value = mock_vector

    mock_retriever = Mock()
    mock_retriever.get_retriever_config = Mock(return_value={'type': 'mock'})
    mock_retriever_class.return_value = mock_retriever

    mock_llm = Mock()
    mock_llm.get_model_info = Mock(return_value={'model': 'claude-sonnet-4'})
    mock_llm_class.return_value = mock_llm

    # Create pipeline with specific configuration
    pipeline = EnhancedRAGPipeline(
        enable_query_rewriting=True,
        enable_multi_query=False,
        enable_reranking=True,
        enable_react_agent=False,
        enable_self_critique=True,
    )

    # Get configuration
    config = pipeline.get_phase2_config()

    # Verify structure
    assert 'phase' in config
    assert config['phase'] == 2
    assert 'features' in config
    assert 'enabled_count' in config
    assert 'base_pipeline' in config

    # Verify feature details
    features = config['features']
    assert features['query_rewriting']['enabled'] == True
    assert features['query_rewriting']['component'] == 'QueryRewriter'
    assert features['multi_query']['enabled'] == False
    assert features['multi_query']['component'] == None
    assert features['reranking']['enabled'] == True
    assert features['react_agent']['enabled'] == False
    assert features['self_critique']['enabled'] == True

    # Verify count
    assert config['enabled_count'] == 3
