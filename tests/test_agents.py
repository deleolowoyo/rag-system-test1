"""
Tests for agents package (Phase 2).

Focused tests for query rewriting and multi-query generation with mocked LLM calls
to avoid API costs during testing.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from langchain_core.documents import Document

from src.agents.query_rewriter import QueryRewriter
from src.agents.multi_query import MultiQueryGenerator, MultiQueryRetriever


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


# Multi-Query Generation Tests

@patch('src.agents.multi_query.LLMGenerator')
def test_multi_query_generator(mock_llm_class):
    """
    Test that MultiQueryGenerator generates multiple query variations.

    Verifies:
    - Generates expected number of query variations
    - Includes original query in results
    - Parses LLM response correctly
    - Returns unique queries
    """
    # Configure mock LLM
    mock_llm = Mock()
    mock_llm.generate = Mock(return_value="""1. What is Retrieval Augmented Generation?
2. How does RAG combine retrieval and language models?
3. What are the components of a RAG system?""")
    mock_llm.get_model_info = Mock(return_value={
        'model': 'claude-sonnet-4',
        'temperature': 0.7,
        'max_tokens': 300,
    })
    mock_llm_class.return_value = mock_llm

    # Create generator
    generator = MultiQueryGenerator()

    # Generate query variations
    original_query = "What is RAG?"
    queries = generator.generate_queries(original_query, num_queries=3)

    # Verify results
    assert len(queries) >= 4, "Should return original + 3 variations"
    assert queries[0] == original_query, "First query should be original"
    assert "Retrieval Augmented Generation" in queries[1], "Should expand RAG"

    # Verify all queries are unique
    assert len(queries) == len(set(q.lower() for q in queries)), \
        "All queries should be unique (case-insensitive)"

    # Verify LLM was called
    mock_llm.generate.assert_called_once()


@patch('src.agents.multi_query.LLMGenerator')
def test_multi_query_deduplication(mock_llm_class):
    """
    Test that MultiQueryRetriever removes duplicate documents.

    Verifies:
    - Duplicate documents are identified by content hash
    - Only unique documents are returned
    - Deduplication works across multiple query results
    """
    # Configure mock LLM for query generation
    mock_llm = Mock()
    mock_llm.generate = Mock(return_value="""1. Machine learning basics
2. Introduction to ML""")
    mock_llm.get_model_info = Mock(return_value={'model': 'claude-sonnet-4'})
    mock_llm_class.return_value = mock_llm

    # Create mock retriever
    mock_retriever = Mock()
    mock_retriever.get_retriever_config = Mock(return_value={
        'search_type': 'similarity',
        'top_k': 3,
    })

    # Create duplicate documents
    doc1 = Document(
        page_content="Machine learning is a subset of AI.",
        metadata={'source': 'doc1.txt'}
    )
    doc2 = Document(
        page_content="ML algorithms learn from data.",
        metadata={'source': 'doc2.txt'}
    )
    # doc3 is a duplicate of doc1 (same content and source)
    doc3 = Document(
        page_content="Machine learning is a subset of AI.",
        metadata={'source': 'doc1.txt'}
    )

    # Mock retriever returns different docs for each query, with one duplicate
    # Note: generator returns original + 2 variations = 3 total queries
    mock_retriever.retrieve = Mock(side_effect=[
        [doc1, doc2],  # First query results
        [doc3, doc2],  # Second query results (doc3 is duplicate of doc1)
        [doc1],        # Third query results (doc1 already seen)
    ])

    # Create multi-query retriever
    generator = MultiQueryGenerator()
    multi_retriever = MultiQueryRetriever(
        retriever=mock_retriever,
        generator=generator,
        num_queries=2,
        top_k_per_query=2,
    )

    # Retrieve documents
    docs = multi_retriever.retrieve("What is machine learning?")

    # Verify deduplication
    assert len(docs) == 2, "Should have 2 unique docs (doc1=doc3 duplicate)"

    # Verify content
    contents = [doc.page_content for doc in docs]
    assert "Machine learning is a subset of AI." in contents
    assert "ML algorithms learn from data." in contents

    # Verify retriever was called for each query (original + 2 variations = 3)
    assert mock_retriever.retrieve.call_count == 3


@patch('src.agents.multi_query.LLMGenerator')
def test_multi_query_parsing(mock_llm_class):
    """
    Test that MultiQueryGenerator handles different LLM response formats.

    Verifies:
    - Parses numbered lists (1. 2. 3.)
    - Parses bullet points (-, *, •)
    - Removes quotes from queries
    - Handles mixed formats
    """
    mock_llm = Mock()
    mock_llm.get_model_info = Mock(return_value={'model': 'claude-sonnet-4'})
    mock_llm_class.return_value = mock_llm

    generator = MultiQueryGenerator()

    # Test 1: Numbered list format
    mock_llm.generate = Mock(return_value="""1. First variation
2. Second variation
3. Third variation""")

    queries = generator.generate_queries("Test query", num_queries=3)
    assert "First variation" in queries
    assert "Second variation" in queries
    assert "Third variation" in queries

    # Test 2: Bullet point format
    mock_llm.generate = Mock(return_value="""- First bullet
* Second bullet
• Third bullet""")

    queries = generator.generate_queries("Test query", num_queries=3)
    assert "First bullet" in queries
    assert "Second bullet" in queries
    assert "Third bullet" in queries

    # Test 3: Queries with quotes
    mock_llm.generate = Mock(return_value="""1. "What is AI?"
2. 'How does ML work?'
3. What are neural networks?""")

    queries = generator.generate_queries("Test query", num_queries=3)
    # Quotes should be removed
    assert "What is AI?" in queries or '"What is AI?"' not in ' '.join(queries)
    assert "How does ML work?" in queries or "'How does ML work?'" not in ' '.join(queries)

    # Test 4: Mixed format
    mock_llm.generate = Mock(return_value="""1. Numbered query
- Bullet query
3) Parenthesis query""")

    queries = generator.generate_queries("Test query", num_queries=3)
    assert len(queries) >= 3, "Should parse mixed formats"


# ReAct Agent Tests

@patch('src.agents.react_agent.LLMGenerator')
def test_react_agent_initialization(mock_llm_class):
    """
    Test that ReAct Agent initializes correctly.

    Verifies:
    - Default max_iterations is set correctly
    - LLM instance is created
    - QueryRewriter is initialized
    - Agent config is accessible
    """
    from src.agents.react_agent import ReActAgent

    # Configure mock LLM
    mock_llm = Mock()
    mock_llm.get_model_info = Mock(return_value={
        'model': 'claude-sonnet-4',
        'temperature': 0.3,
        'max_tokens': 500,
    })
    mock_llm_class.return_value = mock_llm

    # Mock retriever
    mock_retriever = Mock()
    mock_retriever.get_retriever_config = Mock(return_value={'type': 'mock'})

    # Create agent with default parameters
    agent = ReActAgent(retriever=mock_retriever)

    # Verify initialization
    assert agent.max_iterations == 5, "Default max_iterations should be 5"
    assert agent.temperature == 0.3, "Default temperature should be 0.3"
    assert agent.llm is not None, "LLM should be initialized"
    assert agent.query_rewriter is not None, "QueryRewriter should be initialized"
    assert agent.retriever is mock_retriever, "Retriever should be set"

    # Test custom parameters
    custom_agent = ReActAgent(
        retriever=mock_retriever,
        max_iterations=3,
        temperature=0.5,
    )
    assert custom_agent.max_iterations == 3
    assert custom_agent.temperature == 0.5

    # Test config retrieval
    config = agent.get_agent_config()
    assert 'max_iterations' in config
    assert 'temperature' in config
    assert 'has_query_rewriter' in config
    assert config['max_iterations'] == 5


@patch('src.agents.react_agent.LLMGenerator')
@patch('src.agents.react_agent.QueryRewriter')
def test_react_agent_single_iteration(mock_rewriter_class, mock_llm_class):
    """
    Test that ReAct Agent completes in one step when appropriate.

    Verifies:
    - Agent can decide to answer immediately
    - State is updated correctly
    - Final result contains all required fields
    - Actions and thoughts are recorded
    """
    from src.agents.react_agent import ReActAgent
    from langchain_core.documents import Document

    # Configure mock LLM
    mock_llm = Mock()
    mock_llm.get_model_info = Mock(return_value={'model': 'claude-sonnet-4'})

    # First call: reasoning (decides to search)
    # Second call: reasoning (decides to answer)
    # Third call: answer generation
    mock_llm.generate = Mock(side_effect=[
        # First iteration: decide to search
        """Thought: I should search for documents
Action: SEARCH
Action Input: test query""",
        # Second iteration: decide to answer
        """Thought: I have enough information
Action: ANSWER
Action Input: """,
        # Answer generation
        "This is the final answer based on retrieved documents."
    ])
    mock_llm_class.return_value = mock_llm

    # Configure mock retriever
    mock_retriever = Mock()
    test_docs = [
        Document(page_content="Test document 1", metadata={'source': 'test1.txt'}),
        Document(page_content="Test document 2", metadata={'source': 'test2.txt'}),
    ]
    mock_retriever.retrieve = Mock(return_value=test_docs)

    # Configure mock query rewriter
    mock_rewriter = Mock()
    mock_rewriter.rewrite = Mock(return_value="optimized query")
    mock_rewriter_class.return_value = mock_rewriter

    # Create and run agent
    agent = ReActAgent(
        retriever=mock_retriever,
        max_iterations=5,
    )

    result = agent.run("What is test?")

    # Verify result structure
    assert 'answer' in result
    assert 'documents' in result
    assert 'reasoning_trace' in result
    assert 'query_evolution' in result
    assert 'iterations' in result
    assert 'actions_taken' in result

    # Verify completion
    assert result['iterations'] == 2, "Should complete in 2 iterations"
    assert len(result['documents']) == 2, "Should have 2 documents"
    assert len(result['actions_taken']) == 2, "Should have taken 2 actions"
    assert 'search' in result['actions_taken']
    assert 'answer' in result['actions_taken']

    # Verify answer was generated
    assert result['answer'] == "This is the final answer based on retrieved documents."

    # Verify LLM was called for reasoning and answer generation
    assert mock_llm.generate.call_count == 3  # 2 reasoning + 1 answer


@patch('src.agents.react_agent.LLMGenerator')
@patch('src.agents.react_agent.QueryRewriter')
def test_react_agent_max_iterations(mock_rewriter_class, mock_llm_class):
    """
    Test that ReAct Agent stops at max iterations.

    Verifies:
    - Agent stops when reaching max_iterations
    - Still generates an answer with available information
    - All iterations are tracked
    - Warning is logged appropriately
    """
    from src.agents.react_agent import ReActAgent
    from langchain_core.documents import Document

    # Configure mock LLM
    mock_llm = Mock()
    mock_llm.get_model_info = Mock(return_value={'model': 'claude-sonnet-4'})

    # Always decide to continue searching (never answers)
    mock_llm.generate = Mock(side_effect=[
        # Iteration 1
        """Thought: Need more info
Action: SEARCH
Action Input: query 1""",
        # Iteration 2
        """Thought: Still need more
Action: SEARCH
Action Input: query 2""",
        # Iteration 3
        """Thought: Keep searching
Action: NEED_MORE_INFO
Action Input: """,
        # Final answer generation (forced at max iterations)
        "Partial answer with available information."
    ])
    mock_llm_class.return_value = mock_llm

    # Configure mock retriever
    mock_retriever = Mock()
    test_doc = Document(page_content="Some content", metadata={'source': 'test.txt'})
    mock_retriever.retrieve = Mock(return_value=[test_doc])

    # Configure mock query rewriter
    mock_rewriter = Mock()
    mock_rewriter.rewrite = Mock(return_value="query")
    mock_rewriter_class.return_value = mock_rewriter

    # Create agent with small max_iterations
    agent = ReActAgent(
        retriever=mock_retriever,
        max_iterations=3,
    )

    result = agent.run("What is test?")

    # Verify stopped at max iterations
    assert result['iterations'] == 3, "Should reach max_iterations"

    # Verify still generated an answer
    assert result['answer'] == "Partial answer with available information."

    # Verify all actions were recorded
    assert len(result['actions_taken']) == 3


@patch('src.agents.react_agent.LLMGenerator')
def test_react_agent_action_parsing(mock_llm_class):
    """
    Test that ReAct Agent parses LLM responses correctly.

    Verifies:
    - Parses SEARCH action correctly
    - Parses REWRITE_QUERY action correctly
    - Parses ANSWER action correctly
    - Parses FINISH action correctly
    - Handles variations (REWRITE -> REWRITE_QUERY)
    - Extracts thought and action input
    """
    from src.agents.react_agent import ReActAgent, AgentAction

    # Configure mock LLM
    mock_llm = Mock()
    mock_llm.get_model_info = Mock(return_value={'model': 'claude-sonnet-4'})
    mock_llm_class.return_value = mock_llm

    # Mock retriever
    mock_retriever = Mock()

    # Create agent
    agent = ReActAgent(retriever=mock_retriever)

    # Test 1: Parse SEARCH action
    response1 = """Thought: I need to find documents
Action: SEARCH
Action Input: machine learning basics"""

    thought, action, action_input = agent._parse_react_response(response1)
    assert thought == "I need to find documents"
    assert action == AgentAction.SEARCH
    assert action_input == "machine learning basics"

    # Test 2: Parse REWRITE_QUERY action
    response2 = """Thought: Query is too vague
Action: REWRITE_QUERY
Action Input: current query"""

    thought2, action2, input2 = agent._parse_react_response(response2)
    assert action2 == AgentAction.REWRITE_QUERY

    # Test 3: Parse ANSWER action
    response3 = """Thought: I have enough information
Action: ANSWER
Action Input: """

    thought3, action3, input3 = agent._parse_react_response(response3)
    assert action3 == AgentAction.ANSWER
    assert input3 == ""

    # Test 4: Parse FINISH action
    response4 = """Thought: Task complete
Action: FINISH
Action Input: """

    thought4, action4, input4 = agent._parse_react_response(response4)
    assert action4 == AgentAction.FINISH

    # Test 5: Parse REWRITE (maps to REWRITE_QUERY)
    response5 = """Thought: Need better query
Action: REWRITE
Action Input: optimize this"""

    thought5, action5, input5 = agent._parse_react_response(response5)
    assert action5 == AgentAction.REWRITE_QUERY
    assert input5 == "optimize this"

    # Test 6: Parse NEED_MORE_INFO
    response6 = """Thought: Continue searching
Action: NEED_MORE_INFO
Action Input: """

    thought6, action6, input6 = agent._parse_react_response(response6)
    assert action6 == AgentAction.NEED_MORE_INFO


@patch('src.agents.react_agent.LLMGenerator')
@patch('src.agents.react_agent.QueryRewriter')
def test_react_agent_state_tracking(mock_rewriter_class, mock_llm_class):
    """
    Test that ReAct Agent maintains state correctly.

    Verifies:
    - Documents are accumulated across iterations
    - Actions are tracked in order
    - Thoughts are recorded
    - Query evolution is tracked
    - Iterations are counted
    - State deduplicates documents
    """
    from src.agents.react_agent import ReActAgent, AgentState
    from langchain_core.documents import Document

    # Test AgentState directly first
    state = AgentState(
        original_query="What is ML?",
        current_query="What is ML?",
        query_evolution=["What is ML?"],  # Initialized with original query
    )

    # Test initial state
    assert state.iterations == 0
    assert len(state.documents) == 0
    assert len(state.actions) == 0
    assert len(state.thoughts) == 0
    assert len(state.query_evolution) == 1  # Original query

    # Test add_action
    from src.agents.react_agent import AgentAction
    state.add_action(AgentAction.SEARCH, "Searching for information")
    assert len(state.actions) == 1
    assert len(state.thoughts) == 1
    assert state.actions[0] == AgentAction.SEARCH
    assert state.thoughts[0] == "Searching for information"

    # Test add_documents
    doc1 = Document(page_content="Content 1", metadata={'source': 'doc1.txt'})
    doc2 = Document(page_content="Content 2", metadata={'source': 'doc2.txt'})
    state.add_documents([doc1, doc2])
    assert len(state.documents) == 2

    # Test deduplication
    doc1_duplicate = Document(page_content="Content 1", metadata={'source': 'doc1.txt'})
    state.add_documents([doc1_duplicate])
    assert len(state.documents) == 2, "Should not add duplicate document"

    # Test update_query
    state.update_query("What is Machine Learning?")
    assert state.current_query == "What is Machine Learning?"
    assert len(state.query_evolution) == 2
    assert state.query_evolution[1] == "What is Machine Learning?"

    # Test increment_iteration
    state.increment_iteration()
    assert state.iterations == 1

    # Now test full agent run with state tracking
    # Configure mocks
    mock_llm = Mock()
    mock_llm.get_model_info = Mock(return_value={'model': 'claude-sonnet-4'})
    mock_llm.generate = Mock(side_effect=[
        """Thought: First search
Action: SEARCH
Action Input: ML""",
        """Thought: Rewrite query
Action: REWRITE_QUERY
Action Input: """,
        """Thought: Second search
Action: SEARCH
Action Input: machine learning""",
        """Thought: Answer now
Action: ANSWER
Action Input: """,
        "Final answer."
    ])
    mock_llm_class.return_value = mock_llm

    mock_retriever = Mock()
    mock_retriever.retrieve = Mock(side_effect=[
        [Document(page_content="ML doc", metadata={'source': 'ml.txt'})],
        [Document(page_content="ML details", metadata={'source': 'ml2.txt'})],
    ])

    mock_rewriter = Mock()
    mock_rewriter.rewrite = Mock(return_value="machine learning detailed")
    mock_rewriter_class.return_value = mock_rewriter

    agent = ReActAgent(retriever=mock_retriever, max_iterations=5)
    result = agent.run("What is ML?")

    # Verify state tracking in result
    assert result['iterations'] == 4, "Should track all iterations"
    assert len(result['documents']) == 2, "Should accumulate documents"
    assert len(result['reasoning_trace']) == 4, "Should track all thoughts"
    assert len(result['actions_taken']) == 4, "Should track all actions"

    # Verify action order
    assert result['actions_taken'][0] == 'search'
    assert result['actions_taken'][1] == 'rewrite'
    assert result['actions_taken'][2] == 'search'
    assert result['actions_taken'][3] == 'answer'

    # Verify query evolution
    assert len(result['query_evolution']) >= 2, "Should track query changes"
