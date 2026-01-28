"""
Tests for LLM generation functionality.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from langchain_core.messages import HumanMessage, AIMessage

from src.generation.llm import LLMGenerator, ChatHistoryManager, create_llm
from src.generation.prompts import format_documents_for_context, RAG_SYSTEM_PROMPT


@pytest.fixture
def mock_llm():
    """Create mock LLM."""
    mock = Mock()
    mock.invoke = Mock(return_value=Mock(content="This is a test response."))
    mock.stream = Mock(return_value=[
        Mock(content="This "),
        Mock(content="is "),
        Mock(content="streaming."),
    ])
    mock.batch = Mock(return_value=[
        Mock(content="Response 1"),
        Mock(content="Response 2"),
    ])
    return mock


class TestLLMGenerator:
    """Test LLMGenerator functionality."""

    @patch('src.generation.llm.ChatAnthropic')
    def test_initialization(self, mock_chat_anthropic):
        """Test LLM generator initializes correctly."""
        mock_chat_anthropic.return_value = Mock()

        generator = LLMGenerator(
            model="claude-sonnet-4-20250514",
            temperature=0.5,
            max_tokens=1024,
            streaming=False
        )

        assert generator.model == "claude-sonnet-4-20250514"
        assert generator.temperature == 0.5
        assert generator.max_tokens == 1024
        assert generator.streaming == False

    @patch('src.generation.llm.ChatAnthropic')
    def test_initialization_with_defaults(self, mock_chat_anthropic):
        """Test initialization with default values from settings."""
        mock_chat_anthropic.return_value = Mock()

        generator = LLMGenerator()

        assert generator.model is not None
        assert generator.temperature >= 0
        assert generator.max_tokens > 0

    @patch('src.generation.llm.ChatAnthropic')
    def test_generate(self, mock_chat_anthropic):
        """Test generating a response."""
        mock_llm = Mock()
        mock_llm.invoke = Mock(return_value=Mock(content="Test response"))
        mock_chat_anthropic.return_value = mock_llm

        generator = LLMGenerator()
        messages = [HumanMessage(content="Test prompt")]

        response = generator.generate(messages)

        assert response == "Test response"
        mock_llm.invoke.assert_called_once_with(messages)

    @patch('src.generation.llm.ChatAnthropic')
    def test_generate_from_prompt(self, mock_chat_anthropic):
        """Test generating from simple prompt."""
        mock_llm = Mock()
        mock_llm.invoke = Mock(return_value=Mock(content="Simple response"))
        mock_chat_anthropic.return_value = mock_llm

        generator = LLMGenerator()

        response = generator.generate_from_prompt("Test prompt")

        assert response == "Simple response"
        assert mock_llm.invoke.called

    @patch('src.generation.llm.ChatAnthropic')
    def test_generate_from_prompt_with_system(self, mock_chat_anthropic):
        """Test generating with system prompt."""
        mock_llm = Mock()
        mock_llm.invoke = Mock(return_value=Mock(content="System response"))
        mock_chat_anthropic.return_value = mock_llm

        generator = LLMGenerator()

        response = generator.generate_from_prompt(
            "User prompt",
            system_prompt="System instructions"
        )

        assert response == "System response"
        assert mock_llm.invoke.called

    @patch('src.generation.llm.ChatAnthropic')
    def test_stream_generate(self, mock_chat_anthropic):
        """Test streaming generation."""
        mock_llm = Mock()
        mock_llm.stream = Mock(return_value=[
            Mock(content="Chunk 1 "),
            Mock(content="Chunk 2 "),
            Mock(content="Chunk 3"),
        ])
        mock_chat_anthropic.return_value = mock_llm

        generator = LLMGenerator(streaming=True)
        messages = [HumanMessage(content="Stream test")]

        chunks = list(generator.stream_generate(messages))

        assert len(chunks) == 3
        assert chunks[0] == "Chunk 1 "
        assert chunks[1] == "Chunk 2 "
        assert chunks[2] == "Chunk 3"

    @patch('src.generation.llm.ChatAnthropic')
    def test_batch_generate(self, mock_chat_anthropic):
        """Test batch generation."""
        mock_llm = Mock()
        mock_llm.batch = Mock(return_value=[
            Mock(content="Response 1"),
            Mock(content="Response 2"),
            Mock(content="Response 3"),
        ])
        mock_chat_anthropic.return_value = mock_llm

        generator = LLMGenerator()
        message_batches = [
            [HumanMessage(content="Prompt 1")],
            [HumanMessage(content="Prompt 2")],
            [HumanMessage(content="Prompt 3")],
        ]

        responses = generator.batch_generate(message_batches)

        assert len(responses) == 3
        assert responses[0] == "Response 1"
        assert responses[1] == "Response 2"
        assert responses[2] == "Response 3"

    @patch('src.generation.llm.ChatAnthropic')
    def test_get_model_info(self, mock_chat_anthropic):
        """Test getting model information."""
        mock_chat_anthropic.return_value = Mock()

        generator = LLMGenerator(
            model="claude-sonnet-4-20250514",
            temperature=0.7,
            max_tokens=2048,
            streaming=True
        )

        info = generator.get_model_info()

        assert info['model'] == "claude-sonnet-4-20250514"
        assert info['temperature'] == 0.7
        assert info['max_tokens'] == 2048
        assert info['streaming'] == True


class TestChatHistoryManager:
    """Test ChatHistoryManager functionality."""

    def test_initialization(self):
        """Test chat history manager initializes."""
        manager = ChatHistoryManager(max_history=10)

        assert len(manager.messages) == 0
        assert manager.max_history == 10

    def test_add_user_message(self):
        """Test adding user message."""
        manager = ChatHistoryManager()

        manager.add_user_message("Hello!")

        assert len(manager.messages) == 1
        assert isinstance(manager.messages[0], HumanMessage)
        assert manager.messages[0].content == "Hello!"

    def test_add_ai_message(self):
        """Test adding AI message."""
        manager = ChatHistoryManager()

        manager.add_ai_message("Hi there!")

        assert len(manager.messages) == 1
        assert isinstance(manager.messages[0], AIMessage)
        assert manager.messages[0].content == "Hi there!"

    def test_conversation_flow(self):
        """Test back-and-forth conversation."""
        manager = ChatHistoryManager()

        manager.add_user_message("Question 1")
        manager.add_ai_message("Answer 1")
        manager.add_user_message("Question 2")
        manager.add_ai_message("Answer 2")

        assert len(manager.messages) == 4
        assert isinstance(manager.messages[0], HumanMessage)
        assert isinstance(manager.messages[1], AIMessage)
        assert isinstance(manager.messages[2], HumanMessage)
        assert isinstance(manager.messages[3], AIMessage)

    def test_get_messages(self):
        """Test getting message history."""
        manager = ChatHistoryManager()

        manager.add_user_message("Test")
        manager.add_ai_message("Response")

        messages = manager.get_messages()

        assert len(messages) == 2
        assert messages is not manager.messages  # Should be a copy

    def test_clear_history(self):
        """Test clearing message history."""
        manager = ChatHistoryManager()

        manager.add_user_message("Test 1")
        manager.add_ai_message("Response 1")
        assert len(manager.messages) == 2

        manager.clear()

        assert len(manager.messages) == 0

    def test_trim_history(self):
        """Test trimming history to max length."""
        manager = ChatHistoryManager(max_history=3)

        # Add more messages than max_history
        manager.add_user_message("Message 1")
        manager.add_ai_message("Response 1")
        manager.add_user_message("Message 2")
        manager.add_ai_message("Response 2")
        manager.add_user_message("Message 3")

        # Should keep only the last 3 messages
        assert len(manager.messages) == 3
        assert manager.messages[0].content == "Message 2"
        assert manager.messages[1].content == "Response 2"
        assert manager.messages[2].content == "Message 3"


class TestPrompts:
    """Test prompt formatting functions."""

    def test_format_documents_for_context(self):
        """Test formatting documents for context."""
        from langchain_core.documents import Document

        documents = [
            Document(
                page_content="Content 1",
                metadata={'source': 'doc1.txt', 'page': 1}
            ),
            Document(
                page_content="Content 2",
                metadata={'source': 'doc2.txt', 'page': 2}
            ),
        ]

        formatted = format_documents_for_context(documents)

        assert isinstance(formatted, str)
        assert "Content 1" in formatted
        assert "Content 2" in formatted
        assert "doc1.txt" in formatted or "Source" in formatted

    def test_rag_system_prompt(self):
        """Test RAG system prompt is defined."""
        assert RAG_SYSTEM_PROMPT is not None
        assert isinstance(RAG_SYSTEM_PROMPT, str)
        assert len(RAG_SYSTEM_PROMPT) > 0

    def test_rag_system_prompt_content(self):
        """Test RAG system prompt contains key instructions."""
        assert "context" in RAG_SYSTEM_PROMPT.lower()
        assert "source" in RAG_SYSTEM_PROMPT.lower() or "cite" in RAG_SYSTEM_PROMPT.lower()


class TestCreateLLM:
    """Test create_llm convenience function."""

    @patch('src.generation.llm.ChatAnthropic')
    def test_create_llm(self, mock_chat_anthropic):
        """Test creating LLM generator."""
        mock_chat_anthropic.return_value = Mock()

        llm = create_llm(
            model="claude-sonnet-4-20250514",
            streaming=True
        )

        assert isinstance(llm, LLMGenerator)
        assert llm.model == "claude-sonnet-4-20250514"
        assert llm.streaming == True

    @patch('src.generation.llm.ChatAnthropic')
    def test_create_llm_with_defaults(self, mock_chat_anthropic):
        """Test creating LLM with defaults."""
        mock_chat_anthropic.return_value = Mock()

        llm = create_llm()

        assert isinstance(llm, LLMGenerator)
