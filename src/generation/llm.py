"""
LLM interface for generation with Claude.
Supports streaming, chat history, and error handling.
"""
import logging
from typing import List, Optional, Iterator
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from src.config.settings import settings

logger = logging.getLogger(__name__)


class LLMGenerator:
    """
    LLM interface for response generation.
    
    Features:
    - Streaming responses
    - Temperature control
    - Token limits
    - Error handling
    """
    
    def __init__(
        self,
        model: str = None,
        temperature: float = None,
        max_tokens: int = None,
        streaming: bool = False,
    ):
        """
        Initialize LLM generator.
        
        Args:
            model: Model name (default from settings)
            temperature: Generation temperature (default from settings)
            max_tokens: Maximum tokens (default from settings)
            streaming: Enable streaming responses
        """
        self.model = model or settings.llm_model
        self.temperature = temperature if temperature is not None else settings.llm_temperature
        self.max_tokens = max_tokens or settings.llm_max_tokens
        self.streaming = streaming
        
        # Initialize Claude
        callbacks = [StreamingStdOutCallbackHandler()] if streaming else None

        self.llm = ChatAnthropic(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            streaming=streaming,
            callbacks=callbacks,
            api_key=settings.anthropic_api_key,
        )
        
        logger.info(
            f"Initialized LLM: model={self.model}, "
            f"temperature={self.temperature}, "
            f"streaming={streaming}"
        )
    
    def generate(
        self,
        messages: List[BaseMessage],
    ) -> str:
        """
        Generate a response from messages.
        
        Args:
            messages: List of chat messages
            
        Returns:
            Generated response text
        """
        try:
            logger.info(f"Generating response with {len(messages)} messages...")
            
            response = self.llm.invoke(messages)
            
            logger.info("Response generated successfully")
            return response.content
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise
    
    def generate_from_prompt(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Generate response from a simple prompt.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            
        Returns:
            Generated response
        """
        messages = []
        
        if system_prompt:
            messages.append(HumanMessage(content=system_prompt))
        
        messages.append(HumanMessage(content=prompt))
        
        return self.generate(messages)
    
    def stream_generate(
        self,
        messages: List[BaseMessage],
    ) -> Iterator[str]:
        """
        Stream generate responses token by token.
        
        Args:
            messages: List of chat messages
            
        Yields:
            Response tokens
        """
        try:
            logger.info("Starting streaming generation...")
            
            for chunk in self.llm.stream(messages):
                if hasattr(chunk, 'content'):
                    yield chunk.content
                    
        except Exception as e:
            logger.error(f"Error during streaming: {str(e)}")
            raise
    
    def batch_generate(
        self,
        message_batches: List[List[BaseMessage]],
    ) -> List[str]:
        """
        Generate responses for multiple message batches.
        
        Args:
            message_batches: List of message lists
            
        Returns:
            List of generated responses
        """
        try:
            logger.info(f"Batch generating {len(message_batches)} responses...")
            
            responses = self.llm.batch(message_batches)
            
            return [response.content for response in responses]
            
        except Exception as e:
            logger.error(f"Error in batch generation: {str(e)}")
            raise
    
    def get_model_info(self) -> dict:
        """Get information about the current model configuration."""
        return {
            'model': self.model,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'streaming': self.streaming,
        }


class ChatHistoryManager:
    """
    Manages chat history for conversational context.
    (Full implementation in Phase 4 with LangGraph)
    """
    
    def __init__(self, max_history: int = 10):
        """
        Initialize chat history manager.
        
        Args:
            max_history: Maximum messages to keep in history
        """
        self.messages: List[BaseMessage] = []
        self.max_history = max_history
        logger.info(f"Initialized chat history (max={max_history})")
    
    def add_user_message(self, content: str):
        """Add a user message to history."""
        self.messages.append(HumanMessage(content=content))
        self._trim_history()
    
    def add_ai_message(self, content: str):
        """Add an AI message to history."""
        self.messages.append(AIMessage(content=content))
        self._trim_history()
    
    def get_messages(self) -> List[BaseMessage]:
        """Get all messages in history."""
        return self.messages.copy()
    
    def clear(self):
        """Clear message history."""
        self.messages = []
        logger.info("Cleared chat history")
    
    def _trim_history(self):
        """Trim history to max length."""
        if len(self.messages) > self.max_history:
            self.messages = self.messages[-self.max_history:]


# Convenience functions
def create_llm(
    model: str = None,
    streaming: bool = False,
    **kwargs,
) -> LLMGenerator:
    """
    Create an LLM generator instance.
    
    Args:
        model: Optional model name
        streaming: Enable streaming
        **kwargs: Additional LLM configuration
        
    Returns:
        LLMGenerator instance
    """
    return LLMGenerator(
        model=model,
        streaming=streaming,
        **kwargs,
    )
