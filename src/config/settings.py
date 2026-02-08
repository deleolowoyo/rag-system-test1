"""
Configuration management for RAG system.
Uses Pydantic for type safety and validation.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from typing import Literal


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    # API Keys
    openai_api_key: str = Field(..., validation_alias="OPENAI_API_KEY")
    anthropic_api_key: str = Field(..., validation_alias="ANTHROPIC_API_KEY")
    
    # Embedding Configuration
    embedding_model: str = Field(
        default="text-embedding-3-small",
        description="OpenAI embedding model to use"
    )
    embedding_dimensions: int = Field(
        default=1536,
        description="Dimension size for embeddings"
    )
    
    # Text Splitting Configuration
    chunk_size: int = Field(
        default=1000,
        description="Maximum tokens per chunk"
    )
    chunk_overlap: int = Field(
        default=200,
        description="Overlap between chunks to preserve context"
    )
    
    # Vector Store Configuration
    vector_db_path: str = Field(
        default="./data/chromadb",
        description="Path to ChromaDB persistence directory"
    )
    collection_name: str = Field(
        default="documents",
        description="Name of the vector store collection"
    )
    
    # Retrieval Configuration
    retrieval_top_k: int = Field(
        default=4,
        description="Number of chunks to retrieve"
    )
    retrieval_score_threshold: float = Field(
        default=0.7,
        description="Minimum similarity score for retrieval"
    )
    search_type: Literal["similarity", "mmr"] = Field(
        default="similarity",
        description="Type of search algorithm"
    )
    
    # LLM Configuration
    llm_model: str = Field(
        default="claude-sonnet-4-20250514",
        description="LLM model for generation"
    )
    llm_temperature: float = Field(
        default=0.0,
        description="Temperature for LLM generation (0 = deterministic)"
    )
    llm_max_tokens: int = Field(
        default=2048,
        description="Maximum tokens in LLM response"
    )

    # Phase 2 Feature Flags
    enable_query_rewriting: bool = Field(
        default=True,
        description="Enable query optimization/rewriting"
    )
    enable_multi_query: bool = Field(
        default=True,
        description="Enable multi-query generation for improved recall"
    )
    enable_reranking: bool = Field(
        default=True,
        description="Enable document re-ranking"
    )
    enable_react_agent: bool = Field(
        default=False,
        description="Enable ReAct agent for multi-step reasoning (expensive)"
    )
    enable_self_critique: bool = Field(
        default=True,
        description="Enable self-critique validation of answers"
    )

    # Phase 2 Configuration
    react_max_iterations: int = Field(
        default=5,
        description="Maximum iterations for ReAct agent"
    )
    multi_query_count: int = Field(
        default=3,
        description="Number of query variations to generate"
    )
    rerank_top_k: int = Field(
        default=5,
        description="Number of documents to keep after re-ranking"
    )

    # System Configuration
    log_level: str = Field(
        default="INFO",
        description="Logging level"
    )
    enable_tracing: bool = Field(
        default=False,
        description="Enable LangSmith tracing"
    )


# Global settings instance
settings = Settings()
