"""
Phase 3 Pipeline with MCP integration.
Combines RAG knowledge with live MCP data.
"""
import logging
from typing import Dict, Any, Optional

from src.pipeline_v2 import EnhancedRAGPipeline
from src.mcp.manager import MCPManager, get_mcp_manager
from src.agents.mcp_agent import MCPAgent
from src.generation.prompts import format_documents_for_context
from src.config.settings import settings

logger = logging.getLogger(__name__)


class MCPEnabledPipeline(EnhancedRAGPipeline):
    """
    Pipeline with MCP (Model Context Protocol) integration.

    Capabilities:
    - Phase 1: Core RAG
    - Phase 2: Enhanced RAG with agents
    - Phase 3: MCP tool integration for live data
    """

    def __init__(
        self,
        collection_name: str = None,
        persist_directory: str = None,
        enable_mcp: bool = True,
        **kwargs
    ):
        """
        Initialize MCP-enabled pipeline.

        Args:
            collection_name: Vector store collection name
            persist_directory: Vector store persistence directory
            enable_mcp: Enable MCP tool integration
            **kwargs: Additional arguments for parent classes
        """
        # Initialize Phase 1 & 2
        super().__init__(
            collection_name=collection_name,
            persist_directory=persist_directory,
            **kwargs
        )

        # Initialize MCP components
        self.enable_mcp = enable_mcp and settings.enable_mcp

        if self.enable_mcp:
            try:
                self.mcp_manager = get_mcp_manager()
                self.mcp_manager.initialize()

                self.mcp_agent = MCPAgent(
                    mcp_manager=self.mcp_manager,
                    llm=self.llm
                )

                logger.info("MCP integration enabled")
                logger.info(f"Available MCP servers: {settings.mcp_servers_enabled}")
            except Exception as e:
                logger.error(f"Failed to initialize MCP: {e}", exc_info=True)
                self.enable_mcp = False
                self.mcp_manager = None
                self.mcp_agent = None
        else:
            self.mcp_manager = None
            self.mcp_agent = None
            logger.info("MCP integration disabled")

    def query_v3(
        self,
        question: str,
        use_mcp: bool = True,
        use_rag: bool = True,
        return_sources: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Query with MCP integration (Phase 3).

        Args:
            question: User question
            use_mcp: Use MCP tools for live data
            use_rag: Use RAG for static knowledge
            return_sources: Include source information
            **kwargs: Additional arguments

        Returns:
            Enhanced result with MCP data and RAG context
        """
        logger.info(f"Phase 3 query: {question}")

        # Check if MCP is available
        if use_mcp and not self.enable_mcp:
            logger.warning("MCP requested but not available, using RAG only")
            use_mcp = False

        # Get RAG context if needed
        rag_context = None
        rag_documents = []

        if use_rag:
            # Use Phase 2 retrieval
            rag_documents = self.retriever.retrieve(
                query=question,
                return_scores=True
            )

            if rag_documents:
                docs_only = [doc for doc, _ in rag_documents]
                rag_context = format_documents_for_context(docs_only)
                logger.info(f"Retrieved {len(docs_only)} RAG documents")

        # Use MCP if enabled
        if use_mcp and self.mcp_agent:
            result = self.mcp_agent.run(
                query=question,
                rag_context=rag_context
            )

            # Enhance with RAG sources if available
            if return_sources and rag_documents:
                result["rag_sources"] = [
                    {
                        "content": doc.page_content[:200],
                        "metadata": doc.metadata,
                        "score": score
                    }
                    for doc, score in rag_documents
                ]

            result["phase"] = "phase3_mcp"
            return result

        # Fallback to Phase 2
        logger.info("Using Phase 2 pipeline (no MCP)")
        result = self.query_v2(
            question=question,
            return_sources=return_sources,
            **kwargs
        )
        result["phase"] = "phase2_fallback"
        return result

    def get_mcp_tools(self) -> Dict[str, Any]:
        """Get available MCP tools."""
        if not self.enable_mcp or not self.mcp_manager:
            return {}

        return self.mcp_manager.get_available_tools()

    def shutdown(self):
        """Clean shutdown of MCP connections."""
        if self.mcp_manager:
            logger.info("Shutting down MCP connections")
            self.mcp_manager.shutdown()


# Convenience function
def create_mcp_pipeline(**kwargs) -> MCPEnabledPipeline:
    """Create MCP-enabled pipeline with default settings."""
    return MCPEnabledPipeline(**kwargs)
