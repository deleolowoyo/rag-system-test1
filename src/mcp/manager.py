"""
High-level MCP manager for tool orchestration.
Handles async context and provides sync interface.
"""
import asyncio
import logging
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor

from src.mcp.client import MCPClient, get_mcp_client
from src.mcp.servers import MCPServerRegistry
from src.config.settings import settings

logger = logging.getLogger(__name__)


class MCPManager:
    """
    High-level manager for MCP operations.

    Provides:
    - Automatic server initialization
    - Tool discovery
    - Tool execution (sync and async)
    - Connection lifecycle management
    """

    def __init__(self, config=None):
        """Initialize MCP manager."""
        self.config = config or settings
        self.client: Optional[MCPClient] = None
        self.initialized = False
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._executor = ThreadPoolExecutor(max_workers=1)
        logger.info("Initialized MCPManager")

    async def _initialize_async(self):
        """Initialize MCP client and connect to servers (async)."""
        if self.initialized:
            logger.debug("MCP already initialized")
            return

        logger.info("Initializing MCP servers...")

        # Get client
        self.client = await get_mcp_client()

        # Get enabled servers
        servers = MCPServerRegistry.get_enabled_servers(self.config)

        if not servers:
            logger.warning("No MCP servers enabled")
            return

        # Connect to each server
        for server_config in servers:
            try:
                await self.client.connect_to_server(
                    server_name=server_config.name,
                    command=server_config.command,
                    args=server_config.args,
                    env=server_config.env
                )
                logger.info(f"Connected to MCP server: {server_config.name}")
            except Exception as e:
                logger.error(
                    f"Failed to connect to {server_config.name}: {e}",
                    exc_info=True
                )

        self.initialized = True
        logger.info("MCP initialization complete")

    def initialize(self):
        """Initialize MCP (sync wrapper)."""
        if self.initialized:
            return

        # Run async initialization
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._initialize_async())
        finally:
            self._loop = loop

    async def _get_available_tools_async(self) -> Dict[str, List[Dict]]:
        """Get all available tools from all servers (async)."""
        if not self.initialized:
            await self._initialize_async()

        if not self.client:
            return {}

        return await self.client.list_all_tools()

    def get_available_tools(self) -> Dict[str, List[Dict]]:
        """Get all available tools (sync wrapper)."""
        if not self._loop:
            self.initialize()

        return self._loop.run_until_complete(
            self._get_available_tools_async()
        )

    async def _call_tool_async(
        self,
        server_name: str,
        tool_name: str,
        arguments: Dict[str, Any]
    ) -> Any:
        """Call a tool (async)."""
        if not self.initialized:
            await self._initialize_async()

        if not self.client:
            raise RuntimeError("MCP client not initialized")

        result = await self.client.call_tool(
            server_name=server_name,
            tool_name=tool_name,
            arguments=arguments
        )

        return result

    def call_tool(
        self,
        server_name: str,
        tool_name: str,
        arguments: Dict[str, Any]
    ) -> Any:
        """
        Call a tool (sync wrapper).

        Args:
            server_name: MCP server name
            tool_name: Tool name
            arguments: Tool arguments

        Returns:
            Tool execution result
        """
        if not self._loop:
            self.initialize()

        return self._loop.run_until_complete(
            self._call_tool_async(server_name, tool_name, arguments)
        )

    def find_tool(self, tool_name: str) -> Optional[tuple]:
        """
        Find a tool by name across all servers.

        Returns:
            Tuple of (server_name, tool_schema) or None
        """
        all_tools = self.get_available_tools()

        for server_name, tools in all_tools.items():
            for tool in tools:
                if tool["name"] == tool_name:
                    return (server_name, tool)

        return None

    async def _shutdown_async(self):
        """Shutdown MCP client (async)."""
        if self.client:
            await self.client.close()
            self.client = None
        self.initialized = False

    def shutdown(self):
        """Shutdown MCP connections (sync wrapper)."""
        if self._loop and self.initialized:
            self._loop.run_until_complete(self._shutdown_async())

        if self._loop:
            self._loop.close()
            self._loop = None

    def __enter__(self):
        """Context manager entry."""
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown()


# Global manager instance
_manager: Optional[MCPManager] = None


def get_mcp_manager() -> MCPManager:
    """Get or create global MCP manager."""
    global _manager
    if _manager is None:
        _manager = MCPManager()
    return _manager
