"""
MCP (Model Context Protocol) Module

Provides infrastructure for connecting to and using MCP servers via
Anthropic's official MCP SDK, enabling the RAG system to access live
data sources (filesystem, databases, APIs, etc.).

Key Components:
- MCPClient: Async client for connecting to MCP servers and calling tools
- get_mcp_client(): Get the global singleton MCPClient
- MCPServerConfig/MCPServerRegistry: Server config factory (servers.py)
- MCPTool/MCPServer: Custom in-process tool infrastructure (support layer)
- Exceptions: Custom exception classes for error handling

Official MCP SDK Usage (Phase 3):
    >>> import asyncio
    >>> from src.mcp import MCPClient

    >>> async def main():
    ...     async with MCPClient() as client:
    ...         # Connect to filesystem server
    ...         await client.connect_to_server(
    ...             server_name="filesystem",
    ...             command="npx",
    ...             args=["-y", "@modelcontextprotocol/server-filesystem", "/data"]
    ...         )
    ...
    ...         # Discover all tools
    ...         tools = await client.list_all_tools()
    ...
    ...         # Call a tool
    ...         result = await client.call_tool(
    ...             server_name="filesystem",
    ...             tool_name="read_file",
    ...             arguments={"path": "/data/doc.txt"}
    ...         )

    >>> asyncio.run(main())
"""

# Official MCP SDK client (Phase 3)
from .client import (
    MCPClient,
    get_mcp_client,
    reset_mcp_client,
)

# MCP server subprocess configuration (Phase 3)
from .servers import (
    MCPServerConfig,
    MCPServerRegistry,
)

# Custom in-process tool infrastructure (support layer)
from .base import (
    MCPTool,
    MCPToolSchema,
    MCPToolParameter,
)
from .server import (
    MCPServer,
    MCPServerRegistry as MCPToolRegistry,  # aliased to avoid name collision
)

# Exceptions
from .exceptions import (
    MCPError,
    MCPToolNotFoundError,
    MCPParameterValidationError,
    MCPExecutionError,
    MCPToolRegistrationError,
    MCPConnectionError,
)

__all__ = [
    # Official MCP SDK client
    "MCPClient",
    "get_mcp_client",
    "reset_mcp_client",
    # MCP server subprocess configuration
    "MCPServerConfig",
    "MCPServerRegistry",
    # Custom tool infrastructure
    "MCPTool",
    "MCPToolSchema",
    "MCPToolParameter",
    "MCPServer",
    "MCPToolRegistry",  # renamed from MCPServerRegistry in server.py
    # Exceptions
    "MCPError",
    "MCPToolNotFoundError",
    "MCPParameterValidationError",
    "MCPExecutionError",
    "MCPToolRegistrationError",
    "MCPConnectionError",
]

__version__ = "1.0.0"
