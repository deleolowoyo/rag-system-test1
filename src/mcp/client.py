"""
MCP Client for connecting to and managing MCP servers.
Uses Anthropic's official MCP SDK.
"""
import asyncio
import logging
from typing import Dict, List, Optional, Any
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

logger = logging.getLogger(__name__)


class MCPClient:
    """
    Client for managing multiple MCP server connections.

    Supports:
    - Multiple server connections (filesystem, SQLite, Google Drive, etc.)
    - Tool discovery via MCP protocol
    - Tool execution via MCP protocol
    - Server lifecycle management with AsyncExitStack

    All methods that interact with servers are async since MCP SDK is async.

    Example:
        >>> import asyncio
        >>> from src.mcp.client import MCPClient
        >>>
        >>> async def main():
        ...     client = MCPClient()
        ...
        ...     # Connect to filesystem server
        ...     await client.connect_to_server(
        ...         server_name="filesystem",
        ...         command="npx",
        ...         args=["-y", "@modelcontextprotocol/server-filesystem", "/path/to/files"]
        ...     )
        ...
        ...     # Discover tools
        ...     tools = await client.list_all_tools()
        ...
        ...     # Use a tool
        ...     result = await client.call_tool(
        ...         server_name="filesystem",
        ...         tool_name="read_file",
        ...         arguments={"path": "/path/to/files/doc.txt"}
        ...     )
        ...
        ...     await client.close()
        >>>
        >>> asyncio.run(main())
    """

    def __init__(self):
        """Initialize MCP client."""
        self.sessions: Dict[str, ClientSession] = {}
        self.exit_stack = AsyncExitStack()
        self.initialized = False
        logger.info("Initialized MCPClient")

    async def connect_to_server(
        self,
        server_name: str,
        command: str,
        args: List[str],
        env: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Connect to an MCP server via stdio.

        The server runs as a subprocess. Communication happens through
        the subprocess's stdin/stdout using the MCP protocol.

        Args:
            server_name: Unique name for this server connection
            command: Executable to start (e.g., "npx", "python", "uvx")
            args: Arguments for the command
                - For filesystem: ["-y", "@modelcontextprotocol/server-filesystem", "/root/path"]
                - For sqlite: ["-y", "@modelcontextprotocol/server-sqlite", "--db-path", "/path/to/db"]
            env: Optional environment variables for the server subprocess

        Raises:
            RuntimeError: If connection or initialization fails
        """
        if server_name in self.sessions:
            logger.warning(f"Server '{server_name}' already connected. Skipping.")
            return

        logger.info(f"Connecting to MCP server: {server_name} ({command} {' '.join(args)})")

        server_params = StdioServerParameters(
            command=command,
            args=args,
            env=env
        )

        try:
            stdio_transport = await self.exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            stdio, write = stdio_transport

            session = await self.exit_stack.enter_async_context(
                ClientSession(stdio, write)
            )

            await session.initialize()

            self.sessions[server_name] = session

            # Discover tools - list_tools() is async and returns ListToolsResult
            tools_result = await session.list_tools()
            tool_count = len(tools_result.tools)

            logger.info(f"Connected to '{server_name}': {tool_count} tools available")
            if tool_count > 0:
                tool_names = [t.name for t in tools_result.tools]
                logger.debug(f"Tools on '{server_name}': {tool_names}")

        except Exception as e:
            logger.error(f"Failed to connect to server '{server_name}': {e}")
            raise RuntimeError(
                f"Failed to connect to MCP server '{server_name}': {str(e)}"
            ) from e

    async def list_all_tools(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        List all available tools from all connected servers.

        Returns:
            Dict mapping server_name -> list of tool schema dicts.
            Each tool dict has: name, description, input_schema

        Example:
            {
                "filesystem": [
                    {
                        "name": "read_file",
                        "description": "Read the contents of a file",
                        "input_schema": {"type": "object", "properties": {"path": ...}}
                    },
                    ...
                ],
                "sqlite": [...]
            }
        """
        all_tools: Dict[str, List[Dict[str, Any]]] = {}

        for server_name, session in self.sessions.items():
            try:
                # list_tools() is async and returns ListToolsResult with .tools attribute
                tools_result = await session.list_tools()

                all_tools[server_name] = [
                    {
                        "name": tool.name,
                        "description": tool.description or "",
                        "input_schema": tool.inputSchema,
                    }
                    for tool in tools_result.tools
                ]

                logger.debug(
                    f"Server '{server_name}': {len(all_tools[server_name])} tools"
                )

            except Exception as e:
                logger.error(f"Failed to list tools from '{server_name}': {e}")
                all_tools[server_name] = []

        return all_tools

    async def list_tools_flat(self) -> List[Dict[str, Any]]:
        """
        Get a flat list of all tools across all servers.

        Adds a 'server' key to each tool for identification.

        Returns:
            Flat list of tool schema dicts, each including 'server' field

        Example:
            [
                {"server": "filesystem", "name": "read_file", ...},
                {"server": "filesystem", "name": "write_file", ...},
                {"server": "sqlite", "name": "query", ...},
            ]
        """
        all_tools_by_server = await self.list_all_tools()
        flat: List[Dict[str, Any]] = []

        for server_name, tools in all_tools_by_server.items():
            for tool in tools:
                flat.append({**tool, "server": server_name})

        return flat

    async def call_tool(
        self,
        server_name: str,
        tool_name: str,
        arguments: Dict[str, Any]
    ) -> Any:
        """
        Call a tool on a specific MCP server.

        Args:
            server_name: Name of the connected server
            tool_name: Name of the tool to call
            arguments: Arguments dict matching the tool's input_schema

        Returns:
            CallToolResult from the MCP SDK. Access results via result.content,
            which is a list of content blocks (TextContent, ImageContent, etc.)

        Raises:
            ValueError: If the server is not connected
            RuntimeError: If the tool call fails

        Example:
            result = await client.call_tool(
                server_name="filesystem",
                tool_name="read_file",
                arguments={"path": "/data/doc.txt"}
            )
            # result.content is a list of content blocks
            for block in result.content:
                print(block.text)  # for TextContent blocks
        """
        if server_name not in self.sessions:
            available = list(self.sessions.keys())
            raise ValueError(
                f"Server '{server_name}' not connected. "
                f"Available servers: {available}"
            )

        session = self.sessions[server_name]

        logger.info(f"Calling tool '{tool_name}' on server '{server_name}'")
        logger.debug(f"Arguments: {arguments}")

        try:
            # call_tool() is async and returns CallToolResult
            result = await session.call_tool(tool_name, arguments)

            logger.info(f"Tool '{tool_name}' completed successfully")
            return result

        except Exception as e:
            logger.error(
                f"Tool '{tool_name}' on server '{server_name}' failed: {e}"
            )
            raise RuntimeError(
                f"MCP tool call failed: '{tool_name}' on '{server_name}': {str(e)}"
            ) from e

    async def call_tool_text(
        self,
        server_name: str,
        tool_name: str,
        arguments: Dict[str, Any]
    ) -> str:
        """
        Call a tool and return its text content as a single string.

        Convenience wrapper around call_tool() that extracts text from
        all TextContent blocks in the result.

        Args:
            server_name: Name of the connected server
            tool_name: Name of the tool to call
            arguments: Tool arguments

        Returns:
            Concatenated text from all TextContent blocks in result
        """
        result = await self.call_tool(server_name, tool_name, arguments)

        text_parts = []
        for block in result.content:
            if hasattr(block, 'text'):
                text_parts.append(block.text)

        return "\n".join(text_parts)

    def is_connected(self, server_name: str) -> bool:
        """Check if a server is connected."""
        return server_name in self.sessions

    def connected_servers(self) -> List[str]:
        """Get list of connected server names."""
        return list(self.sessions.keys())

    async def close(self) -> None:
        """
        Close all server connections and clean up resources.

        Should be called when done with the client to properly
        terminate server subprocesses.
        """
        server_count = len(self.sessions)
        logger.info(f"Closing {server_count} MCP server connection(s)")

        await self.exit_stack.aclose()

        self.sessions.clear()
        self.initialized = False

        logger.info("All MCP server connections closed")

    async def __aenter__(self):
        """Support async context manager usage."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Close connections on context manager exit."""
        await self.close()

    def __repr__(self) -> str:
        """String representation."""
        servers = list(self.sessions.keys())
        return f"MCPClient(connected_servers={servers})"


# Module-level singleton
_mcp_client: Optional[MCPClient] = None


async def get_mcp_client() -> MCPClient:
    """
    Get or create the global MCPClient singleton.

    Returns:
        The global MCPClient instance

    Note:
        The returned client has no servers connected yet.
        Call connect_to_server() to add connections.
    """
    global _mcp_client
    if _mcp_client is None:
        _mcp_client = MCPClient()
    return _mcp_client


async def reset_mcp_client() -> None:
    """
    Close and reset the global MCPClient singleton.

    Useful for testing or reconfiguring the client.
    """
    global _mcp_client
    if _mcp_client is not None:
        await _mcp_client.close()
        _mcp_client = None
