"""
MCP Server

Server class for managing and executing MCP tools.
"""
import logging
from typing import Dict, Any, List, Optional

from .base import MCPTool
from .exceptions import (
    MCPToolNotFoundError,
    MCPToolRegistrationError,
    MCPParameterValidationError,
    MCPExecutionError,
)

logger = logging.getLogger(__name__)


class MCPServer:
    """
    MCP Server for managing and executing multiple tools.

    The server maintains a registry of available tools and provides
    methods for tool discovery, validation, and execution.
    """

    def __init__(self, name: str = "default"):
        """
        Initialize MCP server.

        Args:
            name: Server name/identifier
        """
        self.name = name
        self._tools: Dict[str, MCPTool] = {}
        logger.info(f"MCP Server '{name}' initialized")

    def register_tool(self, tool: MCPTool, override: bool = False) -> None:
        """
        Register a tool with the server.

        Args:
            tool: MCPTool instance to register
            override: If True, allow overriding existing tool with same name

        Raises:
            MCPToolRegistrationError: If tool registration fails
        """
        if not isinstance(tool, MCPTool):
            raise MCPToolRegistrationError(
                tool_name=str(tool),
                reason=f"Tool must be an instance of MCPTool, got {type(tool).__name__}"
            )

        tool_name = tool.name

        if tool_name in self._tools and not override:
            raise MCPToolRegistrationError(
                tool_name=tool_name,
                reason=f"Tool '{tool_name}' already registered. Use override=True to replace."
            )

        self._tools[tool_name] = tool
        logger.info(
            f"Registered tool '{tool_name}' (category: {tool.category}) "
            f"with server '{self.name}'"
        )

    def unregister_tool(self, tool_name: str) -> bool:
        """
        Unregister a tool from the server.

        Args:
            tool_name: Name of tool to unregister

        Returns:
            True if tool was unregistered, False if tool didn't exist
        """
        if tool_name in self._tools:
            del self._tools[tool_name]
            logger.info(f"Unregistered tool '{tool_name}' from server '{self.name}'")
            return True

        logger.warning(f"Attempted to unregister non-existent tool '{tool_name}'")
        return False

    def get_tool(self, tool_name: str) -> MCPTool:
        """
        Get a registered tool by name.

        Args:
            tool_name: Name of the tool

        Returns:
            MCPTool instance

        Raises:
            MCPToolNotFoundError: If tool is not registered
        """
        if tool_name not in self._tools:
            raise MCPToolNotFoundError(
                tool_name=tool_name,
                available_tools=list(self._tools.keys())
            )

        return self._tools[tool_name]

    def has_tool(self, tool_name: str) -> bool:
        """
        Check if a tool is registered.

        Args:
            tool_name: Name of the tool

        Returns:
            True if tool is registered
        """
        return tool_name in self._tools

    def list_tools(self) -> List[Dict[str, Any]]:
        """
        Get list of all registered tools with their schemas.

        Returns:
            List of tool schema dictionaries

        Example:
            [
                {
                    "name": "read_file",
                    "description": "Read contents of a file",
                    "category": "filesystem",
                    "parameters": {...}
                },
                ...
            ]
        """
        tools = []

        for tool_name, tool in self._tools.items():
            tools.append(tool.get_schema_dict())

        logger.debug(f"Listed {len(tools)} tools from server '{self.name}'")
        return tools

    def list_tool_names(self) -> List[str]:
        """
        Get list of registered tool names.

        Returns:
            List of tool names
        """
        return list(self._tools.keys())

    def get_tools_by_category(self, category: str) -> List[Dict[str, Any]]:
        """
        Get tools filtered by category.

        Args:
            category: Tool category to filter by

        Returns:
            List of tool schemas matching the category
        """
        return [
            tool.get_schema_dict()
            for tool in self._tools.values()
            if tool.category == category
        ]

    def execute_tool(
        self,
        tool_name: str,
        parameters: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute a tool by name with given parameters.

        Args:
            tool_name: Name of the tool to execute
            parameters: Dictionary of parameters (alternative to **kwargs)
            **kwargs: Tool parameters (merged with parameters dict)

        Returns:
            Dictionary with execution results

        Raises:
            MCPToolNotFoundError: If tool is not registered
            MCPParameterValidationError: If parameters are invalid
            MCPExecutionError: If execution fails

        Example:
            result = server.execute_tool(
                "read_file",
                file_path="/path/to/file.txt"
            )
        """
        logger.info(f"Executing tool '{tool_name}' on server '{self.name}'")

        # Get the tool
        tool = self.get_tool(tool_name)

        # Merge parameters
        params = parameters.copy() if parameters else {}
        params.update(kwargs)

        # Execute tool
        try:
            result = tool.execute(**params)
            logger.info(f"Tool '{tool_name}' execution completed")
            return result

        except (MCPParameterValidationError, MCPExecutionError) as e:
            logger.error(f"Tool '{tool_name}' execution failed: {e}")
            raise

        except Exception as e:
            logger.error(f"Unexpected error executing tool '{tool_name}': {e}")
            raise MCPExecutionError(
                tool_name=tool_name,
                error_message=f"Unexpected error: {str(e)}",
                original_exception=e,
                parameters=params
            )

    def get_tool_schema(self, tool_name: str) -> Dict[str, Any]:
        """
        Get schema for a specific tool.

        Args:
            tool_name: Name of the tool

        Returns:
            Tool schema dictionary

        Raises:
            MCPToolNotFoundError: If tool is not registered
        """
        tool = self.get_tool(tool_name)
        return tool.get_schema_dict()

    def get_server_info(self) -> Dict[str, Any]:
        """
        Get server information and statistics.

        Returns:
            Dictionary with server info:
            - name: Server name
            - tool_count: Number of registered tools
            - tools: List of tool names
            - categories: List of unique categories
        """
        categories = set(tool.category for tool in self._tools.values())

        return {
            "name": self.name,
            "tool_count": len(self._tools),
            "tools": list(self._tools.keys()),
            "categories": sorted(list(categories)),
        }

    def clear(self) -> None:
        """
        Unregister all tools.

        Useful for testing or reinitializing the server.
        """
        tool_count = len(self._tools)
        self._tools.clear()
        logger.info(f"Cleared {tool_count} tools from server '{self.name}'")

    def validate_tool_parameters(
        self,
        tool_name: str,
        parameters: Dict[str, Any]
    ) -> bool:
        """
        Validate parameters for a tool without executing it.

        Args:
            tool_name: Name of the tool
            parameters: Parameters to validate

        Returns:
            True if validation succeeds

        Raises:
            MCPToolNotFoundError: If tool is not registered
            MCPParameterValidationError: If validation fails
        """
        tool = self.get_tool(tool_name)
        return tool.validate_parameters(**parameters)

    def __len__(self) -> int:
        """Return number of registered tools."""
        return len(self._tools)

    def __contains__(self, tool_name: str) -> bool:
        """Check if tool is registered."""
        return tool_name in self._tools

    def __str__(self) -> str:
        """String representation of server."""
        return f"MCPServer(name='{self.name}', tools={len(self._tools)})"

    def __repr__(self) -> str:
        """Detailed string representation."""
        tools_str = ", ".join(self._tools.keys()) if self._tools else "none"
        return f"MCPServer(name='{self.name}', tools=[{tools_str}])"


class MCPServerRegistry:
    """
    Registry for managing multiple MCP servers.

    Allows managing multiple tool servers (e.g., filesystem, database, web)
    from a central location.
    """

    def __init__(self):
        """Initialize server registry."""
        self._servers: Dict[str, MCPServer] = {}
        logger.info("MCP Server Registry initialized")

    def register_server(self, server: MCPServer, override: bool = False) -> None:
        """
        Register a server with the registry.

        Args:
            server: MCPServer instance to register
            override: If True, allow overriding existing server with same name

        Raises:
            ValueError: If server already exists and override=False
        """
        if not isinstance(server, MCPServer):
            raise TypeError(f"Expected MCPServer, got {type(server).__name__}")

        server_name = server.name

        if server_name in self._servers and not override:
            raise ValueError(
                f"Server '{server_name}' already registered. Use override=True to replace."
            )

        self._servers[server_name] = server
        logger.info(f"Registered server '{server_name}' with registry")

    def get_server(self, server_name: str) -> MCPServer:
        """
        Get a registered server by name.

        Args:
            server_name: Name of the server

        Returns:
            MCPServer instance

        Raises:
            KeyError: If server is not registered
        """
        if server_name not in self._servers:
            available = ", ".join(self._servers.keys()) if self._servers else "none"
            raise KeyError(
                f"Server '{server_name}' not found. Available servers: {available}"
            )

        return self._servers[server_name]

    def list_servers(self) -> List[str]:
        """Get list of registered server names."""
        return list(self._servers.keys())

    def get_all_tools(self) -> List[Dict[str, Any]]:
        """
        Get all tools from all registered servers.

        Returns:
            List of tool schemas from all servers
        """
        all_tools = []

        for server_name, server in self._servers.items():
            tools = server.list_tools()
            # Add server name to each tool
            for tool in tools:
                tool['server'] = server_name
            all_tools.extend(tools)

        return all_tools

    def execute_tool(
        self,
        server_name: str,
        tool_name: str,
        **parameters
    ) -> Dict[str, Any]:
        """
        Execute a tool from a specific server.

        Args:
            server_name: Name of the server
            tool_name: Name of the tool
            **parameters: Tool parameters

        Returns:
            Execution result
        """
        server = self.get_server(server_name)
        return server.execute_tool(tool_name, **parameters)

    def __len__(self) -> int:
        """Return number of registered servers."""
        return len(self._servers)

    def __contains__(self, server_name: str) -> bool:
        """Check if server is registered."""
        return server_name in self._servers
