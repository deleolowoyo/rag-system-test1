"""
MCP Tools

Concrete implementations of MCP tools for various data sources.

Available Tools:
- GoogleDriveTool: Search and retrieve files from Google Drive
- DatabaseTool: Query SQL databases (SQLite, PostgreSQL)
- (More tools to be added: Filesystem, Web Search)

Example Usage:
    >>> from src.mcp.tools import GoogleDriveTool, DatabaseTool
    >>> from src.mcp import MCPServer
    >>>
    >>> # Create and register Google Drive tool
    >>> drive_tool = GoogleDriveTool(
    ...     credentials_path="./credentials/google_credentials.json",
    ...     token_path="./credentials/google_token.json"
    ... )
    >>>
    >>> # Create and register Database tool
    >>> db_tool = DatabaseTool(database_url="sqlite:///./data/app.db")
    >>>
    >>> server = MCPServer("data_sources")
    >>> server.register_tool(drive_tool)
    >>> server.register_tool(db_tool)
    >>>
    >>> # Search Google Drive
    >>> result = server.execute_tool(
    ...     "google_drive_search",
    ...     query="name contains 'report'",
    ...     max_results=5
    ... )
    >>>
    >>> # Query database
    >>> result = server.execute_tool(
    ...     "database_query",
    ...     query="SELECT * FROM users WHERE active = :active",
    ...     parameters={"active": True},
    ...     limit=10
    ... )
"""

from .google_drive import GoogleDriveTool
from .database import DatabaseTool

__all__ = [
    "GoogleDriveTool",
    "DatabaseTool",
]
