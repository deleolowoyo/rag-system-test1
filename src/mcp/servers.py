"""
MCP Server configurations and initialization.
Defines how to connect to various MCP servers (filesystem, SQLite, Google Drive).
"""
import os
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class MCPServerConfig:
    """
    Configuration for an MCP server subprocess.

    Holds everything needed to start an MCP server process via stdio:
    - command: the executable (e.g., "npx")
    - args: command-line arguments including the server package and options
    - env: optional environment variables for the subprocess
    - enabled: flag to disable without removing the config
    """
    name: str
    command: str
    args: List[str]
    env: Optional[Dict[str, str]] = None
    enabled: bool = True

    def __post_init__(self):
        """Validate configuration after init."""
        if not self.name:
            raise ValueError("Server name cannot be empty")
        if not self.command:
            raise ValueError("Server command cannot be empty")
        if self.args is None:
            self.args = []

    def __repr__(self) -> str:
        return (
            f"MCPServerConfig(name='{self.name}', "
            f"command='{self.command}', "
            f"enabled={self.enabled})"
        )


class MCPServerRegistry:
    """
    Registry of available MCP server configurations.

    Provides factory methods for each supported server type and a
    helper to build the enabled-server list from application settings.

    All servers use npx to run official MCP npm packages.
    Install requirements: Node.js and npm must be installed.
    """

    @staticmethod
    def get_filesystem_server(root_path: str) -> MCPServerConfig:
        """
        Get filesystem MCP server configuration.

        Uses: @modelcontextprotocol/server-filesystem
        Requires: npm install -g @modelcontextprotocol/server-filesystem
                  (or npx will install on first use with -y flag)

        The server exposes tools for reading, writing, and listing files
        within the specified root directory. Only the root directory and
        its descendants are accessible.

        Args:
            root_path: Absolute path to the allowed directory.
                       MCP requires absolute paths for security.

        Returns:
            MCPServerConfig for the filesystem server

        Raises:
            ValueError: If root_path is not an absolute path
        """
        if not os.path.isabs(root_path):
            raise ValueError(
                f"Filesystem root must be an absolute path, got: '{root_path}'"
            )

        logger.info(f"Creating filesystem MCP server config: root={root_path}")

        return MCPServerConfig(
            name="filesystem",
            command="npx",
            args=[
                "-y",
                "@modelcontextprotocol/server-filesystem",
                root_path,
            ],
            enabled=True,
        )

    @staticmethod
    def get_sqlite_server(db_path: str) -> MCPServerConfig:
        """
        Get SQLite MCP server configuration.

        Uses: @modelcontextprotocol/server-sqlite
        Requires: npm install -g @modelcontextprotocol/server-sqlite
                  (or npx will install on first use with -y flag)

        The server exposes tools for querying and inspecting a SQLite
        database file. The database file will be created if it doesn't exist.

        Args:
            db_path: Absolute path to the SQLite database file.
                     MCP requires absolute paths for security.

        Returns:
            MCPServerConfig for the SQLite server

        Raises:
            ValueError: If db_path is not an absolute path
        """
        if not os.path.isabs(db_path):
            raise ValueError(
                f"SQLite DB path must be an absolute path, got: '{db_path}'"
            )

        logger.info(f"Creating SQLite MCP server config: db={db_path}")

        return MCPServerConfig(
            name="sqlite",
            command="npx",
            args=[
                "-y",
                "@modelcontextprotocol/server-sqlite",
                "--db-path",
                db_path,
            ],
            enabled=True,
        )

    @staticmethod
    def get_google_drive_server(credentials_path: str) -> MCPServerConfig:
        """
        Get Google Drive MCP server configuration.

        Uses: @modelcontextprotocol/server-gdrive
        Requires: npm install -g @modelcontextprotocol/server-gdrive
                  (or npx will install on first use with -y flag)

        The server exposes tools for searching and reading files in
        Google Drive. Requires valid OAuth2 credentials.

        Args:
            credentials_path: Path to Google OAuth2 credentials JSON file.
                               Will be resolved to an absolute path.

        Returns:
            MCPServerConfig for the Google Drive server
        """
        abs_creds = os.path.abspath(credentials_path)

        if not os.path.exists(abs_creds):
            logger.warning(
                f"Google Drive credentials file not found: {abs_creds}. "
                f"Server will fail on first use."
            )

        logger.info(f"Creating Google Drive MCP server config: creds={abs_creds}")

        return MCPServerConfig(
            name="google_drive",
            command="npx",
            args=[
                "-y",
                "@modelcontextprotocol/server-gdrive",
            ],
            env={
                "GDRIVE_CREDENTIALS_PATH": abs_creds,
            },
            enabled=True,
        )

    @staticmethod
    def get_enabled_servers(settings) -> List[MCPServerConfig]:
        """
        Build the list of enabled MCP server configurations from app settings.

        Reads settings.mcp_* fields to determine which servers are active
        and creates the corresponding MCPServerConfig for each.

        Args:
            settings: Application Settings instance (src.config.settings.Settings)

        Returns:
            List of MCPServerConfig for all enabled servers

        Example:
            >>> from src.config.settings import settings
            >>> from src.mcp.servers import MCPServerRegistry
            >>>
            >>> configs = MCPServerRegistry.get_enabled_servers(settings)
            >>> for config in configs:
            ...     print(f"  {config.name}: {config.command} {config.args}")
        """
        servers: List[MCPServerConfig] = []

        # Filesystem server
        if getattr(settings, 'mcp_filesystem_enabled', False):
            root = getattr(settings, 'mcp_filesystem_root', '')
            if not root:
                logger.warning(
                    "Filesystem MCP server enabled but MCP_FILESYSTEM_ROOT not set. "
                    "Skipping filesystem server."
                )
            else:
                try:
                    servers.append(MCPServerRegistry.get_filesystem_server(root))
                except ValueError as e:
                    logger.error(f"Invalid filesystem config: {e}")

        # SQLite server
        if getattr(settings, 'mcp_sqlite_enabled', False):
            db_path = getattr(settings, 'mcp_sqlite_db_path', '')
            if not db_path:
                logger.warning(
                    "SQLite MCP server enabled but MCP_SQLITE_DB_PATH not set. "
                    "Skipping SQLite server."
                )
            else:
                try:
                    servers.append(MCPServerRegistry.get_sqlite_server(db_path))
                except ValueError as e:
                    logger.error(f"Invalid SQLite config: {e}")

        # Google Drive server
        if getattr(settings, 'mcp_gdrive_enabled', False):
            creds_path = getattr(settings, 'mcp_gdrive_credentials_path', '')
            if not creds_path:
                logger.warning(
                    "Google Drive MCP server enabled but "
                    "MCP_GDRIVE_CREDENTIALS_PATH not set. "
                    "Skipping Google Drive server."
                )
            else:
                servers.append(
                    MCPServerRegistry.get_google_drive_server(creds_path)
                )

        logger.info(
            f"MCP servers configured: "
            f"{[s.name for s in servers] or 'none'}"
        )
        return servers
