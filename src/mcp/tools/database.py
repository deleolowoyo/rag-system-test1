"""
Database MCP Tool

Tool for querying SQL databases (SQLite and PostgreSQL) with safety controls.
"""
import logging
import re
from typing import Dict, Any, List, Optional
from urllib.parse import urlparse

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

from ..base import MCPTool, MCPToolSchema, MCPToolParameter
from ..exceptions import MCPExecutionError, MCPConnectionError, MCPParameterValidationError

logger = logging.getLogger(__name__)


class DatabaseTool(MCPTool):
    """
    MCP tool for querying SQL databases.

    Features:
    - Read-only queries (SELECT only)
    - SQL injection prevention via parameterized queries
    - Row limits to prevent excessive data retrieval
    - Support for SQLite and PostgreSQL

    Security:
    - Blocks all write operations (INSERT, UPDATE, DELETE, DROP, etc.)
    - Validates queries before execution
    - Uses parameterized queries exclusively
    """

    # Dangerous SQL keywords that should be blocked
    BLOCKED_KEYWORDS = [
        'INSERT', 'UPDATE', 'DELETE', 'DROP', 'CREATE', 'ALTER',
        'TRUNCATE', 'REPLACE', 'MERGE', 'GRANT', 'REVOKE',
        'EXEC', 'EXECUTE', 'CALL', 'DO'
    ]

    def __init__(self, database_url: str):
        """
        Initialize database tool.

        Args:
            database_url: SQLAlchemy database URL
                Examples:
                - sqlite:///./data/app.db
                - postgresql://user:pass@localhost:5432/dbname

        Raises:
            MCPConnectionError: If database URL is invalid
        """
        self.database_url = database_url
        self.engine: Optional[Engine] = None

        # Validate URL format
        try:
            parsed = urlparse(database_url)
            self.db_type = parsed.scheme.split('+')[0]  # Handle postgresql+psycopg2

            if self.db_type not in ['sqlite', 'postgresql']:
                raise ValueError(
                    f"Unsupported database type: {self.db_type}. "
                    f"Supported: sqlite, postgresql"
                )

            logger.info(f"Initialized DatabaseTool for {self.db_type}")

        except Exception as e:
            raise MCPConnectionError(
                server_name="database",
                reason=f"Invalid database URL: {str(e)}",
                server_config={"url_scheme": database_url.split(':')[0]}
            )

        # Initialize base class
        super().__init__()

    def _create_schema(self) -> MCPToolSchema:
        """Create tool schema."""
        return MCPToolSchema(
            name="database_query",
            description=(
                "Execute read-only SQL queries on configured database. "
                "Only SELECT statements are allowed. "
                "Supports parameterized queries to prevent SQL injection. "
                f"Database type: {self.db_type}"
            ),
            parameters=[
                MCPToolParameter(
                    name="query",
                    type="string",
                    description=(
                        "SQL SELECT query to execute. "
                        "Use :param_name for parameterized values. "
                        "Examples: "
                        "'SELECT * FROM users WHERE id = :user_id', "
                        "'SELECT name, email FROM customers LIMIT 10'"
                    ),
                    required=True
                ),
                MCPToolParameter(
                    name="parameters",
                    type="object",
                    description=(
                        "Optional dictionary of query parameters. "
                        "Keys match parameter names in query. "
                        "Example: {'user_id': 123}"
                    ),
                    required=False,
                    default=None
                ),
                MCPToolParameter(
                    name="limit",
                    type="integer",
                    description="Maximum number of rows to return (safety limit)",
                    required=False,
                    default=100
                ),
            ],
            category="database",
            version="1.0.0"
        )

    def _create_engine(self) -> Engine:
        """
        Create or return SQLAlchemy engine.

        Returns:
            SQLAlchemy Engine instance

        Raises:
            MCPConnectionError: If connection fails
        """
        if self.engine:
            return self.engine

        try:
            logger.info(f"Creating database engine for {self.db_type}")

            # Connection arguments
            connect_args = {}

            # SQLite-specific settings
            if self.db_type == 'sqlite':
                connect_args['check_same_thread'] = False

            # Create engine
            self.engine = create_engine(
                self.database_url,
                connect_args=connect_args,
                pool_pre_ping=True,  # Verify connections before using
                echo=False  # Set to True for SQL debugging
            )

            # Test connection
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))

            logger.info("Database connection established")
            return self.engine

        except SQLAlchemyError as e:
            raise MCPConnectionError(
                server_name="database",
                reason=f"Failed to connect to database: {str(e)}",
                server_config={"db_type": self.db_type}
            )

        except Exception as e:
            raise MCPConnectionError(
                server_name="database",
                reason=f"Unexpected error creating engine: {str(e)}",
                server_config={"db_type": self.db_type}
            )

    def _validate_query(self, query: str) -> bool:
        """
        Validate that query is safe to execute.

        Only SELECT statements are allowed. Blocks all write operations.

        Args:
            query: SQL query string

        Returns:
            True if query is valid

        Raises:
            MCPParameterValidationError: If query contains dangerous operations
        """
        # Normalize query for checking
        query_upper = query.upper().strip()

        # Remove comments
        query_upper = re.sub(r'--.*', '', query_upper)  # Line comments
        query_upper = re.sub(r'/\*.*?\*/', '', query_upper, flags=re.DOTALL)  # Block comments

        # Must start with SELECT (after whitespace)
        if not re.match(r'^\s*SELECT\s', query_upper):
            raise MCPParameterValidationError(
                tool_name=self.name,
                parameter_name="query",
                validation_message=(
                    "Only SELECT queries are allowed. "
                    f"Query starts with: {query_upper.split()[0] if query_upper.split() else 'empty'}"
                )
            )

        # Check for blocked keywords
        for keyword in self.BLOCKED_KEYWORDS:
            # Use word boundaries to avoid false positives
            # e.g., don't block "SELECT" in "SELECT"
            pattern = r'\b' + re.escape(keyword) + r'\b'
            if re.search(pattern, query_upper):
                raise MCPParameterValidationError(
                    tool_name=self.name,
                    parameter_name="query",
                    validation_message=f"Query contains blocked keyword: {keyword}"
                )

        # Check for semicolons (prevent multiple statements)
        if ';' in query.rstrip(';'):  # Allow trailing semicolon
            raise MCPParameterValidationError(
                tool_name=self.name,
                parameter_name="query",
                validation_message="Multiple statements not allowed (found semicolon)"
            )

        logger.debug("Query validation passed")
        return True

    def _execute_query(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
        limit: int = 100
    ) -> Dict[str, Any]:
        """
        Execute SQL query and return results.

        Args:
            query: SQL query string
            parameters: Query parameters
            limit: Maximum rows to return

        Returns:
            Dictionary with query results

        Raises:
            MCPExecutionError: If query execution fails
        """
        engine = self._create_engine()
        parameters = parameters or {}

        try:
            # Apply LIMIT to query if not already present
            query_upper = query.upper()
            if 'LIMIT' not in query_upper:
                query = f"{query.rstrip(';')} LIMIT {limit}"
                logger.debug(f"Applied LIMIT {limit} to query")

            # Execute query
            logger.info(f"Executing query: {query[:100]}...")

            with engine.connect() as conn:
                result = conn.execute(text(query), parameters)

                # Fetch all rows
                rows = result.fetchall()

                # Convert rows to dictionaries
                columns = list(result.keys())
                row_dicts = [
                    {col: val for col, val in zip(columns, row)}
                    for row in rows
                ]

                logger.info(f"Query returned {len(row_dicts)} rows")

                return {
                    "rows": row_dicts,
                    "row_count": len(row_dicts),
                    "columns": columns,
                    "query": query,
                }

        except SQLAlchemyError as e:
            logger.error(f"Database query failed: {e}")
            raise MCPExecutionError(
                tool_name=self.name,
                error_message=f"Database query failed: {str(e)}",
                original_exception=e,
                parameters={"query": query[:100], "params": str(parameters)}
            )

        except Exception as e:
            logger.error(f"Unexpected error executing query: {e}")
            raise MCPExecutionError(
                tool_name=self.name,
                error_message=f"Unexpected error: {str(e)}",
                original_exception=e,
                parameters={"query": query[:100]}
            )

    def _execute(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
        limit: int = 100
    ) -> Dict[str, Any]:
        """
        Execute database query with validation.

        Args:
            query: SQL query string
            parameters: Optional query parameters
            limit: Maximum rows to return

        Returns:
            Dictionary with query results
        """
        try:
            # Validate query safety
            self._validate_query(query)

            # Validate limit
            if limit <= 0 or limit > 10000:
                raise MCPParameterValidationError(
                    tool_name=self.name,
                    parameter_name="limit",
                    validation_message=f"Limit must be between 1 and 10000, got {limit}"
                )

            # Execute query
            result = self._execute_query(query, parameters, limit)

            return {
                "success": True,
                "result": result
            }

        except (MCPParameterValidationError, MCPExecutionError):
            # Re-raise MCP errors
            raise

        except Exception as e:
            logger.error(f"Unexpected error in database tool: {e}")
            raise MCPExecutionError(
                tool_name=self.name,
                error_message=f"Unexpected error: {str(e)}",
                original_exception=e,
                parameters={"query": query}
            )

    def close(self) -> None:
        """Close database connection and dispose of engine."""
        if self.engine:
            logger.info("Closing database connection")
            self.engine.dispose()
            self.engine = None

    def get_tables(self) -> List[str]:
        """
        Get list of tables in the database.

        Returns:
            List of table names

        Raises:
            MCPExecutionError: If query fails
        """
        if self.db_type == 'sqlite':
            query = "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        elif self.db_type == 'postgresql':
            query = """
                SELECT tablename FROM pg_catalog.pg_tables
                WHERE schemaname != 'pg_catalog' AND schemaname != 'information_schema'
                ORDER BY tablename
            """
        else:
            raise MCPExecutionError(
                tool_name=self.name,
                error_message=f"get_tables() not supported for {self.db_type}"
            )

        try:
            result = self._execute_query(query, limit=1000)
            return [row[result['columns'][0]] for row in result['rows']]

        except Exception as e:
            logger.error(f"Failed to get table list: {e}")
            raise MCPExecutionError(
                tool_name=self.name,
                error_message=f"Failed to get tables: {str(e)}",
                original_exception=e
            )

    def get_table_schema(self, table_name: str) -> List[Dict[str, Any]]:
        """
        Get schema information for a table.

        Args:
            table_name: Name of the table

        Returns:
            List of column information dictionaries

        Raises:
            MCPExecutionError: If query fails
        """
        # Validate table name (prevent SQL injection)
        if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', table_name):
            raise MCPParameterValidationError(
                tool_name=self.name,
                parameter_name="table_name",
                validation_message=f"Invalid table name: {table_name}"
            )

        if self.db_type == 'sqlite':
            query = f"PRAGMA table_info({table_name})"
        elif self.db_type == 'postgresql':
            query = f"""
                SELECT column_name, data_type, is_nullable
                FROM information_schema.columns
                WHERE table_name = '{table_name}'
                ORDER BY ordinal_position
            """
        else:
            raise MCPExecutionError(
                tool_name=self.name,
                error_message=f"get_table_schema() not supported for {self.db_type}"
            )

        try:
            result = self._execute_query(query, limit=1000)
            return result['rows']

        except Exception as e:
            logger.error(f"Failed to get table schema: {e}")
            raise MCPExecutionError(
                tool_name=self.name,
                error_message=f"Failed to get schema for table '{table_name}': {str(e)}",
                original_exception=e
            )

    def __del__(self):
        """Cleanup: dispose of engine when object is destroyed."""
        self.close()
