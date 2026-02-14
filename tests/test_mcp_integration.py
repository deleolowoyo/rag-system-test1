"""
Integration tests for MCP functionality.
Tests actual MCP server connections and tool execution.
"""
import pytest
import os
from pathlib import Path

from src.mcp.manager import MCPManager
from src.agents.mcp_agent import MCPAgent
from src.pipeline_v3 import MCPEnabledPipeline


@pytest.fixture
def mcp_manager():
    """Create MCP manager for testing."""
    manager = MCPManager()
    manager.initialize()
    yield manager
    manager.shutdown()


@pytest.fixture
def sample_db_path(tmp_path):
    """Create a temporary sample database."""
    import sqlite3

    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE customers (
            id INTEGER PRIMARY KEY,
            name TEXT,
            email TEXT
        )
    """)

    cursor.execute(
        "INSERT INTO customers VALUES (1, 'Test User', 'test@example.com')"
    )

    conn.commit()
    conn.close()

    return str(db_path.absolute())


@pytest.fixture
def sample_filesystem(tmp_path):
    """Create temporary filesystem structure."""
    fs_root = tmp_path / "filesystem"
    fs_root.mkdir()

    (fs_root / "test.txt").write_text("This is a test file for MCP.")
    (fs_root / "data.txt").write_text("Sample data: 12345")

    return str(fs_root.absolute())


class TestMCPManager:
    """Test MCP Manager functionality."""

    def test_manager_initialization(self, mcp_manager):
        """Test MCP manager initializes correctly."""
        assert mcp_manager is not None
        assert mcp_manager.initialized

    def test_get_available_tools(self, mcp_manager):
        """Test getting available tools."""
        tools = mcp_manager.get_available_tools()

        assert isinstance(tools, dict)
        # Should have at least one server if MCP is configured
        # (may be empty if no servers configured in test environment)

    @pytest.mark.skipif(
        not os.getenv("MCP_FILESYSTEM_ROOT"),
        reason="MCP filesystem not configured"
    )
    def test_filesystem_tool(self, mcp_manager, sample_filesystem):
        """Test filesystem tool execution."""
        # This test requires MCP filesystem server to be running
        # Skip if not configured

        result = mcp_manager.find_tool("read_file")
        if result:
            server_name, tool_schema = result

            # Try to read a file
            file_result = mcp_manager.call_tool(
                server_name=server_name,
                tool_name="read_file",
                arguments={"path": "test.txt"}
            )

            assert file_result is not None


class TestMCPAgent:
    """Test MCP Agent functionality."""

    def test_agent_initialization(self, mcp_manager):
        """Test MCP agent initializes."""
        agent = MCPAgent(mcp_manager=mcp_manager)
        assert agent is not None

    def test_query_analysis(self, mcp_manager):
        """Test query analysis for tool selection."""
        agent = MCPAgent(mcp_manager=mcp_manager)

        plan = agent.analyze_query("What files are in my directory?")

        assert isinstance(plan, dict)
        assert "needs_tools" in plan
        assert "reasoning" in plan

    @pytest.mark.integration
    def test_agent_execution(self, mcp_manager):
        """Test full agent execution flow."""
        agent = MCPAgent(mcp_manager=mcp_manager)

        # Simple query that shouldn't need tools
        result = agent.run("What is 2+2?", rag_context="Math context")

        assert "answer" in result
        assert isinstance(result["answer"], str)


class TestMCPPipeline:
    """Test MCP-enabled pipeline."""

    def test_pipeline_initialization(self):
        """Test pipeline initializes with MCP."""
        pipeline = MCPEnabledPipeline(
            collection_name="test_mcp",
            enable_mcp=True
        )

        assert pipeline is not None
        # enable_mcp may be False if MCP not configured

    def test_pipeline_query_v3(self):
        """Test Phase 3 query method."""
        pipeline = MCPEnabledPipeline(
            collection_name="test_mcp",
            enable_mcp=True
        )

        # Query that should work without MCP if not available
        result = pipeline.query_v3(
            "What is machine learning?",
            use_mcp=False,  # Don't require MCP
            use_rag=False   # Don't require RAG (no docs ingested)
        )

        assert "answer" in result

    def test_pipeline_get_mcp_tools(self):
        """Test getting MCP tools from pipeline."""
        pipeline = MCPEnabledPipeline(enable_mcp=True)

        tools = pipeline.get_mcp_tools()
        assert isinstance(tools, dict)


@pytest.mark.e2e
class TestMCPEndToEnd:
    """End-to-end MCP tests."""

    @pytest.mark.skipif(
        not os.getenv("MCP_FILESYSTEM_ROOT") or not os.getenv("MCP_SQLITE_DB_PATH"),
        reason="MCP not fully configured"
    )
    def test_full_mcp_workflow(self, sample_filesystem, sample_db_path):
        """Test complete MCP workflow."""
        # This test requires full MCP setup
        # Create pipeline with MCP
        pipeline = MCPEnabledPipeline(enable_mcp=True)

        # Query using filesystem
        fs_result = pipeline.query_v3(
            "What files are available?",
            use_mcp=True,
            use_rag=False
        )

        assert "answer" in fs_result

        # Query using database
        db_result = pipeline.query_v3(
            "How many customers are in the database?",
            use_mcp=True,
            use_rag=False
        )

        assert "answer" in db_result

        # Cleanup
        pipeline.shutdown()


# Mark all tests that need actual MCP servers
pytestmark = pytest.mark.integration
