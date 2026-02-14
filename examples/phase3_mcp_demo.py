"""
Phase 3 MCP demonstration.
Shows integration with filesystem and database MCP servers.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline_v3 import MCPEnabledPipeline
from src.config.settings import settings


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80 + "\n")


def demo_mcp_tools():
    """Demonstrate available MCP tools."""
    print_section("Available MCP Tools")

    pipeline = MCPEnabledPipeline(enable_mcp=True)

    tools = pipeline.get_mcp_tools()

    if not tools:
        print("No MCP tools available. Check your MCP configuration.")
        print("\nTroubleshooting:")
        print("1. Install MCP servers: npm install -g @modelcontextprotocol/server-filesystem @modelcontextprotocol/server-sqlite")
        print("2. Set absolute paths in .env")
        print("3. Run: python scripts/setup_mcp_data.py")
        return None

    for server_name, server_tools in tools.items():
        print(f"\nServer: {server_name}")
        print(f"   Tools: {len(server_tools)}")
        for tool in server_tools:
            print(f"   - {tool['name']}: {tool['description']}")

    return pipeline


def demo_filesystem_queries(pipeline):
    """Demonstrate filesystem MCP queries."""
    if not pipeline:
        return

    print_section("Filesystem Queries")

    queries = [
        "List all files in the documents folder",
        "What's in the meeting notes file?",
        "Show me the Q2 sales report",
    ]

    for query in queries:
        print(f"Query: {query}")
        print("-" * 80)

        result = pipeline.query_v3(query, use_rag=False)

        print(f"Answer: {result['answer']}")

        if result.get('tool_results'):
            print(f"\nTool calls made: {len(result['tool_results'])}")
            for tool_name, tool_result in result['tool_results'].items():
                if tool_result['success']:
                    print(f"  {tool_name}")

        print()


def demo_database_queries(pipeline):
    """Demonstrate database MCP queries."""
    if not pipeline:
        return

    print_section("Database Queries")

    queries = [
        "How many customers do we have?",
        "Show me recent orders",
        "What are the top-selling products?",
    ]

    for query in queries:
        print(f"Query: {query}")
        print("-" * 80)

        result = pipeline.query_v3(query, use_rag=False)

        print(f"Answer: {result['answer']}")

        if result.get('tool_results'):
            print(f"\nTool calls made: {len(result['tool_results'])}")
            for tool_name, tool_result in result['tool_results'].items():
                if tool_result['success']:
                    print(f"  {tool_name}")

        print()


def demo_hybrid_queries(pipeline):
    """Demonstrate hybrid RAG + MCP queries."""
    if not pipeline:
        return

    print_section("Hybrid Queries (RAG + MCP)")

    # First ingest some documents about the project
    print("Ingesting project documentation into RAG...")
    pipeline.ingest_documents(directory_path="./data/filesystem/documents")
    print("Documents ingested\n")

    queries = [
        "What is Project Alpha and what's its current status?",
        "Summarize our sales performance and project goals",
    ]

    for query in queries:
        print(f"Query: {query}")
        print("-" * 80)

        result = pipeline.query_v3(query, use_rag=True, use_mcp=True)

        print(f"Answer: {result['answer']}")

        print(f"\nData sources:")
        if result.get('tool_results'):
            print(f"  MCP tools: {len(result['tool_results'])}")
        if result.get('rag_sources'):
            print(f"  RAG documents: {len(result['rag_sources'])}")

        print()


def main():
    """Main demo function."""
    print("=" * 80)
    print(" Phase 3: MCP Integration Demo")
    print("=" * 80)

    print(f"\nMCP Enabled: {settings.enable_mcp}")
    print(f"MCP Servers: {settings.mcp_servers_enabled}")

    # Demo 1: Show available tools
    pipeline = demo_mcp_tools()

    if not pipeline:
        print("\nMCP not properly configured. Please set up MCP first.")
        return

    # Demo 2: Filesystem queries
    demo_filesystem_queries(pipeline)

    # Demo 3: Database queries
    demo_database_queries(pipeline)

    # Demo 4: Hybrid queries
    demo_hybrid_queries(pipeline)

    # Cleanup
    pipeline.shutdown()

    print("=" * 80)
    print(" Demo Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
