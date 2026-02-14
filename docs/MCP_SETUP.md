# MCP (Model Context Protocol) Setup Guide

Complete guide for setting up MCP integration with your RAG system.

**Version**: 3.0.0
**Last Updated**: 2026-02-13
**Status**: Phase 3

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Quick Start](#quick-start)
3. [MCP Architecture](#mcp-architecture)
4. [Server Configuration](#server-configuration)
5. [Environment Variables](#environment-variables)
6. [Troubleshooting](#troubleshooting)
7. [Google Drive Setup](#google-drive-setup)
8. [API Reference](#api-reference)
9. [Testing](#testing)

---

## Prerequisites

1. **Node.js** (v18 or higher)
```bash
node --version  # Should be v18+
```

2. **npm** (comes with Node.js)
```bash
npm --version
```

3. **Python** (3.10+) with project dependencies installed
```bash
pip install -r requirements.txt
```

---

## Quick Start

### 1. Install MCP Servers
```bash
# Install filesystem server
npm install -g @modelcontextprotocol/server-filesystem

# Install SQLite server
npm install -g @modelcontextprotocol/server-sqlite

# (Optional) Install Google Drive server
npm install -g @modelcontextprotocol/server-gdrive
```

### 2. Setup Test Data
```bash
# Create sample database and filesystem
python scripts/setup_mcp_data.py
```

This creates:
- SQLite database at `./data/app.db`
- Sample files at `./data/filesystem/`

### 3. Configure Environment

Update your `.env` file with ABSOLUTE paths:
```bash
# MCP Configuration
ENABLE_MCP=true
MCP_TIMEOUT=30

# Filesystem Server (MUST be absolute path)
MCP_FILESYSTEM_ENABLED=true
MCP_FILESYSTEM_ROOT=/full/path/to/your/project/data/filesystem

# SQLite Server (MUST be absolute path)
MCP_SQLITE_ENABLED=true
MCP_SQLITE_DB_PATH=/full/path/to/your/project/data/app.db

# Google Drive (optional)
MCP_GDRIVE_ENABLED=false
MCP_GDRIVE_CREDENTIALS_PATH=./credentials/google_credentials.json
```

**Important**: MCP requires absolute paths for security. Get your project path:
```bash
# On macOS/Linux
pwd

# On Windows
cd

# Then append /data/filesystem or /data/app.db
```

### 4. Verify Setup
```bash
# Run MCP demo
python examples/phase3_mcp_demo.py
```

If successful, you'll see:
```
Server: filesystem
   Tools: 7
   - read_file
   - write_file
   - list_directory
   ...

Server: sqlite
   Tools: 4
   - read_query
   - write_query
   ...
```

---

## MCP Architecture

```
┌─────────────────────────────────────────────────────┐
│                  MCPEnabledPipeline                 │
│                   (pipeline_v3.py)                  │
│                                                     │
│  ┌──────────────┐        ┌───────────────────────┐  │
│  │  EnhancedRAG │        │      MCPAgent         │  │
│  │  (Phase 1+2) │        │  (agents/mcp_agent)   │  │
│  └──────────────┘        └───────────────────────┘  │
│         │                          │                │
│         │                   ┌──────▼──────┐         │
│         │                   │ MCPManager  │         │
│         │                   │(mcp/manager)│         │
│         │                   └──────┬──────┘         │
│         │                          │                │
└─────────┼──────────────────────────┼────────────────┘
          │                          │
          ▼                          ▼
    Vector Store              ┌──────┴──────┐
    (FAISS/Chroma)            │  MCPClient  │
                              │(mcp/client) │
                              └──────┬──────┘
                                     │  stdio (JSON-RPC)
                        ┌────────────┼────────────┐
                        ▼            ▼             ▼
               ┌──────────────┐ ┌────────┐ ┌──────────┐
               │  Filesystem  │ │ SQLite │ │  Google  │
               │  MCP Server  │ │ Server │ │  Drive   │
               │    (npx)     │ │  (npx) │ │  (npx)   │
               └──────────────┘ └────────┘ └──────────┘
```

### How It Works

1. `MCPManager.initialize()` calls `MCPServerRegistry.get_enabled_servers()` to read settings and build server configs
2. For each enabled server, `MCPClient.connect_to_server()` launches the npm package as a subprocess via `npx`
3. Communication happens over stdio using JSON-RPC (the MCP protocol)
4. `MCPAgent.analyze_query()` uses the LLM to decide which tools to call based on the query
5. Results are synthesized with optional RAG context into a final answer

---

## Server Configuration

### Filesystem Server

Exposes tools for reading, listing, and writing files within a sandboxed root directory.

**Available tools:**
| Tool | Description |
|------|-------------|
| `read_file` | Read the full contents of a file |
| `write_file` | Write content to a file |
| `list_directory` | List files and directories |
| `create_directory` | Create a new directory |
| `move_file` | Move or rename a file |
| `search_files` | Search for files by name pattern |
| `get_file_info` | Get metadata (size, modified date) |

**Configuration:**
```bash
MCP_FILESYSTEM_ENABLED=true
MCP_FILESYSTEM_ROOT=/absolute/path/to/root
```

**Security note**: The server only exposes files within `MCP_FILESYSTEM_ROOT`. No parent directory traversal is possible.

---

### SQLite Server

Exposes tools for querying and managing a SQLite database.

**Available tools:**
| Tool | Description |
|------|-------------|
| `read_query` | Execute a SELECT query |
| `write_query` | Execute INSERT/UPDATE/DELETE |
| `create_table` | Create a new table |
| `list_tables` | List all tables in the database |
| `describe_table` | Get schema for a specific table |

**Configuration:**
```bash
MCP_SQLITE_ENABLED=true
MCP_SQLITE_DB_PATH=/absolute/path/to/database.db
```

**Note**: The database file is created automatically if it doesn't exist.

---

### Google Drive Server

Exposes tools for searching and reading files from Google Drive. Requires OAuth2 credentials (see [Google Drive Setup](#google-drive-setup)).

**Available tools:**
| Tool | Description |
|------|-------------|
| `search` | Search for files in Google Drive |
| `read_file` | Read the contents of a Drive file |

**Configuration:**
```bash
MCP_GDRIVE_ENABLED=true
MCP_GDRIVE_CREDENTIALS_PATH=./credentials/google_credentials.json
```

---

## Environment Variables

Full list of MCP-related environment variables:

| Variable | Default | Required | Description |
|----------|---------|----------|-------------|
| `ENABLE_MCP` | `true` | No | Master switch for all MCP features |
| `MCP_TIMEOUT` | `30` | No | Seconds before tool call times out |
| `MCP_FILESYSTEM_ENABLED` | `true` | No | Enable filesystem server |
| `MCP_FILESYSTEM_ROOT` | _(empty)_ | If enabled | **Absolute** path to filesystem root |
| `MCP_SQLITE_ENABLED` | `true` | No | Enable SQLite server |
| `MCP_SQLITE_DB_PATH` | _(empty)_ | If enabled | **Absolute** path to SQLite DB file |
| `MCP_GDRIVE_ENABLED` | `false` | No | Enable Google Drive server |
| `MCP_GDRIVE_CREDENTIALS_PATH` | `./credentials/google_credentials.json` | If enabled | Path to OAuth2 credentials |

---

## Troubleshooting

### "No MCP tools available"

**Cause**: MCP servers failed to start, or no servers are configured.

**Steps:**
1. Check that Node.js is installed: `node --version`
2. Verify npm packages are installed: `npx @modelcontextprotocol/server-filesystem --help`
3. Check your `.env` for absolute paths — relative paths will be rejected
4. Look at application logs for connection errors

---

### "MCP_FILESYSTEM_ROOT must be an absolute path"

**Cause**: The path in `.env` is relative (starts with `./` or `../`).

**Fix:**
```bash
# Get the absolute path
pwd
# Example output: /Users/yourname/projects/rag-system

# Use it in .env:
MCP_FILESYSTEM_ROOT=/Users/yourname/projects/rag-system/data/filesystem
MCP_SQLITE_DB_PATH=/Users/yourname/projects/rag-system/data/app.db
```

---

### "Failed to connect to MCP server"

**Cause**: `npx` cannot find or install the MCP server package.

**Fix:**
```bash
# Pre-install the packages globally
npm install -g @modelcontextprotocol/server-filesystem
npm install -g @modelcontextprotocol/server-sqlite

# Verify installation
npx @modelcontextprotocol/server-filesystem --version
```

---

### Tool call returns empty results

**Cause**: The file or database path doesn't exist, or the LLM provided wrong arguments.

**Fix:**
1. Run `python scripts/setup_mcp_data.py` to populate test data
2. Check that `MCP_FILESYSTEM_ROOT` points to the correct directory
3. Verify the database has tables: `sqlite3 data/app.db ".tables"`

---

### MCP servers start but queries don't use them

**Cause**: The LLM's tool-selection prompt determined no tools were needed.

**Debugging:**
```python
from src.mcp.manager import get_mcp_manager

manager = get_mcp_manager()
manager.initialize()

# Check which servers connected
print(manager.client.connected_servers())

# List all available tools
tools = manager.get_available_tools()
for server, tool_list in tools.items():
    print(f"{server}: {[t['name'] for t in tool_list]}")
```

---

## Google Drive Setup

### 1. Create a Google Cloud Project

1. Go to [Google Cloud Console](https://console.cloud.google.com)
2. Create a new project (or select an existing one)
3. Enable the **Google Drive API**

### 2. Create OAuth2 Credentials

1. Navigate to **APIs & Services → Credentials**
2. Click **Create Credentials → OAuth client ID**
3. Select **Desktop application**
4. Download the JSON credentials file
5. Save it to `credentials/google_credentials.json`

### 3. Configure Environment

```bash
MCP_GDRIVE_ENABLED=true
MCP_GDRIVE_CREDENTIALS_PATH=./credentials/google_credentials.json
```

### 4. First Run (OAuth Flow)

On first use, the server will open a browser for authorization:
```bash
python examples/phase3_mcp_demo.py
# Browser opens → sign in → grant permissions
```

The OAuth token is saved automatically for subsequent runs.

**Security**: `credentials/*.json` and `credentials/*.token` are excluded from git via `.gitignore`.

---

## API Reference

### MCPManager

```python
from src.mcp.manager import MCPManager, get_mcp_manager

# Singleton
manager = get_mcp_manager()
manager.initialize()

# List all tools
tools = manager.get_available_tools()
# Returns: {"filesystem": [{"name": "read_file", ...}], "sqlite": [...]}

# Find a tool by name
result = manager.find_tool("read_file")
# Returns: ("filesystem", {tool_schema}) or None

# Call a tool
result = manager.call_tool(
    server_name="filesystem",
    tool_name="read_file",
    arguments={"path": "/data/filesystem/documents/notes.txt"}
)

# Context manager
with MCPManager() as manager:
    tools = manager.get_available_tools()
```

### MCPClient (async)

```python
import asyncio
from src.mcp import MCPClient

async def main():
    async with MCPClient() as client:
        await client.connect_to_server(
            server_name="filesystem",
            command="npx",
            args=["-y", "@modelcontextprotocol/server-filesystem", "/data"]
        )

        tools = await client.list_all_tools()

        result = await client.call_tool(
            server_name="filesystem",
            tool_name="read_file",
            arguments={"path": "/data/doc.txt"}
        )

        # result.content is a list of TextContent blocks
        for block in result.content:
            print(block.text)

asyncio.run(main())
```

### MCPEnabledPipeline

```python
from src.pipeline_v3 import MCPEnabledPipeline

pipeline = MCPEnabledPipeline(enable_mcp=True)

# Phase 3: MCP + RAG
result = pipeline.query_v3(
    question="What are the latest sales figures?",
    use_mcp=True,   # Use live MCP data
    use_rag=True,   # Also search vector store
    return_sources=True
)

print(result["answer"])
print(result["phase"])         # "phase3_mcp" or "phase2_fallback"
print(result["tool_results"])  # Which MCP tools were called
print(result["rag_sources"])   # Documents from vector store

# Always shut down to close subprocess connections
pipeline.shutdown()
```

---

## Testing

### Unit tests (no MCP servers required)
```bash
pytest tests/test_mcp_integration.py -m "not integration and not e2e" -v
```

### Integration tests (MCP servers required)
```bash
# First ensure servers are running and .env is configured
pytest tests/test_mcp_integration.py -m integration -v
```

### End-to-end tests (full MCP setup required)
```bash
pytest tests/test_mcp_integration.py -m e2e -v
```

### Run all MCP tests
```bash
pytest tests/test_mcp_integration.py -v
```
