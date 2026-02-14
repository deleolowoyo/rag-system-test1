"""
Setup script for MCP test data.
Creates sample database and filesystem structure.
"""
import os
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta


def create_sample_database(db_path: str):
    """Create sample SQLite database for MCP testing."""
    print(f"Creating sample database: {db_path}")

    # Ensure directory exists
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    # Create database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Customers table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS customers (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            created_at TEXT NOT NULL,
            status TEXT NOT NULL
        )
    """)

    # Insert sample customers
    customers = [
        (1, "Alice Johnson", "alice@example.com", "2024-01-15", "active"),
        (2, "Bob Smith", "bob@example.com", "2024-02-20", "active"),
        (3, "Carol White", "carol@example.com", "2024-03-10", "inactive"),
        (4, "David Brown", "david@example.com", "2024-04-05", "active"),
        (5, "Eve Davis", "eve@example.com", "2024-05-12", "active"),
    ]

    cursor.executemany(
        "INSERT OR REPLACE INTO customers VALUES (?, ?, ?, ?, ?)",
        customers
    )

    # Orders table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS orders (
            id INTEGER PRIMARY KEY,
            customer_id INTEGER NOT NULL,
            product TEXT NOT NULL,
            amount REAL NOT NULL,
            order_date TEXT NOT NULL,
            status TEXT NOT NULL,
            FOREIGN KEY (customer_id) REFERENCES customers (id)
        )
    """)

    # Insert sample orders
    orders = [
        (1, 1, "Laptop", 1299.99, "2024-06-01", "delivered"),
        (2, 1, "Mouse", 29.99, "2024-06-02", "delivered"),
        (3, 2, "Keyboard", 89.99, "2024-06-05", "shipped"),
        (4, 3, "Monitor", 349.99, "2024-06-10", "cancelled"),
        (5, 4, "Headphones", 199.99, "2024-06-15", "delivered"),
        (6, 5, "Webcam", 79.99, "2024-06-20", "processing"),
        (7, 1, "USB Cable", 12.99, "2024-06-25", "delivered"),
        (8, 2, "Laptop", 1499.99, "2024-06-28", "processing"),
    ]

    cursor.executemany(
        "INSERT OR REPLACE INTO orders VALUES (?, ?, ?, ?, ?, ?)",
        orders
    )

    # Products table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS products (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            category TEXT NOT NULL,
            price REAL NOT NULL,
            stock INTEGER NOT NULL
        )
    """)

    # Insert sample products
    products = [
        (1, "Laptop", "Computers", 1299.99, 15),
        (2, "Mouse", "Accessories", 29.99, 50),
        (3, "Keyboard", "Accessories", 89.99, 30),
        (4, "Monitor", "Displays", 349.99, 20),
        (5, "Headphones", "Audio", 199.99, 25),
        (6, "Webcam", "Video", 79.99, 40),
    ]

    cursor.executemany(
        "INSERT OR REPLACE INTO products VALUES (?, ?, ?, ?, ?)",
        products
    )

    conn.commit()
    conn.close()

    print(f"Database created with sample data")
    print(f"  - {len(customers)} customers")
    print(f"  - {len(orders)} orders")
    print(f"  - {len(products)} products")


def create_sample_filesystem(root_path: str):
    """Create sample filesystem structure for MCP testing."""
    print(f"Creating sample filesystem: {root_path}")

    root = Path(root_path)
    root.mkdir(parents=True, exist_ok=True)

    # Create directory structure
    (root / "documents").mkdir(exist_ok=True)
    (root / "reports").mkdir(exist_ok=True)
    (root / "notes").mkdir(exist_ok=True)

    # Create sample files
    files = {
        "documents/project_overview.txt": """
Project Alpha - AI Assistant
============================

Goal: Build an intelligent RAG system with MCP integration.

Key Features:
- Multi-format document processing
- Vector search with ChromaDB
- LLM-powered generation
- Real-time data access via MCP

Status: Phase 3 in progress
""",
        "documents/meeting_notes.md": """
# Team Meeting - June 2024

## Attendees
- Alice (PM)
- Bob (Engineering)
- Carol (Design)

## Agenda
1. Phase 3 MCP integration update
2. Testing strategy
3. Timeline review

## Action Items
- [ ] Complete MCP tool integration
- [ ] Write integration tests
- [ ] Update documentation
""",
        "reports/q2_sales.txt": """
Q2 Sales Report
===============

Total Revenue: $45,234
Total Orders: 156
Average Order Value: $289.96

Top Products:
1. Laptop - 45 units
2. Monitor - 32 units
3. Headphones - 28 units

Growth vs Q1: +15%
""",
        "notes/ideas.txt": """
Future Enhancement Ideas
========================

1. Add more MCP servers
   - GitHub integration
   - Slack integration
   - Calendar integration

2. Improve query routing
   - Better intent detection
   - Multi-tool orchestration

3. Enhanced caching
   - Smart cache invalidation
   - Distributed cache support
"""
    }

    for filepath, content in files.items():
        full_path = root / filepath
        full_path.write_text(content.strip())
        print(f"  Created {filepath}")

    print(f"Filesystem created with {len(files)} sample files")


def main():
    """Main setup function."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from src.config.settings import settings

    print("=" * 60)
    print("MCP Test Data Setup")
    print("=" * 60)
    print()

    # Get absolute paths
    db_path = settings.mcp_sqlite_db_path
    fs_root = settings.mcp_filesystem_root

    if not db_path:
        db_path = str(Path.cwd() / "data" / "app.db")
        print(f"Using default database path: {db_path}")

    if not fs_root:
        fs_root = str(Path.cwd() / "data" / "filesystem")
        print(f"Using default filesystem root: {fs_root}")

    # Create sample data
    create_sample_database(os.path.abspath(db_path))
    print()
    create_sample_filesystem(os.path.abspath(fs_root))

    print()
    print("=" * 60)
    print("Setup complete!")
    print("=" * 60)
    print()
    print("Update your .env file with:")
    print(f"MCP_SQLITE_DB_PATH={os.path.abspath(db_path)}")
    print(f"MCP_FILESYSTEM_ROOT={os.path.abspath(fs_root)}")


if __name__ == "__main__":
    main()
