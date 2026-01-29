"""
Demo script for QueryRewriter functionality (Phase 2).

Shows how query rewriting improves search quality by optimizing
user queries before retrieval.
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.query_rewriter import QueryRewriter


def demo_query_rewriting():
    """Demonstrate query rewriting capabilities."""
    print("=" * 70)
    print("Query Rewriting Demo - Phase 2")
    print("=" * 70)
    print()

    # Initialize rewriter
    print("Initializing QueryRewriter...")
    rewriter = QueryRewriter(temperature=0.3)
    print(f"✓ Ready (temperature={rewriter.temperature})")
    print()

    # Example queries that benefit from rewriting
    test_queries = [
        "What's RAG?",
        "How do I setup the system?",
        "API docs",
        "Tell me about ML models",
        "vector db",
        "What r the best practices for chunking?",
    ]

    print("Testing query rewrites...")
    print("-" * 70)

    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}. Original: \"{query}\"")

        # Rewrite query
        rewritten = rewriter.rewrite(query)

        print(f"   Rewritten: \"{rewritten}\"")

        # Show if unchanged
        if rewritten == query:
            print("   → No changes needed (query already well-formed)")

    print()
    print("-" * 70)
    print()

    # Show configuration
    config = rewriter.get_rewriter_config()
    print("Rewriter Configuration:")
    print(f"  Temperature: {config['temperature']}")
    print(f"  Max Tokens: {config['max_tokens']}")
    print(f"  LLM Model: {config['llm_config']['model']}")
    print()

    print("=" * 70)
    print("✓ Demo completed!")
    print()
    print("Key Benefits:")
    print("  • Expands abbreviations (RAG → Retrieval Augmented Generation)")
    print("  • Adds specificity to vague queries")
    print("  • Removes conversational filler")
    print("  • Improves semantic search accuracy")
    print("=" * 70)


if __name__ == "__main__":
    demo_query_rewriting()
