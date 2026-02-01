"""
Demo script for Multi-Query Generation (Phase 2).

Shows how generating multiple query variations improves
retrieval recall by exploring different angles of the same question.
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.multi_query import MultiQueryGenerator, generate_query_variations


def demo_multi_query_generation():
    """Demonstrate multi-query generation capabilities."""
    print("=" * 70)
    print("Multi-Query Generation Demo - Phase 2")
    print("=" * 70)
    print()

    # Initialize generator
    print("Initializing MultiQueryGenerator...")
    generator = MultiQueryGenerator(temperature=0.7)
    print(f"✓ Ready (temperature={generator.temperature}, diversity mode)")
    print()

    # Example queries to expand
    test_queries = [
        "What is RAG?",
        "How do I setup vector stores?",
        "Best practices for chunking",
    ]

    print("Generating query variations...")
    print("-" * 70)

    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}. Original Query: \"{query}\"")
        print()

        # Generate variations
        variations = generator.generate_queries(query, num_queries=3)

        print(f"   Generated {len(variations)} total queries:")
        for j, variation in enumerate(variations, 1):
            if j == 1:
                print(f"   {j}. {variation} ← (original)")
            else:
                print(f"   {j}. {variation}")

    print()
    print("-" * 70)
    print()

    # Show how variations explore different angles
    print("How Query Variations Improve Retrieval:")
    print()
    print("Single Query Approach:")
    print("  • 'What is RAG?' → Retrieves 4 docs about RAG")
    print()
    print("Multi-Query Approach:")
    print("  • 'What is RAG?'")
    print("  • 'What is Retrieval Augmented Generation?'")
    print("  • 'How does RAG combine retrieval and LLMs?'")
    print("  • 'What are the components of a RAG system?'")
    print("  → Retrieves 4 docs × 4 queries = up to 16 unique docs")
    print("  → Better coverage of the topic!")
    print()

    # Show configuration
    config = generator.get_generator_config()
    print("Generator Configuration:")
    print(f"  Temperature: {config['temperature']} (higher = more diversity)")
    print(f"  Max Tokens: {config['max_tokens']}")
    print(f"  LLM Model: {config['llm_config']['model']}")
    print()

    # Show convenience function
    print("Using the Convenience Function:")
    print("  from src.agents.multi_query import generate_query_variations")
    print("  queries = generate_query_variations('What is ML?', num_queries=3)")
    print()

    print("=" * 70)
    print("✓ Demo completed!")
    print()
    print("Key Benefits:")
    print("  • Explores question from multiple angles")
    print("  • Improves recall (finds more relevant documents)")
    print("  • Uses different keywords and phrasings")
    print("  • Automatic deduplication of results")
    print("  • Configurable number of variations")
    print("=" * 70)


if __name__ == "__main__":
    demo_multi_query_generation()
