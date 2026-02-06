"""
Demo script for Document Re-Ranking functionality (Phase 2).

Shows how re-ranking improves retrieval precision by reordering documents
based on LLM-scored relevance and hybrid scoring strategies.
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_core.documents import Document
from src.reranking import LLMReranker, HybridReranker


def demo_llm_reranking():
    """Demonstrate LLM-based re-ranking capabilities."""
    print("=" * 70)
    print("LLM-Based Document Re-Ranking Demo - Phase 2")
    print("=" * 70)
    print()

    # Initialize reranker
    print("Initializing LLMReranker...")
    reranker = LLMReranker(temperature=0.0, max_doc_length=500)
    print(f"✓ Ready (temperature={reranker.temperature}, "
          f"max_doc_length={reranker.max_doc_length})")
    print()

    # Create sample documents (simulating retrieval results)
    query = "What is Retrieval Augmented Generation?"

    documents = [
        Document(
            page_content="RAG (Retrieval Augmented Generation) is a technique that "
                        "combines retrieval of relevant documents with LLM generation. "
                        "It enhances LLM responses with external knowledge.",
            metadata={'source': 'rag_guide.txt', 'page': 1}
        ),
        Document(
            page_content="Vector databases store embeddings for efficient similarity search. "
                        "Popular options include FAISS, Pinecone, and ChromaDB.",
            metadata={'source': 'vector_db.txt', 'page': 2}
        ),
        Document(
            page_content="Machine learning models can be trained using supervised, "
                        "unsupervised, or reinforcement learning approaches.",
            metadata={'source': 'ml_basics.txt', 'page': 5}
        ),
        Document(
            page_content="Retrieval systems use semantic search to find relevant documents "
                        "based on meaning rather than keyword matching. RAG leverages this.",
            metadata={'source': 'retrieval.txt', 'page': 3}
        ),
    ]

    print(f"Query: \"{query}\"")
    print(f"Documents to re-rank: {len(documents)}")
    print("-" * 70)
    print()

    # Score individual document (demonstration)
    print("Example: Scoring a single document")
    print(f"Document: \"{documents[0].page_content[:80]}...\"")
    score = reranker.score_document(query, documents[0])
    print(f"Relevance Score: {score:.1f}/10")
    print()
    print("-" * 70)
    print()

    # Re-rank all documents
    print("Re-ranking all documents...")
    ranked_docs = reranker.rerank(query, documents)
    print()

    # Display results
    print("Re-Ranked Results:")
    print("=" * 70)
    for i, (doc, score) in enumerate(ranked_docs, 1):
        print(f"\n{i}. Score: {score:.1f}/10")
        print(f"   Source: {doc.metadata.get('source', 'unknown')}")
        print(f"   Content: \"{doc.page_content[:100]}...\"")

    print()
    print("=" * 70)
    print(f"✓ Most relevant document: {ranked_docs[0][0].metadata['source']}")
    print(f"✓ Top score: {ranked_docs[0][1]:.1f}/10")
    print()


def demo_hybrid_reranking():
    """Demonstrate hybrid re-ranking (similarity + LLM scores)."""
    print("=" * 70)
    print("Hybrid Re-Ranking Demo (Vector Similarity + LLM)")
    print("=" * 70)
    print()

    # Initialize rerankers
    print("Initializing Hybrid Re-Ranker...")
    llm_reranker = LLMReranker(temperature=0.0, max_doc_length=500)
    hybrid = HybridReranker(
        llm_reranker=llm_reranker,
        llm_weight=0.7,
        similarity_weight=0.3,
    )
    print(f"✓ Ready (LLM weight={hybrid.llm_weight}, "
          f"Similarity weight={hybrid.similarity_weight})")
    print()

    # Create sample documents with simulated similarity scores
    query = "How does vector search work?"

    # Documents with (content, similarity_score) - higher is more similar
    docs_with_scores = [
        (
            Document(
                page_content="Vector search finds similar items by comparing embeddings "
                            "in high-dimensional space using distance metrics like cosine similarity.",
                metadata={'source': 'vector_search.txt'}
            ),
            0.89  # High similarity score
        ),
        (
            Document(
                page_content="Database indexing improves query performance by creating "
                            "data structures that speed up lookups.",
                metadata={'source': 'databases.txt'}
            ),
            0.45  # Low similarity score (less relevant)
        ),
        (
            Document(
                page_content="Embeddings are dense vector representations of text that capture "
                            "semantic meaning for machine learning tasks.",
                metadata={'source': 'embeddings.txt'}
            ),
            0.72  # Medium similarity score
        ),
    ]

    print(f"Query: \"{query}\"")
    print(f"Documents with similarity scores: {len(docs_with_scores)}")
    print()

    print("Original Vector Similarity Ranking:")
    print("-" * 70)
    for i, (doc, sim_score) in enumerate(docs_with_scores, 1):
        print(f"{i}. Similarity: {sim_score:.2f}")
        print(f"   Source: {doc.metadata['source']}")
        print(f"   Content: \"{doc.page_content[:80]}...\"")
        print()

    print("-" * 70)
    print()

    # Re-rank using hybrid approach
    print("Re-ranking with hybrid scoring (combining similarity + LLM)...")
    ranked_docs = hybrid.rerank(query, docs_with_scores)
    print()

    # Display hybrid results
    print("Hybrid Re-Ranked Results:")
    print("=" * 70)
    for i, (doc, combined_score) in enumerate(ranked_docs, 1):
        print(f"\n{i}. Combined Score: {combined_score:.2f}/10")
        print(f"   Source: {doc.metadata['source']}")
        print(f"   Content: \"{doc.page_content[:80]}...\"")

    print()
    print("=" * 70)
    print("✓ Hybrid re-ranking balances fast vector similarity with accurate LLM scoring")
    print()


def main():
    """Run all demos."""
    try:
        # Demo 1: LLM-based re-ranking
        demo_llm_reranking()

        print("\n" + "=" * 70)
        print("Press Enter to continue to Hybrid Re-Ranking demo...")
        input()
        print()

        # Demo 2: Hybrid re-ranking
        demo_hybrid_reranking()

        print("\n" + "=" * 70)
        print("✅ Demo Complete!")
        print()
        print("Key Takeaways:")
        print("- LLM re-ranking provides accurate relevance scoring (0-10 scale)")
        print("- Hybrid re-ranking combines vector similarity with LLM scoring")
        print("- Re-ranking improves retrieval precision for better RAG results")
        print("=" * 70)

    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
    except Exception as e:
        print(f"\n\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
