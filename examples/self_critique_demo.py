"""
Demo script for Self-Critique Agent functionality (Phase 2).

Shows how the Self-Critique agent validates generated answers by:
- Checking relevance to the question
- Verifying source grounding
- Detecting hallucinations
- Assessing quality
- Providing improvement suggestions
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_core.documents import Document
from src.agents import SelfCritiqueAgent


def demo_good_answer():
    """Demonstrate critique of a good, well-supported answer."""
    print("=" * 70)
    print("Self-Critique Agent Demo - Good Answer")
    print("=" * 70)
    print()

    # Initialize agent
    print("Initializing SelfCritiqueAgent...")
    agent = SelfCritiqueAgent(temperature=0.0)
    print(f"✓ Agent ready (temperature={agent.temperature})")
    print()

    # Sample question and context
    question = "What is Retrieval Augmented Generation (RAG)?"

    context = [
        Document(
            page_content=(
                "Retrieval Augmented Generation (RAG) is a technique that combines "
                "information retrieval with language model generation. It works by "
                "first retrieving relevant documents from a knowledge base, then "
                "using those documents as context for the LLM to generate more "
                "accurate, grounded responses. This approach significantly reduces "
                "hallucinations and enables LLMs to access up-to-date information."
            ),
            metadata={'source': 'rag_guide.txt', 'score': 0.95}
        ),
        Document(
            page_content=(
                "The key components of RAG are: 1) Document retrieval using vector "
                "embeddings for semantic search, 2) Context formatting to prepare "
                "retrieved documents for the LLM, and 3) LLM generation with the "
                "retrieved context. RAG is particularly effective for question-answering "
                "systems where accuracy and source attribution are important."
            ),
            metadata={'source': 'rag_architecture.txt', 'score': 0.89}
        ),
    ]

    # Good answer that's well-supported
    answer = (
        "Retrieval Augmented Generation (RAG) is a technique that combines information "
        "retrieval with language model generation. It first retrieves relevant documents "
        "from a knowledge base using vector embeddings for semantic search, then uses "
        "those documents as context for the LLM to generate accurate, grounded responses. "
        "The key benefit is that RAG significantly reduces hallucinations and enables "
        "LLMs to access up-to-date information. RAG is particularly effective for "
        "question-answering systems where accuracy and source attribution are important."
    )

    print(f"Question: \"{question}\"")
    print()
    print("Answer:")
    print("-" * 70)
    print(answer)
    print("-" * 70)
    print()

    print("Context Documents:")
    for i, doc in enumerate(context, 1):
        print(f"{i}. {doc.page_content[:80]}...")
    print()

    # Critique the answer
    print("Running critique...")
    critique = agent.critique(question, answer, context)
    print()

    # Display results
    print("Critique Results:")
    print("=" * 70)
    print(agent.get_critique_summary(critique))
    print()

    # Check if refinement is needed
    should_refine = agent.should_refine(critique)
    print(f"Should Refine Answer: {should_refine}")
    print()


def demo_poor_answer():
    """Demonstrate critique of a poor answer with issues."""
    print("=" * 70)
    print("Self-Critique Agent Demo - Poor Answer")
    print("=" * 70)
    print()

    agent = SelfCritiqueAgent(temperature=0.0)

    question = "What is Retrieval Augmented Generation (RAG)?"

    context = [
        Document(
            page_content=(
                "Retrieval Augmented Generation (RAG) is a technique that combines "
                "information retrieval with language model generation. It works by "
                "first retrieving relevant documents from a knowledge base."
            ),
            metadata={'source': 'rag_guide.txt'}
        ),
    ]

    # Poor answer - off topic, not supported, potential hallucination
    answer = (
        "RAG is a revolutionary AI system developed in 2023 by OpenAI. It can "
        "generate images, videos, and text simultaneously. RAG uses quantum "
        "computing to achieve speeds 1000x faster than traditional systems. "
        "It's primarily used for gaming and entertainment applications."
    )

    print(f"Question: \"{question}\"")
    print()
    print("Answer (contains issues):")
    print("-" * 70)
    print(answer)
    print("-" * 70)
    print()

    print("Context Documents:")
    for i, doc in enumerate(context, 1):
        print(f"{i}. {doc.page_content[:80]}...")
    print()

    # Critique the answer
    print("Running critique...")
    critique = agent.critique(question, answer, context)
    print()

    # Display results
    print("Critique Results:")
    print("=" * 70)
    print(agent.get_critique_summary(critique))
    print()

    # Check if refinement is needed
    should_refine = agent.should_refine(critique)
    print(f"Should Refine Answer: {should_refine}")
    print()

    if should_refine:
        print("⚠️  This answer should be refined before returning to user")
    print()


def demo_fair_answer():
    """Demonstrate critique of a fair answer that could be improved."""
    print("=" * 70)
    print("Self-Critique Agent Demo - Fair Answer")
    print("=" * 70)
    print()

    agent = SelfCritiqueAgent(temperature=0.0)

    question = "How does RAG reduce hallucinations in LLMs?"

    context = [
        Document(
            page_content=(
                "RAG reduces hallucinations by grounding LLM responses in retrieved "
                "documents from a knowledge base. Instead of relying solely on "
                "parametric knowledge learned during training, the LLM can reference "
                "actual source documents when generating answers. This provides a "
                "factual foundation and reduces the tendency to generate plausible "
                "but incorrect information."
            ),
            metadata={'source': 'rag_benefits.txt'}
        ),
        Document(
            page_content=(
                "The retrieval step in RAG ensures that only relevant, factual "
                "documents are provided as context. The LLM is then instructed to "
                "base its response on these documents, creating a citation-based "
                "approach that minimizes unsupported claims."
            ),
            metadata={'source': 'rag_mechanism.txt'}
        ),
    ]

    # Fair answer - addresses question but lacks depth and citations
    answer = (
        "RAG helps reduce hallucinations by using retrieved documents. The LLM "
        "can reference actual sources when generating answers, which makes the "
        "responses more accurate."
    )

    print(f"Question: \"{question}\"")
    print()
    print("Answer (could be improved):")
    print("-" * 70)
    print(answer)
    print("-" * 70)
    print()

    print("Context Documents:")
    for i, doc in enumerate(context, 1):
        print(f"{i}. {doc.page_content[:80]}...")
    print()

    # Critique the answer
    print("Running critique...")
    critique = agent.critique(question, answer, context)
    print()

    # Display results
    print("Critique Results:")
    print("=" * 70)
    print(agent.get_critique_summary(critique))
    print()

    # Check if refinement is needed
    should_refine = agent.should_refine(critique)
    print(f"Should Refine Answer: {should_refine}")
    print()

    if should_refine:
        print("💡 Improvements suggested - consider refining before returning")
    print()


def demo_agent_config():
    """Demonstrate agent configuration."""
    print("=" * 70)
    print("Self-Critique Agent Demo - Configuration")
    print("=" * 70)
    print()

    # Create agent with custom config
    print("Creating agent with custom configuration...")
    agent = SelfCritiqueAgent(
        temperature=0.0,
        max_tokens=500,
        max_context_length=3000,
    )

    config = agent.get_agent_config()

    print("Agent Configuration:")
    print(f"  - Temperature: {config['temperature']}")
    print(f"  - Max tokens: {config['max_tokens']}")
    print(f"  - Max context length: {config['max_context_length']}")
    print(f"  - LLM model: {config['llm_config']['model']}")
    print()


def main():
    """Run all demos."""
    try:
        # Demo 1: Good answer
        demo_good_answer()

        print("\n" + "=" * 70)
        print("Press Enter to continue to Poor Answer demo...")
        input()
        print()

        # Demo 2: Poor answer
        demo_poor_answer()

        print("\n" + "=" * 70)
        print("Press Enter to continue to Fair Answer demo...")
        input()
        print()

        # Demo 3: Fair answer
        demo_fair_answer()

        print("\n" + "=" * 70)
        print("Press Enter to continue to Configuration demo...")
        input()
        print()

        # Demo 4: Configuration
        demo_agent_config()

        print("=" * 70)
        print("✅ Demo Complete!")
        print()
        print("Key Takeaways:")
        print("- Self-critique validates answer quality before returning to users")
        print("- Evaluates relevance, grounding, citations, and hallucination risk")
        print("- Provides actionable improvement suggestions")
        print("- should_refine() method determines if answer needs refinement")
        print("- Acts as a quality gate for RAG system outputs")
        print("=" * 70)

    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
    except Exception as e:
        print(f"\n\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
