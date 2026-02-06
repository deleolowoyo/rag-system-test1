"""
Demo script for ReAct Agent functionality (Phase 2).

Shows how the ReAct agent performs multi-step reasoning and acting
to answer complex queries through iterative refinement.

Note: This demo shows the basic structure. Full reasoning logic
will be enhanced in subsequent steps.
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_core.documents import Document
from src.agents import ReActAgent, AgentAction
from src.agents.query_rewriter import QueryRewriter
from src.generation.llm import LLMGenerator


# Mock retriever for demonstration
class MockRetriever:
    """Mock retriever that returns sample documents."""

    def retrieve(self, query: str):
        """Return mock documents based on query."""
        # Simulate different results for different queries
        if "RAG" in query or "Retrieval Augmented Generation" in query:
            return [
                Document(
                    page_content=(
                        "RAG (Retrieval Augmented Generation) is a technique that "
                        "combines information retrieval with language model generation. "
                        "It retrieves relevant documents and uses them as context for "
                        "the LLM to generate more accurate, grounded responses."
                    ),
                    metadata={'source': 'rag_guide.txt', 'score': 0.95}
                ),
                Document(
                    page_content=(
                        "The key components of RAG are: 1) Document retrieval using "
                        "vector embeddings, 2) Context formatting, and 3) LLM generation "
                        "with retrieved context. This approach reduces hallucinations."
                    ),
                    metadata={'source': 'rag_architecture.txt', 'score': 0.89}
                ),
            ]
        elif "vector" in query.lower() or "embedding" in query.lower():
            return [
                Document(
                    page_content=(
                        "Vector embeddings are dense numerical representations of text "
                        "that capture semantic meaning. They enable similarity search "
                        "in high-dimensional space using distance metrics like cosine."
                    ),
                    metadata={'source': 'embeddings.txt', 'score': 0.87}
                ),
            ]
        else:
            return []

    def get_retriever_config(self):
        return {'type': 'mock', 'top_k': 3}


def demo_basic_react_loop():
    """Demonstrate basic ReAct agent loop."""
    print("=" * 70)
    print("ReAct Agent Demo - Basic Loop")
    print("=" * 70)
    print()

    # Initialize components
    print("Initializing ReAct Agent...")
    retriever = MockRetriever()
    query_rewriter = QueryRewriter()
    llm = LLMGenerator(temperature=0.3, max_tokens=500)

    agent = ReActAgent(
        retriever=retriever,
        query_rewriter=query_rewriter,
        llm=llm,
        max_iterations=5,
    )

    print(f"✓ Agent ready (max_iterations={agent.max_iterations})")
    print()

    # Run agent on a query
    query = "What's RAG?"
    print(f"Query: \"{query}\"")
    print("-" * 70)
    print()

    print("Starting ReAct loop...")
    print("(This will show the agent's reasoning and action steps)")
    print()

    try:
        result = agent.run(query)

        # Display results
        print("\n" + "=" * 70)
        print("Results")
        print("=" * 70)
        print()

        print(f"Iterations taken: {result['iterations']}")
        print(f"Documents retrieved: {len(result['documents'])}")
        print(f"Query evolution: {len(result['query_evolution'])} versions")
        print()

        print("Query Evolution:")
        for i, q in enumerate(result['query_evolution'], 1):
            print(f"  {i}. \"{q}\"")
        print()

        print("Actions Taken:")
        for i, action in enumerate(result['actions_taken'], 1):
            print(f"  {i}. {action}")
        print()

        print("Reasoning Trace:")
        for i, thought in enumerate(result['reasoning_trace'], 1):
            print(f"  {i}. {thought[:100]}...")
        print()

        print("Final Answer:")
        print("-" * 70)
        print(result['answer'])
        print()

    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()


def demo_agent_state_tracking():
    """Demonstrate agent state tracking during execution."""
    print("=" * 70)
    print("ReAct Agent Demo - State Tracking")
    print("=" * 70)
    print()

    from src.agents.react_agent import AgentState

    # Create and manipulate agent state
    print("Creating agent state...")
    state = AgentState(
        original_query="What is machine learning?",
        current_query="What is machine learning?",
    )

    print(f"✓ Initial state created")
    print(f"  - Original query: \"{state.original_query}\"")
    print(f"  - Documents: {len(state.documents)}")
    print(f"  - Iterations: {state.iterations}")
    print()

    # Simulate agent actions
    print("Simulating agent actions...")
    print()

    print("1. REWRITE_QUERY")
    state.add_action(
        AgentAction.REWRITE_QUERY,
        "Query is too broad, rewriting for specificity"
    )
    state.update_query("What is machine learning and how does it work?")
    print(f"   ✓ Query updated to: \"{state.current_query}\"")
    print()

    print("2. SEARCH")
    state.add_action(
        AgentAction.SEARCH,
        "Searching with optimized query"
    )
    mock_docs = [
        Document(page_content="ML is a subset of AI...", metadata={'source': 'ml.txt'}),
        Document(page_content="ML algorithms learn from data...", metadata={'source': 'ml2.txt'}),
    ]
    state.add_documents(mock_docs)
    print(f"   ✓ Retrieved {len(mock_docs)} documents")
    print()

    print("3. ANSWER")
    state.add_action(
        AgentAction.ANSWER,
        "Have enough information, generating answer"
    )
    state.increment_iteration()
    print(f"   ✓ Ready to generate answer")
    print()

    # Show final state
    print("Final State:")
    print("-" * 70)
    print(f"Iterations: {state.iterations}")
    print(f"Documents: {len(state.documents)}")
    print(f"Actions taken: {[a.value for a in state.actions]}")
    print(f"Query evolution: {state.query_evolution}")
    print()


def demo_agent_config():
    """Demonstrate agent configuration."""
    print("=" * 70)
    print("ReAct Agent Demo - Configuration")
    print("=" * 70)
    print()

    retriever = MockRetriever()

    # Create agent with custom config
    print("Creating agent with custom configuration...")
    agent = ReActAgent(
        retriever=retriever,
        max_iterations=3,
        temperature=0.5,
    )

    config = agent.get_agent_config()

    print("Agent Configuration:")
    print(f"  - Max iterations: {config['max_iterations']}")
    print(f"  - Temperature: {config['temperature']}")
    print(f"  - Has query rewriter: {config['has_query_rewriter']}")
    print(f"  - LLM model: {config['llm_config']['model']}")
    print()


def main():
    """Run all demos."""
    try:
        # Demo 1: Basic ReAct loop
        demo_basic_react_loop()

        print("\n" + "=" * 70)
        print("Press Enter to continue to State Tracking demo...")
        input()
        print()

        # Demo 2: State tracking
        demo_agent_state_tracking()

        print("\n" + "=" * 70)
        print("Press Enter to continue to Configuration demo...")
        input()
        print()

        # Demo 3: Configuration
        demo_agent_config()

        print("=" * 70)
        print("✅ Demo Complete!")
        print()
        print("Key Takeaways:")
        print("- ReAct agents iteratively reason and act to answer queries")
        print("- State tracking provides full visibility into agent decisions")
        print("- Agents can rewrite queries and search multiple times")
        print("- Final answers are grounded in retrieved documents")
        print("=" * 70)

    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
    except Exception as e:
        print(f"\n\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
