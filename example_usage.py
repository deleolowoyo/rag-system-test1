"""
Example usage of the RAG system.
This script demonstrates the complete workflow from ingestion to querying.
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline import RAGPipeline
from src.config.settings import settings


def main():
    """Main example workflow."""
    
    print("=" * 80)
    print("RAG SYSTEM - EXAMPLE USAGE")
    print("=" * 80)
    print()
    
    # Step 1: Initialize Pipeline
    print("Step 1: Initializing RAG Pipeline...")
    pipeline = RAGPipeline(
        collection_name="example_docs",
        reinitialize=True,  # Start fresh
    )
    print("✓ Pipeline initialized")
    print()
    
    # Step 2: Ingest Documents
    print("Step 2: Ingesting documents...")
    
    # Create sample documents directory
    data_dir = Path("./data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a sample document
    sample_doc = data_dir / "sample.txt"
    if not sample_doc.exists():
        sample_doc.write_text("""
# Sample Document for RAG System

## Introduction
This is a sample document to demonstrate the RAG (Retrieval Augmented Generation) system.
The system uses advanced natural language processing to answer questions based on document content.

## Key Features
1. Document Ingestion: Supports PDF, DOCX, TXT, and MD files
2. Intelligent Chunking: Splits documents while preserving context
3. Vector Search: Uses embeddings for semantic similarity
4. Citation: Always cites sources in responses

## Technical Details
The system uses:
- OpenAI embeddings (text-embedding-3-small)
- ChromaDB for vector storage
- Claude Sonnet 4 for generation
- LangChain for orchestration

## Use Cases
Common applications include:
- Knowledge base Q&A
- Document analysis
- Research assistance
- Customer support automation
""")
        print(f"✓ Created sample document: {sample_doc}")
    
    # Ingest documents
    stats = pipeline.ingest_documents(
        directory_path=str(data_dir)
    )
    
    print(f"✓ Documents ingested successfully")
    print(f"  - Files processed: {stats['documents_loaded']}")
    print(f"  - Chunks created: {stats['chunks_created']}")
    print(f"  - Avg chunk size: {stats['chunk_stats']['avg_chunk_size']:.0f} chars")
    print()
    
    # Step 3: Query the System
    print("Step 3: Querying the RAG system...")
    print()
    
    # Example queries
    queries = [
        "What are the key features of the RAG system?",
        "What technologies does the system use?",
        "What are some use cases for this system?",
        "How does the chunking work?",  # Test retrieval quality
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"Query {i}: {query}")
        print("-" * 80)
        
        result = pipeline.query(
            question=query,
            return_sources=True,
        )
        
        print(f"Answer: {result['answer']}")
        print()
        print(f"Sources used: {result['num_sources']}")
        
        # Show source details
        if result.get('sources'):
            for j, source in enumerate(result['sources'], 1):
                print(f"  Source {j}:")
                print(f"    - File: {source['metadata'].get('file_name', 'unknown')}")
                print(f"    - Score: {source['score']:.3f}")
                print(f"    - Preview: {source['content'][:100]}...")
        
        print()
        print("=" * 80)
        print()
    
    # Step 4: System Stats
    print("Step 4: System Statistics")
    print("-" * 80)
    stats = pipeline.get_stats()
    
    print(f"Vector Store: {stats['vector_store']['document_count']} documents")
    print(f"Retrieval: top_k={stats['retriever']['top_k']}, "
          f"threshold={stats['retriever']['score_threshold']}")
    print(f"LLM: {stats['llm']['model']}, temp={stats['llm']['temperature']}")
    print()
    
    # Step 5: Streaming Example
    print("Step 5: Streaming Query Example")
    print("-" * 80)
    
    streaming_query = "Explain how the RAG system works in simple terms."
    print(f"Query: {streaming_query}")
    print()
    print("Streaming response:")
    
    for token in pipeline.query_stream(streaming_query):
        print(token, end='', flush=True)
    
    print()
    print()
    print("=" * 80)
    print("Example completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    # Check for API keys
    if not settings.openai_api_key or settings.openai_api_key == "your_openai_api_key_here":
        print("ERROR: Please set OPENAI_API_KEY in .env file")
        print("Copy .env.template to .env and add your API key")
        sys.exit(1)
    
    if not settings.anthropic_api_key or settings.anthropic_api_key == "your_anthropic_api_key_here":
        print("ERROR: Please set ANTHROPIC_API_KEY in .env file")
        print("Copy .env.template to .env and add your API key")
        sys.exit(1)
    
    main()
