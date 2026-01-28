# RAG System - Phase 1: Foundation

A production-ready Retrieval Augmented Generation (RAG) system built with LangChain, ChromaDB, and Claude.

## 🎯 Overview

This is Phase 1 of a comprehensive RAG system that will eventually include:
- **Phase 1**: RAG Foundation (✅ Current)
- **Phase 2**: Enhanced RAG with Agent Reasoning
- **Phase 3**: MCP Integration
- **Phase 4**: LangGraph Orchestration
- **Phase 5**: Production Hardening

## 🏗️ Architecture

```
Documents → Loader → Splitter → Embedder → Vector Store
                                                ↓
User Query → Embedder → Retriever → Context → LLM → Answer
```

### Components

1. **Ingestion Pipeline**
   - `loaders.py`: Multi-format document loading (PDF, DOCX, TXT, MD)
   - `splitters.py`: Intelligent recursive text splitting
   - `embedder.py`: OpenAI embedding generation

2. **Storage Layer**
   - `vector_store.py`: ChromaDB vector storage with persistence

3. **Retrieval Layer**
   - `retriever.py`: Similarity search with score thresholding

4. **Generation Layer**
   - `prompts.py`: RAG-optimized prompt templates
   - `llm.py`: Claude Sonnet 4 integration with streaming

5. **Pipeline Orchestration**
   - `pipeline.py`: End-to-end RAG workflow

## 📋 Prerequisites

- Python 3.10+
- OpenAI API key (for embeddings)
- Anthropic API key (for Claude)

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or navigate to project
cd rag-system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

```bash
# Copy environment template
cp .env.template .env

# Edit .env and add your API keys
# OPENAI_API_KEY=sk-...
# ANTHROPIC_API_KEY=sk-ant-...
```

### 3. Run Example

```bash
python example_usage.py
```

This will:
1. Initialize the RAG pipeline
2. Ingest sample documents
3. Run example queries
4. Show streaming responses

## 💻 Usage

### Basic Usage

```python
from src.pipeline import RAGPipeline

# Initialize pipeline
pipeline = RAGPipeline(collection_name="my_docs")

# Ingest documents
pipeline.ingest_documents(directory_path="./data/raw")

# Query
result = pipeline.query("What is the main topic?")
print(result['answer'])
```

### Advanced Usage

```python
# Query with custom retrieval settings
result = pipeline.query(
    question="Explain the methodology",
    top_k=5,  # Retrieve more documents
    return_sources=True,  # Include source citations
    return_context=True,  # Include retrieved context
)

# Access detailed results
print(f"Answer: {result['answer']}")
print(f"Sources: {result['num_sources']}")
for source in result['sources']:
    print(f"  - {source['metadata']['file_name']} (score: {source['score']:.3f})")
```

### Streaming Responses

```python
# Stream tokens as they're generated
for token in pipeline.query_stream("Summarize the key findings"):
    print(token, end='', flush=True)
```

## ⚙️ Configuration

Edit `src/config/settings.py` or use environment variables:

### Embedding Settings
- `EMBEDDING_MODEL`: OpenAI model (default: `text-embedding-3-small`)
- `EMBEDDING_DIMENSIONS`: Vector dimensions (default: `1536`)

### Text Splitting
- `CHUNK_SIZE`: Maximum tokens per chunk (default: `1000`)
- `CHUNK_OVERLAP`: Overlap between chunks (default: `200`)

### Retrieval
- `RETRIEVAL_TOP_K`: Number of documents to retrieve (default: `4`)
- `RETRIEVAL_SCORE_THRESHOLD`: Minimum similarity score (default: `0.7`)
- `SEARCH_TYPE`: Search algorithm (`similarity` or `mmr`)

### LLM
- `LLM_MODEL`: Claude model (default: `claude-sonnet-4-20250514`)
- `LLM_TEMPERATURE`: Generation temperature (default: `0.0`)
- `LLM_MAX_TOKENS`: Maximum response tokens (default: `2048`)

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_ingestion.py

# Run with coverage
pytest --cov=src tests/
```

## 📊 Performance

### Typical Metrics (Phase 1)

- **Query Latency**: ~2-3 seconds end-to-end
- **Embedding Cost**: ~$0.02 per 1M tokens
- **Storage**: Local (ChromaDB), ~0 cost
- **LLM Cost**: ~$3 per 1M input tokens

### Optimization Tips

1. **Chunk Size**: Tune based on document type
   - Technical docs: 500-800 tokens
   - Narrative text: 1000-1500 tokens

2. **Top-K**: Balance precision vs recall
   - Precise answers: k=3-4
   - Comprehensive coverage: k=5-8

3. **Score Threshold**: Filter noise
   - High precision: 0.8+
   - Balanced: 0.7
   - High recall: 0.6

## 🔍 Common Issues

### "No relevant documents found"
- **Cause**: Query embedding doesn't match document embeddings
- **Fix**: Use more specific queries or lower score threshold

### "Context too long"
- **Cause**: Retrieved chunks exceed LLM context window
- **Fix**: Reduce `top_k` or `chunk_size`

### "API Key Error"
- **Cause**: Missing or invalid API keys
- **Fix**: Check `.env` file has correct keys

## 📁 Project Structure

```
rag-system/
├── src/
│   ├── config/
│   │   └── settings.py          # Configuration management
│   ├── ingestion/
│   │   ├── loaders.py           # Document loading
│   │   ├── splitters.py         # Text chunking
│   │   └── embedder.py          # Embedding generation
│   ├── storage/
│   │   └── vector_store.py      # Vector database
│   ├── retrieval/
│   │   └── retriever.py         # Document retrieval
│   ├── generation/
│   │   ├── prompts.py           # Prompt templates
│   │   └── llm.py               # LLM interface
│   └── pipeline.py              # Main orchestration
├── tests/
│   └── test_ingestion.py        # Unit tests
├── data/
│   ├── raw/                     # Input documents
│   └── chromadb/                # Vector store persistence
├── requirements.txt             # Python dependencies
├── .env.template                # Environment template
└── example_usage.py             # Usage examples
```

## 🎓 Key Learnings for Engineering Teams

### 1. **Text Splitting Strategy**
- Use recursive splitting for semantic coherence
- Always include overlap to preserve context
- Tune chunk size based on domain

### 2. **Embedding Consistency**
- **Critical**: Use same embedding model for documents AND queries
- Mismatched embeddings = poor retrieval

### 3. **Prompt Engineering**
- Ground LLM in retrieved context
- Require citations for verifiability
- Provide "I don't know" escape hatch

### 4. **Metadata Management**
- Preserve source information throughout pipeline
- Enable filtering by metadata (source, date, etc.)

### 5. **Error Handling**
- Validate inputs at each stage
- Graceful degradation (e.g., "no documents found")
- Comprehensive logging

## 🚧 Next Steps (Phase 2)

Phase 2 will add:
- **Query Rewriting**: LLM-enhanced query optimization
- **Multi-Query Retrieval**: Generate query variations
- **Re-ranking**: Improve retrieved document ordering
- **Agent Reasoning**: ReAct-style agent loops

## 📚 Resources

- [LangChain Documentation](https://python.langchain.com/docs/get_started/introduction)
- [Anthropic Claude API](https://docs.anthropic.com/claude/reference/getting-started-with-the-api)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [RAG Best Practices](https://www.anthropic.com/index/claude-2-1-prompting#retrieval-augmented-generation-rag)

## 📝 License

MIT License - see LICENSE file for details

## 🤝 Contributing

This is a teaching project. Focus areas for improvement:
- Additional document loaders (HTML, CSV, etc.)
- Alternative embedding models
- Evaluation metrics and benchmarks
- More comprehensive tests

---

**Status**: Phase 1 Complete ✅  
**Next Phase**: Enhanced RAG with Agent Reasoning
