# RAG System - Quick Reference Guide

## 🚀 Getting Started (5 minutes)

```bash
# 1. Setup
python setup.py

# 2. Configure API keys
# Edit .env file with your OpenAI and Anthropic keys

# 3. Run example
python example_usage.py
```

---

## 📁 Project Structure

```
rag-system/
├── src/                      # Source code
│   ├── config/              # Settings & configuration
│   ├── ingestion/           # Document loading & chunking
│   ├── storage/             # Vector database
│   ├── retrieval/           # Search & retrieval
│   ├── generation/          # LLM & prompts
│   └── pipeline.py          # Main orchestration
├── tests/                   # Unit tests
├── docs/                    # Documentation
├── notebooks/               # Jupyter notebooks
├── data/                    # Data storage
│   ├── raw/                # Input documents (add yours here!)
│   └── chromadb/           # Vector store
└── example_usage.py        # Working example
```

---

## 💻 Essential Commands

### Basic Usage

```python
from src.pipeline import RAGPipeline

# Initialize
pipeline = RAGPipeline()

# Ingest documents
pipeline.ingest_documents(directory_path="./data/raw")

# Query
result = pipeline.query("Your question here?")
print(result['answer'])
```

### With Options

```python
# Query with parameters
result = pipeline.query(
    question="What are the key findings?",
    top_k=5,                    # Retrieve 5 chunks
    return_sources=True,        # Include source info
    return_context=True,        # Include raw context
)

# Access results
print(f"Answer: {result['answer']}")
print(f"Sources: {result['num_sources']}")
for source in result['sources']:
    print(f"  - {source['metadata']['file_name']}: {source['score']:.3f}")
```

### Streaming

```python
# Stream response token by token
for token in pipeline.query_stream("Explain this concept"):
    print(token, end='', flush=True)
```

---

## ⚙️ Configuration

### Via Environment Variables (.env file)

```bash
# Required
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# Optional - Override defaults
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
RETRIEVAL_TOP_K=4
RETRIEVAL_SCORE_THRESHOLD=0.7
LLM_MODEL=claude-sonnet-4-20250514
```

### Via Code

```python
from src.config.settings import settings

# View current settings
print(f"Chunk size: {settings.chunk_size}")
print(f"Top K: {settings.retrieval_top_k}")

# Override in pipeline
from src.ingestion.splitters import DocumentSplitter

splitter = DocumentSplitter(
    chunk_size=500,      # Smaller chunks
    chunk_overlap=100,   # Less overlap
)
```

---

## 🎯 Common Tasks

### Task 1: Add New Documents

```bash
# 1. Add files to data/raw/
cp my_document.pdf data/raw/

# 2. Re-run ingestion
python -c "
from src.pipeline import RAGPipeline
p = RAGPipeline()
p.ingest_documents(directory_path='./data/raw')
"
```

### Task 2: Reset Vector Store

```python
# Start fresh
pipeline = RAGPipeline(reinitialize=True)

# Or manually
pipeline.reset()
```

### Task 3: Query Specific Source

```python
# Filter by metadata (Phase 2 will enhance this)
from src.retrieval.retriever import AdvancedRetriever

retriever = AdvancedRetriever(vector_store=pipeline.vector_store)

# Retrieve from specific file
docs = retriever.retrieve(
    query="your question",
    filter={'file_name': 'specific.pdf'}
)
```

### Task 4: Export Results

```python
import json

result = pipeline.query("question", return_sources=True)

# Save to JSON
with open('results.json', 'w') as f:
    json.dump(result, f, indent=2)
```

---

## 🔍 Debugging

### Check Vector Store Contents

```python
stats = pipeline.get_stats()
print(f"Documents: {stats['vector_store']['document_count']}")
```

### Inspect Retrieved Chunks

```python
result = pipeline.query(
    "your question",
    return_context=True,
    return_sources=True,
)

# See what was sent to LLM
print("Context:")
print(result['context'])

# See retrieval scores
for source in result['sources']:
    print(f"Score {source['score']:.3f}: {source['content'][:100]}...")
```

### View Logs

```python
import logging

logging.basicConfig(level=logging.DEBUG)

# Now run your queries - you'll see detailed logs
```

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_ingestion.py::TestDocumentLoader

# With coverage
pytest --cov=src tests/

# Verbose output
pytest -v tests/
```

---

## 📊 Performance Tuning

### Faster Queries (Lower Quality)

```python
pipeline = RAGPipeline()
result = pipeline.query(
    question="...",
    top_k=2,  # Fewer chunks = faster
)
```

### Better Quality (Slower)

```python
pipeline = RAGPipeline()
result = pipeline.query(
    question="...",
    top_k=8,  # More chunks = better context
)
```

### Optimize Chunk Size

```python
# For technical docs
splitter = DocumentSplitter(chunk_size=500, chunk_overlap=100)

# For narrative text
splitter = DocumentSplitter(chunk_size=1500, chunk_overlap=300)
```

---

## 🐛 Common Issues

### "No module named 'src'"

```bash
# Add project to Python path
export PYTHONPATH="${PYTHONPATH}:/path/to/rag-system"

# Or in code
import sys
sys.path.insert(0, '/path/to/rag-system')
```

### "API Key Error"

```bash
# Check .env file exists and has keys
cat .env

# Verify keys are loaded
python -c "from src.config.settings import settings; print(settings.openai_api_key[:10])"
```

### "ChromaDB Error"

```bash
# Delete and recreate
rm -rf data/chromadb/*

# Reinitialize
python -c "from src.pipeline import RAGPipeline; RAGPipeline(reinitialize=True)"
```

---

## 📈 Cost Estimation

### Per 1000 Queries

Assuming average query retrieves 4 chunks (4000 tokens):

```
Embeddings:
  - Query embedding: 1000 * 50 tokens = $0.001
  
LLM Generation:
  - Input: 1000 * 4000 tokens = $12
  - Output: 1000 * 500 tokens = $15
  
Total: ~$27 per 1000 queries
```

### Per 1000 Documents Ingested

Assuming average document = 10 pages = 20,000 tokens = 20 chunks:

```
Embeddings:
  - 1000 docs * 20 chunks * 1000 tokens = $0.40
  
Storage:
  - ChromaDB local: $0
  
Total: ~$0.40 per 1000 documents
```

---

## 🎓 Learning Path

1. **Week 1**: Understand architecture (read TEAM_GUIDE.md)
2. **Week 2**: Run examples, modify prompts
3. **Week 3**: Ingest your own data, tune parameters
4. **Week 4**: Implement custom features
5. **Week 5**: Prepare for Phase 2 (agents)

---

## 🔗 Important Files

| File | Purpose |
|------|---------|
| `README.md` | Project overview & setup |
| `TEAM_GUIDE.md` | Deep dive for engineers |
| `example_usage.py` | Working code examples |
| `src/pipeline.py` | Main API |
| `src/config/settings.py` | Configuration |
| `notebooks/exploration.ipynb` | Interactive tutorial |

---

## 🆘 Getting Help

1. **Check logs**: Set `LOG_LEVEL=DEBUG` in .env
2. **Read error messages**: They're usually informative
3. **Review TEAM_GUIDE.md**: Covers common issues
4. **Inspect source code**: Well-commented
5. **Run tests**: `pytest -v tests/`

---

## 🚀 Next Steps

Ready for more? Here's what's coming in future phases:

- **Phase 2**: Query rewriting, re-ranking, multi-query
- **Phase 3**: MCP integration with Google Drive, databases
- **Phase 4**: LangGraph workflows, multi-agent systems
- **Phase 5**: Production deployment, monitoring, evaluation

---

**Happy Building!** 🎉
