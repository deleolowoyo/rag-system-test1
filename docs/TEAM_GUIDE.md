# RAG System - Engineering Team Training Guide

## 🎯 Purpose

This guide helps engineering teams understand the RAG system architecture, make informed decisions, and build production-ready applications.

---

## 📚 Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Key Design Decisions](#key-design-decisions)
3. [Common Patterns](#common-patterns)
4. [Troubleshooting Guide](#troubleshooting-guide)
5. [Performance Optimization](#performance-optimization)
6. [Security & Privacy](#security--privacy)
7. [Testing Strategy](#testing-strategy)

---

## Architecture Overview

### System Flow

```
INPUT → INGESTION → STORAGE → RETRIEVAL → GENERATION → OUTPUT
```

### Component Responsibilities

| Component | Responsibility | Key Classes |
|-----------|---------------|-------------|
| **Ingestion** | Load & chunk documents | `DocumentLoader`, `DocumentSplitter` |
| **Storage** | Vector persistence | `VectorStoreManager` |
| **Retrieval** | Find relevant chunks | `AdvancedRetriever` |
| **Generation** | Synthesize answers | `LLMGenerator` |
| **Pipeline** | Orchestrate flow | `RAGPipeline` |

---

## Key Design Decisions

### 1. Why Recursive Text Splitting?

**Decision**: Use `RecursiveCharacterTextSplitter` over fixed-size splitting

**Rationale**:
- Preserves semantic coherence
- Tries natural boundaries (paragraphs → sentences → words)
- Reduces mid-sentence splits

**Trade-offs**:
- ✅ Better context quality
- ✅ More coherent chunks
- ❌ Variable chunk sizes
- ❌ Slightly more complex

**When to use alternatives**:
- Fixed-size: When consistency matters more than quality
- Semantic: For research papers with clear structure
- Custom: For domain-specific formats (code, legal docs)

---

### 2. Why 1000 Token Chunks with 200 Overlap?

**Decision**: Default to 1000-token chunks with 200-token overlap

**Rationale**:
```python
# Chunk size considerations
Too small (100-300 tokens):
  ✅ Fast retrieval
  ❌ Lost context
  ❌ Noisy results

Sweet spot (800-1200 tokens):
  ✅ Good context
  ✅ Manageable size
  ✅ LLM can process

Too large (2000+ tokens):
  ✅ Full context
  ❌ Diluted relevance
  ❌ Expensive embeddings
```

**Overlap importance**:
- Prevents information loss at boundaries
- Critical for concepts that span chunks
- 15-20% overlap is industry standard

**Tuning guidance**:
```python
Document Type          → Chunk Size → Overlap
Technical docs         → 500-800    → 100-150
Narrative/articles     → 1000-1500  → 200-300
Short Q&A             → 300-500    → 50-100
Legal documents       → 800-1200   → 200-300
Code                  → 500-1000   → 100-200
```

---

### 3. Why ChromaDB for Phase 1?

**Decision**: Use ChromaDB for vector storage

**Rationale**:
- **Local-first**: No cloud dependency for MVP
- **Persistent**: Survives restarts
- **Simple**: Minimal configuration
- **Free**: No costs for development

**Migration path**:
```
Phase 1: ChromaDB (local)
    ↓
Phase 3: Consider Pinecone/Weaviate (scale)
    ↓
Production: Managed service with replication
```

**When to migrate**:
- > 1M documents
- Need multi-region
- Require 99.9% SLA
- Team lacks vector DB expertise

---

### 4. Why Claude Sonnet 4 for Generation?

**Decision**: Use Claude Sonnet 4 over GPT-4

**Rationale**:
- **Context window**: 200K tokens (vs GPT-4's 128K)
- **Accuracy**: Better at following instructions
- **Citations**: More reliable source attribution
- **Cost**: Competitive pricing
- **Latency**: Fast streaming responses

**Alternative scenarios**:
```python
Use Case              → Model Choice
Cost-sensitive        → Claude Haiku 4
Maximum reasoning     → Claude Opus 4
OpenAI ecosystem      → GPT-4 Turbo
Local/offline         → Llama 2/3
```

---

### 5. Why Similarity Search over MMR?

**Decision**: Default to similarity search, not MMR

**Rationale**:
- **Simpler**: Easier to understand and debug
- **Faster**: No diversity calculation overhead
- **Predictable**: Pure relevance ranking

**When to use MMR** (Phase 2):
- User wants diverse perspectives
- Avoiding redundant chunks
- Exploratory queries

**Example**:
```python
# Similarity: Returns most similar chunks
# Query: "What is X?"
# Results: X definition, X properties, X characteristics

# MMR: Returns diverse chunks
# Query: "What is X?"
# Results: X definition, X history, X alternatives
```

---

## Common Patterns

### Pattern 1: Ingestion Pipeline

```python
# Standard ingestion pattern
def ingest_knowledge_base(data_path: str):
    pipeline = RAGPipeline(collection_name="kb")
    
    # Batch ingest
    stats = pipeline.ingest_documents(directory_path=data_path)
    
    # Validate
    assert stats['success'], "Ingestion failed"
    assert stats['chunks_created'] > 0, "No chunks created"
    
    # Log metrics
    logger.info(f"Ingested {stats['documents_loaded']} docs, "
                f"{stats['chunks_created']} chunks")
    
    return pipeline
```

**Best practices**:
- ✅ Validate stats after ingestion
- ✅ Log chunk statistics
- ✅ Handle failures gracefully
- ✅ Use batch processing for large datasets

---

### Pattern 2: Query with Error Handling

```python
# Robust query pattern
def safe_query(pipeline: RAGPipeline, question: str) -> dict:
    try:
        result = pipeline.query(
            question=question,
            return_sources=True,
        )
        
        # Validate result
        if result['num_sources'] == 0:
            logger.warning(f"No sources for: {question}")
        
        return {
            'success': True,
            'answer': result['answer'],
            'sources': result['num_sources'],
        }
        
    except Exception as e:
        logger.error(f"Query failed: {e}")
        return {
            'success': False,
            'error': str(e),
            'answer': "I encountered an error processing your question.",
        }
```

**Best practices**:
- ✅ Always use try-except
- ✅ Log failures
- ✅ Return user-friendly errors
- ✅ Track zero-source queries

---

### Pattern 3: Metadata Filtering

```python
# Filter by source
def query_specific_document(pipeline: RAGPipeline, 
                            question: str, 
                            source_file: str) -> dict:
    # This pattern will be enhanced in Phase 2
    # For now, retrieve and filter manually
    
    results = pipeline.retriever.retrieve(
        query=question,
        return_scores=True,
    )
    
    # Filter by source
    filtered = [
        (doc, score) for doc, score in results
        if doc.metadata.get('file_name') == source_file
    ]
    
    if not filtered:
        return {'answer': f"No information in {source_file}"}
    
    # Generate answer from filtered docs
    # ... (implementation details)
```

---

## Troubleshooting Guide

### Issue 1: "No relevant documents found"

**Symptoms**:
- Queries return empty results
- `num_sources = 0`

**Root causes**:
1. **Query-document mismatch**: Query terminology differs from documents
2. **Threshold too high**: Score threshold filters all results
3. **Empty vector store**: Documents not ingested

**Debugging steps**:
```python
# 1. Check vector store
stats = pipeline.get_stats()
print(f"Documents in store: {stats['vector_store']['document_count']}")

# 2. Lower threshold temporarily
result = pipeline.retriever.retrieve(
    query=question,
    return_scores=True,
)
print(f"Top scores: {[score for _, score in result[:3]]}")

# 3. Try broader query
result = pipeline.query("general overview")
```

**Solutions**:
- Lower `retrieval_score_threshold` to 0.6
- Rephrase query to match document terminology
- Re-ingest documents if store is empty

---

### Issue 2: "Context too long" errors

**Symptoms**:
- LLM errors about context length
- Truncated responses

**Root cause**:
- Too many chunks retrieved
- Chunks too large
- Combined context exceeds LLM limit

**Solutions**:
```python
# Reduce retrieved chunks
pipeline.query(question, top_k=3)  # Instead of 5-10

# Or reduce chunk size
splitter = DocumentSplitter(chunk_size=500)  # Instead of 1000
```

---

### Issue 3: Poor answer quality

**Symptoms**:
- Generic answers
- Missing key information
- Incorrect citations

**Debugging**:
```python
# Get full context to inspect
result = pipeline.query(
    question=question,
    return_context=True,
    return_sources=True,
)

# Check what LLM received
print("Context sent to LLM:")
print(result['context'])

# Check source quality
for source in result['sources']:
    print(f"Score: {source['score']:.3f}")
    print(f"Content: {source['content'][:200]}")
```

**Common fixes**:
- Increase `top_k` to get more context
- Improve document quality (remove noise)
- Tune chunk size for domain
- Adjust prompt template

---

## Performance Optimization

### Latency Breakdown

```
Total Query Time (~2-3s):
├─ Embedding query: ~200ms
├─ Vector search: ~100ms
├─ LLM generation: ~2000ms
└─ Overhead: ~100ms
```

### Optimization Strategies

#### 1. Reduce Embedding Cost
```python
# Batch embed during ingestion (already optimized)
# Use smaller embedding model for development
embeddings = EmbeddingGenerator(
    model="text-embedding-3-small"  # Faster, cheaper
)
```

#### 2. Optimize Vector Search
```python
# Reduce search space
retriever = AdvancedRetriever(
    top_k=3,  # Fewer results = faster
    score_threshold=0.75,  # Higher threshold = fewer candidates
)
```

#### 3. Parallel Processing (Phase 5)
```python
# For multiple queries
import asyncio

async def batch_query(questions: List[str]):
    tasks = [pipeline.query_async(q) for q in questions]
    return await asyncio.gather(*tasks)
```

---

## Security & Privacy

### Sensitive Data Handling

**⚠️ Critical Rules**:

1. **Never log user queries** containing PII
2. **Sanitize documents** before ingestion
3. **Use environment variables** for API keys
4. **Implement access controls** in production

### Example Sanitization

```python
import re

def sanitize_document(content: str) -> str:
    """Remove PII before ingestion."""
    
    # Remove email addresses
    content = re.sub(r'\S+@\S+', '[EMAIL]', content)
    
    # Remove phone numbers
    content = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE]', content)
    
    # Remove SSN
    content = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[SSN]', content)
    
    return content

# Use before ingestion
documents = loader.load_file(path)
for doc in documents:
    doc.page_content = sanitize_document(doc.page_content)
```

---

## Testing Strategy

### Test Pyramid

```
                    /\
                   /  \
                  / E2E \ (5%)
                 /--------\
                /          \
               / Integration\ (15%)
              /--------------\
             /                \
            /   Unit Tests     \ (80%)
           /____________________\
```

### Unit Test Example

```python
def test_chunk_overlap():
    """Verify chunks have proper overlap."""
    splitter = DocumentSplitter(chunk_size=100, chunk_overlap=20)
    doc = Document(page_content="A" * 500)
    
    chunks = splitter.split_documents([doc])
    
    # Verify overlap exists
    assert len(chunks) > 1
    
    # Check consecutive chunks share content
    for i in range(len(chunks) - 1):
        chunk1_end = chunks[i].page_content[-10:]
        chunk2_start = chunks[i+1].page_content[:10]
        # Should have some overlap
        assert any(c in chunk2_start for c in chunk1_end)
```

### Integration Test Example

```python
@pytest.mark.integration
def test_end_to_end_query():
    """Test full pipeline from ingestion to query."""
    pipeline = RAGPipeline(collection_name="test", reinitialize=True)
    
    # Ingest test document
    test_doc = "The capital of France is Paris."
    # ... create document ...
    
    stats = pipeline.ingest_documents(file_paths=[test_file])
    assert stats['success']
    
    # Query
    result = pipeline.query("What is the capital of France?")
    
    # Verify answer contains expected information
    assert "Paris" in result['answer']
    assert result['num_sources'] > 0
```

---

## Key Takeaways for Teams

### ✅ Do's

1. **Start simple**: Use defaults, then optimize
2. **Log everything**: Chunk stats, scores, errors
3. **Test with real data**: Not just toy examples
4. **Monitor costs**: Track embedding and LLM usage
5. **Version prompts**: Treat prompts as code
6. **Validate sources**: Check citation accuracy

### ❌ Don'ts

1. **Don't skip overlap**: Context loss at boundaries
2. **Don't ignore metadata**: Critical for citations
3. **Don't hardcode thresholds**: Make them configurable
4. **Don't trust first results**: Iterate and evaluate
5. **Don't forget error handling**: LLMs can fail
6. **Don't expose API keys**: Use environment variables

---

## Next Phase Preview

**Phase 2: Enhanced RAG**
- Query rewriting with LLM
- Multi-query retrieval
- Re-ranking algorithms
- ReAct-style agents

**Phase 3: MCP Integration**
- Connect to Google Drive
- Database querying
- Real-time data sources

**Phase 4: LangGraph**
- Complex workflows
- State management
- Human-in-the-loop
- Multi-agent systems

**Phase 5: Production**
- Observability (LangSmith)
- Evaluation metrics
- API deployment
- Monitoring & alerts

---

## Resources for Deep Dives

### Papers
- "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (Lewis et al., 2020)
- "Lost in the Middle" (Liu et al., 2023)

### Documentation
- [LangChain RAG Guide](https://python.langchain.com/docs/use_cases/question_answering/)
- [Anthropic Prompt Engineering](https://docs.anthropic.com/claude/docs/prompt-engineering)
- [ChromaDB Documentation](https://docs.trychroma.com/)

### Best Practices
- [Pinecone's RAG Guide](https://www.pinecone.io/learn/retrieval-augmented-generation/)
- [LlamaIndex RAG Concepts](https://docs.llamaindex.ai/en/stable/getting_started/concepts.html)

---

**Questions?** Review the README.md or explore the source code with detailed comments.
