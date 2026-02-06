# Code Coverage Report - RAG System

**Generated**: February 1, 2026
**Test Suite**: 118 tests passed, 2 skipped
**Total Tests**: 120

## Overall Coverage

```
Total Coverage: 86% (672 statements, 91 missing)
```

## Coverage by Module

### Excellent Coverage (90-100%)

| Module | Statements | Missing | Coverage | Status |
|--------|-----------|---------|----------|--------|
| `src/__init__.py` | 4 | 0 | **100%** | ✅ |
| `src/config/__init__.py` | 0 | 0 | **100%** | ✅ |
| `src/config/settings.py` | 22 | 0 | **100%** | ✅ |
| `src/agents/__init__.py` | 4 | 0 | **100%** | ✅ |
| `src/generation/__init__.py` | 0 | 0 | **100%** | ✅ |
| `src/ingestion/__init__.py` | 0 | 0 | **100%** | ✅ |
| `src/retrieval/__init__.py` | 0 | 0 | **100%** | ✅ |
| `src/storage/__init__.py` | 0 | 0 | **100%** | ✅ |
| **`src/agents/multi_query.py`** | 120 | 1 | **99%** | ✅ |
| **`src/agents/query_rewriter.py`** | 60 | 2 | **97%** | ✅ |
| **`src/pipeline.py`** | 96 | 6 | **94%** | ✅ |

### Good Coverage (80-89%)

| Module | Statements | Missing | Coverage | Missing Lines |
|--------|-----------|---------|----------|---------------|
| `src/generation/llm.py` | 71 | 9 | **87%** | 86-88, 134-136, 158-160 |
| `src/ingestion/loaders.py` | 53 | 8 | **85%** | 79-81, 106, 114, 129-131 |
| `src/generation/prompts.py` | 25 | 5 | **80%** | 61, 87, 102-109 |
| `src/ingestion/splitters.py` | 35 | 7 | **80%** | 108-117, 130, 165-169 |

### Needs Improvement (< 80%)

| Module | Statements | Missing | Coverage | Missing Lines |
|--------|-----------|---------|----------|---------------|
| `src/storage/vector_store.py` | 99 | 22 | **78%** | 84-86, 123-125, 173-184, 219-221, 240-244, 262-264 |
| `src/retrieval/retriever.py` | 47 | 14 | **70%** | 104-106, 125-148, 167, 195-196, 218-220, 238 |
| **`src/ingestion/embedder.py`** | 36 | 17 | **53%** ⚠️ | 65-74, 86-92, 96, 110-111 |

## Phase 2 Components Coverage

### ✅ Query Rewriter (Phase 2 - Step 1)
- **Coverage**: 97% (60 statements, 2 missing)
- **Missing**: Lines 145, 156 (edge cases in batch processing)
- **Status**: Excellent

### ✅ Multi-Query Generator (Phase 2 - Step 3)
- **Coverage**: 99% (120 statements, 1 missing)
- **Missing**: Line 183 (single edge case)
- **Status**: Excellent

## Areas Needing Attention

### 1. Embedder Module (53% coverage) ⚠️
**Missing Coverage**:
- Error handling paths (lines 65-74, 86-92)
- Convenience function `get_embeddings()` (lines 110-111)
- Edge cases in embedding generation

**Recommendation**: Add tests for:
- Error scenarios (API failures, empty inputs)
- The `get_embeddings()` convenience function
- Batch embedding edge cases

### 2. Retriever Module (70% coverage)
**Missing Coverage**:
- `retrieve_with_context()` method (lines 125-148)
- `retrieve_by_source()` method (lines 167)
- `MultiQueryRetriever` stub (lines 195-196, 218-220)

**Recommendation**: Add tests for:
- Context retrieval with rich metadata
- Source filtering functionality
- Future: Test actual MultiQueryRetriever (now implemented in agents/)

### 3. Vector Store Module (78% coverage)
**Missing Coverage**:
- MMR search functionality (lines 173-184)
- Collection deletion (lines 219-221)
- Some error handling paths

**Recommendation**: Add tests for:
- MMR (Maximal Marginal Relevance) search
- Collection management operations
- Error scenarios

## Test Distribution

### By Test File

| Test File | Tests | Coverage Focus |
|-----------|-------|----------------|
| `test_agents.py` | 8 | Query rewriting (focused tests) |
| `test_query_rewriter.py` | 17 | Query rewriting (comprehensive) |
| `test_multi_query.py` | 24 | Multi-query generation |
| `test_generation.py` | 20 | LLM generation, prompts |
| `test_ingestion.py` | 15 | Document loading, splitting |
| `test_pipeline.py` | 12 | End-to-end pipeline |
| `test_retrieval.py` | 13 | Document retrieval |
| `test_vector_store.py` | 11 | Vector storage |

**Total**: 120 tests

## Coverage Trends

### Phase 1 Modules
- Pipeline: 94%
- Generation: 87% (LLM), 80% (prompts)
- Ingestion: 85% (loaders), 80% (splitters), 53% (embedder) ⚠️
- Storage: 78%
- Retrieval: 70%

### Phase 2 Modules (New)
- Query Rewriter: **97%** ✅
- Multi-Query: **99%** ✅

## Recommendations

### High Priority
1. **Improve embedder.py coverage** (53% → 80%+)
   - Add error handling tests
   - Test convenience functions
   - Add edge case tests

2. **Improve retriever.py coverage** (70% → 85%+)
   - Test `retrieve_with_context()`
   - Test `retrieve_by_source()`
   - Add metadata filtering tests

### Medium Priority
3. **Improve vector_store.py coverage** (78% → 85%+)
   - Add MMR search tests
   - Test collection management
   - Add more error scenarios

4. **Improve prompts.py coverage** (80% → 90%+)
   - Test template creation functions
   - Test context-aware prompts

### Low Priority
5. **Improve generation/llm.py coverage** (87% → 92%+)
   - Test streaming error paths
   - Test batch generation edge cases

## HTML Report

A detailed HTML coverage report has been generated at:
```
htmlcov/index.html
```

Open in browser:
```bash
open htmlcov/index.html  # macOS
```

The HTML report includes:
- Line-by-line coverage visualization
- Branch coverage details
- Function and class coverage
- Interactive navigation

## Summary

### Strengths ✅
- **Overall coverage**: 86% (good)
- **Phase 2 components**: 97-99% (excellent)
- **Core pipeline**: 94% (excellent)
- **Configuration**: 100% (perfect)

### Areas to Improve ⚠️
- **Embedder**: 53% (needs significant work)
- **Retriever**: 70% (needs improvement)
- **Vector Store**: 78% (close to target)

### Coverage Goals
- **Current**: 86%
- **Target**: 90%
- **Gap**: 4% (approximately 30-35 additional lines)

### Next Steps
1. Add embedder error handling tests (+17 lines)
2. Add retriever context/filter tests (+14 lines)
3. Add vector store MMR tests (+10 lines)

This would bring total coverage from 86% to ~92%.
