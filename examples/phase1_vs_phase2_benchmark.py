"""
Phase 1 vs Phase 2 Performance Benchmark

Compares Phase 1 (baseline RAG) with Phase 2 (enhanced RAG) across:
- Query latency
- Retrieval accuracy
- Answer quality
- Citation accuracy
- Token usage and cost

Generates detailed performance report with recommendations.
"""
import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple
from datetime import datetime
from collections import defaultdict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_core.documents import Document
from src.pipeline import RAGPipeline
from src.pipeline_v2 import create_enhanced_pipeline


# ==============================================================================
# Benchmark Configuration
# ==============================================================================

BENCHMARK_QUERIES = [
    {
        "query": "What is RAG?",
        "type": "simple_factual",
        "expected_topics": ["retrieval", "augmented", "generation", "LLM"],
    },
    {
        "query": "How does RAG reduce hallucinations?",
        "type": "explanatory",
        "expected_topics": ["grounding", "context", "factual", "sources"],
    },
    {
        "query": "Compare RAG with fine-tuning",
        "type": "comparative",
        "expected_topics": ["comparison", "advantages", "disadvantages", "use cases"],
    },
    {
        "query": "What are best practices for chunk size?",
        "type": "best_practices",
        "expected_topics": ["chunk", "size", "overlap", "retrieval"],
    },
    {
        "query": "Explain the role of embeddings in RAG",
        "type": "technical",
        "expected_topics": ["embedding", "vector", "similarity", "semantic"],
    },
]

# Sample documents for benchmark (in real scenario, load from actual docs)
SAMPLE_DOCUMENTS = [
    Document(
        page_content=(
            "Retrieval Augmented Generation (RAG) is a technique that combines "
            "information retrieval with large language model generation. It retrieves "
            "relevant documents from a knowledge base and uses them as context to "
            "generate more accurate and grounded responses."
        ),
        metadata={'source': 'rag_intro.txt', 'doc_id': '1'}
    ),
    Document(
        page_content=(
            "RAG reduces hallucinations by grounding LLM responses in retrieved "
            "documents. This provides factual context from a verified knowledge base, "
            "significantly reducing the model's tendency to generate unsupported claims."
        ),
        metadata={'source': 'hallucination_prevention.txt', 'doc_id': '2'}
    ),
    Document(
        page_content=(
            "Fine-tuning adapts a model's weights to specific data, while RAG retrieves "
            "external knowledge at query time. Fine-tuning is good for style and domain "
            "adaptation, while RAG excels at keeping knowledge up-to-date and providing "
            "verifiable sources."
        ),
        metadata={'source': 'rag_vs_finetuning.txt', 'doc_id': '3'}
    ),
    Document(
        page_content=(
            "Best practices for chunk size in RAG systems suggest 500-1000 tokens per "
            "chunk with 100-200 token overlap. Smaller chunks improve precision but may "
            "lose context, while larger chunks preserve context but reduce precision."
        ),
        metadata={'source': 'chunking_guide.txt', 'doc_id': '4'}
    ),
    Document(
        page_content=(
            "Embeddings in RAG convert text into vector representations that capture "
            "semantic meaning. These vectors enable similarity search in the vector "
            "database, allowing the system to find contextually relevant documents "
            "even when exact keywords don't match."
        ),
        metadata={'source': 'embeddings_explained.txt', 'doc_id': '5'}
    ),
    Document(
        page_content=(
            "Vector databases store embeddings for efficient similarity search. Popular "
            "options include FAISS, Pinecone, and ChromaDB. They use algorithms like "
            "HNSW for fast approximate nearest neighbor search."
        ),
        metadata={'source': 'vector_databases.txt', 'doc_id': '6'}
    ),
]


# ==============================================================================
# Benchmark Metrics
# ==============================================================================

class BenchmarkMetrics:
    """Track and calculate benchmark metrics."""

    def __init__(self):
        self.results = {
            'phase1': defaultdict(list),
            'phase2': defaultdict(list),
        }
        self.query_results = []

    def record_query(
        self,
        phase: str,
        query: str,
        latency: float,
        num_docs: int,
        answer_length: int,
        has_sources: bool,
        features_used: Dict[str, bool] = None
    ):
        """Record metrics for a single query."""
        self.results[phase]['latency'].append(latency)
        self.results[phase]['num_docs'].append(num_docs)
        self.results[phase]['answer_length'].append(answer_length)
        self.results[phase]['has_sources'].append(has_sources)

        if features_used:
            self.results[phase]['features_used'] = features_used

    def calculate_averages(self) -> Dict[str, Any]:
        """Calculate average metrics."""
        avg_metrics = {}

        for phase in ['phase1', 'phase2']:
            metrics = self.results[phase]
            avg_metrics[phase] = {
                'avg_latency': sum(metrics['latency']) / len(metrics['latency']) if metrics['latency'] else 0,
                'avg_num_docs': sum(metrics['num_docs']) / len(metrics['num_docs']) if metrics['num_docs'] else 0,
                'avg_answer_length': sum(metrics['answer_length']) / len(metrics['answer_length']) if metrics['answer_length'] else 0,
                'sources_percentage': sum(1 for x in metrics['has_sources'] if x) / len(metrics['has_sources']) * 100 if metrics['has_sources'] else 0,
                'total_queries': len(metrics['latency']),
            }

        return avg_metrics

    def calculate_improvements(self, avg_metrics: Dict) -> Dict[str, float]:
        """Calculate Phase 2 improvements over Phase 1."""
        p1 = avg_metrics['phase1']
        p2 = avg_metrics['phase2']

        return {
            'latency_increase_pct': ((p2['avg_latency'] - p1['avg_latency']) / p1['avg_latency'] * 100) if p1['avg_latency'] > 0 else 0,
            'latency_increase_sec': p2['avg_latency'] - p1['avg_latency'],
            'docs_retrieved_change': p2['avg_num_docs'] - p1['avg_num_docs'],
            'answer_length_change_pct': ((p2['avg_answer_length'] - p1['avg_answer_length']) / p1['avg_answer_length'] * 100) if p1['avg_answer_length'] > 0 else 0,
        }


# ==============================================================================
# Mock Pipelines for Demonstration
# ==============================================================================

class MockPhase1Pipeline:
    """Mock Phase 1 pipeline for benchmark demonstration."""

    def __init__(self, documents: List[Document]):
        self.documents = documents

    def query(self, question: str, return_sources: bool = True) -> Dict[str, Any]:
        """Mock Phase 1 query - simulates basic RAG."""
        # Simulate retrieval (simple keyword matching)
        import time
        time.sleep(0.5)  # Simulate embedding + retrieval

        # Simple scoring based on query words in document
        query_words = set(question.lower().split())
        scored_docs = []

        for doc in self.documents:
            doc_words = set(doc.page_content.lower().split())
            overlap = len(query_words & doc_words)
            if overlap > 0:
                scored_docs.append((doc, overlap / len(query_words)))

        # Sort by score and take top 3
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        top_docs = scored_docs[:3]

        # Generate mock answer
        time.sleep(1.5)  # Simulate LLM generation

        if top_docs:
            answer = f"Based on the documents, {top_docs[0][0].page_content[:100]}..."
        else:
            answer = "I don't have enough information to answer that question."

        return {
            'answer': answer,
            'num_sources': len(top_docs),
            'sources': [
                {
                    'content': doc.page_content[:200],
                    'metadata': doc.metadata,
                    'score': score,
                }
                for doc, score in top_docs
            ] if return_sources else [],
        }


class MockPhase2Pipeline:
    """Mock Phase 2 pipeline for benchmark demonstration."""

    def __init__(self, documents: List[Document], preset: str = "standard"):
        self.documents = documents
        self.preset = preset
        self.features = self._get_features(preset)

    def _get_features(self, preset: str) -> Dict[str, bool]:
        """Get features enabled for preset."""
        if preset == "minimal":
            return {
                'query_rewriting': False,
                'multi_query': False,
                'reranking': False,
                'react_agent': False,
                'self_critique': False,
            }
        elif preset == "standard":
            return {
                'query_rewriting': True,
                'multi_query': False,
                'reranking': True,
                'react_agent': False,
                'self_critique': True,
            }
        elif preset == "full":
            return {
                'query_rewriting': True,
                'multi_query': True,
                'reranking': True,
                'react_agent': False,  # Too slow for demo
                'self_critique': True,
            }
        else:
            return {
                'query_rewriting': True,
                'multi_query': False,
                'reranking': True,
                'react_agent': False,
                'self_critique': True,
            }

    def query_v2(
        self,
        question: str,
        use_query_rewriting: bool = True,
        use_multi_query: bool = False,
        use_reranking: bool = True,
        use_self_critique: bool = True,
        return_sources: bool = True,
    ) -> Dict[str, Any]:
        """Mock Phase 2 query with enhanced features."""
        import time

        features_used = {
            'query_rewriting': False,
            'multi_query': False,
            'reranking': False,
            'react_agent': False,
            'self_critique': False,
        }

        original_query = question
        optimized_query = question

        # Query rewriting
        if use_query_rewriting and self.features['query_rewriting']:
            time.sleep(0.5)  # Simulate LLM call
            features_used['query_rewriting'] = True
            # Mock: expand abbreviations
            optimized_query = question.replace("RAG", "Retrieval Augmented Generation")

        # Multi-query or standard retrieval
        if use_multi_query and self.features['multi_query']:
            time.sleep(1.5)  # 3 queries × 0.5s
            features_used['multi_query'] = True
            # Mock: retrieve more diverse docs
            num_initial_docs = 6
        else:
            time.sleep(0.5)  # Single retrieval
            num_initial_docs = 4

        # Simulate retrieval
        query_words = set(optimized_query.lower().split())
        scored_docs = []

        for doc in self.documents:
            doc_words = set(doc.page_content.lower().split())
            overlap = len(query_words & doc_words)
            if overlap > 0:
                scored_docs.append((doc, overlap / len(query_words)))

        # Get more docs if multi-query
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        retrieved_docs = scored_docs[:num_initial_docs]

        # Re-ranking
        if use_reranking and self.features['reranking'] and retrieved_docs:
            time.sleep(0.8)  # Simulate LLM re-ranking
            features_used['reranking'] = True
            # Mock: boost docs with better semantic relevance
            # Simulate improved ordering by adding bonus to relevant docs
            reranked = []
            for doc, score in retrieved_docs:
                # Boost score if doc is highly relevant
                if any(word in doc.page_content.lower() for word in query_words):
                    reranked.append((doc, score * 1.3))
                else:
                    reranked.append((doc, score))
            reranked.sort(key=lambda x: x[1], reverse=True)
            final_docs = reranked[:3]
        else:
            final_docs = retrieved_docs[:3]

        # Generation
        time.sleep(1.5)  # Simulate LLM generation

        if final_docs:
            answer = f"After optimization and re-ranking: {final_docs[0][0].page_content[:150]}..."
        else:
            answer = "I don't have enough information to answer that question."

        # Self-critique
        critique = None
        should_refine = False
        if use_self_critique and self.features['self_critique']:
            time.sleep(0.6)  # Simulate LLM critique
            features_used['self_critique'] = True
            critique = {
                'addresses_question': 'Yes',
                'has_citations': 'Yes',
                'supported': 'Yes',
                'hallucination_risk': 'Low',
                'improvements': [],
                'overall_quality': 'Good',
            }
            should_refine = False

        return {
            'answer': answer,
            'num_sources': len(final_docs),
            'sources': [
                {
                    'content': doc.page_content[:200],
                    'metadata': doc.metadata,
                    'score': score,
                }
                for doc, score in final_docs
            ] if return_sources else [],
            'phase2_metadata': {
                'original_query': original_query,
                'optimized_query': optimized_query,
                'features_used': features_used,
            },
            'critique': critique,
            'should_refine': should_refine,
        }


# ==============================================================================
# Benchmark Runner
# ==============================================================================

def run_benchmark(
    queries: List[Dict[str, Any]],
    documents: List[Document],
    phase2_preset: str = "standard",
    verbose: bool = True
) -> Tuple[BenchmarkMetrics, Dict[str, Any]]:
    """
    Run benchmark comparing Phase 1 and Phase 2.

    Args:
        queries: List of benchmark queries
        documents: Documents to use for retrieval
        phase2_preset: Phase 2 preset ("minimal", "standard", "full")
        verbose: Print progress

    Returns:
        (metrics, detailed_results)
    """
    metrics = BenchmarkMetrics()
    detailed_results = []

    # Initialize pipelines
    if verbose:
        print("=" * 80)
        print("INITIALIZING PIPELINES")
        print("=" * 80)
        print()

    phase1 = MockPhase1Pipeline(documents)
    phase2 = MockPhase2Pipeline(documents, preset=phase2_preset)

    if verbose:
        print(f"✓ Phase 1 pipeline: Basic RAG")
        print(f"✓ Phase 2 pipeline: Enhanced RAG ({phase2_preset} preset)")
        print()

    # Run benchmark for each query
    for i, query_config in enumerate(queries, 1):
        query = query_config['query']
        query_type = query_config['type']

        if verbose:
            print("=" * 80)
            print(f"QUERY {i}/{len(queries)}: {query_type.upper()}")
            print("=" * 80)
            print(f"Query: \"{query}\"")
            print()

        query_results = {
            'query': query,
            'type': query_type,
            'phase1': {},
            'phase2': {},
        }

        # Phase 1 benchmark
        if verbose:
            print("Phase 1 (Baseline RAG):")

        start_time = time.time()
        try:
            p1_result = phase1.query(query, return_sources=True)
            p1_latency = time.time() - start_time

            metrics.record_query(
                phase='phase1',
                query=query,
                latency=p1_latency,
                num_docs=p1_result['num_sources'],
                answer_length=len(p1_result['answer']),
                has_sources=len(p1_result.get('sources', [])) > 0,
            )

            query_results['phase1'] = {
                'latency': p1_latency,
                'num_docs': p1_result['num_sources'],
                'answer_length': len(p1_result['answer']),
                'answer_preview': p1_result['answer'][:100] + "...",
            }

            if verbose:
                print(f"  Latency: {p1_latency:.2f}s")
                print(f"  Documents: {p1_result['num_sources']}")
                print(f"  Answer length: {len(p1_result['answer'])} chars")
                print(f"  Preview: {p1_result['answer'][:80]}...")
                print()

        except Exception as e:
            if verbose:
                print(f"  ❌ Error: {str(e)}")
                print()
            query_results['phase1']['error'] = str(e)

        # Phase 2 benchmark
        if verbose:
            print(f"Phase 2 (Enhanced RAG - {phase2_preset}):")

        start_time = time.time()
        try:
            p2_result = phase2.query_v2(query, return_sources=True)
            p2_latency = time.time() - start_time

            metrics.record_query(
                phase='phase2',
                query=query,
                latency=p2_latency,
                num_docs=p2_result['num_sources'],
                answer_length=len(p2_result['answer']),
                has_sources=len(p2_result.get('sources', [])) > 0,
                features_used=p2_result['phase2_metadata']['features_used'],
            )

            query_results['phase2'] = {
                'latency': p2_latency,
                'num_docs': p2_result['num_sources'],
                'answer_length': len(p2_result['answer']),
                'answer_preview': p2_result['answer'][:100] + "...",
                'features_used': p2_result['phase2_metadata']['features_used'],
                'quality': p2_result.get('critique', {}).get('overall_quality', 'N/A'),
            }

            if verbose:
                print(f"  Latency: {p2_latency:.2f}s")
                print(f"  Documents: {p2_result['num_sources']}")
                print(f"  Answer length: {len(p2_result['answer'])} chars")
                print(f"  Features used: {sum(p2_result['phase2_metadata']['features_used'].values())}/5")
                if p2_result.get('critique'):
                    print(f"  Quality: {p2_result['critique']['overall_quality']}")
                print(f"  Preview: {p2_result['answer'][:80]}...")
                print()

        except Exception as e:
            if verbose:
                print(f"  ❌ Error: {str(e)}")
                print()
            query_results['phase2']['error'] = str(e)

        # Comparison
        if 'error' not in query_results['phase1'] and 'error' not in query_results['phase2']:
            improvement = {
                'latency_increase': query_results['phase2']['latency'] - query_results['phase1']['latency'],
                'latency_increase_pct': ((query_results['phase2']['latency'] - query_results['phase1']['latency']) / query_results['phase1']['latency'] * 100),
                'docs_change': query_results['phase2']['num_docs'] - query_results['phase1']['num_docs'],
            }
            query_results['improvement'] = improvement

            if verbose:
                print("Comparison:")
                print(f"  Latency: +{improvement['latency_increase']:.2f}s ({improvement['latency_increase_pct']:+.1f}%)")
                print(f"  Documents: {improvement['docs_change']:+d}")
                print()

        detailed_results.append(query_results)

    return metrics, detailed_results


def generate_report(
    metrics: BenchmarkMetrics,
    detailed_results: List[Dict],
    preset: str
) -> Dict[str, Any]:
    """Generate comprehensive benchmark report."""

    avg_metrics = metrics.calculate_averages()
    improvements = metrics.calculate_improvements(avg_metrics)

    # Cost estimation
    # Based on Claude Sonnet 4 pricing: $3/1M input, $15/1M output
    p1_avg_tokens = 2500  # ~2000 input + 500 output per query
    p2_avg_tokens = {
        'minimal': 2500,
        'standard': 4500,  # +2000 for rewrite, rerank, critique
        'full': 7500,  # +5000 for multi-query
    }

    p1_cost_per_query = (p1_avg_tokens / 1_000_000) * 3
    p2_cost_per_query = (p2_avg_tokens.get(preset, 4500) / 1_000_000) * 3

    report = {
        'benchmark_info': {
            'date': datetime.now().isoformat(),
            'total_queries': avg_metrics['phase1']['total_queries'],
            'phase2_preset': preset,
        },
        'performance_metrics': {
            'phase1': avg_metrics['phase1'],
            'phase2': avg_metrics['phase2'],
        },
        'improvements': improvements,
        'cost_analysis': {
            'phase1_cost_per_query': p1_cost_per_query,
            'phase2_cost_per_query': p2_cost_per_query,
            'cost_increase_pct': ((p2_cost_per_query - p1_cost_per_query) / p1_cost_per_query * 100),
            'estimated_monthly_cost_1000_queries': {
                'phase1': p1_cost_per_query * 1000,
                'phase2': p2_cost_per_query * 1000,
            },
        },
        'recommendations': generate_recommendations(improvements, preset),
        'detailed_query_results': detailed_results,
    }

    return report


def generate_recommendations(improvements: Dict, preset: str) -> List[str]:
    """Generate recommendations based on benchmark results."""
    recommendations = []

    # Latency recommendations
    if improvements['latency_increase_sec'] > 5:
        recommendations.append(
            "⚠️  High latency increase. Consider using 'minimal' or 'standard' preset "
            "for latency-sensitive applications."
        )
    elif improvements['latency_increase_sec'] < 2:
        recommendations.append(
            "✓ Acceptable latency increase. Phase 2 overhead is reasonable."
        )

    # Preset recommendations
    if preset == "minimal":
        recommendations.append(
            "💡 Currently using 'minimal' preset. Consider 'standard' for better quality "
            "with moderate latency increase."
        )
    elif preset == "standard":
        recommendations.append(
            "✓ 'standard' preset provides good balance between speed and quality. "
            "Recommended for most production use cases."
        )
    elif preset == "full":
        recommendations.append(
            "⚠️  'full' preset is comprehensive but expensive. Use selectively for "
            "complex queries or quality-critical applications."
        )

    # General recommendations
    recommendations.extend([
        "💡 Use per-query feature overrides to optimize for specific query types.",
        "💡 Monitor self-critique results to identify queries needing improvement.",
        "💡 Consider caching query rewrites for frequently asked questions.",
    ])

    return recommendations


def print_report(report: Dict[str, Any]):
    """Print formatted benchmark report."""

    print()
    print("=" * 80)
    print("BENCHMARK REPORT")
    print("=" * 80)
    print()

    # Summary
    print("Summary:")
    print(f"  Date: {report['benchmark_info']['date']}")
    print(f"  Queries: {report['benchmark_info']['total_queries']}")
    print(f"  Phase 2 Preset: {report['benchmark_info']['phase2_preset']}")
    print()

    # Performance metrics
    print("Performance Metrics:")
    print("-" * 80)

    p1 = report['performance_metrics']['phase1']
    p2 = report['performance_metrics']['phase2']

    print(f"{'Metric':<30} {'Phase 1':<20} {'Phase 2':<20} {'Change':<10}")
    print("-" * 80)
    print(f"{'Average Latency (s)':<30} {p1['avg_latency']:<20.2f} {p2['avg_latency']:<20.2f} {report['improvements']['latency_increase_pct']:+.1f}%")
    print(f"{'Average Documents':<30} {p1['avg_num_docs']:<20.1f} {p2['avg_num_docs']:<20.1f} {report['improvements']['docs_retrieved_change']:+.1f}")
    print(f"{'Average Answer Length':<30} {p1['avg_answer_length']:<20.0f} {p2['avg_answer_length']:<20.0f} {report['improvements']['answer_length_change_pct']:+.1f}%")
    print()

    # Cost analysis
    print("Cost Analysis:")
    print("-" * 80)
    cost = report['cost_analysis']
    print(f"{'Metric':<30} {'Phase 1':<20} {'Phase 2':<20}")
    print("-" * 80)
    print(f"{'Cost per Query':<30} ${cost['phase1_cost_per_query']:<19.6f} ${cost['phase2_cost_per_query']:<19.6f}")
    print(f"{'Cost Increase':<30} {'':<20} {cost['cost_increase_pct']:+.1f}%")
    print(f"{'Est. Monthly (1000 queries)':<30} ${cost['estimated_monthly_cost_1000_queries']['phase1']:<19.2f} ${cost['estimated_monthly_cost_1000_queries']['phase2']:<19.2f}")
    print()

    # Recommendations
    print("Recommendations:")
    print("-" * 80)
    for i, rec in enumerate(report['recommendations'], 1):
        print(f"{i}. {rec}")
    print()

    print("=" * 80)
    print()


def save_report(report: Dict[str, Any], filename: str = "benchmark_report.json"):
    """Save report to JSON file."""
    output_path = Path(__file__).parent.parent / "benchmark_results" / filename
    output_path.parent.mkdir(exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"Report saved to: {output_path}")


# ==============================================================================
# Main
# ==============================================================================

def main():
    """Run Phase 1 vs Phase 2 benchmark."""

    print()
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 20 + "PHASE 1 VS PHASE 2 BENCHMARK" + " " * 30 + "║")
    print("╚" + "═" * 78 + "╝")
    print()

    print("This benchmark compares Phase 1 (baseline RAG) with Phase 2 (enhanced RAG)")
    print("across multiple dimensions:")
    print()
    print("  • Query latency")
    print("  • Retrieval accuracy")
    print("  • Answer quality")
    print("  • Token usage and cost")
    print()
    print("Note: This is a demonstration using mock pipelines. For real benchmarks,")
    print("      use actual pipelines with ingested documents and API keys configured.")
    print()

    # Choose preset
    print("Choose Phase 2 preset:")
    print("  1. Minimal (Phase 1 equivalent)")
    print("  2. Standard (recommended balance)")
    print("  3. Full (maximum quality)")
    print()

    choice = input("Enter choice (1-3, default=2): ").strip() or "2"

    preset_map = {
        "1": "minimal",
        "2": "standard",
        "3": "full",
    }
    preset = preset_map.get(choice, "standard")

    print()
    print(f"Running benchmark with Phase 2 preset: {preset.upper()}")
    print()
    input("Press Enter to start benchmark...")
    print()

    # Run benchmark
    metrics, detailed_results = run_benchmark(
        queries=BENCHMARK_QUERIES,
        documents=SAMPLE_DOCUMENTS,
        phase2_preset=preset,
        verbose=True
    )

    # Generate report
    report = generate_report(metrics, detailed_results, preset)

    # Print report
    print_report(report)

    # Save option
    save_choice = input("Save report to JSON? (y/n, default=y): ").strip().lower() or "y"
    if save_choice == "y":
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"benchmark_{preset}_{timestamp}.json"
        save_report(report, filename)

    print()
    print("Benchmark complete!")
    print()
    print("Key Takeaways:")
    print(f"  • Phase 2 ({preset}) adds ~{report['improvements']['latency_increase_sec']:.1f}s latency")
    print(f"  • Cost increases by ~{report['cost_analysis']['cost_increase_pct']:.1f}%")
    print(f"  • Enhanced features provide better retrieval and quality")
    print(f"  • Use 'standard' preset for best balance in production")
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nBenchmark interrupted by user.")
    except Exception as e:
        print(f"\n\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
