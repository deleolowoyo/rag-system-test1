"""
Enhanced RAG Pipeline (Phase 2) with advanced features.

Extends the basic RAGPipeline with:
- Query Rewriting for optimized retrieval
- Multi-Query Generation for improved recall
- Document Re-Ranking for better precision
- ReAct Agent for multi-step reasoning (optional)

Architecture:
1. Document Loading → 2. Splitting → 3. Embedding → 4. Storage
5. Query Enhancement (rewrite/multi-query) → 6. Retrieval → 7. Re-Ranking
8. Generation/Agent → 9. Self-Critique → 10. Response
"""
import logging
from typing import List, Optional, Dict, Any, Tuple

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

from src.pipeline import RAGPipeline
from src.generation.prompts import format_documents_for_context, RAG_SYSTEM_PROMPT
from src.agents import (
    QueryRewriter,
    MultiQueryRetriever,
    ReActAgent,
    SelfCritiqueAgent,
)
from src.reranking import LLMReranker, HybridReranker
from src.config.settings import settings

logging.basicConfig(
    level=settings.log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EnhancedRAGPipeline(RAGPipeline):
    """
    Enhanced RAG pipeline with Phase 2 advanced features.

    Extends RAGPipeline with:
    - Query optimization (rewriting, multi-query)
    - Document re-ranking (LLM-based, hybrid)
    - Multi-step reasoning (ReAct agent)
    - Answer validation (self-critique)

    All Phase 2 features can be enabled/disabled individually.
    """

    def __init__(
        self,
        collection_name: str = None,
        persist_directory: str = None,
        reinitialize: bool = False,
        # Phase 2 feature flags
        enable_query_rewriting: bool = True,
        enable_multi_query: bool = True,
        enable_reranking: bool = True,
        enable_react_agent: bool = False,
        enable_self_critique: bool = True,
        # Phase 2 configuration
        reranking_mode: str = "llm",  # "llm" or "hybrid"
        react_max_iterations: int = 5,
        llm_rerank_weight: float = 0.7,
        similarity_weight: float = 0.3,
    ):
        """
        Initialize Enhanced RAG Pipeline with Phase 2 components.

        Args:
            collection_name: Vector store collection name
            persist_directory: Path for vector store persistence
            reinitialize: If True, delete existing collection and start fresh

            # Phase 2 Feature Flags
            enable_query_rewriting: Enable query optimization (default: True)
            enable_multi_query: Enable multi-query generation (default: True)
            enable_reranking: Enable document re-ranking (default: True)
            enable_react_agent: Enable ReAct multi-step reasoning (default: False)
            enable_self_critique: Enable answer validation (default: True)

            # Phase 2 Configuration
            reranking_mode: "llm" for LLM-only, "hybrid" for similarity+LLM (default: "llm")
            react_max_iterations: Max iterations for ReAct agent (default: 5)
            llm_rerank_weight: Weight for LLM scores in hybrid mode (default: 0.7)
            similarity_weight: Weight for similarity scores in hybrid mode (default: 0.3)
        """
        # Initialize base pipeline
        logger.info("=" * 70)
        logger.info("Initializing Enhanced RAG Pipeline (Phase 2)...")
        logger.info("=" * 70)

        super().__init__(
            collection_name=collection_name,
            persist_directory=persist_directory,
            reinitialize=reinitialize,
        )

        # Store feature flags
        self.enable_query_rewriting = enable_query_rewriting
        self.enable_multi_query = enable_multi_query
        self.enable_reranking = enable_reranking
        self.enable_react_agent = enable_react_agent
        self.enable_self_critique = enable_self_critique

        # Initialize Phase 2 components
        logger.info("\nInitializing Phase 2 Components:")
        logger.info("-" * 70)

        # 1. Query Rewriter
        if self.enable_query_rewriting:
            logger.info("✓ Query Rewriting: ENABLED")
            self.query_rewriter = QueryRewriter()
            logger.info("  - Optimizes queries for better retrieval")
            logger.info("  - Adds context and specificity")
        else:
            logger.info("✗ Query Rewriting: DISABLED")
            self.query_rewriter = None

        # 2. Multi-Query Retriever
        if self.enable_multi_query:
            logger.info("✓ Multi-Query Generation: ENABLED")
            self.multi_query_retriever = MultiQueryRetriever(
                retriever=self.retriever
            )
            logger.info("  - Generates query variations")
            logger.info("  - Improves recall with diverse perspectives")
        else:
            logger.info("✗ Multi-Query Generation: DISABLED")
            self.multi_query_retriever = None

        # 3. Document Re-Ranker
        if self.enable_reranking:
            logger.info(f"✓ Document Re-Ranking: ENABLED (mode={reranking_mode})")

            # Initialize LLM reranker (base component)
            llm_reranker = LLMReranker(temperature=0.0, max_doc_length=1000)

            if reranking_mode == "hybrid":
                # Hybrid mode: combine similarity + LLM scores
                self.reranker = HybridReranker(
                    llm_reranker=llm_reranker,
                    llm_weight=llm_rerank_weight,
                    similarity_weight=similarity_weight,
                )
                logger.info(f"  - Mode: Hybrid (LLM weight={llm_rerank_weight}, "
                           f"Similarity weight={similarity_weight})")
                logger.info("  - Combines vector similarity with LLM scoring")
            else:
                # LLM-only mode
                self.reranker = llm_reranker
                logger.info("  - Mode: LLM-only")
                logger.info("  - Uses LLM for relevance scoring (0-10 scale)")

            self.reranking_mode = reranking_mode
        else:
            logger.info("✗ Document Re-Ranking: DISABLED")
            self.reranker = None
            self.reranking_mode = None

        # 4. ReAct Agent (optional, expensive)
        if self.enable_react_agent:
            logger.info("✓ ReAct Agent: ENABLED")
            self.react_agent = ReActAgent(
                retriever=self.retriever,
                query_rewriter=self.query_rewriter if self.enable_query_rewriting else None,
                llm=self.llm,
                max_iterations=react_max_iterations,
            )
            logger.info(f"  - Max iterations: {react_max_iterations}")
            logger.info("  - Multi-step reasoning with Think-Act-Observe loop")
            logger.warning("  ⚠️  WARNING: ReAct mode is more expensive (multiple LLM calls)")
        else:
            logger.info("✗ ReAct Agent: DISABLED")
            self.react_agent = None

        # 5. Self-Critique Agent
        if self.enable_self_critique:
            logger.info("✓ Self-Critique: ENABLED")
            self.self_critique_agent = SelfCritiqueAgent(temperature=0.0)
            logger.info("  - Validates answer quality and accuracy")
            logger.info("  - Detects hallucinations and unsupported claims")
        else:
            logger.info("✗ Self-Critique: DISABLED")
            self.self_critique_agent = None

        # Summary
        logger.info("-" * 70)
        enabled_count = sum([
            self.enable_query_rewriting,
            self.enable_multi_query,
            self.enable_reranking,
            self.enable_react_agent,
            self.enable_self_critique,
        ])
        logger.info(f"Phase 2 Features: {enabled_count}/5 enabled")
        logger.info("=" * 70)
        logger.info("Enhanced RAG Pipeline initialized successfully\n")

    def query_v2(
        self,
        question: str,
        # Feature usage flags (override initialization defaults)
        use_query_rewriting: bool = True,
        use_multi_query: bool = False,
        use_reranking: bool = True,
        use_react_agent: bool = False,
        use_self_critique: bool = True,
        # Standard query parameters
        top_k: int = None,
        return_sources: bool = True,
        return_context: bool = False,
        # Additional options
        **kwargs
    ) -> Dict[str, Any]:
        """
        Enhanced query method with Phase 2 features.

        Orchestrates the complete Phase 2 pipeline:
        1. Query optimization (rewriting/multi-query)
        2. Document retrieval
        3. Re-ranking
        4. Answer generation or ReAct reasoning
        5. Self-critique validation

        Args:
            question: User question

            # Feature Usage Flags (override initialization settings)
            use_query_rewriting: Use query rewriting if available (default: True)
            use_multi_query: Use multi-query generation if available (default: False)
            use_reranking: Use re-ranking if available (default: True)
            use_react_agent: Use ReAct agent if available (default: False)
            use_self_critique: Use self-critique if available (default: True)

            # Standard Parameters
            top_k: Number of documents to retrieve
            return_sources: Include source documents in response
            return_context: Include formatted context in response

            # Additional Options
            **kwargs: Additional arguments passed to base pipeline

        Returns:
            Dictionary with:
            - answer: Generated answer
            - sources: Source documents (if return_sources=True)
            - num_sources: Number of sources used
            - phase2_metadata: Information about which features were used
            - critique: Self-critique results (if use_self_critique=True)
            - (additional fields from ReAct agent if used)

        Examples:
            >>> # Standard enhanced query
            >>> result = pipeline.query_v2("What is RAG?")

            >>> # Use ReAct agent for complex reasoning
            >>> result = pipeline.query_v2(
            ...     "How does RAG compare to fine-tuning?",
            ...     use_react_agent=True
            ... )

            >>> # Minimal processing (fastest)
            >>> result = pipeline.query_v2(
            ...     "Quick question",
            ...     use_query_rewriting=False,
            ...     use_reranking=False,
            ...     use_self_critique=False
            ... )
        """
        logger.info(f"Processing enhanced query: '{question}'")

        # Track which features were actually used
        features_used = {
            'query_rewriting': False,
            'multi_query': False,
            'reranking': False,
            'react_agent': False,
            'self_critique': False,
        }

        # Store original query for metadata
        original_query = question
        optimized_query = question

        # =====================================================================
        # STEP A: Check if ReAct agent should be used
        # =====================================================================
        if use_react_agent and self.enable_react_agent and self.react_agent:
            logger.info("Using ReAct Agent for multi-step reasoning")
            features_used['react_agent'] = True

            # ReAct agent handles the entire query internally
            result = self.react_agent.run(question)

            # Add Phase 2 metadata
            result['phase2_metadata'] = {
                'original_query': original_query,
                'features_used': features_used,
            }

            # Optionally apply self-critique to ReAct result
            if use_self_critique and self.enable_self_critique and self.self_critique_agent:
                logger.info("Applying self-critique to ReAct answer")
                features_used['self_critique'] = True

                critique = self.self_critique_agent.critique(
                    question=original_query,
                    answer=result['answer'],
                    context=result.get('documents', []),
                )

                result['critique'] = critique
                result['should_refine'] = self.self_critique_agent.should_refine(critique)

                # Update metadata
                result['phase2_metadata']['features_used'] = features_used

            logger.info("ReAct agent query complete")
            return result

        # =====================================================================
        # STEP B: Standard enhanced pipeline
        # =====================================================================

        # B1: Apply query rewriting if enabled
        if use_query_rewriting and self.enable_query_rewriting and self.query_rewriter:
            logger.info("Applying query rewriting")
            features_used['query_rewriting'] = True

            try:
                optimized_query = self.query_rewriter.rewrite(question)
                logger.info(f"Query rewritten: '{question}' -> '{optimized_query}'")
            except Exception as e:
                logger.warning(f"Query rewriting failed: {str(e)}, using original")
                optimized_query = question

        # B2: Retrieve documents (multi-query OR standard)
        retrieved_docs = []

        if use_multi_query and self.enable_multi_query and self.multi_query_retriever:
            # Multi-query retrieval
            logger.info("Using multi-query retrieval")
            features_used['multi_query'] = True

            try:
                retrieved_docs = self.multi_query_retriever.retrieve(
                    query=optimized_query,
                    top_k=top_k,
                )
                logger.info(f"Multi-query retrieved {len(retrieved_docs)} documents")
            except Exception as e:
                logger.error(f"Multi-query retrieval failed: {str(e)}")
                # Fallback to standard retrieval
                retrieved_docs = self.retriever.retrieve(
                    query=optimized_query,
                    top_k=top_k,
                )

        else:
            # Standard retrieval
            logger.info("Using standard retrieval")
            retrieved_docs = self.retriever.retrieve(
                query=optimized_query,
                top_k=top_k,
                return_scores=True,
            )

        if not retrieved_docs:
            logger.warning("No relevant documents found")
            return {
                'answer': "I don't have enough information to answer that question.",
                'sources': [],
                'num_sources': 0,
                'phase2_metadata': {
                    'original_query': original_query,
                    'optimized_query': optimized_query,
                    'features_used': features_used,
                },
            }

        # B3: Apply re-ranking if enabled
        documents = []
        scores = []

        if use_reranking and self.enable_reranking and self.reranker:
            logger.info(f"Applying re-ranking to {len(retrieved_docs)} documents")
            features_used['reranking'] = True

            try:
                # Check if docs have scores (for hybrid reranking)
                if isinstance(retrieved_docs[0], tuple):
                    # Format: [(doc, score), ...]
                    ranked_results = self.reranker.rerank(
                        query=optimized_query,
                        documents_with_scores=retrieved_docs,
                    )
                else:
                    # Format: [doc, ...] - use LLM-only reranking
                    from src.reranking import HybridReranker
                    if isinstance(self.reranker, HybridReranker):
                        logger.warning("Hybrid reranker requires scores, using LLM-only")
                        # Convert to LLM reranker
                        ranked_results = self.reranker.llm_reranker.rerank(
                            query=optimized_query,
                            documents=retrieved_docs,
                        )
                    else:
                        ranked_results = self.reranker.rerank(
                            query=optimized_query,
                            documents=retrieved_docs,
                        )

                # Extract documents and scores
                documents = [doc for doc, _ in ranked_results]
                scores = [score for _, score in ranked_results]

                logger.info(f"Re-ranking complete: top score={scores[0]:.2f}")

            except Exception as e:
                logger.error(f"Re-ranking failed: {str(e)}, using original order")
                # Fallback: extract from retrieved_docs
                if isinstance(retrieved_docs[0], tuple):
                    documents = [doc for doc, _ in retrieved_docs]
                    scores = [score for _, score in retrieved_docs]
                else:
                    documents = retrieved_docs
                    scores = [1.0] * len(documents)

        else:
            # No re-ranking: extract documents from retrieved results
            if isinstance(retrieved_docs[0], tuple):
                documents = [doc for doc, _ in retrieved_docs]
                scores = [score for _, score in retrieved_docs]
            else:
                documents = retrieved_docs
                scores = [1.0] * len(documents)

        logger.info(f"Using {len(documents)} documents for generation")

        # =====================================================================
        # STEP C: Generate answer using Phase 1 logic
        # =====================================================================

        # Format context
        context = format_documents_for_context(documents)

        # Create prompt
        prompt = f"{context}\n\nQuestion: {optimized_query}\n\nAnswer:"

        # Generate response
        messages = [
            HumanMessage(content=RAG_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]

        answer = self.llm.generate(messages)

        # =====================================================================
        # STEP D: Apply self-critique if enabled
        # =====================================================================

        critique_result = None
        should_refine = False

        if use_self_critique and self.enable_self_critique and self.self_critique_agent:
            logger.info("Applying self-critique to generated answer")
            features_used['self_critique'] = True

            try:
                critique_result = self.self_critique_agent.critique(
                    question=original_query,
                    answer=answer,
                    context=documents,
                )

                should_refine = self.self_critique_agent.should_refine(critique_result)

                if should_refine:
                    logger.warning(
                        f"Self-critique suggests refinement: "
                        f"quality={critique_result.get('overall_quality', 'Unknown')}"
                    )
                else:
                    logger.info(
                        f"Self-critique passed: "
                        f"quality={critique_result.get('overall_quality', 'Unknown')}"
                    )

            except Exception as e:
                logger.error(f"Self-critique failed: {str(e)}")
                critique_result = {'error': str(e)}

        # =====================================================================
        # Build final response
        # =====================================================================

        response = {
            'answer': answer,
            'num_sources': len(documents),
            'phase2_metadata': {
                'original_query': original_query,
                'optimized_query': optimized_query,
                'features_used': features_used,
            },
        }

        # Add sources if requested
        if return_sources:
            response['sources'] = [
                {
                    'content': doc.page_content[:200] + '...',  # Truncate for brevity
                    'metadata': doc.metadata,
                    'score': score,
                }
                for doc, score in zip(documents, scores)
            ]

        # Add context if requested
        if return_context:
            response['context'] = context

        # Add critique if performed
        if critique_result:
            response['critique'] = critique_result
            response['should_refine'] = should_refine

        logger.info("Enhanced query complete")
        return response

    def get_phase2_config(self) -> Dict[str, Any]:
        """
        Get Phase 2 configuration and component status.

        Returns:
            Dictionary with Phase 2 feature flags and component info
        """
        config = {
            'phase': 2,
            'features': {
                'query_rewriting': {
                    'enabled': self.enable_query_rewriting,
                    'component': 'QueryRewriter' if self.query_rewriter else None,
                },
                'multi_query': {
                    'enabled': self.enable_multi_query,
                    'component': 'MultiQueryRetriever' if self.multi_query_retriever else None,
                },
                'reranking': {
                    'enabled': self.enable_reranking,
                    'mode': self.reranking_mode,
                    'component': self.reranker.__class__.__name__ if self.reranker else None,
                },
                'react_agent': {
                    'enabled': self.enable_react_agent,
                    'component': 'ReActAgent' if self.react_agent else None,
                    'max_iterations': self.react_agent.max_iterations if self.react_agent else None,
                },
                'self_critique': {
                    'enabled': self.enable_self_critique,
                    'component': 'SelfCritiqueAgent' if self.self_critique_agent else None,
                },
            },
            'enabled_count': sum([
                self.enable_query_rewriting,
                self.enable_multi_query,
                self.enable_reranking,
                self.enable_react_agent,
                self.enable_self_critique,
            ]),
        }

        # Add base pipeline stats
        config['base_pipeline'] = super().get_stats()

        return config

    def get_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive pipeline statistics including Phase 2 components.

        Returns:
            Dictionary with system statistics
        """
        # Get base stats
        stats = super().get_stats()

        # Add Phase 2 info
        stats['phase2_config'] = self.get_phase2_config()

        return stats


# Convenience function
def create_enhanced_pipeline(
    collection_name: str = None,
    reinitialize: bool = False,
    # Quick presets
    preset: str = "standard",  # "minimal", "standard", "full", "custom"
    # Custom configuration (if preset="custom")
    enable_query_rewriting: bool = True,
    enable_multi_query: bool = True,
    enable_reranking: bool = True,
    enable_react_agent: bool = False,
    enable_self_critique: bool = True,
) -> EnhancedRAGPipeline:
    """
    Create an Enhanced RAG pipeline with preset configurations.

    Args:
        collection_name: Optional collection name
        reinitialize: Whether to start fresh

        preset: Configuration preset:
            - "minimal": Only query rewriting (fastest, cheapest)
            - "standard": Query rewriting + multi-query + reranking (recommended)
            - "full": All features including ReAct agent (most powerful, expensive)
            - "custom": Use individual enable flags

        # Custom flags (only used if preset="custom")
        enable_query_rewriting: Enable query optimization
        enable_multi_query: Enable multi-query generation
        enable_reranking: Enable document re-ranking
        enable_react_agent: Enable ReAct reasoning
        enable_self_critique: Enable answer validation

    Returns:
        EnhancedRAGPipeline instance

    Examples:
        >>> # Standard configuration (recommended)
        >>> pipeline = create_enhanced_pipeline(preset="standard")

        >>> # Full features including ReAct agent
        >>> pipeline = create_enhanced_pipeline(preset="full")

        >>> # Minimal configuration (fastest)
        >>> pipeline = create_enhanced_pipeline(preset="minimal")

        >>> # Custom configuration
        >>> pipeline = create_enhanced_pipeline(
        ...     preset="custom",
        ...     enable_query_rewriting=True,
        ...     enable_multi_query=False,
        ...     enable_reranking=True,
        ...     enable_react_agent=False,
        ...     enable_self_critique=True,
        ... )
    """
    # Apply presets
    if preset == "minimal":
        # Minimal: Just query rewriting
        enable_query_rewriting = True
        enable_multi_query = False
        enable_reranking = False
        enable_react_agent = False
        enable_self_critique = False
        logger.info("Using preset: MINIMAL (fastest, cheapest)")

    elif preset == "standard":
        # Standard: Query optimization + reranking
        enable_query_rewriting = True
        enable_multi_query = True
        enable_reranking = True
        enable_react_agent = False
        enable_self_critique = True
        logger.info("Using preset: STANDARD (recommended)")

    elif preset == "full":
        # Full: All features including ReAct
        enable_query_rewriting = True
        enable_multi_query = True
        enable_reranking = True
        enable_react_agent = True
        enable_self_critique = True
        logger.info("Using preset: FULL (most powerful, more expensive)")

    elif preset == "custom":
        # Custom: Use provided flags
        logger.info("Using preset: CUSTOM")
    else:
        logger.warning(f"Unknown preset '{preset}', using STANDARD")
        preset = "standard"
        enable_query_rewriting = True
        enable_multi_query = True
        enable_reranking = True
        enable_react_agent = False
        enable_self_critique = True

    return EnhancedRAGPipeline(
        collection_name=collection_name,
        reinitialize=reinitialize,
        enable_query_rewriting=enable_query_rewriting,
        enable_multi_query=enable_multi_query,
        enable_reranking=enable_reranking,
        enable_react_agent=enable_react_agent,
        enable_self_critique=enable_self_critique,
    )
