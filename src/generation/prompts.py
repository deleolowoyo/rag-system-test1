"""
Prompt templates for RAG system.
Includes system prompts, context formatting, and citation requirements.
"""
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from typing import List
from langchain_core.documents import Document


# System prompt for RAG
RAG_SYSTEM_PROMPT = """You are a helpful AI assistant that answers questions based on the provided context.

CRITICAL RULES:
1. Answer ONLY using information from the provided context
2. If the answer is not in the context, say "I don't have enough information to answer that question."
3. ALWAYS cite your sources using [Source: filename, page X] format
4. Be concise and accurate
5. If multiple sources support your answer, cite all of them

Remember: It's better to say you don't know than to make up information."""


# Context formatting template
CONTEXT_TEMPLATE = """Context from documents:
---
{context}
---"""


# Full RAG prompt template
RAG_PROMPT_TEMPLATE = """You are a helpful AI assistant that answers questions based on the provided context.

CRITICAL RULES:
1. Answer ONLY using information from the provided context
2. If the answer is not in the context, say "I don't have enough information to answer that question."
3. ALWAYS cite your sources using [Source: filename, page X] format
4. Be concise and accurate
5. If multiple sources support your answer, cite all of them

Context from documents:
---
{context}
---

Question: {question}

Answer: """


def format_documents_for_context(documents: List[Document]) -> str:
    """
    Format retrieved documents into a context string.
    
    Args:
        documents: List of retrieved documents
        
    Returns:
        Formatted context string with sources
    """
    if not documents:
        return "No relevant documents found."
    
    formatted_chunks = []
    
    for i, doc in enumerate(documents, 1):
        # Extract metadata
        source = doc.metadata.get('file_name', doc.metadata.get('source', 'unknown'))
        page = doc.metadata.get('page', 'N/A')
        
        # Format chunk with source citation
        chunk = f"""Document {i}:
{doc.page_content}
[Source: {source}, page {page}]
"""
        formatted_chunks.append(chunk)
    
    return "\n".join(formatted_chunks)


def create_rag_prompt() -> ChatPromptTemplate:
    """
    Create the RAG chat prompt template.
    
    Returns:
        ChatPromptTemplate for RAG
    """
    return ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)


def create_context_aware_prompt(
    system_prompt: str = None,
) -> ChatPromptTemplate:
    """
    Create a context-aware prompt with custom system message.
    
    Args:
        system_prompt: Optional custom system prompt
        
    Returns:
        ChatPromptTemplate
    """
    system = system_prompt or RAG_SYSTEM_PROMPT
    
    template = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", CONTEXT_TEMPLATE + "\n\nQuestion: {question}"),
    ])
    
    return template


# Prompt for query rewriting (Phase 2)
QUERY_REWRITE_TEMPLATE = """Given the following question, rewrite it to be more specific and searchable.
Make it clear and focused on the key information needed.

Original question: {question}

Rewritten question:"""


# Prompt for summarization
SUMMARIZATION_TEMPLATE = """Summarize the following text concisely, preserving key information:

Text:
{text}

Summary:"""


# Export commonly used prompts
__all__ = [
    'RAG_SYSTEM_PROMPT',
    'RAG_PROMPT_TEMPLATE',
    'format_documents_for_context',
    'create_rag_prompt',
    'create_context_aware_prompt',
]
