"""
RAG System - Retrieval Augmented Generation with LangChain and FAISS.

This package initializes the RAG system and configures warning filters.
"""
import warnings

# Suppress Pydantic V1 compatibility warnings from LangChain
# This is a known issue with LangChain's internal use of Pydantic V1 for backwards compatibility
# It doesn't affect functionality and will be resolved in future LangChain versions
warnings.filterwarnings(
    "ignore",
    message="Core Pydantic V1 functionality isn't compatible with Python 3.14 or greater.",
    category=UserWarning,
    module="langchain_core._api.deprecation"
)

__version__ = "1.0.0"
__author__ = "Dele Olowoyo"
