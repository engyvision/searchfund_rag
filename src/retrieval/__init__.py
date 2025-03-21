"""
Retrieval functionality for the Web Scraper project.

This package contains modules for retrieving documents using various techniques,
including vector similarity, BM25, and hybrid approaches with LLM reranking.
"""

from src.retrieval.retrieval import DocumentRetriever
from src.retrieval.contextual_embeddings import ContextualEmbeddingRetriever
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.reranker import LLMReranker

__all__ = [
    'DocumentRetriever',
    'ContextualEmbeddingRetriever',
    'BM25Retriever',
    'HybridRetriever',
    'LLMReranker'
]