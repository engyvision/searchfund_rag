"""
Data processing functionality for the Web Scraper project.

This package contains modules for data processing, embedding generation,
and vector indexing.
"""

from src.data.preprocessing import TextPreprocessor, DocumentProcessor
from src.data.embeddings import EmbeddingService, EmbeddingManager
from src.data.indexing import FAISSIndexer

__all__ = [
    'TextPreprocessor',
    'DocumentProcessor',
    'EmbeddingService',
    'EmbeddingManager',
    'FAISSIndexer'
]