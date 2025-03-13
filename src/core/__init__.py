"""
Core functionality for the Web Scraper project.

This package contains core modules that are used throughout the application.
"""

from src.core.config import (
    CONFIG, 
    OPENAI_CONFIG, 
    FAISS_CONFIG, 
    PDF_CONFIG, 
    OPENAI_API_KEY
)

from src.core.logging import get_logger, setup_logging

__all__ = [
    'CONFIG', 
    'OPENAI_CONFIG', 
    'FAISS_CONFIG', 
    'PDF_CONFIG', 
    'OPENAI_API_KEY',
    'get_logger',
    'setup_logging'
]