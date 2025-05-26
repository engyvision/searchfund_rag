"""
Query clarification using LLMs for the Web Scraper project.

This module provides functionality for clarifying and enhancing user queries
using language models to improve retrieval performance.
"""

from typing import Dict, Any, Optional

from src.core import get_logger, LLM_CONFIG
from src.llm.providers import get_provider

# Initialize logger
logger = get_logger("llm.query_clarification")

class QueryClarifier:
    """Clarify and enhance user queries using LLMs."""
    
    def __init__(self):
        """Initialize the query clarifier."""
        self.provider = get_provider(LLM_CONFIG.config_data, "query_clarification")
        logger.info(f"Initialized QueryClarifier with provider: {self.provider.provider_name}")
    
    def clarify_query(
        self, 
        query: str,
        max_completion_tokens: int = 100,
        temperature: float = 0.5
    ) -> str:
        """Clarify a user query to improve retrieval performance.
        
        Args:
            query: The user's query
            max_completion_tokens: Maximum tokens for the response
            temperature: Sampling temperature
            
        Returns:
            str: The clarified query
        """
        logger.info(f"Clarifying query: {query}")
        return self.provider.clarify_query(query, max_completion_tokens, temperature)