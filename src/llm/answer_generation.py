"""
Answer generation using LLMs for the Web Scraper project.

This module provides functionality for generating answers to user queries
based on retrieved document context using language models.
"""

from typing import Dict, List, Any, Optional

from src.core import get_logger, LLM_CONFIG
from src.llm.providers import get_provider

# Initialize logger
logger = get_logger("llm.answer_generation")

class AnswerGenerator:
    """Generate answers to user queries using LLMs."""
    
    def __init__(self):
        """Initialize the answer generator."""
        self.provider = get_provider(LLM_CONFIG.config_data, "answer_generation")
        logger.info(f"Initialized AnswerGenerator with provider: {self.provider.provider_name}")
    
    def generate_answer(
        self, 
        query: str,
        context: str,
        max_completion_tokens: int = 10000,
        reasoning_effort: str = "medium",
        include_source_explanations: bool = True
    ) -> str:
        """Generate an answer to a query based on context.
        
        Args:
            query: The user's query
            context: The context information from retrieved documents
            max_completion_tokens: Maximum tokens for the response
            reasoning_effort: Reasoning effort for the model (low, medium, high)
            include_source_explanations: Whether to include explanations of source relevance
            
        Returns:
            str: The generated answer
        """
        logger.info(f"Generating answer for query: {query}")
        return self.provider.generate_answer(
            query=query,
            context=context,
            max_completion_tokens=max_completion_tokens,
            reasoning_effort=reasoning_effort,
            include_source_explanations=include_source_explanations
        )