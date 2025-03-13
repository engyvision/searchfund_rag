"""
LLM integration for the Web Scraper project.

This package contains modules for integrating with language models
for query clarification and answer generation.
"""

from src.llm.query_clarification import QueryClarifier
from src.llm.answer_generation import AnswerGenerator

__all__ = ['QueryClarifier', 'AnswerGenerator']