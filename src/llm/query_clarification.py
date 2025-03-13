"""
Query clarification using LLMs for the Web Scraper project.

This module provides functionality for clarifying and enhancing user queries
using language models to improve retrieval performance.
"""

from typing import Dict, Any, Optional
from openai import OpenAI

from src.core import get_logger, OPENAI_CONFIG

# Initialize logger
logger = get_logger("llm.query_clarification")

class QueryClarifier:
    """Clarify and enhance user queries using LLMs."""
    
    def __init__(
        self, 
        api_key: str = OPENAI_CONFIG.api_key,
        model: str = OPENAI_CONFIG.clarification_model
    ):
        """Initialize the query clarifier.
        
        Args:
            api_key: OpenAI API key
            model: Model to use for query clarification
        """
        self.client = OpenAI(api_key=api_key)
        self.model = model
        logger.info(f"Initialized QueryClarifier with model: {model}")
    
    def clarify_query(
        self, 
        query: str,
        max_tokens: int = 100,
        temperature: float = 0.5
    ) -> str:
        """Clarify a user query to improve retrieval performance.
        
        Args:
            query: The user's query
            max_tokens: Maximum tokens for the response
            temperature: Sampling temperature
            
        Returns:
            str: The clarified query
        """
        logger.info(f"Clarifying query: {query}")
        
        system_prompt = """
        Understand the user query, infer the type of user (such as a searcher, investor, or business owner), and rephrase the query to be more specific and suitable for analysis by an AI specialized in search funds. If the type of user cannot be inferred from the query, assume the user is a searcher.

        # Steps

        1. **Analyze the query**: Break down the user's question to identify key elements that might indicate the user's identity or goals.
        2. **Infer user type**: Use indicators from the query to deduce whether the user is a searcher, investor, business owner, or another relevant type. If unclear, revert to the default assumption of the user being a searcher (aka search fund entrepreneur).
        3. **Rephrase the query**: Reformulate the user's query to enhance specificity and clarity, ensuring it is tailored to the user's inferred identity and intentions.
        4. **Ensure clarity**: Confirm that the rephrased query removes ambiguity and is designed for precise interpretation by the AI focused on search funds.

        # Output Format

        - Present the rephrased query clearly.

        # Examples

        **Example 1:**

        - **Original Query:** "How does market volatility affect my investments?"
        - **User Type Analysis:** Investor inferred from mention of "investments."
        - **Rephrased Query:** "As an investor, how do fluctuations in the market impact the search fund sector specifically?"

        **Example 2:**

        - **Original Query:** "Why should I sell to a search fund ?"
        - **User Type Analysis:** Business owner inferred from "sell to a search fund."
        - **Rephrased Query:** "For a small business owner, what are the main advantages of selling their business specifically to a search fund entrepreneur rather than to a PE fund or strategic buyer?"

        **Example 3:**

        - **Original Query:** "What are key factors in evaluating a company?"
        - **User Type Analysis:** Cannot specifically infer user type, assume search fund entrepreneur.
        - **Rephrased Query:** "As a search fund entrepreneur, what key factors should be considered when evaluating a company's potential for acquisition?"

        # Notes

        - In cases where elements pointing to a specific user type are ambiguous, default to assuming a search fund entrepreneur.
        - Ensure that the rephrased query is distinct and clearly outlines specific concerns related to the search fund context.
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ],
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            clarified_query = response.choices[0].message.content.strip()
            logger.info(f"Clarified query: {clarified_query}")
            return clarified_query
        except Exception as e:
            logger.error(f"Error clarifying query: {e}")
            # Fall back to original query if clarification fails
            return query