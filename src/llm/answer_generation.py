"""
Answer generation using LLMs for the Web Scraper project.

This module provides functionality for generating answers to user queries
based on retrieved document context using language models.
"""

from typing import Dict, List, Any, Optional
from openai import OpenAI

from src.core import get_logger, OPENAI_CONFIG

# Initialize logger
logger = get_logger("llm.answer_generation")

class AnswerGenerator:
    """Generate answers to user queries using LLMs."""
    
    def __init__(
        self, 
        api_key: str = OPENAI_CONFIG.api_key,
        model: str = OPENAI_CONFIG.completion_model
    ):
        """Initialize the answer generator.
        
        Args:
            api_key: OpenAI API key
            model: Model to use for answer generation
        """
        self.client = OpenAI(api_key=api_key)
        self.model = model
        logger.info(f"Initialized AnswerGenerator with model: {model}")
    
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
        
        system_prompt = """
        You are tasked with acting as a search fund expert to answer a specific question based on a provided context and augmented with your own knowledge and web results, if available. Ensure your response targets the correct user group—either a searcher, search fund investor, business owner, or intermediary—based on the question's details. 
        
        The provided context has been retrieved using an advanced hybrid retrieval system that combines multiple methods:
        1. Contextual embeddings that consider the query when retrieving documents
        2. Keyword-based BM25 search for stronger term matches
        3. LLM-based reranking that evaluates document relevance to the query
        
        The context includes relevance scores from these methods. Higher scores indicate greater relevance. The provided context should be considered the most reliable source unless newer or strongly contradictory evidence is found through your knowledge or online sources, in which case this should be highlighted in your response.

        # Steps

        1. **Understanding the Question**: Carefully read and comprehend the question to ensure it is appropriately targeted and complete.

        2. **Analyzing the Context**: Review the provided context thoroughly as it serves as the primary source of information for the question.

        3. **Identifying the Audience**: Determine whether the context is more relevant to search fund investors, searchers, business owners, or intermediaries and adjust your response accordingly.

        4. **Applying AI Knowledge**: Incorporate your existing knowledge of the search fund landscape to enhance the response.

        5. **Checking Web Results**: If necessary, obtain current web information or updates on the topic that may not be covered by the context or your knowledge.

        6. **Comparison**: Compare findings from the context, AI knowledge, and web results. Identify any contradictions, especially if newer information invalidates the context.

        7. **Construct the Response**: Synthesize all the data to provide a comprehensive, well-reasoned answer enriched by multiple sources, targeting the correct user group.

        # Output Format

        Provide a structured and well-organized response. Highlight the sources of each piece of information within, and ensure the answer is aligned with the identified audience.

        # Notes

        - Clearly explain any substantial contradictions between the provided context and new findings.
        - Always consider the timeliness and reliability of web sources used in the response.
        - Ensure clarity in explanations and conclusions, while targeting the identified audience.
        - When generating answers, always indicate which parts of the response come directly from the retrieved context using phrases like "According to the [document source name])...". 
        - If any part of the response relies on your own knowledge, clearly preface it with "Additionally, based on [my own knowledge] or [specific web source]...".
        """
        
        user_prompt = f"""
        Context:
        {context}

        Question: {query}

        Answer:
        """
        
        if include_source_explanations:
            user_prompt += """
            
            After your answer, please include a brief explanation of which sources were most helpful for answering this query and why.
            """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_completion_tokens=max_completion_tokens,
                reasoning_effort=reasoning_effort,
                # temperature=0.2  # Lower temperature for more deterministic responses
            )
            
            answer = response.choices[0].message.content.strip()
            
            # Log token usage if available
            usage = getattr(response, "usage", None)
            if usage:
                logger.info(f"Token usage: Prompt: {usage.prompt_tokens}, Completion: {usage.completion_tokens}, Total: {usage.total_tokens}")
                
                # Check if we're close to the token limit
                if usage.completion_tokens > max_completion_tokens * 0.9:
                    logger.warning(f"Answer completion is using {usage.completion_tokens} tokens, which is close to the max limit of {max_completion_tokens}")
            
            logger.info("Answer generated successfully")
            return answer
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return f"I apologize, but I encountered an error while generating the answer: {str(e)}"