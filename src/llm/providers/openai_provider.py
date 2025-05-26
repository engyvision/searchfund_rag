"""
OpenAI provider implementation.

This module implements the LLM provider interface for OpenAI.
"""

import os
import json
from typing import Dict, List, Any, Optional, Union, Tuple, ClassVar
from tenacity import retry, wait_random_exponential, stop_after_attempt
from openai import OpenAI

from src.llm.providers.base_provider import BaseLLMProvider
from src.core import get_logger

# Initialize logger
logger = get_logger("llm.providers.openai")


class OpenAIProvider(BaseLLMProvider):
    """OpenAI provider implementation."""
    
    # Default environment variable name for API key
    DEFAULT_API_KEY_ENV: ClassVar[str] = "OPENAI_API_KEY"
    
    # Static provider name
    PROVIDER_NAME: ClassVar[str] = "openai"
    
    # Default model names
    DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
    DEFAULT_COMPLETION_MODEL = "o3-mini-2025-01-31"
    DEFAULT_CLARIFICATION_MODEL = "gpt-4o-mini"
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the OpenAI provider from configuration.
        
        Args:
            config: Provider configuration dictionary with all function types
        """
        # Get API key from config or environment variable
        api_key = self._get_api_key_from_config(config)
        
        # Initialize client
        self.client = OpenAI(api_key=api_key)
        
        # Get model names from config
        self.embedding_model = self._get_model_for_function(config, "embeddings", self.DEFAULT_EMBEDDING_MODEL)
        self.completion_model = self._get_model_for_function(config, "answer_generation", self.DEFAULT_COMPLETION_MODEL)
        self.clarification_model = self._get_model_for_function(config, "query_clarification", self.DEFAULT_CLARIFICATION_MODEL)
        
        # Save config for future reference
        self.config = config
        
        logger.info(f"Initialized OpenAIProvider with models: embedding={self.embedding_model}, completion={self.completion_model}, clarification={self.clarification_model}")
    
    def _get_api_key_from_config(self, config: Dict[str, Any]) -> str:
        """Get API key from config or environment variable.
        
        Args:
            config: Provider configuration dictionary
            
        Returns:
            str: API key
            
        Raises:
            ValueError: If API key is not found
        """
        # Try to get API key from each function type config
        for function_type, function_config in config.items():
            if function_config.get("provider") == "openai" and "api_key" in function_config:
                return function_config["api_key"]
        
        # If not found in config, try environment variable
        api_key = os.getenv(self.DEFAULT_API_KEY_ENV)
        if not api_key:
            raise ValueError(f"OpenAI API key not found in config or environment variable {self.DEFAULT_API_KEY_ENV}")
        
        return api_key
    
    def _get_model_for_function(self, config: Dict[str, Any], function_type: str, default_model: str) -> str:
        """Get model name for a specific function type.
        
        Args:
            config: Provider configuration dictionary
            function_type: Function type (embeddings, answer_generation, etc.)
            default_model: Default model name to use if not specified
            
        Returns:
            str: Model name
        """
        # Check if this function type is configured for this provider
        if function_type in config and config[function_type].get("provider") == "openai":
            return config[function_type].get("model", default_model)
        
        # Otherwise return default
        return default_model
    
    @property
    def provider_name(self) -> str:
        """Get the name of the provider.
        
        Returns:
            str: The provider name
        """
        return "openai"
    
    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def get_embedding(self, text: str) -> Tuple[List[float], Optional[int]]:
        """Get embeddings for a single text.
        
        Args:
            text: The text to embed
            
        Returns:
            Tuple[List[float], Optional[int]]: The embedding and token count
        """
        try:
            response = self.client.embeddings.create(input=text, model=self.embedding_model)
            embedding = response.data[0].embedding
            tokens_used = response.usage.total_tokens if hasattr(response, "usage") and hasattr(response.usage, "total_tokens") else None
            return embedding, tokens_used
        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            raise
    
    def get_embedding_vector(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for a list of texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List[List[float]]: List of embeddings
        """
        try:
            response = self.client.embeddings.create(input=texts, model=self.embedding_model)
            embeddings = [data.embedding for data in response.data]
            return embeddings
        except Exception as e:
            logger.error(f"Error getting embeddings for batch: {e}")
            raise
    
    def generate_answer(
        self,
        query: str,
        context: str,
        max_completion_tokens: int = 10000,
        reasoning_effort: str = "medium",
        include_source_explanations: bool = True
    ) -> str:
        """Generate an answer based on context.
        
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
                model=self.completion_model,
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
        
        system_prompt = """
        Understand the user query, infer the type of user (such as a searcher, investor, or business owner), and rephrase the query to be more specific and suitable for analysis by an AI specialized in search funds. If the type of user cannot be inferred from the query, assume the user is a searcher.

        # Steps

        1. **Analyze the query**: Break down the user's question to identify key elements that might indicate the user's identity or goals.
        2. **Infer user type**: Use indicators from the query to deduce whether the user is a searcher, investor, business owner, or another relevant type. If unclear, revert to the default assumption of the user being a searcher (aka search fund entrepreneur).
        3. **Rephrase the query**: Reformulate the user's query to enhance specificity and clarity, ensuring it is tailored to the user's inferred identity and intentions.
        4. **Ensure clarity**: Confirm that the rephrased query removes ambiguity and is designed for precise interpretation by the AI focused on search funds.

        # Output Format

        - Present only the rephrased query clearly.

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
                model=self.clarification_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ],
                max_completion_tokens=max_completion_tokens,
                temperature=temperature
            )
            
            clarified_query = response.choices[0].message.content.strip()
            logger.info(f"Clarified query: {clarified_query}")
            return clarified_query
        except Exception as e:
            logger.error(f"Error clarifying query: {e}")
            # Fall back to original query if clarification fails
            return query
    
    def rerank(
        self,
        query: str,
        retrieved_documents: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Rerank documents using an LLM.
        
        Args:
            query: User query
            retrieved_documents: Documents retrieved by a retriever
            
        Returns:
            List[Dict[str, Any]]: Reranked list of documents
        """
        if not retrieved_documents:
            return []
            
        prompt = self._construct_reranking_prompt(query, retrieved_documents)
        
        try:
            response = self.client.chat.completions.create(
                model=self.completion_model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that evaluates document relevance."},
                    {"role": "user", "content": prompt}
                ],
                # temperature=0.0
            )
            
            result_text = response.choices[0].message.content
            
            # Extract JSON from the response
            result_json = json.loads(result_text)
            reranked_indices = sorted(
                result_json["reranked_documents"],
                key=lambda x: x["relevance_score"],
                reverse=True
            )
            
            # Reorder the original documents
            reranked_documents = []
            for item in reranked_indices:
                doc_index = item["document_index"]
                if doc_index < len(retrieved_documents):
                    doc = retrieved_documents[doc_index].copy()
                    doc["relevance_score"] = item["relevance_score"]
                    doc["explanation"] = item["explanation"]
                    
                    # Log reranking information
                    doc_preview = doc["document"][:50].replace('\n', ' ').strip()
                    file_info = doc.get("file", "unknown")
                    if isinstance(file_info, str) and len(file_info) > 20:
                        file_info = file_info[:20] + "..."
                    
                    logger.info(f"Reranked doc {doc_index+1} → position {len(reranked_documents)+1}: score={item['relevance_score']}/10, file={file_info}, '{doc_preview}...'")
                    
                    reranked_documents.append(doc)
            
            return reranked_documents
            
        except Exception as e:
            logger.error(f"Error during document reranking: {str(e)}")
            # Fall back to original ranking
            return retrieved_documents
    
    def _construct_reranking_prompt(self, query: str, documents: List[Dict[str, Any]]) -> str:
        """Construct a prompt for the LLM to rerank documents.
        
        Args:
            query: User query
            documents: List of retrieved documents
        
        Returns:
            Prompt string for the LLM
        """
        prompt = f"""
        You are an expert at determining if a document is relevant to a query.
        
        Query: {query}
        
        Below are documents retrieved by a search system. For each document, rate its relevance to the query on a scale from 0 to 10,
        where 0 means completely irrelevant and 10 means perfectly relevant.
        
        For each document, provide ONLY:
        1. A relevance score (0-10)
        2. A very brief (1-2 sentence) explanation
        
        Format your response as valid JSON with the following structure:
        {{
            "reranked_documents": [
                {{
                    "document_index": 0,
                    "relevance_score": 8,
                    "explanation": "Explanation here"
                }},
                ...
            ]
        }}
        
        Documents:
        """
        
        # Only use a shorter preview of each document to save tokens
        for i, doc in enumerate(documents):
            # Get a short preview - first 500 chars is usually enough for relevance judgment
            doc_preview = doc['document'][:500] + "..."
            prompt += f"\nDocument {i}:\n{doc_preview}\n"
        
        return prompt