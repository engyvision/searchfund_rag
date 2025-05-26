"""
Base provider interface for LLM integrations.

This module defines the abstract base class that all LLM providers must implement.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union, Tuple, ClassVar


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    # Default environment variable names for API keys
    DEFAULT_API_KEY_ENV: ClassVar[str] = ""
    
    # Provider name - can be overridden by subclasses
    PROVIDER_NAME: ClassVar[str] = ""
    
    @abstractmethod
    def __init__(self, config: Dict[str, Any]):
        """Initialize the provider from configuration.
        
        Args:
            config: Provider configuration dictionary with all function types
        """
        pass
    
    @property
    def provider_name(self) -> str:
        """Get the name of the provider.
        
        Returns:
            str: The provider name
        """
        # First check for static PROVIDER_NAME class variable
        if self.__class__.PROVIDER_NAME:
            return self.__class__.PROVIDER_NAME
        
        # Fallback to deriving from class name
        class_name = self.__class__.__name__
        if class_name.endswith('Provider'):
            return class_name[:-8].lower()  # Remove 'Provider' and lowercase
        return class_name.lower()
    
    @abstractmethod
    def get_embedding(self, text: str) -> Tuple[List[float], Optional[int]]:
        """Get embeddings for a single text.
        
        Args:
            text: The text to embed
            
        Returns:
            Tuple[List[float], Optional[int]]: The embedding and token count
        """
        pass
    
    @abstractmethod
    def get_embedding_vector(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for a list of texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List[List[float]]: List of embeddings
        """
        pass
    
    @abstractmethod
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
        pass
    
    @abstractmethod
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
        pass
    
    @abstractmethod
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
        pass