from typing import List, Dict, Any
import json
from src.core import get_logger, LLM_CONFIG
from src.llm.providers import get_provider

logger = get_logger(__name__)

class LLMReranker:
    """Uses an LLM to rerank retrieved documents based on query relevance."""
    
    def __init__(self):
        """
        Initialize the LLM reranker.
        """
        self.provider = get_provider(LLM_CONFIG.config_data, "reranking")
        logger.info(f"Initialized LLMReranker with provider: {self.provider.provider_name}")
    
    def rerank(self, query: str, retrieved_documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Rerank documents using an LLM.
        
        Args:
            query: User query
            retrieved_documents: Documents retrieved by a retriever
            
        Returns:
            Reranked list of documents
        """
        if not retrieved_documents:
            return []
        
        return self.provider.rerank(query, retrieved_documents)