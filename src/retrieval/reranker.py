from typing import List, Dict, Any
from openai import OpenAI
import json
from src.core import OPENAI_CONFIG, get_logger

logger = get_logger(__name__)

class LLMReranker:
    """Uses an LLM to rerank retrieved documents based on query relevance."""
    
    def __init__(self, model_name: str = "o3-mini-2025-01-31"):
        """
        Initialize the LLM reranker.
        
        Args:
            model_name: Name of the LLM model to use
        """
        self.client = OpenAI(api_key=OPENAI_CONFIG.api_key)
        self.model_name = model_name
    
    def _construct_reranking_prompt(self, query: str, documents: List[Dict[str, Any]]) -> str:
        """
        Construct a prompt for the LLM to rerank documents.
        
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
            
        prompt = self._construct_reranking_prompt(query, retrieved_documents)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
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