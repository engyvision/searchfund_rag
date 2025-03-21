from typing import List, Dict, Any
import numpy as np
import re
from rank_bm25 import BM25Okapi
from src.data.preprocessing import preprocess_text
from src.core.logging import get_logger

# Initialize logger
logger = get_logger("retrieval.bm25")

class BM25Retriever:
    """Retriever that uses BM25 algorithm for keyword-based retrieval."""
    
    def __init__(self, top_k: int = 10):
        """
        Initialize the BM25 retriever.
        
        Args:
            top_k: Number of documents to retrieve
        """
        self.top_k = top_k
        self.bm25 = None
        self.tokenized_corpus = None
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        # Simple tokenization - you may want to use a more sophisticated approach
        text = preprocess_text(text)
        return re.findall(r'\w+', text.lower())
    
    def fit(self, documents: List[str]):
        """
        Fit the BM25 model on the corpus.
        
        Args:
            documents: List of document texts
        """
        self.tokenized_corpus = [self._tokenize(doc) for doc in documents]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
    
    def retrieve(self, query: str, documents: List[str], document_metadata: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Retrieve documents using BM25.
        
        Args:
            query: User query
            documents: List of document texts
            document_metadata: Metadata for each document
            
        Returns:
            List of retrieved documents with their metadata and scores
        """
        if self.bm25 is None:
            self.fit(documents)
            
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top k indices
        top_indices = np.argsort(scores)[-self.top_k:][::-1]
        
        # Return results
        results = [
            {
                "document": documents[i],
                "metadata": document_metadata[i],
                "score": float(scores[i])
            }
            for i in top_indices
        ]
        
        # Log BM25 retrieval results
        for i, result in enumerate(results):
            doc_preview = result["document"][:50].replace('\n', ' ').strip()
            score = result.get("score", 0)
            
            # Get metadata for logging
            meta = result.get("metadata", {})
            file_info = meta.get("file", "unknown")
            if isinstance(file_info, str) and len(file_info) > 20:
                file_info = file_info[:20] + "..."
                
            logger.debug(f"BM25 result {i+1}: file={file_info}, score={score:.3f} - '{doc_preview}...'")
        
        return results