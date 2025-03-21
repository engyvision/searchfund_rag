from typing import List, Dict, Any
from src.retrieval.contextual_embeddings import ContextualEmbeddingRetriever
from src.retrieval.bm25 import BM25Retriever
from src.core.logging import get_logger

# Initialize logger
logger = get_logger("retrieval.hybrid")

class HybridRetriever:
    """Combines multiple retrieval methods for better results."""
    
    def __init__(
        self, 
        embedding_retriever: ContextualEmbeddingRetriever,
        bm25_retriever: BM25Retriever,
        top_k: int = 10
    ):
        """
        Initialize the hybrid retriever.
        
        Args:
            embedding_retriever: Contextual embedding retriever
            bm25_retriever: BM25 retriever
            top_k: Number of documents to retrieve
        """
        self.embedding_retriever = embedding_retriever
        self.bm25_retriever = bm25_retriever
        self.top_k = top_k
    
    def retrieve(self, query: str, documents: List[str], document_metadata: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Retrieve documents using hybrid approach.
        
        Args:
            query: User query
            documents: List of document texts
            document_metadata: Metadata for each document
            
        Returns:
            List of retrieved documents with their metadata and scores
        """
        # Get results from both retrievers
        embedding_results = self.embedding_retriever.retrieve(query, documents, document_metadata)
        bm25_results = self.bm25_retriever.retrieve(query, documents, document_metadata)
        
        # Combine results - use a simple set union with score normalization
        combined_results = {}
        
        # Make sure all results have the required keys
        # The contextual embedding retriever returns "score" not "embedding_score"
        # So we need to standardize the keys first
        standardized_embedding_results = []
        for result in embedding_results:
            standardized_result = {
                "document": result["document"],
                "metadata": result["metadata"],
                "embedding_score": result["score"],  # Rename score to embedding_score
                "score": result["score"]  # Keep original score key too
            }
            standardized_embedding_results.append(standardized_result)
            
            # Add to combined results - just focus on preserving content and metadata
            doc_metadata = result["metadata"]
            
            # Generate a stable ID for deduplication
            doc_id = hash(result["document"][:100])  # Use content hash as a stable ID
                
            if doc_id not in combined_results:
                # Ensure content is directly included
                if "content" not in doc_metadata:
                    doc_metadata["content"] = result["document"]
                
                combined_results[doc_id] = {
                    "document": result["document"],
                    "metadata": doc_metadata,  # Ensure complete metadata
                    "embedding_score": result["score"],
                    "bm25_score": 0.0
                }
                
        # Process BM25 results
        standardized_bm25_results = []
        for result in bm25_results:
            standardized_result = {
                "document": result["document"],
                "metadata": result["metadata"],
                "bm25_score": result["score"],  # Rename score to bm25_score
                "score": result["score"]  # Keep original score key too
            }
            standardized_bm25_results.append(standardized_result)
            
            # Add to combined results with same content-focused approach
            doc_metadata = result["metadata"]
            
            # Generate a stable ID for deduplication
            doc_id = hash(result["document"][:100])  # Use content hash as a stable ID
                
            if doc_id in combined_results:
                combined_results[doc_id]["bm25_score"] = result["score"]
            else:
                # Ensure content is directly included
                if "content" not in doc_metadata:
                    doc_metadata["content"] = result["document"]
                    
                combined_results[doc_id] = {
                    "document": result["document"],
                    "metadata": doc_metadata,  # Ensure complete metadata
                    "embedding_score": 0.0,
                    "bm25_score": result["score"]
                }
        
        # Calculate combined score (simple weighted sum)
        for doc_id, result in combined_results.items():
            # Normalize scores between 0 and 1
            max_embedding_score = max([r["embedding_score"] for r in standardized_embedding_results]) if standardized_embedding_results else 1
            max_bm25_score = max([r["bm25_score"] for r in standardized_bm25_results]) if standardized_bm25_results else 1
            
            norm_embedding_score = result["embedding_score"] / max_embedding_score if max_embedding_score > 0 else 0
            norm_bm25_score = result["bm25_score"] / max_bm25_score if max_bm25_score > 0 else 0
            
            # Weighted combination (adjust weights as needed)
            result["score"] = 0.7 * norm_embedding_score + 0.3 * norm_bm25_score
        
        # Sort and limit to top_k
        sorted_results = sorted(combined_results.values(), key=lambda x: x["score"], reverse=True)[:self.top_k]
        
        # Log hybrid retrieval results
        for i, result in enumerate(sorted_results):
            doc_preview = result["document"][:50].replace('\n', ' ').strip()
            emb_score = result.get("embedding_score", 0)
            bm25_score = result.get("bm25_score", 0)
            combined_score = result.get("score", 0)
            
            # Get metadata for logging
            meta = result.get("metadata", {})
            file_info = meta.get("file", "unknown")
            if isinstance(file_info, str) and len(file_info) > 20:
                file_info = file_info[:20] + "..."
                
            logger.debug(f"Hybrid result {i+1}: file={file_info}, emb={emb_score:.3f}, bm25={bm25_score:.3f}, combined={combined_score:.3f} - '{doc_preview}...'")
        
        return sorted_results