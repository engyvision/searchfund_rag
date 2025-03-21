"""
Document retrieval using advanced techniques for the Web Scraper project.

This module provides functionality for retrieving relevant documents
using multiple strategies including vector similarity, BM25, and reranking.
"""

import os
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
import json

from src.core.logging import get_logger
from src.core.config import OPENAI_CONFIG
from src.data.embeddings import EmbeddingService
from src.data.indexing import FAISSIndexer
from src.retrieval.contextual_embeddings import ContextualEmbeddingRetriever
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.reranker import LLMReranker

# Initialize logger
logger = get_logger("retrieval")

class DocumentRetriever:
    """Retrieve relevant documents using advanced retrieval techniques."""
    
    def __init__(
        self,
        embedding_service: Optional[EmbeddingService] = None,
        faiss_indexer: Optional[FAISSIndexer] = None,
        use_hybrid_retrieval: bool = True,
        use_reranking: bool = True
    ):
        """Initialize the document retriever.
        
        Args:
            embedding_service: Service for generating embeddings
            faiss_indexer: FAISS indexer for vector similarity search
            use_hybrid_retrieval: Whether to use hybrid retrieval (contextual + BM25)
            use_reranking: Whether to use LLM reranking
        """
        self.embedding_service = embedding_service or EmbeddingService()
        self.faiss_indexer = faiss_indexer or FAISSIndexer()
        self.use_hybrid_retrieval = use_hybrid_retrieval
        self.use_reranking = use_reranking
        
        # Load the FAISS index
        if not self.faiss_indexer.load():
            logger.error("Failed to load FAISS index")
            raise ValueError("FAISS index could not be loaded")
        
        # Set up advanced retrieval components if enabled
        if use_hybrid_retrieval:
            # Create a wrapper function for the embedding service with error handling
            def embedding_wrapper(texts):
                embeddings = []
                for text in texts:
                    try:
                        # Get embedding and handle potential errors
                        embedding, _ = self.embedding_service.get_embedding(text)
                        embeddings.append(embedding)
                    except Exception as e:
                        logger.error(f"Error generating embedding (using fallback): {str(e)}")
                        # Create a zero vector as fallback to avoid crashing
                        # The dimension comes from FAISS configuration
                        embeddings.append([0.0] * 1536)  # text-embedding-3-small has 1536 dimensions
                return embeddings
            
            self.contextual_retriever = ContextualEmbeddingRetriever(
                index=self.faiss_indexer.index,
                embedding_function=embedding_wrapper,
                top_k=20  # Retrieve more for hybrid
            )
            
            self.bm25_retriever = BM25Retriever(top_k=20)  # Retrieve more for hybrid
            
            self.hybrid_retriever = HybridRetriever(
                embedding_retriever=self.contextual_retriever,
                bm25_retriever=self.bm25_retriever,
                top_k=10  # Final number after hybrid combination
            )
        
        # Set up reranker if enabled
        if use_reranking:
            self.reranker = LLMReranker()
            
        logger.info(f"Initialized DocumentRetriever with hybrid={use_hybrid_retrieval}, reranking={use_reranking}")
    
    def retrieve_documents(
        self, 
        query: str, 
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Retrieve documents for a query.
        
        Args:
            query: The query text
            top_k: Number of documents to retrieve
            
        Returns:
            List[Dict[str, Any]]: List of retrieved documents
        """
        logger.info(f"Retrieving top {top_k} documents for query: {query}")
        
        if not self.use_hybrid_retrieval:
            # Use traditional vector search if hybrid is disabled
            embedding, _ = self.embedding_service.get_embedding(query)
            query_vector = np.array(embedding, dtype=np.float32)
            
            # Search for similar documents
            distances, indices, results = self.faiss_indexer.search(
                query_vector=query_vector,
                top_k=top_k
            )
            
            # Enhance results with distance information
            enhanced_results = []
            for i, result in enumerate(results):
                enhanced_result = result.copy()
                enhanced_result["score"] = float(1.0 / (1.0 + distances[0][i]))
                enhanced_result["distance"] = float(distances[0][i])
                enhanced_results.append(enhanced_result)
            
            results = enhanced_results
        else:
            # Use hybrid retrieval (contextual embeddings + BM25)
            logger.info("Using hybrid retrieval (contextual embeddings + BM25)")
            
            # Get all documents and metadata from the FAISS index
            all_documents = []
            all_metadata = []
            
            # First, use traditional vector search to pre-filter candidates
            # This limits the number of documents we need to process with contextual embeddings
            embedding, _ = self.embedding_service.get_embedding(query)
            query_vector = np.array(embedding, dtype=np.float32)
            
            # Get fewer candidates to speed up processing while still maintaining quality
            distances, indices, results = self.faiss_indexer.search(
                query_vector=query_vector,
                top_k=20  # Reduced from 50 to 20 to improve speed
            )
            
            # Extract documents and metadata from the pre-filtered results
            for i, result in enumerate(results):
                if "content" in result and result["content"]:
                    # Truncate document for logging purposes
                    doc_preview = result["content"][:50].replace('\n', ' ').strip()
                    file_info = result.get("file", "unknown")
                    chunk_info = f"chunk_{result.get('chunk_index', 'unknown')}" if result.get('chunk_index') is not None else ""
                    score_info = f"score={result.get('score', 0):.3f}"
                    
                    logger.debug(f"Pre-filtered doc {i+1}: {file_info} {chunk_info} {score_info} - '{doc_preview}...'")
                    
                    all_documents.append(result["content"])
                    all_metadata.append(result)
            
            logger.info(f"Pre-filtered to {len(all_documents)} documents for hybrid search")
            
            if len(all_documents) == 0:
                # No documents found in pre-filtering, use the fallback
                logger.warning("No documents found in pre-filtering, using fallback vector search")
                enhanced_results = []
                for i, result in enumerate(results):
                    enhanced_result = result.copy()
                    enhanced_result["score"] = float(1.0 / (1.0 + distances[0][i]))
                    enhanced_result["distance"] = float(distances[0][i])
                    enhanced_results.append(enhanced_result)
                
                results = enhanced_results[:top_k]
            else:
                # Use hybrid retriever on the pre-filtered results
                results = self.hybrid_retriever.retrieve(query, all_documents, all_metadata)
                
                # Cut to desired number if more retrieved
                results = results[:top_k]
        
        # Apply reranking if enabled
        if self.use_reranking and results:
            logger.info("Applying LLM reranking")
            results = self.reranker.rerank(query, results)
            
            # Cut to desired number if reranking added any metadata
            results = results[:top_k]
        
        # Log final retrieved documents
        logger.info(f"Retrieved {len(results)} documents")
        for i, result in enumerate(results):
            doc_preview = result["document"][:50].replace('\n', ' ').strip()
            score = result.get("score", 0)
            relevance = result.get("relevance_score", "N/A")
            
            logger.info(f"Final document {i+1}: score={score:.3f}, LLM relevance={relevance}, text='{doc_preview}...'")
        
        return results
    
    def retrieve_with_content(
        self,
        query: str,
        top_k: int = 5,
        content_dir: str = None  # No longer used, kept for API compatibility
    ) -> List[Dict[str, Any]]:
        """Retrieve documents with their content.
        
        Args:
            query: The query text
            top_k: Number of documents to retrieve
            content_dir: (Deprecated) Not used, content is stored directly in vector DB
            
        Returns:
            List[Dict[str, Any]]: List of retrieved documents with content
        """
        # Get documents with metadata
        results = self.retrieve_documents(query, top_k)
        
        # Process results to ensure content is available
        for result in results:
            # If content is missing but document is available, use document as content
            if ("content" not in result or not result["content"]) and "document" in result and result["document"]:
                result["content"] = result["document"]
                
            # If content is still missing, set a placeholder
            if "content" not in result or not result["content"]:
                result["content"] = "Content not available for this document"
                
            # If metadata is missing but exists in another field, standardize it
            if "metadata" not in result and isinstance(result.get("document_metadata"), dict):
                result["metadata"] = result["document_metadata"]
            elif "metadata" not in result:
                # Create basic metadata if none exists
                result["metadata"] = {}
                
        return results
    
    def retrieve_and_format(
        self,
        query: str,
        top_k: int = 5,
        content_dir: str = "data/preprocessed_data",
        include_metadata: bool = True
    ) -> str:
        """Retrieve documents and format them for use in prompts.
        
        Args:
            query: The query text
            top_k: Number of documents to retrieve
            content_dir: Directory containing document content files
            include_metadata: Whether to include metadata in the output
            
        Returns:
            str: Formatted document context for use in prompts
        """
        results = self.retrieve_with_content(query, top_k, content_dir)
        
        formatted_results = []
        for i, result in enumerate(results):
            section = f"Document {i+1}:"
            
            if include_metadata:
                metadata = []
                if "file" in result:
                    metadata.append(f"Source: {result['file']}")
                if "score" in result:
                    metadata.append(f"Relevance: {result['score']:.2f}")
                if "relevance_score" in result:
                    metadata.append(f"Relevance (LLM): {result['relevance_score']}/10")
                
                if metadata:
                    section += " " + ", ".join(metadata)
            
            if "content" in result:
                section += f"\n{result['content']}"
            
            formatted_results.append(section)
        
        return "\n\n".join(formatted_results)