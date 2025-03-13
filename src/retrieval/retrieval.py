"""
Document retrieval using vector similarity for the Web Scraper project.

This module provides functionality for retrieving relevant documents
based on query embeddings and vector similarity search.
"""

import os
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
import json

from src.core import get_logger, OPENAI_CONFIG
from src.data.embeddings import EmbeddingService
from src.data.indexing import FAISSIndexer

# Initialize logger
logger = get_logger("retrieval")

class DocumentRetriever:
    """Retrieve relevant documents using vector similarity."""
    
    def __init__(
        self,
        embedding_service: Optional[EmbeddingService] = None,
        faiss_indexer: Optional[FAISSIndexer] = None
    ):
        """Initialize the document retriever.
        
        Args:
            embedding_service: Service for generating embeddings
            faiss_indexer: FAISS indexer for vector similarity search
        """
        self.embedding_service = embedding_service or EmbeddingService()
        self.faiss_indexer = faiss_indexer or FAISSIndexer()
        
        # Load the FAISS index
        if not self.faiss_indexer.load():
            logger.error("Failed to load FAISS index")
            raise ValueError("FAISS index could not be loaded")
            
        logger.info("Initialized DocumentRetriever")
    
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
        
        # Generate query embedding
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
            
        logger.info(f"Retrieved {len(enhanced_results)} documents")
        return enhanced_results
    
    def retrieve_with_content(
        self,
        query: str,
        top_k: int = 5,
        content_dir: str = "data/preprocessed_data"
    ) -> List[Dict[str, Any]]:
        """Retrieve documents and include their content.
        
        Args:
            query: The query text
            top_k: Number of documents to retrieve
            content_dir: Directory containing document content files
            
        Returns:
            List[Dict[str, Any]]: List of retrieved documents with content
        """
        # Get document metadata
        results = self.retrieve_documents(query, top_k)
        
        # Enhance with document content
        for result in results:
            try:
                file_name = result.get("file")
                chunk_index = result.get("chunk_index")
                
                if file_name:
                    file_path = os.path.join(content_dir, file_name)
                    
                    if os.path.exists(file_path):
                        with open(file_path, "r", encoding="utf-8") as f:
                            content = f.read()
                        
                        # If this is a chunk, we could try to extract just that chunk
                        # For now, include the whole document
                        result["content"] = content
                    else:
                        logger.warning(f"Content file not found: {file_path}")
                        result["content"] = "Content not found"
            except Exception as e:
                logger.error(f"Error retrieving content for {result.get('file')}: {e}")
                result["content"] = "Error retrieving content"
        
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
                
                if metadata:
                    section += " " + ", ".join(metadata)
            
            if "content" in result:
                section += f"\n{result['content']}"
            
            formatted_results.append(section)
        
        return "\n\n".join(formatted_results)