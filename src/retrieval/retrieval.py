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
        content_dir: str = "data/preprocessed_data"  # Kept for backward compatibility
    ) -> List[Dict[str, Any]]:
        """Retrieve documents and include their content.
        
        Args:
            query: The query text
            top_k: Number of documents to retrieve
            content_dir: Directory containing document content files (only used as fallback)
            
        Returns:
            List[Dict[str, Any]]: List of retrieved documents with content
        """
        # Get document metadata
        results = self.retrieve_documents(query, top_k)
        
        # Check if content is already in the metadata (updated implementation)
        for result in results:
            if "content" not in result or not result["content"]:
                try:
                    file_name = result.get("file")
                    chunk_index = result.get("chunk_index")
                    
                    logger.debug(f"Content not found in metadata for {file_name}, using fallback")
                    
                    if file_name:
                        # Remove chunk suffix from file name if it exists in the metadata
                        actual_file_name = file_name
                        if "_chunk_" in file_name and chunk_index is None:
                            # Extract the actual file name (remove _chunk_X suffix)
                            actual_file_name = file_name.split("_chunk_")[0]
                        
                        # Try multiple possible paths for content files
                        content_file_paths = [
                            os.path.join(content_dir, actual_file_name),  # Original path
                            os.path.join("data", "preprocessed_data", actual_file_name),  # From project root
                            os.path.join("/workspaces/webscraper_project/data/preprocessed_data", actual_file_name)  # Absolute path
                        ]
                        
                        file_path = None
                        for path in content_file_paths:
                            if os.path.exists(path):
                                file_path = path
                                logger.debug(f"Found content file at: {file_path}")
                                break
                        
                        if file_path and os.path.exists(file_path):
                            try:
                                with open(file_path, "r", encoding="utf-8") as f:
                                    content = f.read()
                                
                                # Check if content is JSON formatted and extract text if needed
                                if content.startswith('{') and '"pages":' in content:
                                    logger.debug(f"File appears to be JSON formatted, attempting to extract text")
                                    try:
                                        import json
                                        content_json = json.loads(content)
                                        # Extract text from all pages
                                        text_content = []
                                        for page in content_json.get('pages', []):
                                            for text_block in page.get('text', []):
                                                if isinstance(text_block, str):
                                                    text_content.append(text_block)
                                        
                                        # Join all text blocks
                                        if text_content:
                                            content = "\n\n".join(text_content)
                                            logger.debug(f"Successfully extracted text from JSON")
                                    except Exception as json_e:
                                        logger.warning(f"Failed to parse JSON content: {json_e}")
                                
                                result["content"] = content
                                logger.debug(f"Successfully loaded content for {file_name}, content length: {len(content)}")
                            except Exception as read_err:
                                logger.error(f"Error reading file {file_path}: {read_err}")
                                result["content"] = f"Error reading file: {str(read_err)}"
                        else:
                            logger.warning(f"Content file not found: {file_path}")
                            result["content"] = "Content not found"
                except Exception as e:
                    logger.error(f"Error retrieving content for {result.get('file')}: {e}")
                    result["content"] = f"Error retrieving content: {str(e)}"
                
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