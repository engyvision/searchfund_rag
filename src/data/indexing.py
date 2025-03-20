"""
Vector indexing with FAISS for the Web Scraper project.

This module provides functionality for building and managing FAISS vector indices
for efficient similarity search of document embeddings.
"""

import os
import json
import faiss
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path

from src.core import get_logger, FAISS_CONFIG

# Initialize logger
logger = get_logger("data.indexing")

class FAISSIndexer:
    """FAISS vector indexing and searching."""
    
    def __init__(
        self, 
        embedding_dim: int = FAISS_CONFIG.embedding_dim,
        index_file: str = FAISS_CONFIG.index_file,
        metadata_file: str = FAISS_CONFIG.metadata_file
    ):
        """Initialize the FAISS indexer.
        
        Args:
            embedding_dim: Dimension of the embeddings
            index_file: Path to save/load the FAISS index
            metadata_file: Path to save/load the metadata
        """
        self.embedding_dim = embedding_dim
        self.index_file = index_file
        self.metadata_file = metadata_file
        self.index = None
        self.metadata = {"ids": [], "metadata": []}
        
        logger.info(f"Initialized FAISSIndexer with dim={embedding_dim}, index_file={index_file}")
    
    def load(self) -> bool:
        """Load the index and metadata from disk if they exist.
        
        Returns:
            bool: True if loaded successfully, False otherwise
        """
        # Load index if it exists
        if os.path.exists(self.index_file):
            try:
                self.index = faiss.read_index(self.index_file)
                logger.info(f"Loaded FAISS index from {self.index_file} with {self.index.ntotal} vectors")
            except Exception as e:
                logger.error(f"Error loading FAISS index: {e}")
                return False
        else:
            logger.info(f"FAISS index file {self.index_file} not found, will create new index")
        
        # Load metadata if it exists
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)
                logger.info(f"Loaded metadata from {self.metadata_file} with {len(self.metadata['ids'])} entries")
            except Exception as e:
                logger.error(f"Error loading metadata: {e}")
                return False
        else:
            logger.info(f"Metadata file {self.metadata_file} not found, will create new metadata")
        
        return True if self.index else False
    
    def save(self) -> bool:
        """Save the index and metadata to disk.
        
        Returns:
            bool: True if saved successfully, False otherwise
        """
        if self.index is None:
            logger.error("Cannot save index: No index is loaded or built")
            return False
        
        try:
            # Create directories if they don't exist
            os.makedirs(os.path.dirname(self.index_file), exist_ok=True)
            os.makedirs(os.path.dirname(self.metadata_file), exist_ok=True)
            
            # Save index
            faiss.write_index(self.index, self.index_file)
            logger.info(f"Saved FAISS index to {self.index_file}")
            
            # Save metadata
            with open(self.metadata_file, "w", encoding="utf-8") as f:
                json.dump(self.metadata, f)
            logger.info(f"Saved metadata to {self.metadata_file}")
            
            return True
        except Exception as e:
            logger.error(f"Error saving index and metadata: {e}")
            return False
    
    def build_index_from_embeddings(
        self, 
        embeddings_file: str
    ) -> bool:
        """Build a FAISS index from embeddings stored in a JSON file.
        
        Args:
            embeddings_file: Path to the JSON file containing embeddings
            
        Returns:
            bool: True if index was built successfully, False otherwise
        """
        try:
            # Load embeddings
            with open(embeddings_file, "r", encoding="utf-8") as f:
                embeddings_data = json.load(f)
            
            # Initialize empty lists for vectors and metadata
            vector_list = []
            metadata_list = []
            id_list = []
            
            # Process each file's embeddings
            for file_name, embed_data in embeddings_data.items():
                # Check if embed_data is a list of chunk embeddings
                if isinstance(embed_data, list):
                    # Check if the list contains dictionaries (chunked embeddings)
                    if len(embed_data) > 0 and isinstance(embed_data[0], dict):
                        for chunk in embed_data:
                            unique_id = f"{file_name}_chunk_{chunk['chunk_index']}"
                            vector_list.append(chunk["embedding"])
                            id_list.append(unique_id)
                            # Load content from file
                            content = None
                            # Try multiple possible paths for content files
                            content_file_paths = [
                                os.path.join(os.path.dirname(embeddings_file), "..", "preprocessed_data", file_name),  # Original path
                                os.path.join(os.path.dirname(embeddings_file), "preprocessed_data", file_name),  # Direct subdirectory
                                os.path.join("data", "preprocessed_data", file_name),  # From project root
                                os.path.join("/workspaces/webscraper_project/data/preprocessed_data", file_name)  # Absolute path
                            ]
                            
                            for path in content_file_paths:
                                if os.path.exists(path):
                                    content_file_path = path
                                    logger.debug(f"Found content file at: {content_file_path}")
                                    break
                            else:
                                content_file_path = None
                                logger.warning(f"Content file not found for: {file_name} after trying multiple paths")
                            
                            try:
                                if content_file_path and os.path.exists(content_file_path):
                                    with open(content_file_path, "r", encoding="utf-8") as f:
                                        file_content = f.read()
                                    
                                    # Process JSON content if needed
                                    if file_content.startswith("{") and '"pages":' in file_content:
                                        try:
                                            content_json = json.loads(file_content)
                                            text_content = []
                                            for page in content_json.get('pages', []):
                                                for text_block in page.get('text', []):
                                                    if isinstance(text_block, str):
                                                        text_content.append(text_block)
                                            if text_content:
                                                content = "\n\n".join(text_content)
                                        except Exception as json_err:
                                            logger.warning(f"Failed to parse JSON content for {file_name}: {json_err}")
                                            content = file_content
                                    else:
                                        content = file_content
                                else:
                                    logger.warning(f"Content file not found: {content_file_path}")
                            except Exception as e:
                                logger.warning(f"Could not load content for {file_name}: {e}")
                            
                            # Store file reference, chunk index, and content
                            metadata_list.append({
                                "file": file_name,
                                "chunk_index": chunk["chunk_index"],
                                "content": content if content else f"Content not available for {file_name}"
                            })
                    else:
                        # The list is just a single embedding vector
                        unique_id = file_name
                        vector_list.append(embed_data)
                        id_list.append(unique_id)
                        # Load content from file for single embeddings
                        content = None
                        # Try multiple possible paths for content files
                        content_file_paths = [
                            os.path.join(os.path.dirname(embeddings_file), "..", "preprocessed_data", file_name),  # Original path
                            os.path.join(os.path.dirname(embeddings_file), "preprocessed_data", file_name),  # Direct subdirectory
                            os.path.join("data", "preprocessed_data", file_name),  # From project root
                            os.path.join("/workspaces/webscraper_project/data/preprocessed_data", file_name)  # Absolute path
                        ]
                        
                        for path in content_file_paths:
                            if os.path.exists(path):
                                content_file_path = path
                                logger.debug(f"Found content file at: {content_file_path}")
                                break
                        else:
                            content_file_path = None
                            logger.warning(f"Content file not found for: {file_name} after trying multiple paths")
                        
                        try:
                            if content_file_path and os.path.exists(content_file_path):
                                with open(content_file_path, "r", encoding="utf-8") as f:
                                    file_content = f.read()
                                
                                # Process JSON content if needed
                                if file_content.startswith("{") and '"pages":' in file_content:
                                    try:
                                        content_json = json.loads(file_content)
                                        text_content = []
                                        for page in content_json.get('pages', []):
                                            for text_block in page.get('text', []):
                                                if isinstance(text_block, str):
                                                    text_content.append(text_block)
                                        if text_content:
                                            content = "\n\n".join(text_content)
                                    except Exception as json_err:
                                        logger.warning(f"Failed to parse JSON content for {file_name}: {json_err}")
                                        content = file_content
                                else:
                                    content = file_content
                            else:
                                logger.warning(f"Content file not found: {content_file_path}")
                        except Exception as e:
                            logger.warning(f"Could not load content for {file_name}: {e}")
                        
                        metadata_list.append({
                            "file": file_name,
                            "content": content if content else f"Content not available for {file_name}"
                        })
                else:
                    # Single embedding that's not in a list
                    unique_id = file_name
                    vector_list.append(embed_data)
                    id_list.append(unique_id)
                    
                    # Load content from file for single embeddings
                    content = None
                    # Try multiple possible paths for content files
                    content_file_paths = [
                        os.path.join(os.path.dirname(embeddings_file), "..", "preprocessed_data", file_name),  # Original path
                        os.path.join(os.path.dirname(embeddings_file), "preprocessed_data", file_name),  # Direct subdirectory
                        os.path.join("data", "preprocessed_data", file_name),  # From project root
                        os.path.join("/workspaces/webscraper_project/data/preprocessed_data", file_name)  # Absolute path
                    ]
                    
                    for path in content_file_paths:
                        if os.path.exists(path):
                            content_file_path = path
                            logger.debug(f"Found content file at: {content_file_path}")
                            break
                    else:
                        content_file_path = None
                        logger.warning(f"Content file not found for: {file_name} after trying multiple paths")
                    
                    try:
                        if content_file_path and os.path.exists(content_file_path):
                            with open(content_file_path, "r", encoding="utf-8") as f:
                                file_content = f.read()
                            
                            # Process JSON content if needed
                            if file_content.startswith("{") and '"pages":' in file_content:
                                try:
                                    content_json = json.loads(file_content)
                                    text_content = []
                                    for page in content_json.get('pages', []):
                                        for text_block in page.get('text', []):
                                            if isinstance(text_block, str):
                                                text_content.append(text_block)
                                    if text_content:
                                        content = "\n\n".join(text_content)
                                except Exception as json_err:
                                    logger.warning(f"Failed to parse JSON content for {file_name}: {json_err}")
                                    content = file_content
                            else:
                                content = file_content
                        else:
                            logger.warning(f"Content file not found: {content_file_path}")
                    except Exception as e:
                        logger.warning(f"Could not load content for {file_name}: {e}")
                    
                    metadata_list.append({
                        "file": file_name,
                        "content": content if content else f"Content not available for {file_name}"
                    })
            
            # Convert to numpy array
            vectors_np = np.array(vector_list, dtype=np.float32)
            
            # Create and populate the index
            self.index = faiss.IndexFlatL2(self.embedding_dim)
            self.index.add(vectors_np)
            
            # Store metadata
            self.metadata = {
                "ids": id_list,
                "metadata": metadata_list
            }
            
            logger.info(f"Built FAISS index with {self.index.ntotal} vectors from {len(embeddings_data)} files")
            return True
        except Exception as e:
            logger.error(f"Error building index: {e}")
            return False
    
    def search(
        self, 
        query_vector: np.ndarray, 
        top_k: int = 5
    ) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        """Search the index for similar vectors.
        
        Args:
            query_vector: The query vector
            top_k: Number of results to return
            
        Returns:
            Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]: 
                Distances, indices, and metadata for the results
                
        Raises:
            ValueError: If no index is loaded or built
        """
        if self.index is None:
            raise ValueError("No index is loaded or built")
        
        # Ensure query vector is 2D
        if len(query_vector.shape) == 1:
            query_vector = np.expand_dims(query_vector, axis=0)
        
        # Search the index
        distances, indices = self.index.search(query_vector, top_k)
        
        # Get metadata for results
        results = []
        for idx in indices[0]:
            try:
                vector_id = self.metadata["ids"][idx]
                result_info = self.metadata["metadata"][self.metadata["ids"].index(vector_id)]
                results.append(result_info)
                logger.debug(f"Retrieved result: {result_info}")
            except Exception as e:
                logger.error(f"Error processing index {idx}: {e}")
        
        return distances, indices, results