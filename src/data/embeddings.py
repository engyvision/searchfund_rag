"""
Embedding generation and management for the Web Scraper project.

This module handles creating and storing embeddings for document chunks,
with support for batching and token counting.
"""

import os
import json
from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
import tiktoken
from tenacity import retry, wait_random_exponential, stop_after_attempt
from openai import OpenAI

from src.core import get_logger, OPENAI_CONFIG

# Initialize logger
logger = get_logger("data.embeddings")

# Set pricing (in dollars) per 1K tokens for the embedding model
EMBEDDING_COST_PER_K_TOKEN = 0.00002

# Token limit for embedding models
TOKEN_LIMIT = 8192

class EmbeddingService:
    """Service for generating and managing embeddings."""
    
    def __init__(self, api_key: str = OPENAI_CONFIG.api_key, model: str = OPENAI_CONFIG.embedding_model):
        """Initialize the embedding service.
        
        Args:
            api_key: OpenAI API key
            model: Embedding model to use
        """
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        logger.info(f"Initialized EmbeddingService with model: {model}")
    
    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text string.
        
        Args:
            text: The text to count tokens for
            
        Returns:
            int: The number of tokens
        """
        return len(self.tokenizer.encode(text))
    
    def split_text_into_chunks(
        self, 
        text: str, 
        max_tokens: int = TOKEN_LIMIT, 
        overlap: int = 50
    ) -> List[str]:
        """Split text into chunks of at most max_tokens tokens.
        
        Args:
            text: The text to split
            max_tokens: Maximum tokens per chunk
            overlap: Number of tokens to overlap between chunks
            
        Returns:
            List[str]: List of text chunks
        """
        tokens = self.tokenizer.encode(text)
        chunks = []
        start = 0
        
        # Loop over tokens in increments ensuring overlap
        while start < len(tokens):
            end = start + max_tokens
            chunk_tokens = tokens[start:end]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            chunks.append(chunk_text)
            start += max_tokens - overlap  # Advance with overlap
        
        logger.debug(f"Split text into {len(chunks)} chunks with max_tokens={max_tokens}, overlap={overlap}")
        return chunks
    
    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def get_embedding(self, text: str) -> Tuple[List[float], Optional[int]]:
        """Get an embedding for a text string.
        
        Args:
            text: The text to embed
            
        Returns:
            Tuple[List[float], Optional[int]]: The embedding and token count
            
        Raises:
            Exception: If the API call fails
        """
        try:
            response = self.client.embeddings.create(input=text, model=self.model)
            embedding = response.data[0].embedding
            tokens_used = response.usage.total_tokens if hasattr(response, "usage") and hasattr(response.usage, "total_tokens") else None
            return embedding, tokens_used
        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            raise
    
    def embed_document(
        self, 
        document: str, 
        file_id: str
    ) -> Dict[str, Any]:
        """Embed a document, chunking if necessary.
        
        Args:
            document: The document text to embed
            file_id: Identifier for the document
            
        Returns:
            Dict[str, Any]: Embedding results including metadata
        """
        estimated_tokens = self.count_tokens(document)
        logger.info(f"Processing {file_id} - Estimated tokens: {estimated_tokens}")
        
        result = {
            "id": file_id,
            "estimated_tokens": estimated_tokens,
            "embeddings": [],
            "total_tokens": 0,
            "total_cost": 0.0,
            "status": "success"
        }
        
        try:
            # If document exceeds token limit, chunk it
            if estimated_tokens > TOKEN_LIMIT:
                logger.info(f"{file_id} exceeds token limit. Splitting into chunks.")
                chunks = self.split_text_into_chunks(document, TOKEN_LIMIT, 50)
                
                for idx, chunk in enumerate(chunks):
                    try:
                        embedding, tokens_used = self.get_embedding(chunk)
                        
                        # Use the token count provided by the API if available; otherwise use estimator
                        tokens_for_chunk = tokens_used if tokens_used is not None else self.count_tokens(chunk)
                        
                        chunk_cost = (tokens_for_chunk / 1000.0) * EMBEDDING_COST_PER_K_TOKEN
                        
                        result["embeddings"].append({
                            "chunk_index": idx,
                            "embedding": embedding,
                            "tokens_used": tokens_for_chunk,
                            "cost": chunk_cost
                        })
                        
                        result["total_tokens"] += tokens_for_chunk
                        result["total_cost"] += chunk_cost
                        
                        logger.info(f"Generated embedding for {file_id} chunk {idx} with {tokens_for_chunk} tokens, cost ${chunk_cost:.4f}")
                    except Exception as e:
                        logger.error(f"Error generating embedding for {file_id} chunk {idx}: {e}")
                        result["status"] = "partial_failure"
                        result["error_chunks"] = result.get("error_chunks", []) + [{"index": idx, "error": str(e)}]
            else:
                # Process document as a single chunk
                try:
                    embedding, tokens_used = self.get_embedding(document)
                    
                    tokens_used = tokens_used if tokens_used is not None else estimated_tokens
                    cost = (tokens_used / 1000.0) * EMBEDDING_COST_PER_K_TOKEN
                    
                    result["embeddings"].append({
                        "chunk_index": 0,
                        "embedding": embedding,
                        "tokens_used": tokens_used,
                        "cost": cost
                    })
                    
                    result["total_tokens"] = tokens_used
                    result["total_cost"] = cost
                    
                    logger.info(f"Generated embedding for {file_id} with {tokens_used} tokens, cost ${cost:.4f}")
                except Exception as e:
                    logger.error(f"Error generating embedding for {file_id}: {e}")
                    result["status"] = "failure"
                    result["error"] = str(e)
        except Exception as e:
            logger.error(f"Unexpected error processing {file_id}: {e}")
            result["status"] = "failure"
            result["error"] = str(e)
        
        return result


class EmbeddingManager:
    """Manager for embedding generation and storage."""
    
    def __init__(
        self, 
        output_file: str = "data/embeddings.json",
        log_file: str = "data/embedding_log.json",
        service: Optional[EmbeddingService] = None
    ):
        """Initialize the embedding manager.
        
        Args:
            output_file: Path to output embeddings JSON file
            log_file: Path to log file
            service: EmbeddingService instance (created if not provided)
        """
        self.output_file = output_file
        self.log_file = log_file
        self.service = service or EmbeddingService()
        
        # Load existing embeddings and logs if they exist
        self.embeddings = self._load_json(output_file, {})
        self.log_data = self._load_json(log_file, {})
        
        # Summary counters
        self.embedding_count = 0
        self.total_tokens = 0
        self.total_cost = 0.0
        
        logger.info(f"Initialized EmbeddingManager with output: {output_file}, log: {log_file}")
    
    def _load_json(self, file_path: str, default: Dict[str, Any]) -> Dict[str, Any]:
        """Load JSON from a file, returning a default if the file doesn't exist.
        
        Args:
            file_path: Path to the JSON file
            default: Default value to return if file doesn't exist
            
        Returns:
            Dict[str, Any]: Loaded JSON data or default
        """
        if os.path.exists(file_path):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
        return default
    
    def _save_json(self, file_path: str, data: Dict[str, Any]) -> None:
        """Save JSON data to a file.
        
        Args:
            file_path: Path to the output file
            data: Data to save
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f)
                
            logger.info(f"Saved data to {file_path}")
        except Exception as e:
            logger.error(f"Error saving to {file_path}: {e}")
    
    def process_directory(self, directory_path: str, file_ext: str = ".txt") -> None:
        """Process all files in a directory.
        
        Args:
            directory_path: Path to the directory containing text files
            file_ext: File extension to filter for
        """
        logger.info(f"Processing directory: {directory_path}")
        
        for filename in os.listdir(directory_path):
            if not filename.endswith(file_ext):
                continue
                
            file_path = os.path.join(directory_path, filename)
            
            # Skip if already processed successfully
            if filename in self.embeddings:
                logger.info(f"Skipping {filename} as embedding already exists.")
                continue
            
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read().strip()
                
                # Generate embeddings
                result = self.service.embed_document(text, filename)
                
                # Update embeddings and logs
                self.embeddings[filename] = result["embeddings"]
                self.log_data[filename] = {
                    "status": result["status"],
                    "total_tokens": result["total_tokens"],
                    "total_cost": result["total_cost"],
                }
                
                if "error" in result:
                    self.log_data[filename]["error"] = result["error"]
                
                # Update counters
                self.embedding_count += len(result["embeddings"])
                self.total_tokens += result["total_tokens"]
                self.total_cost += result["total_cost"]
                
                # Save after each file to avoid data loss
                self._save_json(self.output_file, self.embeddings)
                self._save_json(self.log_file, self.log_data)
                
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                self.log_data[filename] = {"status": "error", "error": str(e)}
                self._save_json(self.log_file, self.log_data)
    
    def save(self) -> None:
        """Save all data and print summary."""
        self._save_json(self.output_file, self.embeddings)
        self._save_json(self.log_file, self.log_data)
        
        # Print summary
        logger.info(f"Summary:")
        logger.info(f"Embeddings generated (including all chunks): {self.embedding_count}")
        logger.info(f"Total tokens used: {self.total_tokens}")
        logger.info(f"Total cost: ${self.total_cost:.4f}")
        logger.info(f"Embeddings saved to: {self.output_file}")
        logger.info(f"Log data saved to: {self.log_file}")
    
    def get_embeddings(self, file_id: str) -> Union[List[Dict[str, Any]], None]:
        """Get embeddings for a specific file.
        
        Args:
            file_id: File identifier
            
        Returns:
            Union[List[Dict[str, Any]], None]: Embeddings if found, None otherwise
        """
        return self.embeddings.get(file_id)