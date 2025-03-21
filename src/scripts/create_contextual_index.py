#!/usr/bin/env python3
"""
Script to create a FAISS index with smaller chunks for contextual embedding.

This script takes the original document content, creates smaller chunks
suitable for contextual embedding, and creates a new FAISS index.
"""

import argparse
import os
import json
import pickle
from pathlib import Path
import numpy as np
import faiss  # Add missing import

from src.core.logging import get_logger
from src.data.embeddings import EmbeddingService
from src.data.indexing import FAISSIndexer

# Initialize logger
logger = get_logger("scripts.create_contextual_index")

def parse_args():
    """Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Create a FAISS index with smaller chunks for contextual embedding")
    
    parser.add_argument(
        "--metadata-file",
        type=str,
        default="/workspaces/webscraper_project/data/faiss_metadata.json",
        help="Existing metadata file with document content"
    )
    
    parser.add_argument(
        "--output-index",
        type=str,
        default="/workspaces/webscraper_project/data/faiss_contextual_index.idx",
        help="Output file for new FAISS index"
    )
    
    parser.add_argument(
        "--output-metadata",
        type=str,
        default="/workspaces/webscraper_project/data/faiss_contextual_metadata.json",
        help="Output file for new metadata"
    )
    
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Maximum chunk size in words"
    )
    
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        help="Overlap between chunks in words"
    )
    
    parser.add_argument(
        "--checkpoint-file",
        type=str,
        default="/workspaces/webscraper_project/data/contextual_index_checkpoint.pkl",
        help="File to save processing checkpoint for resuming"
    )
    
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume processing from checkpoint if it exists"
    )
    
    return parser.parse_args()

def chunk_text(text, chunk_size, chunk_overlap):
    """Split text into chunks with overlap.
    
    Args:
        text: Text to split
        chunk_size: Maximum chunk size in words
        chunk_overlap: Overlap between chunks in words
        
    Returns:
        List[str]: Chunked text
    """
    words = text.split()
    chunks = []
    
    if len(words) <= chunk_size:
        return [text]
    
    for i in range(0, len(words), chunk_size - chunk_overlap):
        chunk = words[i:i + chunk_size]
        if len(chunk) < 50:  # Skip very small chunks at the end
            continue
        chunks.append(" ".join(chunk))
    
    return chunks

def save_checkpoint(checkpoint_file, processed_data):
    """Save processing checkpoint to file.
    
    Args:
        checkpoint_file: Path to save checkpoint
        processed_data: Dictionary with processed data
    """
    try:
        os.makedirs(os.path.dirname(checkpoint_file), exist_ok=True)
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(processed_data, f)
        logger.info(f"Saved checkpoint to {checkpoint_file}")
    except Exception as e:
        logger.error(f"Error saving checkpoint: {e}")

def load_checkpoint(checkpoint_file):
    """Load processing checkpoint from file.
    
    Args:
        checkpoint_file: Path to load checkpoint from
        
    Returns:
        Dict or None: Processed data if checkpoint exists, None otherwise
    """
    if not os.path.exists(checkpoint_file):
        return None
    
    try:
        with open(checkpoint_file, 'rb') as f:
            checkpoint = pickle.load(f)
        logger.info(f"Loaded checkpoint from {checkpoint_file}")
        return checkpoint
    except Exception as e:
        logger.error(f"Error loading checkpoint: {e}")
        return None

def main():
    """Main function to create contextual FAISS index."""
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output_index), exist_ok=True)
    os.makedirs(os.path.dirname(args.output_metadata), exist_ok=True)
    
    logger.info(f"Creating contextual FAISS index with chunk size {args.chunk_size} and overlap {args.chunk_overlap}")
    
    # Check for checkpoint
    checkpoint = None
    if args.resume:
        checkpoint = load_checkpoint(args.checkpoint_file)
    
    if checkpoint:
        logger.info(f"Resuming from checkpoint with {len(checkpoint['all_chunks'])} chunks already processed")
        all_chunks = checkpoint['all_chunks']
        all_chunk_metadata = checkpoint['all_chunk_metadata']
        all_embeddings = checkpoint['all_embeddings']
        last_doc_index = checkpoint.get('last_doc_index', 0)
    else:
        # Start fresh
        all_chunks = []
        all_chunk_metadata = []
        all_embeddings = []
        last_doc_index = 0
    
    # Load existing metadata
    try:
        with open(args.metadata_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)
            logger.info(f"Loaded metadata from {args.metadata_file}")
    except Exception as e:
        logger.error(f"Error loading metadata: {e}")
        return
    
    # Initialize embedding service
    embedding_service = EmbeddingService()
    
    # Process documents and create chunks
    try:
        # Start from where we left off if resuming
        for i, doc_metadata in enumerate(metadata["metadata"][last_doc_index:], start=last_doc_index):
            if "content" not in doc_metadata or not doc_metadata["content"]:
                logger.warning(f"Document {i} has no content, skipping")
                continue
            
            doc_content = doc_metadata["content"]
            file_name = doc_metadata.get("file", f"document_{i}")
            
            # Create chunks
            chunks = chunk_text(doc_content, args.chunk_size, args.chunk_overlap)
            logger.info(f"Created {len(chunks)} chunks for document {file_name}")
            
            # Process each chunk
            for j, chunk in enumerate(chunks):
                try:
                    # Generate embedding
                    embedding, _ = embedding_service.get_embedding(chunk)
                    
                    # Store chunk and metadata
                    chunk_id = f"{file_name}_chunk_{j}"
                    all_chunks.append(chunk)
                    all_embeddings.append(embedding)
                    
                    # Create metadata for this chunk (copy original and add chunk-specific info)
                    chunk_metadata = doc_metadata.copy()
                    chunk_metadata["content"] = chunk
                    chunk_metadata["chunk_index"] = j
                    chunk_metadata["chunk_id"] = chunk_id
                    chunk_metadata["original_file"] = file_name
                    all_chunk_metadata.append(chunk_metadata)
                    
                    # Save checkpoint every 20 chunks
                    if len(all_chunks) % 20 == 0:
                        logger.info(f"Processed {len(all_chunks)} chunks so far")
                        checkpoint_data = {
                            'all_chunks': all_chunks,
                            'all_chunk_metadata': all_chunk_metadata,
                            'all_embeddings': all_embeddings,
                            'last_doc_index': i
                        }
                        save_checkpoint(args.checkpoint_file, checkpoint_data)
                    
                except Exception as e:
                    logger.error(f"Error processing chunk {j} of document {file_name}: {e}")
                    # Continue processing other chunks
    
    except Exception as e:
        # Save checkpoint on error
        logger.error(f"Error during processing: {e}")
        checkpoint_data = {
            'all_chunks': all_chunks,
            'all_chunk_metadata': all_chunk_metadata,
            'all_embeddings': all_embeddings,
            'last_doc_index': last_doc_index
        }
        save_checkpoint(args.checkpoint_file, checkpoint_data)
        return
    
    # Final stats
    logger.info(f"Created {len(all_chunks)} total chunks, generating embeddings")
    
    # Convert embeddings to numpy array
    embeddings_np = np.array(all_embeddings, dtype=np.float32)
    
    if len(embeddings_np) == 0:
        logger.error("No valid embeddings were created. Aborting index creation.")
        return
    
    # Create and save the new FAISS index
    try:
        # Initialize FAISS indexer
        indexer = FAISSIndexer(
            embedding_dim=len(all_embeddings[0]),
            index_file=args.output_index,
            metadata_file=args.output_metadata
        )
        
        # Create the index
        index = faiss.IndexFlatL2(len(all_embeddings[0]))
        index.add(embeddings_np)
        indexer.index = index
        
        # Create and store metadata
        chunk_ids = [meta.get("chunk_id", f"chunk_{i}") for i, meta in enumerate(all_chunk_metadata)]
        indexer.metadata = {
            "ids": chunk_ids,
            "metadata": all_chunk_metadata
        }
        
        # Save index and metadata
        if indexer.save():
            logger.info(f"Successfully created and saved contextual FAISS index with {len(all_chunks)} chunks")
            logger.info(f"Index saved to {args.output_index}")
            logger.info(f"Metadata saved to {args.output_metadata}")
            
            # Remove checkpoint after successful completion
            if os.path.exists(args.checkpoint_file):
                os.remove(args.checkpoint_file)
                logger.info(f"Removed checkpoint file {args.checkpoint_file}")
        else:
            logger.error("Failed to save FAISS index")
            
    except Exception as e:
        logger.error(f"Error creating contextual FAISS index: {e}")
        # Save checkpoint on error
        checkpoint_data = {
            'all_chunks': all_chunks,
            'all_chunk_metadata': all_chunk_metadata,
            'all_embeddings': all_embeddings,
            'last_doc_index': last_doc_index
        }
        save_checkpoint(args.checkpoint_file, checkpoint_data)

if __name__ == "__main__":
    main()