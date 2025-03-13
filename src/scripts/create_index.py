#!/usr/bin/env python3
"""
Script to create a FAISS index from embeddings.

This script takes an embeddings JSON file and creates a FAISS
index for efficient vector similarity search.
"""

import argparse
import os
from pathlib import Path

from src.core import get_logger
from src.data import FAISSIndexer

# Initialize logger
logger = get_logger("scripts.create_index")

def parse_args():
    """Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Create a FAISS index from embeddings")
    
    parser.add_argument(
        "--embeddings-file",
        type=str,
        default="data/embeddings.json",
        help="JSON file containing embeddings"
    )
    
    parser.add_argument(
        "--index-file",
        type=str,
        default="data/faiss_index.idx",
        help="Output file for FAISS index"
    )
    
    parser.add_argument(
        "--metadata-file",
        type=str,
        default="data/faiss_metadata.json",
        help="Output file for metadata"
    )
    
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=1536,
        help="Dimension of the embeddings"
    )
    
    return parser.parse_args()

def main():
    """Main function to create a FAISS index."""
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.index_file), exist_ok=True)
    os.makedirs(os.path.dirname(args.metadata_file), exist_ok=True)
    
    logger.info(f"Creating FAISS index from embeddings in {args.embeddings_file}")
    
    # Initialize FAISS indexer
    indexer = FAISSIndexer(
        embedding_dim=args.embedding_dim,
        index_file=args.index_file,
        metadata_file=args.metadata_file
    )
    
    # Build index
    if indexer.build_index_from_embeddings(args.embeddings_file):
        # Save index and metadata
        if indexer.save():
            logger.info("Successfully created and saved FAISS index")
        else:
            logger.error("Failed to save FAISS index")
    else:
        logger.error("Failed to build FAISS index")

if __name__ == "__main__":
    main()