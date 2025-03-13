#!/usr/bin/env python3
"""
Script to generate embeddings for preprocessed text files.

This script takes preprocessed text files and generates embeddings
for them using the OpenAI API, saving the results to a JSON file.
"""

import argparse
import os
from pathlib import Path

from src.core import get_logger
from src.data import EmbeddingManager

# Initialize logger
logger = get_logger("scripts.generate_embeddings")

def parse_args():
    """Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Generate embeddings for preprocessed text files")
    
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/preprocessed_data",
        help="Directory containing preprocessed text files"
    )
    
    parser.add_argument(
        "--output-file",
        type=str,
        default="data/embeddings.json",
        help="Output file for embeddings"
    )
    
    parser.add_argument(
        "--log-file",
        type=str,
        default="data/embedding_log.json",
        help="Output file for embedding log"
    )
    
    parser.add_argument(
        "--file-ext",
        type=str,
        default=".txt",
        help="File extension to filter for"
    )
    
    return parser.parse_args()

def main():
    """Main function to generate embeddings."""
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    logger.info(f"Generating embeddings for files in {args.input_dir}")
    
    # Initialize embedding manager
    embedding_manager = EmbeddingManager(
        output_file=args.output_file,
        log_file=args.log_file
    )
    
    # Process directory
    embedding_manager.process_directory(
        directory_path=args.input_dir,
        file_ext=args.file_ext
    )
    
    # Save results
    embedding_manager.save()

if __name__ == "__main__":
    main()