#!/usr/bin/env python3
"""
Script to test the RAG pipeline.

This script tests the full RAG pipeline, including query clarification,
document retrieval, and answer generation.
"""

import argparse
import os
from pathlib import Path

from src.core.logging import get_logger
from src.llm.query_clarification import QueryClarifier
from src.llm.answer_generation import AnswerGenerator
from src.retrieval.retrieval import DocumentRetriever

# Initialize logger
logger = get_logger("scripts.test_rag")

def parse_args():
    """Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Test the RAG pipeline")
    
    parser.add_argument(
        "query",
        type=str,
        help="The query to test"
    )
    
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of documents to retrieve"
    )
    
    parser.add_argument(
        "--content-dir",
        type=str,
        default="data/preprocessed_data",
        help="Directory containing document content"
    )
    
    return parser.parse_args()

def test_pipeline(query: str, top_k: int = 3, content_dir: str = "data/preprocessed_data"):
    """Test the full RAG pipeline.
    
    Args:
        query: The query to test
        top_k: Number of documents to retrieve
        content_dir: Directory containing document content
        
    Returns:
        tuple: Clarified query, retrieved documents, and answer
    """
    logger.info(f"Testing RAG pipeline with query: {query}")
    
    # Initialize components
    query_clarifier = QueryClarifier()
    document_retriever = DocumentRetriever()
    answer_generator = AnswerGenerator()
    
    # Step 1: Clarify the query
    clarified_query = query_clarifier.clarify_query(query)
    logger.info(f"Clarified query: {clarified_query}")
    
    # Step 2: Retrieve documents
    context = document_retriever.retrieve_and_format(
        clarified_query, 
        top_k=top_k,
        content_dir=content_dir,
        include_metadata=True
    )
    logger.info(f"Retrieved {top_k} documents")
    
    # Step 3: Generate answer
    answer = answer_generator.generate_answer(
        clarified_query,
        context
    )
    logger.info("Generated answer")
    
    return clarified_query, context, answer

def main():
    """Main function to test the RAG pipeline."""
    args = parse_args()
    
    print(f"Original Query: {args.query}\n")
    
    # Test the pipeline
    clarified_query, context, answer = test_pipeline(
        args.query,
        args.top_k,
        args.content_dir
    )
    
    print(f"Clarified Query: {clarified_query}\n")
    print("Retrieved Documents:")
    print(f"{context}\n")
    print("Generated Answer:")
    print(answer)

if __name__ == "__main__":
    main()