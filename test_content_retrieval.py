#!/usr/bin/env python3
"""
Test script to verify that document content is properly stored and retrieved from FAISS metadata.
"""

import os
import sys
import json
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.core import get_logger
from src.retrieval import DocumentRetriever

# Initialize logger
logger = get_logger("test_content_retrieval")

def main():
    """Test content retrieval functionality."""
    # Instantiate the document retriever
    retriever = DocumentRetriever()
    
    # Test queries to run
    test_queries = [
        "private equity investments",
        "mergers and acquisitions",
        "search fund financials"
    ]
    
    print("Testing document retrieval with content:")
    print("-" * 50)
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        
        # Retrieve documents with content
        results = retriever.retrieve_with_content(query, top_k=2)
        
        for i, result in enumerate(results):
            print(f"\nResult {i+1}:")
            print(f"File: {result.get('file', 'N/A')}")
            print(f"Score: {result.get('score', 'N/A')}")
            
            # Check if content is available
            if 'content' in result and result['content'] and "not found" not in result['content']:
                content_preview = result['content'][:150] + "..." if len(result['content']) > 150 else result['content']
                print(f"Content preview: {content_preview}")
                print(f"Content length: {len(result['content'])} characters")
            else:
                print(f"Content issue: {result.get('content', 'No content available')}")
                
    print("\nTest completed.")

if __name__ == "__main__":
    main()