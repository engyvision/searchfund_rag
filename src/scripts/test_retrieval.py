#!/usr/bin/env python3
"""
Test script for the enhanced retrieval pipeline.

This script evaluates the enhanced retrieval system with various configurations
including hybrid retrieval and reranking.
"""

import argparse
import json
import os
import time
from typing import Dict, List, Any, Optional

from src.core.logging import get_logger
from src.retrieval.retrieval import DocumentRetriever
from src.llm.answer_generation import AnswerGenerator

# Initialize logger
logger = get_logger("scripts.test_retrieval")

def parse_args():
    """Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Test the enhanced retrieval pipeline")
    
    parser.add_argument(
        "--queries",
        type=str,
        default="data/test_queries.json",
        help="JSON file with test queries"
    )
    
    parser.add_argument(
        "--results-dir",
        type=str,
        default="data/retrieval_results",
        help="Directory to store results"
    )
    
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of documents to retrieve"
    )
    
    parser.add_argument(
        "--configurations",
        type=str,
        nargs="+",
        default=["baseline", "hybrid", "reranking", "hybrid+reranking"],
        help="Retrieval configurations to test"
    )
    
    return parser.parse_args()

def load_test_queries(query_file: str) -> List[Dict[str, Any]]:
    """Load test queries from a JSON file.
    
    Args:
        query_file: Path to the JSON file with test queries
        
    Returns:
        List[Dict[str, Any]]: List of test queries
    """
    # Create sample queries if the file doesn't exist
    if not os.path.exists(query_file):
        logger.info(f"Test query file {query_file} not found, creating sample queries")
        
        sample_queries = [
            {
                "id": "q1",
                "query": "What are the key characteristics of successful search fund entrepreneurs?",
                "expected_keywords": ["search fund", "entrepreneur", "characteristics", "success"]
            },
            {
                "id": "q2",
                "query": "How is the valuation of a search fund target company typically calculated?",
                "expected_keywords": ["valuation", "multiple", "EBITDA", "target company"]
            },
            {
                "id": "q3",
                "query": "What are the main differences between search funds and traditional private equity?",
                "expected_keywords": ["search fund", "private equity", "differences", "comparison"]
            },
            {
                "id": "q4",
                "query": "What are common challenges during the first 100 days after acquisition?",
                "expected_keywords": ["100 days", "acquisition", "challenges", "post-acquisition"]
            }
        ]
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(query_file), exist_ok=True)
        
        # Save sample queries
        with open(query_file, "w", encoding="utf-8") as f:
            json.dump(sample_queries, f, indent=2)
        
        return sample_queries
    
    # Load existing queries
    try:
        with open(query_file, "r", encoding="utf-8") as f:
            queries = json.load(f)
        logger.info(f"Loaded {len(queries)} test queries from {query_file}")
        return queries
    except Exception as e:
        logger.error(f"Error loading test queries from {query_file}: {e}")
        raise

def evaluate_retrieval_configuration(
    queries: List[Dict[str, Any]],
    use_hybrid_retrieval: bool,
    use_reranking: bool,
    top_k: int
) -> Dict[str, Any]:
    """Evaluate a specific retrieval configuration.
    
    Args:
        queries: List of test queries
        use_hybrid_retrieval: Whether to use hybrid retrieval
        use_reranking: Whether to use LLM reranking
        top_k: Number of documents to retrieve
        
    Returns:
        Dict[str, Any]: Evaluation results
    """
    config_name = []
    if use_hybrid_retrieval:
        config_name.append("hybrid")
    if use_reranking:
        config_name.append("reranking")
    if not config_name:
        config_name = ["baseline"]
    
    config_str = "+".join(config_name)
    logger.info(f"Evaluating configuration: {config_str}")
    
    # Initialize retriever with the specified configuration
    retriever = DocumentRetriever(
        use_hybrid_retrieval=use_hybrid_retrieval,
        use_reranking=use_reranking
    )
    
    # Initialize answer generator
    answer_generator = AnswerGenerator()
    
    # Evaluate each query
    results = []
    for query_data in queries:
        query = query_data["query"]
        query_id = query_data["id"]
        expected_keywords = query_data.get("expected_keywords", [])
        
        logger.info(f"Processing query {query_id}: {query}")
        
        # Measure retrieval time
        start_time = time.time()
        retrieved_docs = retriever.retrieve_with_content(query, top_k=top_k)
        retrieval_time = time.time() - start_time
        
        # Format context
        context = retriever.retrieve_and_format(query, top_k=top_k)
        
        # Generate answer
        answer = answer_generator.generate_answer(query, context)
        
        # Basic evaluation: check if expected keywords are in the retrieved documents
        found_keywords = set()
        for doc in retrieved_docs:
            content = doc.get("content", "")
            for keyword in expected_keywords:
                if keyword.lower() in content.lower():
                    found_keywords.add(keyword)
        
        keyword_coverage = len(found_keywords) / len(expected_keywords) if expected_keywords else 0
        
        # Compile results for this query
        query_result = {
            "query_id": query_id,
            "query": query,
            "retrieval_time_seconds": retrieval_time,
            "num_docs_retrieved": len(retrieved_docs),
            "keyword_coverage": keyword_coverage,
            "found_keywords": list(found_keywords),
            "expected_keywords": expected_keywords,
            "answer": answer,
            "top_sources": [doc.get("file", "unknown") for doc in retrieved_docs]
        }
        
        results.append(query_result)
    
    # Calculate aggregate statistics
    avg_retrieval_time = sum(r["retrieval_time_seconds"] for r in results) / len(results)
    avg_keyword_coverage = sum(r["keyword_coverage"] for r in results) / len(results)
    
    return {
        "configuration": config_str,
        "use_hybrid_retrieval": use_hybrid_retrieval,
        "use_reranking": use_reranking,
        "top_k": top_k,
        "num_queries": len(queries),
        "avg_retrieval_time_seconds": avg_retrieval_time,
        "avg_keyword_coverage": avg_keyword_coverage,
        "query_results": results
    }

def main():
    """Main function to test the retrieval pipeline."""
    args = parse_args()
    
    # Create results directory if it doesn't exist
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Load test queries
    queries = load_test_queries(args.queries)
    
    # Configure the retrieval configurations to test
    configs = []
    if "baseline" in args.configurations:
        configs.append({"use_hybrid_retrieval": False, "use_reranking": False})
    if "hybrid" in args.configurations:
        configs.append({"use_hybrid_retrieval": True, "use_reranking": False})
    if "reranking" in args.configurations:
        configs.append({"use_hybrid_retrieval": False, "use_reranking": True})
    if "hybrid+reranking" in args.configurations:
        configs.append({"use_hybrid_retrieval": True, "use_reranking": True})
    
    # Evaluate each configuration
    all_results = {}
    for config in configs:
        use_hybrid = config["use_hybrid_retrieval"]
        use_reranking = config["use_reranking"]
        
        config_name = []
        if use_hybrid:
            config_name.append("hybrid")
        if use_reranking:
            config_name.append("reranking")
        if not config_name:
            config_name = ["baseline"]
        
        config_str = "+".join(config_name)
        
        # Evaluate the configuration
        result = evaluate_retrieval_configuration(
            queries=queries,
            use_hybrid_retrieval=use_hybrid,
            use_reranking=use_reranking,
            top_k=args.top_k
        )
        
        # Save the individual result
        result_file = os.path.join(args.results_dir, f"result_{config_str}.json")
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        
        # Store for comparison
        all_results[config_str] = {
            "avg_retrieval_time_seconds": result["avg_retrieval_time_seconds"],
            "avg_keyword_coverage": result["avg_keyword_coverage"]
        }
        
        logger.info(f"Config {config_str}: avg_time={result['avg_retrieval_time_seconds']:.2f}s, keyword_coverage={result['avg_keyword_coverage']:.2f}")
    
    # Save comparison results
    comparison_file = os.path.join(args.results_dir, "comparison.json")
    with open(comparison_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    
    # Print comparison
    logger.info("\nRetrieval Configuration Comparison:")
    logger.info("===================================")
    for config, metrics in all_results.items():
        logger.info(f"{config.ljust(16)}: Time {metrics['avg_retrieval_time_seconds']:.2f}s, Coverage {metrics['avg_keyword_coverage']:.2f}")

if __name__ == "__main__":
    main()