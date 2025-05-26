"""
Test script for the LLM provider abstraction.

This script demonstrates how to use the LLM provider abstraction.
"""

import os
import sys
import argparse
from typing import Dict, Any

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.core import get_logger, LLM_CONFIG
from src.llm.providers import get_provider, PROVIDER_REGISTRY
from src.data.embeddings import EmbeddingService
from src.llm.query_clarification import QueryClarifier
from src.llm.answer_generation import AnswerGenerator
from src.retrieval.reranker import LLMReranker

# Initialize logger
logger = get_logger("test_provider")

def main():
    """Run the test provider script."""
    parser = argparse.ArgumentParser(description="Test LLM provider abstraction")
    parser.add_argument("--query", type=str, default="What is a search fund?", help="Query to test with")
    parser.add_argument("--text", type=str, default="A search fund is an investment vehicle through which an entrepreneur acquires a company they did not found.", help="Text to test embedding with")
    parser.add_argument("--show-config", action="store_true", help="Show full LLM configuration")
    parser.add_argument("--register-demo-provider", action="store_true", help="Register the demo Anthropic provider")
    args = parser.parse_args()
    
    # Register demo provider if requested
    if args.register_demo_provider:
        try:
            # Import the demo provider
            from examples.anthropic_provider import AnthropicProvider
            
            # Register it directly
            PROVIDER_REGISTRY["anthropic"] = AnthropicProvider
            logger.info(f"Registered demo Anthropic provider")
            
            # Set up a dummy API key for demo purposes
            import os
            os.environ["ANTHROPIC_API_KEY"] = "demo_key_1234567890"
        except ImportError:
            logger.error("Could not import demo Anthropic provider. Make sure the examples directory is in your Python path.")
        except Exception as e:
            logger.error(f"Error registering demo provider: {e}")
    
    logger.info("Testing LLM provider abstraction...")
    logger.info(f"Using LLM config: {LLM_CONFIG.config_file}")
    
    # Show available providers
    logger.info(f"Available providers: {', '.join(PROVIDER_REGISTRY.keys())}")
    
    # Show full configuration if requested
    if args.show_config:
        logger.info("Current LLM configuration:")
        for function_type, config in LLM_CONFIG.config_data.items():
            logger.info(f"  {function_type}:")
            for key, value in config.items():
                logger.info(f"    {key}: {value}")
    
    # Test embedding service
    logger.info("Testing embedding service...")
    try:
        embedding_service = EmbeddingService()
        provider = embedding_service.provider
        provider_name = provider.provider_name
        logger.info(f"Using provider: {provider_name}")
        logger.info(f"Embedding model: {provider.embedding_model}")
        
        embedding, tokens = embedding_service.get_embedding(args.text)
        logger.info(f"Embedding successful. Dimensions: {len(embedding)}, Tokens: {tokens}")
    except Exception as e:
        logger.error(f"Embedding test failed: {e}")
    
    # Test query clarifier
    logger.info("Testing query clarifier...")
    try:
        query_clarifier = QueryClarifier()
        provider = query_clarifier.provider
        provider_name = provider.provider_name
        logger.info(f"Using provider: {provider_name}")
        logger.info(f"Clarification model: {provider.clarification_model}")
        
        clarified_query = query_clarifier.clarify_query(args.query)
        logger.info(f"Original query: '{args.query}'")
        logger.info(f"Clarified query: '{clarified_query}'")
    except Exception as e:
        logger.error(f"Query clarification test failed: {e}")
    
    # Test answer generator
    logger.info("Testing answer generator...")
    try:
        answer_generator = AnswerGenerator()
        provider = answer_generator.provider
        provider_name = provider.provider_name
        logger.info(f"Using provider: {provider_name}")
        logger.info(f"Completion model: {provider.completion_model}")
        
        context = """
        A search fund is an investment vehicle through which an entrepreneur acquires a company they did not found.
        The searcher raises capital from investors to search for and acquire a small to mid-sized business,
        then operates the business as CEO for a period of years with the goal of growing the business and eventually exiting.
        
        The search fund model was pioneered at Stanford Graduate School of Business in 1984 and has become increasingly
        popular as an entrepreneurial path. There are generally two stages of investment in a search fund:
        
        1. The search capital stage, where investors provide capital to fund the searcher's efforts to find and evaluate acquisition targets.
        2. The acquisition capital stage, where investors provide capital to actually acquire the target business.
        
        Typical searchers are recent MBA graduates or experienced professionals who want to be entrepreneurs but don't have a startup idea or prefer to work with an established business rather than start from scratch.
        """
        
        answer = answer_generator.generate_answer(args.query, context)
        logger.info(f"Generated answer: '{answer[:200]}...'")
    except Exception as e:
        logger.error(f"Answer generation test failed: {e}")
    
    # Test reranker
    logger.info("Testing reranker...")
    try:
        reranker = LLMReranker()
        provider = reranker.provider
        provider_name = provider.provider_name
        logger.info(f"Using provider: {provider_name}")
        
        # Different model might be used for reranking compared to answer generation
        if provider_name == "openai":
            logger.info(f"Reranking model: {provider.completion_model}")
        elif provider_name == "perplexity":
            logger.info(f"Reranking model: {provider.completion_model}")
        
        documents = [
            {
                "document": "A search fund is an investment vehicle through which an entrepreneur acquires a company they did not found.",
                "metadata": {"file": "doc1.txt"},
                "score": 0.8
            },
            {
                "document": "Venture capital is a form of private equity financing provided by firms or funds to startups with high growth potential.",
                "metadata": {"file": "doc2.txt"},
                "score": 0.6
            }
        ]
        
        reranked_docs = reranker.rerank(args.query, documents)
        logger.info(f"Reranking successful. Scores: {[doc.get('relevance_score', 0) for doc in reranked_docs]}")
    except Exception as e:
        logger.error(f"Reranking test failed: {e}")
    
    logger.info("Tests completed.")

if __name__ == "__main__":
    main()