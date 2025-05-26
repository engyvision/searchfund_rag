"""
Demo script for registering a provider directly.

This demonstrates how to register a provider dynamically without using entry points,
which can be useful during development.
"""

import os
import sys

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the LLM providers module
from src.llm.providers import PROVIDER_REGISTRY, get_provider
from src.core import LLM_CONFIG, get_logger
from examples.anthropic_provider import AnthropicProvider

# Initialize logger
logger = get_logger("register_provider_demo")

def main():
    """Run the demo."""
    # Register the Anthropic provider directly
    PROVIDER_REGISTRY["anthropic"] = AnthropicProvider
    logger.info(f"Registered Anthropic provider directly")
    
    # Show all registered providers
    logger.info(f"Registered providers: {', '.join(PROVIDER_REGISTRY.keys())}")
    
    # Try to get a provider for a function type
    try:
        # Make sure the environment variable is set
        os.environ["ANTHROPIC_API_KEY"] = "demo_key_1234567890"
        
        # Set up a demo config
        provider_config = {
            "embeddings": {
                "provider": "openai",
                "model": "text-embedding-3-small",
                "api_key": "demo_key_openai"
            },
            "query_clarification": {
                "provider": "anthropic",
                "model": "claude-3-haiku-20240307",
                "api_key": "demo_key_anthropic"
            }
        }
        
        # Get the provider for query clarification
        provider = get_provider(provider_config, "query_clarification")
        logger.info(f"Got provider for query clarification: {provider.provider_name}")
        logger.info(f"Model: {provider.clarification_model}")
        
        # Try to clarify a query
        clarified_query = provider.clarify_query("What is a search fund?")
        logger.info(f"Clarified query: {clarified_query}")
        
    except Exception as e:
        logger.error(f"Error: {e}")

if __name__ == "__main__":
    main()