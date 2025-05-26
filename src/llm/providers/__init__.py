"""
Provider module for LLM integrations.

This module provides a consistent interface for different LLM providers
and a factory function for creating provider instances.

The provider system supports dynamic discovery of providers through
two mechanisms:
1. Built-in providers (directly imported)
2. Entry points (for third-party providers)

Entry points should be registered under the 'webscraper.llm_providers' group.
"""

import os
import importlib
import pkgutil
import inspect
from typing import Dict, Any, Type, List, Tuple
import pkg_resources

from src.llm.providers.base_provider import BaseLLMProvider
from src.core import get_logger

# Initialize logger
logger = get_logger("llm.providers")

# Registry of provider classes
PROVIDER_REGISTRY: Dict[str, Type[BaseLLMProvider]] = {}

# Cache for provider instances
_provider_cache: Dict[str, BaseLLMProvider] = {}


def _is_provider_class(obj: Any) -> bool:
    """Check if an object is a provider class (subclass of BaseLLMProvider).
    
    Args:
        obj: Object to check
        
    Returns:
        bool: True if the object is a provider class
    """
    return (
        inspect.isclass(obj) and 
        issubclass(obj, BaseLLMProvider) and 
        obj is not BaseLLMProvider
    )


def _get_provider_name(provider_class: Type[BaseLLMProvider]) -> str:
    """Get the name of a provider class.
    
    This uses the provider_name property if it's a class attribute, 
    otherwise it derives the name from the class name.
    
    Args:
        provider_class: Provider class
        
    Returns:
        str: Provider name
    """
    # Check if provider_name is defined as a class attribute
    if hasattr(provider_class, 'provider_name') and isinstance(getattr(provider_class, 'provider_name'), property):
        # Create a temporary instance to get the provider_name
        try:
            # This is a hack to get the provider name without initializing the class
            # Only works if provider_name is a simple property that doesn't depend on initialized state
            provider_name = provider_class.provider_name.fget(None)
            if provider_name:
                return provider_name
        except:
            pass
    
    # Fallback: derive name from class name
    class_name = provider_class.__name__
    if class_name.endswith('Provider'):
        return class_name[:-8].lower()  # Remove 'Provider' and lowercase
    return class_name.lower()


def discover_builtin_providers() -> None:
    """Discover and register built-in providers from this package."""
    logger.debug("Discovering built-in providers...")
    
    # Import built-in provider modules
    from src.llm.providers import openai_provider, perplexity_provider
    
    # Scan modules for provider classes
    modules_to_scan = [
        openai_provider,
        perplexity_provider
    ]
    
    for module in modules_to_scan:
        for name, obj in inspect.getmembers(module):
            if _is_provider_class(obj):
                provider_name = _get_provider_name(obj)
                PROVIDER_REGISTRY[provider_name] = obj
                logger.debug(f"Registered built-in provider: {provider_name} ({obj.__name__})")


def discover_entry_point_providers() -> None:
    """Discover and register providers from entry points."""
    logger.debug("Discovering entry point providers...")
    
    try:
        # Look for entry points in the 'webscraper.llm_providers' group
        for entry_point in pkg_resources.iter_entry_points(group='webscraper.llm_providers'):
            try:
                # Load the entry point
                provider_class = entry_point.load()
                
                # Verify it's a provider class
                if _is_provider_class(provider_class):
                    # Get provider name - use entry point name as a fallback
                    provider_name = _get_provider_name(provider_class)
                    
                    # Register the provider
                    PROVIDER_REGISTRY[provider_name] = provider_class
                    logger.debug(f"Registered entry point provider: {provider_name} ({provider_class.__name__})")
                else:
                    logger.warning(f"Entry point {entry_point.name} does not point to a valid provider class")
            except Exception as e:
                logger.error(f"Error loading provider from entry point {entry_point.name}: {e}")
    except Exception as e:
        logger.error(f"Error discovering entry point providers: {e}")


def discover_providers() -> None:
    """Discover and register all available providers."""
    discover_builtin_providers()
    discover_entry_point_providers()
    
    if not PROVIDER_REGISTRY:
        logger.warning("No providers found!")
    else:
        logger.info(f"Discovered {len(PROVIDER_REGISTRY)} providers: {', '.join(PROVIDER_REGISTRY.keys())}")


# Discover providers at module import time
discover_providers()


def get_provider(provider_config: Dict[str, Any], function_type: str) -> BaseLLMProvider:
    """Get a provider instance for the specified function type.
    
    Args:
        provider_config: Configuration dictionary with provider settings
        function_type: Type of function (embeddings, query_clarification, answer_generation, reranking)
        
    Returns:
        BaseLLMProvider: A provider instance
        
    Raises:
        ValueError: If the provider is not found or if required config is missing
    """
    if function_type not in provider_config:
        raise ValueError(f"Configuration for function type '{function_type}' not found")
    
    function_config = provider_config[function_type]
    provider_name = function_config.get("provider")
    
    if not provider_name:
        raise ValueError(f"Provider name not specified for function type '{function_type}'")
    
    if provider_name not in PROVIDER_REGISTRY:
        raise ValueError(f"Provider '{provider_name}' not found in registry")
    
    # If we've already initialized this provider, return it from cache
    if provider_name in _provider_cache:
        logger.debug(f"Using cached provider for {provider_name}")
        return _provider_cache[provider_name]
    
    # Get provider class
    provider_class = PROVIDER_REGISTRY[provider_name]
    
    # Initialize provider with full configuration
    try:
        provider = provider_class(config=provider_config)
        
        # Cache the provider
        _provider_cache[provider_name] = provider
        
        logger.info(f"Initialized {provider_name} provider for {function_type}")
        return provider
    except Exception as e:
        logger.error(f"Error initializing {provider_name} provider: {e}")
        raise


__all__ = ["BaseLLMProvider", "get_provider", "PROVIDER_REGISTRY"]