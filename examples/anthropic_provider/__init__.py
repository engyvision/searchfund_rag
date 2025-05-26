"""
Example Anthropic provider for the web scraper project.

This is a demonstration of how to create a third-party provider
that can be dynamically discovered by the provider system.
"""

from .anthropic_provider import AnthropicProvider

__all__ = ["AnthropicProvider"]