"""
Setup script for the Anthropic provider example.

This demonstrates how to create a third-party provider that can be
discovered by the Web Scraper project.
"""

from setuptools import setup, find_packages

setup(
    name="anthropic_provider",
    version="0.1.0",
    description="Anthropic provider for Web Scraper project",
    author="Example Author",
    author_email="example@example.com",
    packages=find_packages(),
    install_requires=[
        "webscraper",  # This would depend on the main project
    ],
    entry_points={
        # This is the key part - register the provider class as an entry point
        "webscraper.llm_providers": [
            "anthropic = anthropic_provider:AnthropicProvider",
        ],
    },
)