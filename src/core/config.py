"""
Configuration management for the Web Scraper project.

This module centralizes all configuration loading and validation, supporting
multiple environments (local development, Streamlit Cloud) and configuration sources
(environment variables, YAML files, Streamlit secrets).
"""

import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path
from pydantic import BaseModel, Field, ValidationError
from dotenv import load_dotenv
import logging

# Initialize logger
logger = logging.getLogger(__name__)

class OpenAIConfig(BaseModel):
    """Configuration for OpenAI API."""
    api_key: str = Field(..., description="OpenAI API key")
    embedding_model: str = Field("text-embedding-3-small", description="Model to use for embeddings")
    completion_model: str = Field("o3-mini-2025-01-31", description="Model to use for completions")
    clarification_model: str = Field("gpt-4o-mini", description="Model to use for query clarification")

class FAISSConfig(BaseModel):
    """Configuration for FAISS vector database."""
    index_file: str = Field("data/faiss_index.idx", description="Path to FAISS index file")
    metadata_file: str = Field("data/faiss_metadata.json", description="Path to FAISS metadata file")
    embedding_dim: int = Field(1536, description="Dimension of embeddings")

class PDFConfig(BaseModel):
    """Configuration for PDF processing."""
    config_file: str = Field("src/pdf_config.yaml", description="Path to PDF config YAML")
    
    # These will be populated from the YAML file
    pdf_urls: Optional[Dict[str, str]] = None
    output_directory: Optional[str] = None

class AppConfig(BaseModel):
    """Master configuration for the application."""
    openai: OpenAIConfig
    faiss: FAISSConfig
    pdf: PDFConfig
    project_root: str = Field(..., description="Root directory of the project")
    env: str = Field("development", description="Environment (development, production)")

def get_project_root() -> Path:
    """Get the project root directory.
    
    Returns:
        Path: The project root directory path
    """
    # Current file is in src/core, so go up two levels
    return Path(__file__).parent.parent.parent

def load_yaml_config(file_path: str) -> Dict[str, Any]:
    """Load configuration from a YAML file.
    
    Args:
        file_path: Path to the YAML configuration file
        
    Returns:
        Dict[str, Any]: The loaded configuration
        
    Raises:
        FileNotFoundError: If the YAML file doesn't exist
    """
    try:
        with open(file_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {file_path}")
        raise

def load_config() -> AppConfig:
    """Load and validate the application configuration.
    
    Tries different configuration sources in this order:
    1. Streamlit secrets (if running in Streamlit Cloud)
    2. Environment variables (loaded from .env file if it exists)
    3. YAML configuration files
    
    Returns:
        AppConfig: Validated application configuration
        
    Raises:
        ValueError: If required configuration values are missing
    """
    # Get project root
    project_root = str(get_project_root())
    
    # Try to load OpenAI API key from Streamlit secrets first (for cloud deployment)
    openai_api_key = None
    try:
        import streamlit as st
        openai_api_key = st.secrets.get("OPENAI_API_KEY")
        env = "production"
        logger.info("Loaded API key from Streamlit secrets")
    except (ModuleNotFoundError, AttributeError, FileNotFoundError):
        # Not running in Streamlit or secrets not found
        env = "development"
    
    # If not found in Streamlit secrets, try environment variables
    if not openai_api_key:
        load_dotenv()  # Load from .env file if it exists
        openai_api_key = os.getenv("OPENAI_API_KEY")
        logger.info("Loaded API key from environment variables")
    
    if not openai_api_key:
        raise ValueError("OpenAI API key not found. Please set it in .env or Streamlit secrets.")
    
    # Load PDF configuration from YAML
    pdf_config_path = os.path.join(project_root, "src", "pdf_config.yaml")
    try:
        pdf_config_data = load_yaml_config(pdf_config_path)
        logger.info(f"Loaded PDF configuration from {pdf_config_path}")
    except FileNotFoundError:
        pdf_config_data = {}
        logger.warning(f"PDF configuration file not found: {pdf_config_path}")
    
    # Build and validate the configuration
    try:
        config = AppConfig(
            openai=OpenAIConfig(
                api_key=openai_api_key,
                # Use environment variables if set, otherwise use defaults
                embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
                completion_model=os.getenv("OPENAI_COMPLETION_MODEL", "o3-mini-2025-01-31"),
                clarification_model=os.getenv("OPENAI_CLARIFICATION_MODEL", "gpt-4o-mini"),
            ),
            faiss=FAISSConfig(
                index_file=os.path.join(project_root, "data", "faiss_index.idx"),
                metadata_file=os.path.join(project_root, "data", "faiss_metadata.json"),
                embedding_dim=1536,
            ),
            pdf=PDFConfig(
                config_file=pdf_config_path,
                pdf_urls=pdf_config_data.get("pdf_urls", {}),
                output_directory=pdf_config_data.get("output_directory", "data/IESE"),
            ),
            project_root=project_root,
            env=env,
        )
        return config
    except ValidationError as e:
        logger.error(f"Configuration validation error: {e}")
        raise

# Create a singleton instance
CONFIG = load_config()

# Export individual configs for convenience
OPENAI_CONFIG = CONFIG.openai
FAISS_CONFIG = CONFIG.faiss
PDF_CONFIG = CONFIG.pdf

# Backwards compatibility for direct imports
OPENAI_API_KEY = OPENAI_CONFIG.api_key