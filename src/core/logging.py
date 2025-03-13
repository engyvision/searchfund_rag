"""
Centralized logging configuration for the Web Scraper project.

This module provides a consistent logging setup across the application,
with support for both console and file logging, as well as different
log levels for development and production environments.
"""

import os
import logging
import json
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path
import sys

# Import configuration to check environment
try:
    from src.core.config import CONFIG
    ENV = CONFIG.env
except ImportError:
    # Default to development if config not available
    ENV = "development"

# Log levels based on environment
LOG_LEVELS = {
    "development": logging.DEBUG,
    "production": logging.INFO,
}

# Default log directory
LOG_DIR = "logs"

class StructuredLogAdapter(logging.LoggerAdapter):
    """Custom log adapter that adds structured context to log records."""
    
    def process(self, msg: str, kwargs: Dict[str, Any]) -> tuple:
        """Process the logging message and arguments.
        
        Args:
            msg: The log message
            kwargs: Additional keyword arguments
            
        Returns:
            tuple: Processed message and keywords
        """
        # Add timestamp and context to message
        extra = kwargs.get("extra", {})
        
        if not isinstance(extra, dict):
            extra = {"extra": extra}
            
        # Add timestamp if not exists
        if "timestamp" not in extra:
            extra["timestamp"] = datetime.now().isoformat()
            
        # Format as JSON if needed
        if kwargs.pop("json", False):
            # Combine message and extra into a single JSON object
            data = {"message": msg}
            data.update(extra)
            msg = json.dumps(data)
        
        kwargs["extra"] = extra
        return msg, kwargs

def setup_logging(
    name: str, 
    log_level: Optional[int] = None,
    log_to_file: bool = True,
    log_to_console: bool = True,
    log_file_name: Optional[str] = None
) -> logging.Logger:
    """Set up a logger with the specified configuration.
    
    Args:
        name: The name of the logger
        log_level: The logging level (if None, uses environment-based default)
        log_to_file: Whether to log to a file
        log_to_console: Whether to log to the console
        log_file_name: Custom log file name (if None, uses {name}.log)
        
    Returns:
        logging.Logger: Configured logger
    """
    # Get the logger
    logger = logging.getLogger(name)
    
    # Clear any existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Set level based on environment if not specified
    if log_level is None:
        log_level = LOG_LEVELS.get(ENV, logging.INFO)
    
    logger.setLevel(log_level)
    
    # Create formatters
    console_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] [%(name)s] %(message)s"
    )
    
    file_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] [%(name)s] [%(filename)s:%(lineno)d] %(message)s"
    )
    
    # Add console handler
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    # Add file handler
    if log_to_file:
        # Create log directory if it doesn't exist
        Path(LOG_DIR).mkdir(exist_ok=True)
        
        # Set log file name
        if log_file_name is None:
            log_file_name = f"{name.replace('.', '_')}.log"
        
        log_file_path = os.path.join(LOG_DIR, log_file_name)
        
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    # Wrap the logger with the adapter
    return StructuredLogAdapter(logger, {})

# Initialize root logger
root_logger = setup_logging("webscraper")

def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name.
    
    Creates a child logger of the root logger with the specified name.
    
    Args:
        name: The name of the logger
        
    Returns:
        logging.Logger: The logger
    """
    if name == "webscraper":
        return root_logger
    
    # Create child logger - inherits handlers from parent
    logger_name = f"webscraper.{name}"
    return logging.getLogger(logger_name)