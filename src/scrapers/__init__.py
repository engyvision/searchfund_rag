"""
Web and PDF scraping functionality for the Web Scraper project.

This package contains modules for scraping web pages and extracting
text from PDF documents.
"""

from src.scrapers.web_scraper import IESEScraper
from src.scrapers.pdf_extractor import PDFExtractor

__all__ = ['IESEScraper', 'PDFExtractor']