#!/usr/bin/env python3
"""
Script to scrape documents from the IESE search funds website.

This script downloads PDF documents from the IESE website and optionally
extracts their text content.
"""

import argparse
import os
from pathlib import Path

from src.core import get_logger
from src.scrapers import IESEScraper, PDFExtractor

# Initialize logger
logger = get_logger("scripts.scrape_documents")

def parse_args():
    """Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Scrape PDF documents from IESE website")
    
    parser.add_argument(
        "--base-url",
        type=str,
        default="https://www.iese.edu/entrepreneurship/search-funds/",
        help="Base URL for the IESE search funds page"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/IESE",
        help="Directory to save downloaded files"
    )
    
    parser.add_argument(
        "--extract-text",
        action="store_true",
        help="Extract text from downloaded PDFs"
    )
    
    parser.add_argument(
        "--text-output-dir",
        type=str,
        default="data/processed_txt",
        help="Directory to save extracted text"
    )
    
    return parser.parse_args()

def main():
    """Main function to scrape documents."""
    args = parse_args()
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    if args.extract_text:
        os.makedirs(args.text_output_dir, exist_ok=True)
    
    # Initialize scraper
    logger.info(f"Initializing scraper with base_url={args.base_url}, output_dir={args.output_dir}")
    scraper = IESEScraper(base_url=args.base_url, output_dir=args.output_dir)
    
    # Scrape PDFs
    downloaded_pdfs = scraper.scrape()
    
    # Optional text extraction
    if args.extract_text and downloaded_pdfs:
        logger.info(f"Extracting text from {len(downloaded_pdfs)} downloaded PDFs")
        
        extractor = PDFExtractor()
        
        for pdf_info in downloaded_pdfs:
            section_folder = os.path.join(args.output_dir, pdf_info['section'].replace(' ', '-'))
            pdf_path = os.path.join(section_folder, pdf_info['filename'])
            
            output_path = os.path.join(
                args.text_output_dir, 
                f"{os.path.splitext(pdf_info['filename'])[0]}.txt"
            )
            
            extractor.process_file(pdf_path, output_path)
    
    logger.info("Scraping completed")

if __name__ == "__main__":
    main()