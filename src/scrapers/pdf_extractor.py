"""
PDF text extraction with header/footer removal.

This module extracts text from PDFs while filtering out headers and footers
based on their vertical position on the page.
"""

import os
import fitz  # PyMuPDF
from typing import Dict, List, Any, Optional, Tuple, Union
import logging

from src.core import get_logger

# Initialize logger
logger = get_logger("scrapers.pdf_extractor")

class PDFExtractor:
    """PDF text extraction with header/footer removal."""
    
    def __init__(
        self, 
        header_threshold: int = 100,
        footer_threshold: int = 150
    ):
        """Initialize the PDF extractor.
        
        Args:
            header_threshold: Vertical threshold for header content (pixels from top)
            footer_threshold: Vertical threshold for footer content (pixels from bottom)
        """
        self.header_threshold = header_threshold
        self.footer_threshold = footer_threshold
        
        logger.info(f"Initialized PDFExtractor with header_threshold={header_threshold}, footer_threshold={footer_threshold}")
    
    def extract_text(self, pdf_path: str) -> str:
        """Extract text from a PDF, removing headers and footers.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            str: Extracted text content
        """
        try:
            logger.info(f"Extracting text from {pdf_path}")
            doc = fitz.open(pdf_path)
        except Exception as e:
            logger.error(f"Error opening {pdf_path}: {e}")
            return ""
        
        extracted_text = []
        
        for page_num, page in enumerate(doc):
            page_height = page.rect.height
            blocks = page.get_text("blocks")  # Gets blocks in natural reading order
            
            page_text = []
            for block in blocks:
                # Unpack block coordinates and text
                x0, y0, x1, y1, text, block_no, block_type = block
                
                # Filter header/footer regions
                if (y0 > self.header_threshold) and (y1 < (page_height - self.footer_threshold)):
                    clean_text = text.strip()
                    if clean_text:
                        page_text.append(clean_text)
            
            if page_text:
                extracted_text.append("\n".join(page_text))
                
            logger.debug(f"Processed page {page_num+1}/{len(doc)}")
        
        text = "\n\n".join(extracted_text)
        logger.info(f"Extracted {len(text)} characters from {pdf_path}")
        
        return text
    
    def process_file(
        self, 
        pdf_path: str, 
        output_path: Optional[str] = None
    ) -> Tuple[str, bool]:
        """Process a PDF file and optionally save the extracted text.
        
        Args:
            pdf_path: Path to the PDF file
            output_path: Path to save the extracted text (if None, won't save)
            
        Returns:
            Tuple[str, bool]: Extracted text and success status
        """
        # Extract text
        text = self.extract_text(pdf_path)
        
        if not text:
            logger.warning(f"No text extracted from {pdf_path}")
            return text, False
        
        # Optionally save to file
        if output_path:
            try:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(text)
                logger.info(f"Saved extracted text to {output_path}")
            except Exception as e:
                logger.error(f"Error saving text to {output_path}: {e}")
                return text, False
        
        return text, True
    
    def process_directory(
        self, 
        input_dir: str, 
        output_dir: str,
        file_ext: str = ".pdf"
    ) -> Dict[str, bool]:
        """Process all PDFs in a directory.
        
        Args:
            input_dir: Input directory containing PDFs
            output_dir: Output directory for text files
            file_ext: File extension to filter for
            
        Returns:
            Dict[str, bool]: Dictionary mapping file paths to success/failure
        """
        logger.info(f"Processing directory: {input_dir}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        results = {}
        
        # Process each PDF file
        for filename in os.listdir(input_dir):
            if not filename.lower().endswith(file_ext):
                continue
                
            pdf_path = os.path.join(input_dir, filename)
            base_name = os.path.splitext(filename)[0]
            output_path = os.path.join(output_dir, f"{base_name}.txt")
            
            _, success = self.process_file(pdf_path, output_path)
            results[pdf_path] = success
        
        # Log summary
        success_count = sum(1 for success in results.values() if success)
        logger.info(f"Processed {len(results)} files, {success_count} succeeded, {len(results) - success_count} failed")
        
        return results


def main():
    """Main function to execute the PDF extractor."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract text from PDF files")
    parser.add_argument("--input", type=str, required=True, help="Input PDF file or directory")
    parser.add_argument("--output", type=str, required=True, help="Output text file or directory")
    parser.add_argument("--header", type=int, default=100, help="Header threshold (pixels from top)")
    parser.add_argument("--footer", type=int, default=150, help="Footer threshold (pixels from bottom)")
    
    args = parser.parse_args()
    
    extractor = PDFExtractor(header_threshold=args.header, footer_threshold=args.footer)
    
    if os.path.isdir(args.input):
        extractor.process_directory(args.input, args.output)
    else:
        extractor.process_file(args.input, args.output)


if __name__ == "__main__":
    main()