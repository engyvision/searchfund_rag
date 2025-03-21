"""
Text preprocessing for the Web Scraper project.

This module handles text extraction, cleaning, normalization, and chunking
to prepare data for embedding and indexing.
"""

import os
import re
import json
from typing import Dict, List, Any, Optional, Union, Callable
from pathlib import Path

from src.core.logging import get_logger

# Initialize logger
logger = get_logger("data.preprocessing")

def preprocess_text(text: str) -> str:
    """Preprocess text for BM25 and other retrieval methods.
    
    Args:
        text: Text to preprocess
        
    Returns:
        str: Preprocessed text
    """
    # Convert to lowercase
    text = text.lower()
    
    # Replace multiple whitespace with single space
    text = re.sub(r'\s+', ' ', text)
    
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    
    # Remove extra whitespace
    text = text.strip()
    
    return text


class TextPreprocessor:
    """Text preprocessing for documents."""
    
    def __init__(self):
        """Initialize the text preprocessor."""
        logger.info("Initialized TextPreprocessor")
    
    def clean_text(self, text: str) -> str:
        """Clean text by removing extra whitespace, normalizing, etc.
        
        Args:
            text: The text to clean
            
        Returns:
            str: The cleaned text
        """
        # Replace multiple newlines with a single newline
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Replace multiple spaces with a single space
        text = re.sub(r' {2,}', ' ', text)
        
        # Remove non-printable characters
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]', '', text)
        
        # Trim leading/trailing whitespace from each line
        text = '\n'.join(line.strip() for line in text.splitlines())
        
        # Trim leading/trailing whitespace from the entire text
        text = text.strip()
        
        return text
    
    def split_by_section(
        self, 
        text: str, 
        section_patterns: List[str] = None,
        min_section_length: int = 100
    ) -> List[Dict[str, str]]:
        """Split text into sections based on patterns.
        
        Args:
            text: The text to split
            section_patterns: Regular expression patterns for section headers
            min_section_length: Minimum length for a section to be included
            
        Returns:
            List[Dict[str, str]]: List of sections with titles and content
        """
        if section_patterns is None:
            # Default patterns for common section headers
            section_patterns = [
                r'^#{1,6}\s+(.+)$',  # Markdown headers
                r'^([A-Z][A-Za-z\s]{2,50})$',  # Capitalized lines that could be headers
                r'^(\d+\.\s+[A-Z][A-Za-z\s]{2,50})$',  # Numbered sections
            ]
        
        # Compile patterns
        compiled_patterns = [re.compile(pattern, re.MULTILINE) for pattern in section_patterns]
        
        # Find all potential section headers
        headers = []
        for pattern in compiled_patterns:
            for match in pattern.finditer(text):
                headers.append((match.start(), match.end(), match.group(1)))
        
        # Sort headers by position
        headers.sort()
        
        # Extract sections
        sections = []
        for i, (start, end, title) in enumerate(headers):
            # Get section content (from after this header to start of next header)
            if i < len(headers) - 1:
                content = text[end:headers[i+1][0]].strip()
            else:
                content = text[end:].strip()
            
            # Only include if content is substantial
            if len(content) >= min_section_length:
                sections.append({
                    "title": title.strip(),
                    "content": content
                })
        
        # If no sections were found or first header isn't at the start, add the beginning content
        if not headers or headers[0][0] > 0:
            first_content = text[:headers[0][0] if headers else None].strip()
            if first_content and len(first_content) >= min_section_length:
                sections.insert(0, {
                    "title": "Introduction",
                    "content": first_content
                })
        
        logger.info(f"Split text into {len(sections)} sections")
        return sections
    
    def extract_metadata(self, text: str) -> Dict[str, Any]:
        """Extract metadata from document text.
        
        This is a placeholder for document-specific metadata extraction.
        
        Args:
            text: The document text
            
        Returns:
            Dict[str, Any]: Extracted metadata
        """
        metadata = {
            "length": len(text),
            "word_count": len(text.split()),
        }
        
        # Try to extract a title
        title_match = re.search(r'^#\s+(.+)$|^Title:\s*(.+)$', text, re.MULTILINE)
        if title_match:
            metadata["title"] = title_match.group(1) or title_match.group(2)
        
        # Try to extract a date
        date_match = re.search(r'Date:\s*(.+)$|(\d{4}-\d{2}-\d{2})', text, re.MULTILINE)
        if date_match:
            metadata["date"] = date_match.group(1) or date_match.group(2)
        
        return metadata
    

class DocumentProcessor:
    """Processing pipeline for documents."""
    
    def __init__(
        self, 
        input_dir: str,
        output_dir: str,
        preprocessor: Optional[TextPreprocessor] = None
    ):
        """Initialize the document processor.
        
        Args:
            input_dir: Directory containing input files
            output_dir: Directory for output files
            preprocessor: Text preprocessor instance (created if not provided)
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.preprocessor = preprocessor or TextPreprocessor()
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"Initialized DocumentProcessor with input_dir={input_dir}, output_dir={output_dir}")
    
    def process_file(
        self, 
        file_path: str,
        custom_processing: Optional[Callable[[str], str]] = None
    ) -> bool:
        """Process a single file.
        
        Args:
            file_path: Path to the file to process
            custom_processing: Optional custom processing function
            
        Returns:
            bool: True if processing succeeded, False otherwise
        """
        try:
            # Determine output file path
            file_name = os.path.basename(file_path)
            output_path = os.path.join(self.output_dir, file_name)
            
            # Read input file
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # Apply preprocessing
            text = self.preprocessor.clean_text(text)
            
            # Apply custom processing if provided
            if custom_processing:
                text = custom_processing(text)
            
            # Extract metadata
            metadata = self.preprocessor.extract_metadata(text)
            
            # Write processed text
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(text)
            
            # Save metadata if needed
            metadata_path = os.path.join(self.output_dir, f"{os.path.splitext(file_name)[0]}_metadata.json")
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Processed {file_path} -> {output_path}")
            return True
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            return False
    
    def process_directory(
        self, 
        file_ext: str = ".txt",
        custom_processing: Optional[Callable[[str], str]] = None
    ) -> Dict[str, bool]:
        """Process all files in the input directory.
        
        Args:
            file_ext: File extension to filter for
            custom_processing: Optional custom processing function
            
        Returns:
            Dict[str, bool]: Dictionary mapping file paths to success/failure
        """
        results = {}
        
        for filename in os.listdir(self.input_dir):
            if not filename.endswith(file_ext):
                continue
                
            file_path = os.path.join(self.input_dir, filename)
            results[file_path] = self.process_file(file_path, custom_processing)
        
        # Log summary
        success_count = sum(1 for success in results.values() if success)
        logger.info(f"Processed {len(results)} files, {success_count} succeeded, {len(results) - success_count} failed")
        
        return results