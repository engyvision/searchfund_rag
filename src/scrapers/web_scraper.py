"""
Web scraper for IESE search fund resources.

This module scrapes PDF documents and their metadata from the IESE website's
search funds section.
"""

import requests
from bs4 import BeautifulSoup
import os
import re
from urllib.parse import urljoin
import logging
from typing import Dict, List, Any, Optional

from src.core import get_logger

# Initialize logger
logger = get_logger("scrapers.web_scraper")

class IESEScraper:
    """Web scraper for IESE search fund resources."""
    
    def __init__(
        self, 
        base_url: str = "https://www.iese.edu/entrepreneurship/search-funds/",
        output_dir: Optional[str] = None
    ):
        """Initialize the IESE scraper.
        
        Args:
            base_url: Base URL for the IESE search funds page
            output_dir: Directory to save downloaded files (default: data/IESE)
        """
        self.base_url = base_url
        
        # Set default output directory if not provided
        if output_dir is None:
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            self.output_dir = os.path.join(project_root, "data", "IESE")
        else:
            self.output_dir = output_dir
        
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        
        logger.info(f"Initialized IESEScraper with base_url={base_url}, output_dir={self.output_dir}")
    
    def clean_filename(self, title: str) -> str:
        """Generate safe filenames from document titles.
        
        Args:
            title: The document title
            
        Returns:
            str: A cleaned filename
        """
        # Remove special characters and normalize whitespace
        clean = re.sub(r'[^a-zA-Z0-9- ]', '', title)
        clean = re.sub(r'\s+', ' ', clean).strip()
        return clean.lower().replace(' ', '-')[:100]
    
    def extract_section_data(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """Extract PDF metadata from all collapsible sections.
        
        Args:
            soup: BeautifulSoup object for the page
            
        Returns:
            List[Dict[str, Any]]: List of sections with PDF links
        """
        sections = []
        
        # Find all section containers
        for section in soup.find_all('div', class_='collapse-item'):
            section_title = section.find('a', class_='btn-collapse').get_text(strip=True)
            section_links = []
            
            # Extract PDF links and titles
            for item in section.find_all('div', class_='collapsed-content'):
                for a in item.find_all('a', href=True):
                    if '.pdf' not in a['href'].lower():
                        continue
                    
                    # Extract document title
                    doc_title = a.get_text(strip=True)
                    if not doc_title or len(doc_title) < 5:
                        continue  # Skip empty/invalid titles
                    
                    # Extract publication year
                    year_match = re.search(r'(20\d{2})', item.text)
                    year_suffix = f"-{year_match.group(1)}" if year_match else ''
                    
                    # Generate filename
                    filename = f"{self.clean_filename(doc_title)}{year_suffix}.pdf"
                    
                    section_links.append({
                        'url': urljoin(self.base_url, a['href']),
                        'filename': filename,
                        'section': section_title,
                        'title': doc_title,
                        'year': year_match.group(1) if year_match else None
                    })
            
            if section_links:
                sections.append({
                    'name': section_title,
                    'links': section_links
                })
        
        return sections
    
    def download_pdf(self, link_info: Dict[str, Any]) -> bool:
        """Download PDF file.
        
        Args:
            link_info: Dictionary with PDF metadata
            
        Returns:
            bool: True if download was successful, False otherwise
        """
        # Create section folder
        section_folder = os.path.join(self.output_dir, link_info['section'].replace(' ', '-'))
        os.makedirs(section_folder, exist_ok=True)
        
        pdf_path = os.path.join(section_folder, link_info['filename'])
        
        try:
            # Download PDF
            logger.info(f"Downloading {link_info['url']} to {pdf_path}")
            response = requests.get(link_info['url'], headers=self.headers, stream=True, timeout=20)
            
            if response.status_code != 200:
                logger.error(f"Failed to download {link_info['filename']}: HTTP {response.status_code}")
                return False
            
            with open(pdf_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            logger.info(f"Successfully downloaded {link_info['filename']}")
            return True
        
        except Exception as e:
            logger.error(f"Error downloading {link_info['filename']}: {str(e)}")
            return False
    
    def scrape(self) -> List[Dict[str, Any]]:
        """Scrape PDFs from the IESE website.
        
        Returns:
            List[Dict[str, Any]]: Information about downloaded PDFs
        """
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Fetch main page
        logger.info(f"Fetching {self.base_url}")
        response = requests.get(self.base_url, headers=self.headers)
        
        if response.status_code != 200:
            logger.error(f"Failed to fetch {self.base_url}: HTTP {response.status_code}")
            return []
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract all PDF metadata
        sections = self.extract_section_data(soup)
        
        # Process all documents
        downloaded_pdfs = []
        total_downloaded = 0
        
        for section in sections:
            logger.info(f"Processing section: {section['name']}")
            
            for link in section['links']:
                if self.download_pdf(link):
                    downloaded_pdfs.append(link)
                    total_downloaded += 1
        
        logger.info(f"Successfully downloaded {total_downloaded} documents to {self.output_dir}")
        return downloaded_pdfs


def main():
    """Main function to execute the scraper."""
    scraper = IESEScraper()
    scraper.scrape()


if __name__ == "__main__":
    main()