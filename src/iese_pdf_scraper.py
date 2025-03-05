#!/usr/bin/env python3
import requests
from bs4 import BeautifulSoup
import os
import re
from urllib.parse import urljoin
from PyPDF2 import PdfReader

# ------------------------------ CONFIGURATION ------------------------------
BASE_URL = "https://www.iese.edu/entrepreneurship/search-funds/"
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
OUTPUT_DIR = os.path.join(BASE_DIR, "IESE")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

def clean_filename(title):
    """Generate safe filenames from document titles"""
    # Remove special characters and normalize whitespace
    clean = re.sub(r'[^a-zA-Z0-9- ]', '', title)
    clean = re.sub(r'\s+', ' ', clean).strip()
    return clean.lower().replace(' ', '-')[:100]

def extract_section_data(soup):
    """Extract PDF metadata from all collapsible sections"""
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
                filename = f"{clean_filename(doc_title)}{year_suffix}.pdf"
                
                section_links.append({
                    'url': urljoin(BASE_URL, a['href']),
                    'filename': filename,
                    'section': section_title
                })
        
        if section_links:
            sections.append({
                'name': section_title,
                'links': section_links
            })
    
    return sections

def download_pdf(link_info, output_dir):
    """Download PDF and extract text content"""
    # Create section folder
    section_folder = os.path.join(output_dir, link_info['section'].replace(' ', '-'))
    os.makedirs(section_folder, exist_ok=True)
    
    pdf_path = os.path.join(section_folder, link_info['filename'])
    
    try:
        # Download PDF
        response = requests.get(link_info['url'], headers=HEADERS, stream=True, timeout=20)
        if response.status_code != 200:
            return False
        
        with open(pdf_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        # Extract text
        txt_path = pdf_path.replace('.pdf', '.txt')
        with open(txt_path, 'w', encoding='utf-8') as txt_file:
            pdf = PdfReader(pdf_path)
            text = '\n'.join([page.extract_text() for page in pdf.pages])
            txt_file.write(text)
        
        return True
    
    except Exception as e:
        print(f"Error processing {link_info['filename']}: {str(e)}")
        return False

def main():
    # Fetch main page
    response = requests.get(BASE_URL, headers=HEADERS)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Extract all PDF metadata
    sections = extract_section_data(soup)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Process all documents
    total_downloaded = 0
    for section in sections:
        print(f"\nProcessing section: {section['name']}")
        for link in section['links']:
            print(f"Downloading: {link['filename']}")
            if download_pdf(link, OUTPUT_DIR):
                total_downloaded += 1
    
    print(f"\nSuccessfully processed {total_downloaded} documents")

if __name__ == "__main__":
    main()
