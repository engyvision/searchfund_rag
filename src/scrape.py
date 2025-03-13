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
    """Create safe filenames from document titles"""
    # Remove special characters and normalize
    clean = re.sub(r'[^a-zA-Z0-9- ]', '', title)
    clean = re.sub(r'\s+', '-', clean).strip().lower()
    return clean[:80] + ".pdf"

def extract_pdf_links(soup):
    """Extract all PDF links with their visible text"""
    pdf_links = []
    
    # Find all links containing PDFs
    for a in soup.find_all('a', href=True):
        if '.pdf' in a['href'].lower():
            # Get link text from either the <a> tag or its parent <li>
            link_text = a.get_text(strip=True)
            if not link_text:
                if a.parent.name == 'li':
                    link_text = a.parent.get_text(strip=True)
            
            if link_text:
                pdf_links.append({
                    'url': urljoin(BASE_URL, a['href']),
                    'title': link_text
                })
    
    return pdf_links

def download_pdf(link, output_dir):
    """Download PDF and extract text"""
    try:
        # Generate filename
        filename = clean_filename(link['title'])
        pdf_path = os.path.join(output_dir, filename)
        
        # Download PDF
        response = requests.get(link['url'], headers=HEADERS, stream=True, timeout=20)
        if response.status_code != 200:
            print(f"Failed to download: {filename}")
            return False
        
        with open(pdf_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        # Extract text
        try:
            with open(pdf_path.replace('.pdf', '.txt'), 'w', encoding='utf-8') as txt_file:
                pdf = PdfReader(pdf_path)
                text = '\n'.join([page.extract_text() for page in pdf.pages])
                txt_file.write(text)
        except Exception as e:
            print(f"Text extraction failed for {filename}: {str(e)}")
        
        print(f"Success: {filename}")
        return True
    
    except Exception as e:
        print(f"Error processing {link['url']}: {str(e)}")
        return False

def main():
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Fetch and parse main page
    response = requests.get(BASE_URL, headers=HEADERS)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Extract all PDF links
    pdf_links = extract_pdf_links(soup)
    print(f"Found {len(pdf_links)} PDF links")
    
    # Download all files
    success_count = 0
    for link in pdf_links:
        if download_pdf(link, OUTPUT_DIR):
            success_count += 1
    
    print(f"\nSuccessfully downloaded {success_count} files to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
