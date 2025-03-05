import os
import pymupdf as fitz
import pathlib
import logging
import pytesseract
from pathvalidate import sanitize_filename
from PIL import Image
#from fitz.utils import getJSON 

# For Windows default installation path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

logging.basicConfig(filename='pdf_processing.log', level=logging.INFO)

def process_pdf(pdf_path, config):
    try:
        doc = fitz.open(pdf_path)
        full_content = {"pages": []}
        
        for page in doc:
            content = process_page(page, config)
            full_content["pages"].append(content)
            
        return full_content
    except Exception as e:
        logging.error(f"Failed to process {pdf_path}: {str(e)}")
    return None

def process_txt(txt_path):
    try:
        with open(txt_path, "r", encoding="utf-8") as f:
            text_content = f.read()
        # Wrap the text to mimic a single-page structure from your PDF output
        return {"pages": [{"text": [text_content]}]}
    except Exception as e:
        logging.error(f"Failed to process TXT file {txt_path}: {str(e)}")
    return None

def process_page(page, config):
    content = {"text": [], "tables": []}
    
    # Extract native text from the PDF page
    blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_LIGATURES)["blocks"]
    native_text = ""
    for b in blocks:
        # Check if the block is a text block (type 0)
        if b.get("type", 1) == 0:
            for line in b.get("lines", []):
                for span in line.get("spans", []):
                    native_text += span.get("text", "") + " "
    native_text = native_text.strip()
    
    if native_text:
        # Use native extraction result if available
        content["text"].append(native_text)
    elif config.get('ocr_enabled', False):
        # Optionally pre-process the image for better OCR performance
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # Increased resolution
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        img = img.convert("L")  # Convert to grayscale for improved OCR accuracy
        # Use Tesseract with any additional config, for example specifying English as language
        ocr_text = pytesseract.image_to_string(img, config=config.get("tesseract_config", ""), lang="eng")
        content["text"].append(ocr_text)
    
    # Table handling remains unchanged
    if config.get('table_detection', False):
        tables = page.find_tables()
        content["tables"] = [tbl.extract() for tbl in tables]  # JSON format

    return content

def process_folder(input_folder, output_folder, config):
    """
    Processes files from the input_folder, handling PDFs and TXT files.
    For PDFs, it uses process_pdf and for TXT files, it uses process_txt.
    The output format is identical to your original PDF-only implementation.
    """
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(input_folder, filename)
            try:
                content = process_pdf(pdf_path, config)
                if content:
                    clean_name = sanitize_filename(os.path.splitext(filename)[0])
                    output_path = os.path.join(output_folder, f"{clean_name}.txt")
                    pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(str(content))
            except Exception as e:
                logging.error(f"Failed {filename}: {str(e)}")
        elif filename.lower().endswith(".txt"):
            txt_path = os.path.join(input_folder, filename)
            try:
                content = process_txt(txt_path)
                if content:
                    clean_name = sanitize_filename(os.path.splitext(filename)[0])
                    output_path = os.path.join(output_folder, f"{clean_name}.txt")
                    pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(str(content))
            except Exception as e:
                logging.error(f"Failed {filename}: {str(e)}")

if __name__ == "__main__":
    config = {
    "table_detection": True,
    "ocr_enabled": True,
    "tesseract_config": "--oem 3 --psm 6"
    }
    
    # Get the directory where the script lives
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # Navigate up from src/ to project root
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

    INPUT_DIR = os.path.join(PROJECT_ROOT, "data", "IESE")
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "preprocessed_data")
    
    process_folder(INPUT_DIR, OUTPUT_DIR, config)
