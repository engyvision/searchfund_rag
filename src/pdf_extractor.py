import os
import fitz  # PyMuPDF

def process_pdf(pdf_path, header_threshold=100, footer_threshold=150):
    """
    Extracts text from PDF while removing headers/footers using coordinate filtering.
    Returns cleaned text as a single string.
    """
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"Error opening {pdf_path}: {e}")
        return ""

    extracted_text = []
    
    for page in doc:
        page_height = page.rect.height
        blocks = page.get_text("blocks")  # Gets blocks in natural reading order
        
        page_text = []
        for block in blocks:
            # Unpack block coordinates and text
            x0, y0, x1, y1, text, block_no, block_type = block
            
            # Filter header/footer regions
            if (y0 > header_threshold) and (y1 < (page_height - footer_threshold)):
                clean_text = text.strip()
                if clean_text:
                    page_text.append(clean_text)
        
        if page_text:
            extracted_text.append("\n".join(page_text))
    
    return "\n\n".join(extracted_text)

def process_folder(input_folder, output_folder):
    """Process all PDFs in input_folder and save results to output_folder"""
    os.makedirs(output_folder, exist_ok=True)
    
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(input_folder, filename)
            print(f"Processing: {pdf_path}")
            
            try:
                text = process_pdf(pdf_path)
                base_name = os.path.splitext(filename)[0]
                output_path = os.path.join(output_folder, f"{base_name}.txt")
                
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(text)
                print(f"Saved: {output_path}\n")
                
            except Exception as e:
                print(f"Failed to process {filename}: {e}")

if __name__ == "__main__":
    # Get the directory where the script lives
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # Navigate up from src/ to project root
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
    
    # Construct correct paths (modify these if your folder names differ)
    INPUT_DIR = os.path.join(
        PROJECT_ROOT,  # Now points to parent of src/
        "data",
        "IESE",
        "old"
    )
    
    OUTPUT_DIR = os.path.join(
        PROJECT_ROOT,
        "data",
        "processed_txt"
    )

    print(f"Looking for PDFs in: {INPUT_DIR}")
    print(f"Output will be saved to: {OUTPUT_DIR}\n")

    if os.path.exists(INPUT_DIR):
        process_folder(INPUT_DIR, OUTPUT_DIR)
    else:
        print(f"Input directory not found: {INPUT_DIR}")
        print("\nTroubleshooting suggestions:")
        print("1. Verify the folder structure exists:")
        print(f"   {os.path.join(PROJECT_ROOT, 'webscraper_project')}")
        print("2. Check for typos in folder names ('webscraper' vs 'webscrapper')")
        print("3. Ensure you have at least one PDF in the 'old' folder")
