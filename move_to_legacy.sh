#!/bin/bash

# Script to move obsolete files to a legacy backup directory after restructuring

# Create legacy backup directory if it doesn't exist
LEGACY_DIR="src/legacy_backup"
mkdir -p "$LEGACY_DIR"

# List of files to be moved
OBSOLETE_FILES=(
  "src/FAISS_index.py"
  "src/config.py"
  "src/gen_embeddings.py"
  "src/iese_pdf_scraper.py"
  "src/pdf_extractor.py"
  "src/pdf_extractor_adv_old"
  "src/pdf_extractor_advanced.py"
  "src/preprocess_data.py"
  "src/rag_test.py"
  "src/retrieval.py"
  "src/scrape.py"
  "src/app.py"
  "src/pdf_processing.log"
)

echo "Moving obsolete files to $LEGACY_DIR..."

# Move each file if it exists
for file in "${OBSOLETE_FILES[@]}"; do
  if [ -f "$file" ]; then
    filename=$(basename "$file")
    echo "Moving $file to $LEGACY_DIR/$filename"
    mv "$file" "$LEGACY_DIR/$filename"
  else
    echo "File $file does not exist, skipping"
  fi
done

echo "Done! The obsolete files have been moved to $LEGACY_DIR."
echo "You can test the new structure and then remove the $LEGACY_DIR directory when ready."