import os
import re

# Define the directory where your .txt files are stored
data_directory = "../data/preprocessed_data"  # Update this to your actual folder name

# Function to clean text
def clean_text(text):
    # Remove extra spaces and newlines, normalize the text
    text = re.sub(r'\s+', ' ', text)  # Replace consecutive whitespace with a single space
    text = re.sub(r'[\r\n]+', ' ', text)  # Replace newlines with a space
    text = re.sub(r'[^\w\s.,!?]', '', text)  # Remove special characters except punctuation
    return text.strip()

# Dictionary to store preprocessed data
preprocessed_data = {}

# Process each .txt file in your data directory
for filename in os.listdir(data_directory):
    if filename.endswith(".txt"):
        file_path = os.path.join(data_directory, filename)
        with open(file_path, "r", encoding="utf-8") as file:
            raw_text = file.read()
            cleaned_text = clean_text(raw_text)
            preprocessed_data[filename] = cleaned_text

# Define output directory for preprocessed files
output_directory = "../data/processed_txt"
os.makedirs(output_directory, exist_ok=True)

# Save the cleaned text to new files in the output directory
for filename, content in preprocessed_data.items():
    output_path = os.path.join(output_directory, filename)
    with open(output_path, "w", encoding="utf-8") as file:
        file.write(content)

# List the preprocessed files for verification
print("Preprocessed files:", os.listdir(output_directory))
