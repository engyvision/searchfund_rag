from dotenv import load_dotenv
import os
import json
import tiktoken  # For accurate token counting
from openai import OpenAI
from tenacity import retry, wait_random_exponential, stop_after_attempt

# Load environment variables from .env file
load_dotenv()

# Create an OpenAI client with your API key
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Define directories and output file paths; adjust as needed.
preprocessed_directory = "../data/preprocessed_data"
embeddings_output_file = "../data/embeddings.json"
log_file = "../data/embedding_log.json"

# Set pricing (in dollars) per 1K tokens.
COST_PER_K_TOKEN = 0.00002

# Token limit for embedding models (e.g., text-embedding-3-small)
TOKEN_LIMIT = 8192

# Load existing embeddings and log data if they exist; otherwise, initialize new dictionaries.
if os.path.exists(embeddings_output_file):
    with open(embeddings_output_file, "r", encoding="utf-8") as file:
        embeddings = json.load(file)
else:
    embeddings = {}

if os.path.exists(log_file):
    with open(log_file, "r", encoding="utf-8") as file:
        log_data = json.load(file)
else:
    log_data = {}

# Summary counters for overall processing.
embedding_count = 0  # Total embedding API calls (including individual chunks)
total_tokens = 0
total_cost = 0.0

# Function to count tokens accurately using tiktoken with the 'cl100k_base' encoding.
def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(string))

# Function to split a long text into chunks, each within the TOKEN_LIMIT.
def split_text_into_chunks(text: str, max_tokens: int, overlap: int = 50, encoding_name: str = "cl100k_base"):
    """
    Splits text into chunks of at most max_tokens tokens.
    An overlap of a specified number of tokens between consecutive chunks is used to preserve context.
    """
    encoding = tiktoken.get_encoding(encoding_name)
    tokens = encoding.encode(text)
    chunks = []
    start = 0
    # Loop over tokens in increments ensuring overlap.
    while start < len(tokens):
        end = start + max_tokens
        chunk_tokens = tokens[start:end]
        chunk_text = encoding.decode(chunk_tokens)
        chunks.append(chunk_text)
        start += max_tokens - overlap  # Overlap for context.
    return chunks

# Wrapper function for getting an embedding with retry logic.
@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
def get_embedding(text: str, model="text-embedding-3-small"):
    response = client.embeddings.create(input=text, model=model)
    embedding = response.data[0].embedding
    tokens_used = response.usage.total_tokens if hasattr(response, "usage") and hasattr(response.usage, "total_tokens") else None
    return embedding, tokens_used

# Process each preprocessed file in the directory.
for filename in os.listdir(preprocessed_directory):
    if filename.endswith(".txt"):
        file_path = os.path.join(preprocessed_directory, filename)

        # Skip this file if an embedding already exists.
        if filename in embeddings:
            print(f"Skipping {filename} as embedding already exists.")
            continue

        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read().strip()

            # Obtain an accurate token count using tiktoken.
            try:
                estimated_tokens = num_tokens_from_string(text, "cl100k_base")
                print(f"Processing {filename} - Estimated tokens: {estimated_tokens}")
            except Exception as e:
                print(f"Error counting tokens for {filename}: {e}")
                log_data[filename] = {"status": "error", "error_message": f"Token counting failed: {e}"}
                continue

            # If text length exceeds TOKEN_LIMIT, use chunking.
            if estimated_tokens > TOKEN_LIMIT:
                print(f"{filename} exceeds token limit. Splitting into chunks.")
                chunks = split_text_into_chunks(text, max_tokens=TOKEN_LIMIT, overlap=50, encoding_name="cl100k_base")
                file_embeddings = []  # List to store embedding details for each chunk.
                for idx, chunk in enumerate(chunks):
                    try:
                        embedding, tokens_used = get_embedding(chunk, model="text-embedding-3-small")
                        # Use the token count provided by the API if available; otherwise, recalc using tiktoken.
                        tokens_for_chunk = tokens_used if tokens_used is not None else num_tokens_from_string(chunk, "cl100k_base")
                        file_embeddings.append({
                            "chunk_index": idx,
                            "embedding": embedding,
                            "tokens_used": tokens_for_chunk
                        })
                        embedding_count += 1
                        total_tokens += tokens_for_chunk
                        total_cost += (tokens_for_chunk / 1000.0) * COST_PER_K_TOKEN
                        print(f"Successfully generated embedding for {filename} chunk {idx} with {tokens_for_chunk} tokens.")
                    except Exception as e:
                        print(f"Error generating embedding for {filename} chunk {idx}: {e}")
                        # Log errors for individual chunks.
                        if filename not in log_data:
                            log_data[filename] = {"chunks": {}}
                        log_data[filename]["chunks"][str(idx)] = {"status": "error", "error_message": str(e)}
                # Store the list of chunk embeddings under the file's key.
                embeddings[filename] = file_embeddings
                log_data[filename] = {
                    "status": "success",
                    "chunks_processed": len(file_embeddings),
                    "estimated_tokens": estimated_tokens
                }
            else:
                # If within token limit, process the file normally.
                try:
                    embedding, tokens_used = get_embedding(text, model="text-embedding-3-small")
                    embeddings[filename] = embedding
                    embedding_count += 1
                    if tokens_used is None:
                        tokens_used = estimated_tokens
                    cost = (tokens_used / 1000.0) * COST_PER_K_TOKEN
                    total_tokens += tokens_used
                    total_cost += cost
                    log_data[filename] = {
                        "status": "success",
                        "tokens_used": tokens_used,
                        "cost": cost,
                        "estimated_tokens": estimated_tokens,
                    }
                    print(f"Successfully generated embedding for {filename} with {tokens_used} tokens, cost ${cost:.4f}")
                except Exception as e:
                    log_data[filename] = {"status": "error", "error_message": str(e)}
                    print(f"Error generating embedding for {filename}: {e}")

# Save embeddings and log data to their respective JSON files.
with open(embeddings_output_file, "w", encoding="utf-8") as output_file:
    json.dump(embeddings, output_file)

with open(log_file, "w", encoding="utf-8") as log_output:
    json.dump(log_data, log_output)

# Print overall summary.
print(f"\nSummary:")
print(f"Embeddings generated (including all chunks): {embedding_count}")
print(f"Total tokens used: {total_tokens}")
print(f"Total cost: ${total_cost:.4f}")
print(f"Embeddings saved to: {embeddings_output_file}")
print(f"Log data saved to: {log_file}")
