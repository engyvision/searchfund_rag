from dotenv import load_dotenv
import os
import numpy as np
import faiss
import json
from openai import OpenAI
import tiktoken


# Load environment variables from .env file
load_dotenv()

# Instantiate the OpenAI client (make sure your API key is loaded)
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def embed_query(query: str, model="text-embedding-3-small"):
    """
    Embeds the user query using the specified embedding model.
    Returns:
       A numpy float32 array representing the query embedding.
    """
    response = client.embeddings.create(input=query, model=model)
    embedding = response.data[0].embedding
    
    return np.array(embedding, dtype=np.float32)

# Define the path to the FAISS index and metadata
faiss_index_file = "../data/faiss_index.idx"
metadata_file = "../data/faiss_metadata.json"

# Load the index
index = faiss.read_index(faiss_index_file)
print(f"Loaded FAISS index with {index.ntotal} vectors.")

# Load metadata mapping
with open(metadata_file, "r", encoding="utf-8") as meta_file:
    metadata = json.load(meta_file)
    # metadata should contain two keys: "ids" (list of unique IDs) and "metadata" (list of associated metadata)

def retrieve(query: str, top_k: int = 5):
    """
    Retrieves the top_k most similar documents based on the user query.
    
    Returns:
       A tuple containing:
         - distances: The Euclidean distances from the query to the retrieved vectors.
         - indices: The index positions in the FAISS index.
         - results: A list of metadata for each retrieved vector.
    """
    # Step 1: Embed the query
    query_vec = embed_query(query)
    query_vec = np.expand_dims(query_vec, axis=0)  # FAISS expects a 2D array
    
    # Step 2: Search the FAISS index for top_k similar vectors
    distances, indices = index.search(query_vec, top_k)
    
    # Step 3: Map the indices to metadata. Note that metadata['ids'] lists 
    # the vector IDs in the same order as they were added, and metadata['metadata'] holds the related info.
    results = []
    for idx in indices[0]:
        # Retrieve metadata using the vector ID.
        # Here we assume the order in metadata['ids'] matches the order in the index.
        result_info = metadata["metadata"][metadata["ids"].index(metadata["ids"][idx])]
        results.append(result_info)
    
    return distances, indices, results

# Example usage:
query_text = "how do i raise capital?"
dists, inds, results = retrieve(query_text, top_k=3)
print("Retrieved results:")
for res in results:
    print(res)
