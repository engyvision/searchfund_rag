import faiss
import numpy as np
import json
import os

# Define the dimension of your embeddings.
# For example, text-embedding-3-small produces vectors of dimension 1536.
embedding_dim = 1536

# Load the embeddings from disk.
embeddings_output_file = "../data/embeddings.json"
with open(embeddings_output_file, "r", encoding="utf-8") as f:
    embeddings = json.load(f)

# Prepare lists to hold vector data and associated metadata.
vector_list = []       # Will contain vectors as NumPy arrays.
metadata_list = []     # Additional info for each vector.
id_list = []           # Unique IDs for each vector entry.

# Iterate over the embeddings dictionary.
for file_name, embed_data in embeddings.items():
    # Check if embed_data is a list.
    if isinstance(embed_data, list):
        # If the list is not empty and its first element is a dictionary (i.e., chunked data)
        if len(embed_data) > 0 and isinstance(embed_data[0], dict):
            # Process each chunk from the file.
            for chunk in embed_data:
                # Create a unique ID using file name and chunk index.
                unique_id = f"{file_name}_chunk_{chunk['chunk_index']}"
                vector_list.append(chunk["embedding"])
                id_list.append(unique_id)
                metadata_list.append({
                    "file": file_name,
                    "chunk_index": chunk["chunk_index"]
                })
        else:
            # The list is not chunked (it's a single embedding vector: a list of floats).
            unique_id = file_name
            vector_list.append(embed_data)
            id_list.append(unique_id)
            metadata_list.append({"file": file_name})
    else:
        # If embed_data is not a list, then it's a single embedding vector.
        unique_id = file_name
        vector_list.append(embed_data)
        id_list.append(unique_id)
        metadata_list.append({"file": file_name})

# Convert the vector list to a NumPy array of type float32.
vectors_np = np.array(vector_list, dtype=np.float32)

# Build a simple FAISS index using FlatL2 (Euclidean distance).
index = faiss.IndexFlatL2(embedding_dim)

# Add vectors to the index.
index.add(vectors_np)
print(f"FAISS index built with {index.ntotal} vectors.")

# Optionally save the FAISS index to disk.
faiss_index_file = "../data/faiss_index.idx"
faiss.write_index(index, faiss_index_file)
print(f"FAISS index saved to {faiss_index_file}")

# Save the metadata mapping to a JSON file.
metadata_output_file = "../data/faiss_metadata.json"
with open(metadata_output_file, "w", encoding="utf-8") as meta_file:
    json.dump({
        "ids": id_list,
        "metadata": metadata_list
    }, meta_file)
print(f"Metadata for the index saved to {metadata_output_file}")
