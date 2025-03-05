import faiss
import numpy as np
import json
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables (e.g., OpenAI API key)
load_dotenv()

# Initialize OpenAI client
client = OpenAI()

# Load FAISS index and metadata
faiss_index_file = "../data/faiss_index.idx"
metadata_file = "../data/faiss_metadata.json"

index = faiss.read_index(faiss_index_file)
with open(metadata_file, "r", encoding="utf-8") as meta_file:
    metadata = json.load(meta_file)

# --- Step 1: Retrieval-Augmented Generation (RAG) ---
def embed_query(query: str, model="text-embedding-3-small"):
    response = client.embeddings.create(input=query, model=model)
    embedding = response.data[0].embedding
    return np.array(embedding, dtype=np.float32)

def retrieve(query: str, top_k: int = 5):
    query_vec = embed_query(query)
    query_vec = np.expand_dims(query_vec, axis=0)
    distances, indices = index.search(query_vec, top_k)
    results = []
    for idx in indices[0]:
        result_info = metadata["metadata"][metadata["ids"].index(metadata["ids"][idx])]
        results.append(result_info)
    return distances, indices, results

def generate_answer_with_context(query, retrieved_results):
    context = "\n\n".join([f"Document {i+1}: {result}" for i, result in enumerate(retrieved_results)])
    prompt = f"""
    You are an expert on search funds. Answer the following question based on the provided context:

    Context:
    {context}

    Question: {query}

    Answer:
    """
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a search fund expert assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=500,
        temperature=0.7,
    )
    return response.choices[0].message.content.strip()

# --- Step 2: Query Clarification ---
def clarify_query(query):
    """
    Dynamically infers user intent (e.g., searcher vs. investor) and clarifies the query.
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": """
                You are a query clarification assistant for a search fund expert system. 
                Rewrite the user's question to clarify their intent (e.g., whether they are a searcher, investor, 
                or business owner) and make it more specific. Do not assume the user's role—infer it from their question.
                """
            },
            {"role": "user", "content": query}
        ],
        max_tokens=50,
        temperature=0.5
    )
    return response.choices[0].message.content.strip()


# --- Test Pipeline ---
def test_pipeline(query):
    # Step 1: Clarify the query
    clarified = clarify_query(query)
    print(f"Original Query: {query}")
    print(f"Clarified Query: {clarified}\n")

    # Step 2: Retrieve documents
    distances, indices, results = retrieve(clarified, top_k=3)
    print(f"Retrieved Metadata: {results}\n")

    # Step 3: Generate answer
    answer = generate_answer_with_context(query, results)
    print(f"Generated Answer:\n{answer}")

# Example Test Case
test_pipeline("What should i consider in an SPA?")
