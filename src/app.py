import os
import logging
import json
import faiss
import numpy as np
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]  # You can add FileHandler if you want a log file
)

# Load environment variables (e.g., OpenAI API key)
load_dotenv()
logging.info("Environment variables loaded.")

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
logging.info("OpenAI client initialized.")

# Load FAISS index and metadata
faiss_index_file = "../data/faiss_index.idx"
metadata_file = "../data/faiss_metadata.json"

logging.info("Loading FAISS index from: %s", faiss_index_file)
index = faiss.read_index(faiss_index_file)
logging.info("FAISS index loaded with %d vectors.", index.ntotal)

with open(metadata_file, "r", encoding="utf-8") as meta_file:
    metadata = json.load(meta_file)
logging.info("Metadata loaded from: %s", metadata_file)

# Function to embed a query
def embed_query(query: str, model="text-embedding-3-small"):
    logging.debug("Embedding query: %s", query)
    response = client.embeddings.create(input=query, model=model)
    embedding = response.data[0].embedding
    logging.debug("Embedding generated with %s tokens.", response.usage.total_tokens if hasattr(response, "usage") and hasattr(response.usage, "total_tokens") else "unknown")
    return np.array(embedding, dtype=np.float32)

# Function to retrieve top-k documents
def retrieve(query: str, top_k: int = 5):
    logging.info("Retrieving top %d documents for query: %s", top_k, query)
    query_vec = embed_query(query)
    query_vec = np.expand_dims(query_vec, axis=0)
    distances, indices = index.search(query_vec, top_k)
    results = []
    for idx in indices[0]:
        try:
            vector_id = metadata["ids"][idx]
            result_info = metadata["metadata"][metadata["ids"].index(vector_id)]
            results.append(result_info)
            logging.debug("Retrieved result: %s", result_info)
        except Exception as e:
            logging.error("Error processing index %d: %s", idx, e)
    return distances, indices, results

# Function to clarify the query using GPT-4o-mini
def clarify_query(query):
    logging.info("Clarifying query: %s", query)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                "You are a query clarification assistant for a search fund expert system. "
                "Rewrite the user's question to clarify their intent (e.g., whether they are a searcher, investor, "
                "or business owner) and make it more specific. Do not assume the user's role—infer it from their question."
                "If ambiguity remains, assume it is a search fund entrepreneur seeking advice."
             )},
            {"role": "user", "content": query}
        ],
        max_tokens=50,
        temperature=0.5
    )
    clarified = response.choices[0].message.content.strip()
    logging.info("Clarified query: %s", clarified)
    return clarified

# Function to generate an answer using GPT-4o
def generate_answer_with_context(query, retrieved_results):
    # Combine retrieved results into a context string
    context = "\n\n".join([f"Document {i+1}: {result}" for i, result in enumerate(retrieved_results)])
    prompt = f"""
    You are an expert on search funds. Answer the following question based on the provided context:

    Context:
    {context}

    Question: {query}

    Answer:
    """
    logging.info("Generating final answer with prompt:\n%s", prompt)
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a search fund expert assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=500,
        temperature=0.7,
    )
    
    # Use getattr to obtain token usage; using dot notation for other attributes
    usage = getattr(response, "usage", None)
    if usage:
        logging.info("Token usage: %s", usage)
    else:
        logging.warning("Token usage information is not available in the response.")
    
    # Access the final answer using dot notation
    final_answer = response.choices[0].message.content.strip()
    logging.info("Final answer generated.")
    return final_answer

# Streamlit UI
st.title("Search Fund Assistant")
st.write("Ask any question about search funds and get expert answers!")

# Input field for user query
query = st.text_input("Enter your query:")

if st.button("Submit"):
    if query.strip():
        with st.spinner("Processing..."):
            # Step 1: Clarify the query
            clarified_query = clarify_query(query)
            st.subheader("Clarified Query")
            st.write(clarified_query)

            # Step 2: Retrieve documents
            distances, indices, results = retrieve(clarified_query, top_k=3)
            st.subheader("Retrieved Documents")
            for i, result in enumerate(results):
                st.write(f"**Document {i+1}:** {result}")

            # Step 3: Generate final answer
            final_answer = generate_answer_with_context(clarified_query, results)
            st.subheader("Final Answer")
            st.write(final_answer)
        
        st.success("Done !")
    else:
        st.error("Please enter a valid query.")
