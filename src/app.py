import os
import logging
import json
import faiss
import numpy as np
import streamlit as st
from openai import OpenAI
from config import OPENAI_API_KEY  # Import from config.py
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
client = OpenAI(api_key=OPENAI_API_KEY)
logging.info("OpenAI client initialized.")

# Get the directory of the current script (app.py)
current_dir = os.path.dirname(os.path.abspath(__file__))  # Returns the path to src/

# Navigate up one level to the project root (where data/ is located)
project_root = os.path.dirname(current_dir)

# Load FAISS index and metadata
faiss_index_file = os.path.join(project_root, "data", "faiss_index.idx")
metadata_file = os.path.join(project_root, "data", "faiss_metadata.json")

# Check if files exist locally or in deployment
if not os.path.exists(faiss_index_file):
    raise FileNotFoundError(f"FAISS index file not found: {faiss_index_file}")
if not os.path.exists(metadata_file):
    raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

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
                "content": """
                Understand the user query, infer the type of user (such as a searcher, investor, or business owner), and rephrase the query to be more specific and suitable for analysis by an AI specialized in search funds. If the type of user cannot be inferred from the query, assume the user is a searcher.

# Steps

1. **Analyze the query**: Break down the user's question to identify key elements that might indicate the user's identity or goals.
2. **Infer user type**: Use indicators from the query to deduce whether the user is a searcher, investor, business owner, or another relevant type. If unclear, revert to the default assumption of the user being a searcher (aka search fund entrepreneur).
3. **Rephrase the query**: Reformulate the user's query to enhance specificity and clarity, ensuring it is tailored to the user's inferred identity and intentions.
4. **Ensure clarity**: Confirm that the rephrased query removes ambiguity and is designed for precise interpretation by the AI focused on search funds.

# Output Format

- Present the rephrased query clearly.

# Examples

**Example 1:**

- **Original Query:** "How does market volatility affect my investments?"
- **User Type Analysis:** Investor inferred from mention of "investments."
- **Rephrased Query:** "As an investor, how do fluctuations in the market impact the search fund sector specifically?"

**Example 2:**

- **Original Query:** "Why should I sell to a search fund ?"
- **User Type Analysis:** Business owner inferred from "sell to a search fund."
- **Rephrased Query:** "For a small business owner, what are the main advantages of selling their business specifically to a search fund entrepreneur rather than to a PE fund or strategic buyer?"

**Example 3:**

- **Original Query:** "What are key factors in evaluating a company?"
- **User Type Analysis:** Cannot specifically infer user type, assume search fund entrepreneur.
- **Rephrased Query:** "As a search fund entrepreneur, what key factors should be considered when evaluating a company's potential for acquisition?"

# Notes

- In cases where elements pointing to a specific user type are ambiguous, default to assuming a search fund entrepreneur.
- Ensure that the rephrased query is distinct and clearly outlines specific concerns related to the search fund context.
                """
            },
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
        model="o3-mini-2025-01-31",
        messages=[
            {"role": "system", "content": """
             You are tasked with acting as a search fund expert to answer a specific question based on a provided context and augmented with your own knowledge and web results, if available. Ensure your response targets the correct user group—either a searcher, search fund investor, business owner, or intermediary—based on the question's details. The provided context should be considered the most reliable source unless newer or strongly contradictory evidence is found through your knowledge or online sources, in which case this should be highlighted in your response.

# Steps

1. **Understanding the Question**: Carefully read and comprehend the question to ensure it is appropriately targeted and complete.

2. **Analyzing the Context**: Review the provided context thoroughly as it serves as the primary source of information for the question.

3. **Identifying the Audience**: Determine whether the context is more relevant to search fund investors, searchers, business owners, or intermediaries and adjust your response accordingly.

4. **Applying AI Knowledge**: Incorporate your existing knowledge of the search fund landscape to enhance the response.

5. **Checking Web Results**: If necessary, obtain current web information or updates on the topic that may not be covered by the context or your knowledge.

6. **Comparison**: Compare findings from the context, AI knowledge, and web results. Identify any contradictions, especially if newer information invalidates the context.

7. **Construct the Response**: Synthesize all the data to provide a comprehensive, well-reasoned answer enriched by multiple sources, targeting the correct user group.

# Output Format

Provide a structured and well-organized response. Highlight the sources of each piece of information within, and ensure the answer is aligned with the identified audience.

# Examples

**Example 1:**

**Input:**
- **Question:** What are the latest trends in the search fund industry for entrepreneurs?
- **Provided Context:** [Details of a 2022 industry report discussing trends in search fund strategies and successes.]

**Processing:**
- **Context Analysis:** Review the industry report from 2022.
- **AI Knowledge:** Recall relevant trends known up to 2023.
- **Web Check:** Search for the latest news or articles on search fund trends as of 2023.
- **Audience Identification:** Ensure context relevance for search fund entrepreneurs.

**Output:**
"According to the 2022 industry report, emerging trends for search fund entrepreneurs include increasing focus on niche markets and digital transformation. My updated knowledge suggests a wave of strategic partnerships as of 2023, confirmed by a recent article from October 2023. Although the context aligns with these points, the emphasis on sustainability noted this year is a key addition missing from the report for entrepreneurs."

(Replace placeholders with real data; real examples should be more detailed.)

# Notes

- Clearly explain any substantial contradictions between the provided context and new findings.
- Always consider the timeliness and reliability of web sources used in the response.
- Ensure clarity in explanations and conclusions, while targeting the identified audience.
- When generating answers, always indicate which parts of the response come directly from the retrieved context using phrases like "According to the [document source name])...". 
- If any part of the response relies on your own knowledge, clearly preface it with "Additionally, based on [my own knowledge] or [specific web source]...".
             """},
            {"role": "user", "content": prompt}
        ],
        reasoning_effort="medium",
        max_completion_tokens=5000,
        # temperature=0.7,
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
