"""
Streamlit web application for the Search Fund Assistant.

This module provides a web interface for users to interact with the
search fund assistant, submitting queries and receiving answers.
"""

import streamlit as st

from src.core import get_logger
from src.llm import QueryClarifier, AnswerGenerator
from src.retrieval import DocumentRetriever

# Initialize logger
logger = get_logger("app.streamlit")

# Initialize components
query_clarifier = None
document_retriever = None
answer_generator = None

def initialize_components():
    """Initialize application components."""
    global query_clarifier, document_retriever, answer_generator
    
    if query_clarifier is None:
        with st.spinner("Loading query clarifier..."):
            query_clarifier = QueryClarifier()
    
    if document_retriever is None:
        with st.spinner("Loading document retriever..."):
            document_retriever = DocumentRetriever()
    
    if answer_generator is None:
        with st.spinner("Loading answer generator..."):
            answer_generator = AnswerGenerator()

def process_query(query: str, top_k: int = 3):
    """Process a user query.
    
    Args:
        query: The user's query
        top_k: Number of documents to retrieve
        
    Returns:
        tuple: Clarified query, retrieved documents, and final answer
    """
    # Step 1: Clarify the query
    clarified_query = query_clarifier.clarify_query(query)
    
    # Step 2: Retrieve documents
    context = document_retriever.retrieve_and_format(
        clarified_query, 
        top_k=top_k,
        include_metadata=True
    )
    
    # Step 3: Generate answer
    answer = answer_generator.generate_answer(
        clarified_query,
        context
    )
    
    return clarified_query, context, answer

def main():
    """Main function for the streamlit app."""
    st.title("Search Fund Assistant")
    st.write("Ask any question about search funds and get expert answers!")
    
    # Initialize components
    initialize_components()
    
    # Create a text input for the user query
    query = st.text_input("Enter your query:")
    
    # Create a button to submit the query
    if st.button("Submit"):
        if query.strip():
            with st.spinner("Processing..."):
                # Process the query
                clarified_query, context, answer = process_query(query)
                
                # Display results
                st.subheader("Clarified Query")
                st.write(clarified_query)
                
                st.subheader("Retrieved Documents")
                st.expander("View Document Context", expanded=False).text(context)
                
                st.subheader("Final Answer")
                st.write(answer)
            
            st.success("Done!")
        else:
            st.error("Please enter a valid query.")

if __name__ == "__main__":
    main()